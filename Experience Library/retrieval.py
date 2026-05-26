"""
Memory Retrieval Engine.

Implements the 9-step retrieval algorithm:

    1. Always retrieve latest latent state.
    2. Always retrieve latest concession node.
    3. Always retrieve strategy stats.
    4. Perform semantic search over session nodes.
    5. For top-2 sessions → expand 1-hop via same_year + temporal_next.
    6. Merge all retrieved nodes.
    7. Deduplicate.
    8. Rank by relevance.
    9. Fit under token budget.

Supports three scoring signals:
    • Semantic similarity  (sentence-transformers embeddings)
    • BM25 keyword match   (rank_bm25 or simple fallback)
    • Graph activation     (spreading activation over neighbours)

Session node hybrid score (paper Appendix C.3, Eq. for score(M^sess)):
    score = λ * sim_sem + (1 - λ) * sim_lex
    where λ = 0.8  (SEMANTIC_WEIGHT)
    Graph activation expands the top-k sessions but does NOT enter the
    session-scoring formula; it is applied separately in Step 5.
"""

import math
import re
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from memory_graph import PersonaMemoryGraph, MemoryNode

# ====================================================================
# OPTIONAL IMPORTS (graceful fallback)
# ====================================================================
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


# ====================================================================
# EMBEDDING ENGINE
# ====================================================================

class EmbeddingEngine:
    """
    Produces dense vector embeddings for text.

    Paper (Appendix C.3) specifies EmbeddingGemma (Schechter Vera et al.,
    2025) as the embedding model.  We use the sentence-transformers
    equivalent (all-MiniLM-L6-v2) as a drop-in; swap `model_name` for the
    official EmbeddingGemma checkpoint when available.
    Fallback: simple TF–IDF bag-of-words (no extra deps).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._global_vocab: Dict[str, int] = {}   # for TF-IDF fallback
        self._idf: Dict[str, float] = {}
        self._load_model()

    def _load_model(self):
        if HAS_SBERT:
            try:
                print(f"  [Embedding] Loading {self.model_name} ...")
                self.model = SentenceTransformer(self.model_name)
                print("  [Embedding] Ready.")
            except Exception as e:
                fallback = "all-MiniLM-L6-v2"
                print(f"  [Embedding] Failed to load {self.model_name}: {e}")
                print(f"  [Embedding] Falling back to {fallback}")
                self.model_name = fallback
                self.model = SentenceTransformer(self.model_name)
        else:
            print("  [Embedding] sentence-transformers not found → TF-IDF fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        if self.model:
            return self.model.encode(text, normalize_embeddings=True)
        return self._tfidf_embed(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if self.model:
            return self.model.encode(texts, normalize_embeddings=True)
        # Build IDF from corpus first
        self._build_idf(texts)
        return np.array([self._tfidf_embed(t) for t in texts])

    # ------------------------------------------------------------------
    # TF-IDF fallback
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_idf(self, texts: List[str]):
        """Build a global vocabulary and IDF from a corpus."""
        doc_freq: Dict[str, int] = defaultdict(int)
        all_tokens: Set[str] = set()
        n = len(texts)

        for t in texts:
            tokens = set(self._tokenize(t))
            all_tokens |= tokens
            for tok in tokens:
                doc_freq[tok] += 1

        self._global_vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}
        self._idf = {
            tok: math.log((n + 1) / (doc_freq.get(tok, 0) + 1)) + 1.0
            for tok in all_tokens
        }

    def _tfidf_embed(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        dim = max(len(self._global_vocab), 1)
        vec = np.zeros(dim)

        tf: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1

        for tok, count in tf.items():
            idx = self._global_vocab.get(tok)
            if idx is not None:
                vec[idx] = count * self._idf.get(tok, 1.0)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ====================================================================
# BM25 ENGINE
# ====================================================================

class BM25Engine:
    """Keyword search over node texts using BM25 (or simple fallback)."""

    def __init__(self):
        self.corpus: List[List[str]] = []
        self.node_ids: List[str] = []
        self.bm25 = None

    def index(self, nodes: Dict[str, MemoryNode]):
        self.corpus = []
        self.node_ids = []
        for nid, node in nodes.items():
            self.corpus.append(self._tokenize(node.text))
            self.node_ids.append(nid)
        if HAS_BM25 and self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.corpus:
            return []
        if self.bm25:
            scores = self.bm25.get_scores(self._tokenize(query))
            pairs = [(self.node_ids[i], float(scores[i])) for i in range(len(scores))]
        else:
            pairs = self._keyword_fallback(query)
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    # ------------------------------------------------------------------

    def _keyword_fallback(self, query: str) -> List[Tuple[str, float]]:
        qtoks = set(self._tokenize(query))
        results = []
        for i, doc_toks in enumerate(self.corpus):
            overlap = len(qtoks & set(doc_toks))
            score = overlap / max(len(qtoks), 1)
            results.append((self.node_ids[i], score))
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())


# ====================================================================
# GRAPH ACTIVATION (Spreading Activation)
# ====================================================================

class GraphActivation:
    """
    Spreading activation: propagate signal from seed nodes through edges.

    Each hop multiplies the signal by `decay`.
    Only goes through the requested edge types.
    """

    def __init__(self, decay: float = 0.5):
        self.decay = decay

    def activate(
        self,
        graph: PersonaMemoryGraph,
        seed_nodes: List[str],
        hops: int = 1,
        edge_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Returns { node_id: activation_score } for all reachable nodes.
        Seed nodes start with activation = 1.0.
        """
        activation: Dict[str, float] = {nid: 1.0 for nid in seed_nodes}
        frontier: Set[str] = set(seed_nodes)

        for _ in range(hops):
            next_frontier: Set[str] = set()
            for nid in frontier:
                cur_act = activation.get(nid, 0.0)
                for nbr_id, etype in graph.get_neighbors(nid, edge_types):
                    spread = cur_act * self.decay
                    if nbr_id not in activation or activation[nbr_id] < spread:
                        activation[nbr_id] = spread
                        next_frontier.add(nbr_id)
            frontier = next_frontier

        return activation


# ====================================================================
# MEMORY RETRIEVER  (9-step algorithm)
# ====================================================================

class MemoryRetriever:
    """
    Orchestrates the full retrieval pipeline.

    Constructor takes a built PersonaMemoryGraph, builds embeddings +
    BM25 index, then `retrieve(query)` executes the 9 steps.
    """

    def __init__(
        self,
        graph: PersonaMemoryGraph,
        embedding_model: str = "all-MiniLM-L6-v2",
        token_budget: int = 2000,
        semantic_weight: float = 0.8,   # λ in paper Appendix C.3
        bm25_weight: float = 0.2,       # (1-λ); must sum to 1 with semantic_weight
        graph_weight: float = 0.5,      # spreading-activation scale for Step-5 nodes
        activation_decay: float = 0.5,
    ):
        self.graph = graph
        self.token_budget = token_budget
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.graph_weight = graph_weight

        # Engines
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.bm25_engine = BM25Engine()
        self.graph_activation = GraphActivation(decay=activation_decay)

        self._build_indices()

    # ------------------------------------------------------------------
    # INDEX BUILDING
    # ------------------------------------------------------------------

    def _build_indices(self):
        """Embed all nodes and build BM25 index (on session nodes only)."""
        node_ids = list(self.graph.nodes.keys())
        texts = [self.graph.nodes[nid].text for nid in node_ids]

        if texts:
            embeddings = self.embedding_engine.embed_batch(texts)
            for i, nid in enumerate(node_ids):
                self.graph.nodes[nid].embedding = embeddings[i]

        # BM25 over session nodes only (Step 4 target)
        session_nodes = {
            nid: node for nid, node in self.graph.nodes.items()
            if node.node_type == "session"
        }
        self.bm25_engine.index(session_nodes)

    # ------------------------------------------------------------------
    # RETRIEVE  (the core 9-step algorithm)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
    ) -> List[MemoryNode]:
        """
        Execute the 9-step retrieval.

        Args:
            query:  Current user message (or combined context).
            conversation_history:  Recent user/agent utterances for context.

        Returns:
            Ordered list of MemoryNode objects, fitted under token budget.
        """
        retrieved: Dict[str, Tuple[MemoryNode, float]] = {}

        # ── STEP 1: latest latent state (always) ──────────────────
        self._must_retrieve(retrieved, "latent_cumulative")

        # ── STEP 2: latest concession node (always) ──────────────
        self._must_retrieve(retrieved, "concession_cumulative")

        # ── STEP 3: strategy stats (always) ──────────────────────
        self._must_retrieve(retrieved, "strategy_cumulative")

        # ── STEP 4: semantic search over session nodes ────────────
        session_nodes = self.graph.get_nodes_by_type("session")
        top_sessions: List[Tuple[str, float]] = []

        if session_nodes:
            # Build expanded query from history
            full_query = query
            if conversation_history:
                full_query = " ".join(conversation_history[-3:]) + " " + query

            # 4a. Semantic scores
            q_emb = self.embedding_engine.embed_text(full_query)
            sem_scores: Dict[str, float] = {}
            for node in session_nodes:
                if node.embedding is not None:
                    sim = self._cosine(q_emb, node.embedding)
                    sem_scores[node.node_id] = sim

            # 4b. BM25 scores
            bm25_raw = self.bm25_engine.search(full_query, top_k=len(session_nodes))
            bm25_scores = dict(bm25_raw)
            mx = max(bm25_scores.values()) if bm25_scores else 1.0
            if mx > 0:
                bm25_scores = {k: v / mx for k, v in bm25_scores.items()}

            # 4c. Combine — paper Appendix C.3:
            #   score = λ * sim_sem + (1-λ) * sim_lex   (λ = semantic_weight = 0.8)
            combined: Dict[str, float] = {}
            for nid in sem_scores:
                s = sem_scores.get(nid, 0.0)
                b = bm25_scores.get(nid, 0.0)
                combined[nid] = self.semantic_weight * s + self.bm25_weight * b
        seed_ids = [nid for nid, _ in top_sessions]
        if seed_ids:
            activations = self.graph_activation.activate(
                self.graph,
                seed_ids,
                hops=1,
                edge_types=["same_year", "temporal_next"],
            )
            for nid, act in activations.items():
                if nid not in retrieved:
                    node = self.graph.get_node(nid)
                    if node:
                        retrieved[nid] = (node, self.graph_weight * act)

        # ── STEP 6 + 7: merge + deduplicate (dict handles this) ──

        # ── STEP 8: rank by relevance ─────────────────────────────
        ranked = sorted(retrieved.values(), key=lambda x: x[1], reverse=True)

        # ── STEP 9: fit under token budget ────────────────────────
        final: List[MemoryNode] = []
        total_tokens = 0

        # Priority types that MUST be included
        priority_types = {"latent_cumulative", "strategy_cumulative", "concession_cumulative"}

        for node, score in ranked:
            est = node.token_estimate()
            if total_tokens + est <= self.token_budget:
                final.append(node)
                total_tokens += est
            elif node.node_type in priority_types:
                # Always include cumulative nodes even if slightly over budget
                final.append(node)
                total_tokens += est

        return final

    def retrieve_with_trace(
        self,
        query: str,
        conversation_history: Optional[List[str]] = None,
    ) -> Tuple[List[MemoryNode], Dict]:
        """
        Same as retrieve() but also returns a detailed trace dict
        showing which nodes were touched at each step and their scores.

        Returns:
            (final_nodes, trace_dict)

        trace_dict keys:
            step_1_latent:     [(node_id, score)]
            step_2_concession: [(node_id, score)]
            step_3_strategy:   [(node_id, score)]
            step_4_semantic:   [(node_id, score)]  -- all session scores
            step_4_bm25:       [(node_id, score)]
            step_4_combined:   [(node_id, score)]  -- top 2
            step_5_expanded:   [(node_id, score)]
            step_8_ranked:     [(node_id, score)]
            step_9_final:      [(node_id, score, tokens)]
            all_scores:        {node_id: score}    -- every touched node
        """
        trace: Dict = {}
        retrieved: Dict[str, Tuple[MemoryNode, float]] = {}

        # ── STEP 1 ──
        self._must_retrieve(retrieved, "latent_cumulative")
        trace["step_1_latent"] = [
            (nid, sc) for nid, (_, sc) in retrieved.items()
        ]

        # ── STEP 2 ──
        self._must_retrieve(retrieved, "concession_cumulative")
        trace["step_2_concession"] = [
            (nid, sc) for nid, (_, sc) in retrieved.items()
            if "concession_cumulative" in nid
        ]

        # ── STEP 3 ──
        self._must_retrieve(retrieved, "strategy_cumulative")
        trace["step_3_strategy"] = [
            (nid, sc) for nid, (_, sc) in retrieved.items()
            if "strategy_cumulative" in nid
        ]

        # ── STEP 4 ──
        session_nodes = self.graph.get_nodes_by_type("session")
        top_sessions: List[Tuple[str, float]] = []

        if session_nodes:
            full_query = query
            if conversation_history:
                full_query = " ".join(conversation_history[-3:]) + " " + query

            q_emb = self.embedding_engine.embed_text(full_query)
            sem_scores: Dict[str, float] = {}
            for node in session_nodes:
                if node.embedding is not None:
                    sim = self._cosine(q_emb, node.embedding)
                    sem_scores[node.node_id] = sim

            bm25_raw = self.bm25_engine.search(full_query, top_k=len(session_nodes))
            bm25_scores = dict(bm25_raw)
            mx = max(bm25_scores.values()) if bm25_scores else 1.0
            if mx > 0:
                bm25_scores = {k: v / mx for k, v in bm25_scores.items()}

            combined: Dict[str, float] = {}
            for nid in sem_scores:
                s = sem_scores.get(nid, 0.0)
                b = bm25_scores.get(nid, 0.0)
                combined[nid] = self.semantic_weight * s + self.bm25_weight * b

            sorted_sessions = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            top_sessions = sorted_sessions[:2]

            for nid, score in top_sessions:
                node = self.graph.get_node(nid)
                if node:
                    retrieved[nid] = (node, score)

            trace["step_4_semantic"] = sorted(
                sem_scores.items(), key=lambda x: x[1], reverse=True
            )
            trace["step_4_bm25"] = sorted(
                bm25_scores.items(), key=lambda x: x[1], reverse=True
            )
            trace["step_4_combined"] = list(top_sessions)
        else:
            trace["step_4_semantic"] = []
            trace["step_4_bm25"] = []
            trace["step_4_combined"] = []

        # ── STEP 5 ──
        seed_ids = [nid for nid, _ in top_sessions]
        expanded_nodes: List[Tuple[str, float]] = []
        if seed_ids:
            activations = self.graph_activation.activate(
                self.graph, seed_ids, hops=1,
                edge_types=["same_year", "temporal_next"],
            )
            for nid, act in activations.items():
                if nid not in retrieved:
                    node = self.graph.get_node(nid)
                    if node:
                        retrieved[nid] = (node, self.graph_weight * act)
                        expanded_nodes.append((nid, self.graph_weight * act))
        trace["step_5_expanded"] = expanded_nodes

        # ── STEP 8 ──
        ranked = sorted(retrieved.values(), key=lambda x: x[1], reverse=True)
        trace["step_8_ranked"] = [(n.node_id, sc) for n, sc in ranked]

        # ── STEP 9 ──
        final: List[MemoryNode] = []
        total_tokens = 0
        priority_types = {"latent_cumulative", "strategy_cumulative", "concession_cumulative"}
        final_trace: List[Tuple[str, float, int]] = []

        for node, score in ranked:
            est = node.token_estimate()
            if total_tokens + est <= self.token_budget:
                final.append(node)
                total_tokens += est
                final_trace.append((node.node_id, score, est))
            elif node.node_type in priority_types:
                final.append(node)
                total_tokens += est
                final_trace.append((node.node_id, score, est))

        trace["step_9_final"] = final_trace
        trace["all_scores"] = {nid: sc for nid, (_, sc) in retrieved.items()}

        return final, trace

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _must_retrieve(self, store: Dict, node_type: str):
        """Add a cumulative node with max relevance."""
        nodes = [n for n in self.graph.nodes.values() if n.node_type == node_type]
        if nodes:
            node = nodes[0]
            store[node.node_id] = (node, 1.0)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity, handling shape mismatches for TF-IDF fallback."""
        if a.shape != b.shape:
            ml = min(len(a), len(b))
            a, b = a[:ml], b[:ml]
        dot = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))


# ====================================================================
# QUICK TEST
# ====================================================================
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))

    test_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "memory", "personas", "P_001.json",
    )
    if os.path.exists(test_path):
        from memory_graph import PersonaMemoryGraph

        g = PersonaMemoryGraph("P_001")
        g.build_from_json(test_path, {
            "personality": "budget-conscious",
            "dominant_trait": "openness",
            "income_range": "Low",
        })
        print(g.summary())

        r = MemoryRetriever(g, token_budget=2000)
        nodes = r.retrieve("I need cheap insurance for my car")
        print(f"\nRetrieved {len(nodes)} nodes:")
        for n in nodes:
            print(f"  {n.node_id} ({n.node_type}) → {n.text[:80]}...")
    else:
        print(f"Test file not found: {test_path}")
