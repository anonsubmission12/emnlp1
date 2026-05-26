"""
Memory Graph Construction.

Builds a connected graph from persona memory JSON files.

Graph Structure (per persona):
──────────────────────────────────────────────────────────────
    persona_root
    ├── session_2015 ── session_2016 ── ... ── session_2025
    │       ↕                ↕                      ↕
    ├── latent_2015 ── latent_2016 ── ... ── latent_2025
    │       ↕                ↕                      ↕
    ├── strategy_2015 ─ strategy_2016 ─ ... ─ strategy_2025
    │       ↕                ↕                      ↕
    └── concession_2015  concession_2016 ... concession_2025

    + cumulative nodes: latent_cumulative, strategy_cumulative,
      concession_cumulative linked to latest year nodes
──────────────────────────────────────────────────────────────

Edge types:
    root_to_child   : persona_root → first year's 4 nodes
    temporal_next   : same type, year N → year N+1
    same_year       : different types within same year (bi-directional)
    cumulative_link : last year node → cumulative aggregate node

Constraints:
    - NO cross-year cross-type edges
      (e.g., session_2015 ✘→ latent_2016)
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any


# ====================================================================
# DATA CLASSES
# ====================================================================

class MemoryNode:
    """A single node in the memory graph."""

    def __init__(
        self,
        node_id: str,
        node_type: str,      # persona | session | latent | strategy | concession
                              # latent_cumulative | strategy_cumulative | concession_cumulative
        year: Optional[int],
        data: Dict[str, Any],
        text: str,            # human-readable for embedding / search
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.year = year
        self.data = data
        self.text = text
        self.embedding = None  # populated by EmbeddingEngine

    def token_estimate(self) -> int:
        """Rough token count (words × 1.3)."""
        return int(len(self.text.split()) * 1.3)

    def __repr__(self):
        return f"MemoryNode({self.node_id}, type={self.node_type}, year={self.year})"


class MemoryEdge:
    """A directed edge in the memory graph."""

    EDGE_TYPES = [
        "root_to_child",
        "temporal_next",
        "same_year",
        "cumulative_link",
    ]

    def __init__(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.edge_type = edge_type
        self.weight = weight

    def __repr__(self):
        return f"Edge({self.source_id} --{self.edge_type}--> {self.target_id})"


# ====================================================================
# GRAPH
# ====================================================================

class PersonaMemoryGraph:
    """
    Memory graph for a single persona.

    Builds nodes (session / latent / strategy / concession per year,
    plus cumulative aggregates) and edges (temporal, same-year,
    root-to-child, cumulative-link) from a persona memory JSON file.
    """

    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: List[MemoryEdge] = []
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # Metadata
        self.personality: Optional[str] = None
        self.dominant_trait: Optional[str] = None
        self.income_range: Optional[str] = None
        self.years: List[int] = []

    # ----------------------------------------------------------------
    # BUILD
    # ----------------------------------------------------------------

    def build_from_json(
        self,
        memory_path: str,
        persona_meta: Optional[Dict] = None,
    ) -> "PersonaMemoryGraph":
        """
        Construct the full graph from a persona JSON file.

        Args:
            memory_path:  Path to persona memory JSON (e.g. memory/personas/P_001.json)
            persona_meta: Optional dict with {personality, dominant_trait, income_range}
        """
        with open(memory_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        persona_id = data.get("persona_id", self.persona_id)

        if persona_meta:
            self.personality = persona_meta.get("personality")
            self.dominant_trait = persona_meta.get("dominant_trait")
            self.income_range = persona_meta.get("income_range")

        sessions = data.get("sessions", [])
        strategy_memory = data.get("strategy_memory", {})
        latent_state = data.get("latent_state", {})
        concession_memory = data.get("concession_memory", {})

        # Sort chronologically
        sessions.sort(key=lambda s: s.get("year", 0))
        self.years = [s["year"] for s in sessions]

        # ----------------------------------------------------------
        # 1.  ROOT NODE
        # ----------------------------------------------------------
        persona_label = (
            f"{self.personality or 'Unknown'}"
            f"+{self.dominant_trait or 'Unknown'}"
            f"+{self.income_range or 'Unknown'}"
        )
        root_text = (
            f"Persona: {persona_label}. "
            f"ID: {persona_id}. "
            f"Years: {self.years[0] if self.years else '?'}"
            f"–{self.years[-1] if self.years else '?'}."
        )
        root_id = f"{persona_id}_root"
        self._add_node(MemoryNode(root_id, "persona", None, {
            "personality": self.personality,
            "dominant_trait": self.dominant_trait,
            "income_range": self.income_range,
        }, root_text))

        # ----------------------------------------------------------
        # 2.  PER-YEAR NODES + EDGES
        # ----------------------------------------------------------
        prev_ids: Dict[str, Optional[str]] = {
            "session": None,
            "latent": None,
            "strategy": None,
            "concession": None,
        }

        for sess in sessions:
            year = sess["year"]
            year_node_ids = []

            # --- session node ---
            sid = f"{persona_id}_session_{year}"
            self._add_node(MemoryNode(
                sid, "session", year, sess,
                self._text_session(sess),
            ))
            year_node_ids.append(sid)

            # --- latent node (per-year snapshot) ---
            lid = f"{persona_id}_latent_{year}"
            latent_yr = {
                "trust": sess.get("trust_estimate", 0.5),
                "price_sensitivity": sess.get("price_sensitivity_estimate", 0.5),
                "decision_readiness": sess.get("decision_readiness_end", 0.0),
            }
            self._add_node(MemoryNode(
                lid, "latent", year, latent_yr,
                self._text_latent_year(year, latent_yr),
            ))
            year_node_ids.append(lid)

            # --- strategy node (per-year usage) ---
            stid = f"{persona_id}_strategy_{year}"
            strat_yr = sess.get("avg_strategy_vector", {})
            self._add_node(MemoryNode(
                stid, "strategy", year, strat_yr,
                self._text_strategy_year(year, strat_yr, sess.get("final_outcome")),
            ))
            year_node_ids.append(stid)

            # --- concession node (per-year negotiation) ---
            cid = f"{persona_id}_concession_{year}"
            conc_yr = {
                "negotiation_moves": sess.get("negotiation_moves", []),
                "final_outcome": sess.get("final_outcome"),
            }
            self._add_node(MemoryNode(
                cid, "concession", year, conc_yr,
                self._text_concession_year(year, conc_yr),
            ))
            year_node_ids.append(cid)

            # ---- root → first year ----
            if prev_ids["session"] is None:
                for nid in year_node_ids:
                    self._add_edge(MemoryEdge(root_id, nid, "root_to_child"))

            # ---- temporal_next (vertical, same type) ----
            type_keys = ["session", "latent", "strategy", "concession"]
            cur_ids = [sid, lid, stid, cid]
            for tkey, cur_id in zip(type_keys, cur_ids):
                if prev_ids[tkey] is not None:
                    self._add_edge(MemoryEdge(prev_ids[tkey], cur_id, "temporal_next"))
                prev_ids[tkey] = cur_id

            # ---- same_year (horizontal, fully connected within year) ----
            for i in range(len(year_node_ids)):
                for j in range(i + 1, len(year_node_ids)):
                    self._add_edge(MemoryEdge(
                        year_node_ids[i], year_node_ids[j], "same_year"))
                    self._add_edge(MemoryEdge(
                        year_node_ids[j], year_node_ids[i], "same_year"))

        # ----------------------------------------------------------
        # 3.  CUMULATIVE NODES
        # ----------------------------------------------------------
        # Latent cumulative
        cum_lid = f"{persona_id}_latent_cumulative"
        self._add_node(MemoryNode(
            cum_lid, "latent_cumulative", None, latent_state,
            self._text_latent_cumulative(latent_state),
        ))
        if prev_ids["latent"]:
            self._add_edge(MemoryEdge(prev_ids["latent"], cum_lid, "cumulative_link"))

        # Strategy cumulative
        cum_stid = f"{persona_id}_strategy_cumulative"
        self._add_node(MemoryNode(
            cum_stid, "strategy_cumulative", None, strategy_memory,
            self._text_strategy_cumulative(strategy_memory),
        ))
        if prev_ids["strategy"]:
            self._add_edge(MemoryEdge(prev_ids["strategy"], cum_stid, "cumulative_link"))

        # Concession cumulative
        cum_cid = f"{persona_id}_concession_cumulative"
        self._add_node(MemoryNode(
            cum_cid, "concession_cumulative", None, concession_memory,
            self._text_concession_cumulative(concession_memory),
        ))
        if prev_ids["concession"]:
            self._add_edge(MemoryEdge(prev_ids["concession"], cum_cid, "cumulative_link"))

        return self

    # ----------------------------------------------------------------
    # ACCESSORS
    # ----------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: str) -> List[MemoryNode]:
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """Return [(neighbor_id, edge_type)] optionally filtered."""
        nbrs = self.adjacency.get(node_id, [])
        if edge_types:
            nbrs = [(nid, et) for nid, et in nbrs if et in edge_types]
        return nbrs

    def get_latest_year(self) -> Optional[int]:
        return self.years[-1] if self.years else None

    def summary(self) -> str:
        return (
            f"PersonaMemoryGraph({self.persona_id}): "
            f"{len(self.nodes)} nodes, {len(self.edges)} edges, "
            f"years={self.years}"
        )

    # ----------------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------------

    def _add_node(self, node: MemoryNode):
        self.nodes[node.node_id] = node

    def _add_edge(self, edge: MemoryEdge):
        self.edges.append(edge)
        self.adjacency[edge.source_id].append((edge.target_id, edge.edge_type))

    # ----------------------------------------------------------------
    # TEXT BUILDERS  (for embedding / search / prompt injection)
    # ----------------------------------------------------------------

    @staticmethod
    def _text_session(sess: Dict) -> str:
        year = sess.get("year", "?")
        summary = sess.get("year_summary", "No summary.")
        outcome = sess.get("final_outcome", "Unknown")
        trust = sess.get("trust_estimate", 0.5)
        ps = sess.get("price_sensitivity_estimate", 0.5)
        dr = sess.get("decision_readiness_end", 0.0)
        n_moves = len(sess.get("negotiation_moves", []))
        persona_profile = sess.get("persona_profile", {})
        persona_text = ""
        if persona_profile:
            persona_text = (
                f" Persona: {persona_profile.get('personality', 'Unknown')}, "
                f"Trait: {persona_profile.get('dominant_trait', 'Unknown')}, "
                f"Income: {persona_profile.get('income_range', 'Unknown')}."
            )
        return (
            f"Year {year} Session: {summary} "
            f"Outcome: {outcome}. Trust: {trust:.2f}, "
            f"Price Sensitivity: {ps:.2f}, "
            f"Decision Readiness: {dr:.2f}. "
            f"Negotiation rounds: {n_moves}."
            f"{persona_text}"
        )

    @staticmethod
    def _text_latent_year(year: int, d: Dict) -> str:
        return (
            f"Year {year} Latent State: "
            f"Trust={d.get('trust', 0.5):.2f}, "
            f"Price Sensitivity={d.get('price_sensitivity', 0.5):.2f}, "
            f"Decision Readiness={d.get('decision_readiness', 0.0):.2f}."
        )

    @staticmethod
    def _text_strategy_year(year: int, d: Dict, outcome: str = None) -> str:
        parts = [f"Year {year} Strategy Profile:"]
        for k, v in d.items():
            parts.append(f"{k}={v:.2f}")
        if outcome:
            parts.append(f"Outcome: {outcome}")
        return " ".join(parts)

    @staticmethod
    def _text_concession_year(year: int, d: Dict) -> str:
        moves = d.get("negotiation_moves", [])
        outcome = d.get("final_outcome", "Unknown")
        if not moves:
            return f"Year {year} Concession: No negotiation. Outcome: {outcome}."
        user_prices = [m.get("user_request", 0) for m in moves if m.get("user_request")]
        agent_prices = [m.get("agent_offer", 0) for m in moves if m.get("agent_offer")]
        uf = user_prices[-1] if user_prices else 0
        af = agent_prices[-1] if agent_prices else 0
        return (
            f"Year {year} Concession: {len(moves)} rounds. "
            f"User target: ₹{uf:,}. Agent final: ₹{af:,}. "
            f"Outcome: {outcome}."
        )

    @staticmethod
    def _text_latent_cumulative(d: Dict) -> str:
        return (
            f"Current Latent Persuasion State: "
            f"Trust={d.get('trust', 0.5):.2f}, "
            f"Price Sensitivity={d.get('price_sensitivity', 0.5):.2f}, "
            f"Commitment={d.get('commitment', 0.0):.2f}, "
            f"Logical Resp={d.get('logical_resp', 0.5):.2f}, "
            f"Emotional Resp={d.get('emotional_resp', 0.5):.2f}, "
            f"Credibility Resp={d.get('credibility_resp', 0.5):.2f}, "
            f"Concession Expectation={d.get('concession_expectation', 0.5):.2f}."
        )

    @staticmethod
    def _text_strategy_cumulative(d: Dict) -> str:
        stats = d.get("stats", {})
        parts = ["Strategy Effectiveness History:"]
        for name, st in stats.items():
            uses = st.get("total_uses", 0)
            succ = st.get("success_count", 0)
            delta = st.get("total_delta_readiness", 0.0)
            if uses > 0:
                rate = succ / uses
                avg_d = delta / uses
                parts.append(
                    f"{name}: {succ}/{uses} success ({rate:.0%}), "
                    f"avg Δreadiness {avg_d:+.3f}"
                )
            else:
                parts.append(f"{name}: no data")
        return " | ".join(parts)

    @staticmethod
    def _text_concession_cumulative(d: Dict) -> str:
        uh = d.get("user_requests_history", [])
        ah = d.get("agent_offers_history", [])
        lu = uh[-1] if uh else 0
        la = ah[-1] if ah else 0
        ru = uh[-5:] if uh else []
        ra = ah[-5:] if ah else []
        return (
            f"Concession History: "
            f"Last user request ₹{lu:,}. Last agent offer ₹{la:,}. "
            f"Recent user asks: {['₹'+str(p) for p in ru]}. "
            f"Recent agent offers: {['₹'+str(p) for p in ra]}. "
            f"Total tracked: {len(uh)} user, {len(ah)} agent."
        )


# ====================================================================
# QUICK TEST
# ====================================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    # Try to build graph for P_001
    test_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "memory", "personas", "P_001.json",
    )
    if os.path.exists(test_path):
        g = PersonaMemoryGraph("P_001")
        g.build_from_json(test_path, {
            "personality": "budget-conscious",
            "dominant_trait": "openness",
            "income_range": "Low",
        })
        print(g.summary())
        print(f"Sample nodes: {list(g.nodes.keys())[:8]} ...")
        print(f"Sample edges: {g.edges[:5]}")
    else:
        print(f"Test file not found: {test_path}")
