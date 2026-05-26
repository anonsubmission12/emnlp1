"""
Prompt Builder for Memory-Augmented Inference.

Takes retrieved memory nodes + persona profile + conversation history
and constructs the final system prompt for the agent LLM.

Output format mirrors the user's specification:

    System:
        You are a motor insurance agent.
    Persona Profile:
        Openness: 0.9 ...
    Latent Persuasion State:
        Commitment: 0.3 ...
    Strategy Effectiveness History:
        Logical works well (+0.2 commitment) ...
    Negotiation History:
        Last year final premium: ₹31,000 ...
    Relevant Past Interactions:
        [2023] ... [2020] ...
    Conversation so far:
        User: ... Agent: ...
    Generate the best next response.
"""

import json
import os
import re
import numpy as np
from typing import Dict, List, Optional
from memory_graph import MemoryNode

# Top-K experience rationales to inject at inference time (paper Appendix C.4)
TOP_K_RATIONALES = 8


# ====================================================================
# EXPERIENCE RATIONALE RETRIEVER  (paper Appendix C.4, Phase 2)
# ====================================================================

class ExperienceRationaleRetriever:
    """
    Retrieves the Top-K most relevant experience rationales from the
    per-persona experience library at inference time.

    Paper Appendix C.4 (Phase 2 — Inference-time retrieval):
        Given query q_t, score every entry in E_p by cosine similarity
        between q_t and the stored user utterance, retrieve Top-K (K=8)
        highest-scoring rationales, and concatenate them into the prompt.

    The retriever uses the same EmbeddingEngine as the MemoryRetriever
    (sentence-transformers all-MiniLM-L6-v2 → EmbeddingGemma when available).
    """

    def __init__(
        self,
        persona_id: str,
        library_dir: Optional[str] = None,
        top_k: int = TOP_K_RATIONALES,
    ):
        self.persona_id = persona_id
        self.top_k = top_k
        self.entries: List[Dict] = []
        self._utterance_embeddings: Optional[np.ndarray] = None
        self._embedding_engine = None

        if library_dir is None:
            library_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "experience_library"
            )

        lib_path = os.path.join(
            library_dir, f"experience_library_{persona_id}.json"
        )
        if os.path.exists(lib_path):
            try:
                with open(lib_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                self._build_index()
                print(
                    f"  [RationaleRetriever] Loaded {len(self.entries)} entries "
                    f"for {persona_id}"
                )
            except Exception as e:
                print(f"  [RationaleRetriever] Failed to load {lib_path}: {e}")

    # ------------------------------------------------------------------

    def _build_index(self):
        """Pre-embed all stored user utterances for fast cosine retrieval."""
        try:
            from retrieval import EmbeddingEngine
            self._embedding_engine = EmbeddingEngine()
            utterances = [e.get("user_utterance", "") for e in self.entries]
            if utterances:
                self._utterance_embeddings = (
                    self._embedding_engine.embed_batch(utterances)
                )
        except Exception as e:
            print(f"  [RationaleRetriever] Index build failed: {e}")

    @property
    def available(self) -> bool:
        return len(self.entries) > 0

    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict]:
        """
        Return Top-K experience entries ranked by cosine similarity between
        `query` and each stored user utterance (paper Appendix C.4).
        """
        if not self.entries:
            return []

        if (
            self._embedding_engine is not None
            and self._utterance_embeddings is not None
        ):
            q_emb = self._embedding_engine.embed_text(query)
            scores = self._batch_cosine(q_emb, self._utterance_embeddings)
        else:
            # Fallback: keyword overlap
            qtoks = set(re.findall(r"\w+", query.lower()))
            scores = []
            for e in self.entries:
                utoks = set(
                    re.findall(r"\w+", e.get("user_utterance", "").lower())
                )
                overlap = len(qtoks & utoks) / max(len(qtoks), 1)
                scores.append(overlap)
            scores = np.array(scores)

        top_indices = np.argsort(scores)[::-1][: self.top_k]
        return [self.entries[i] for i in top_indices]

    def format_for_prompt(self, query: str) -> Optional[str]:
        """
        Retrieve Top-K rationales and format them as a prompt section.
        Returns None if no library is loaded.
        """
        entries = self.retrieve(query)
        if not entries:
            return None

        lines = ["EXPERIENCE RATIONALES (Top-K most similar past situations):"]
        for i, e in enumerate(entries, 1):
            year = e.get("year", "?")
            stage = e.get("stage", "?")
            reward = e.get("reward", "?")
            rationale = e.get("rationale", "")
            best_response = e.get("best_response", "")

            lines.append(
                f"\n  [{i}] Year {year} | Stage: {stage} | "
                f"Referee score: {reward:.2f}"
            )
            if rationale:
                lines.append(f"      Why it worked: {rationale}")
            if best_response:
                lines.append(f"      Best response: {best_response[:200]}")

        return "\n".join(lines)

    # ------------------------------------------------------------------

    @staticmethod
    def _batch_cosine(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between query vector and a 2-D embedding matrix."""
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = matrix / norms
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return np.zeros(len(matrix))
        return normed @ (q / q_norm)


class PromptBuilder:
    """Builds the structured inference prompt from memory context."""

    def __init__(
        self,
        agent_name: str = "Agent",
        company_name: str = "Insurance Co.",
    ):
        self.agent_name = agent_name
        self.company_name = company_name

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================

    def build_system_prompt(
        self,
        persona_profile: Dict,
        retrieved_nodes: List[MemoryNode],
        conversation_history: List[Dict],
        user_message: str,
        current_year: Optional[int] = None,
        experience_library_text: Optional[str] = None,
        rationale_retriever=None,
    ) -> str:
        """
        Assemble the complete system prompt.

        Args:
            persona_profile: {personality, dominant_trait, income_range, big_five_scores}
            retrieved_nodes:  Ranked list from MemoryRetriever.retrieve()
            conversation_history:  [{"role": "user"|"agent", "content": "..."}]
            user_message:  The current user utterance to respond to.
            current_year:  If provided, exclude sessions from years >=
                           current_year to prevent future memory leakage.
            experience_library_text: Pre-formatted rationale text (ignored
                           when rationale_retriever is supplied).
            rationale_retriever: ExperienceRationaleRetriever instance.
                           When provided, Top-K=8 rationales are retrieved
                           and injected per paper Appendix C.4 (Phase 2).

        Returns:
            Full prompt string.
        """
        # Partition nodes by type
        latent = [n for n in retrieved_nodes if "latent" in n.node_type]
        strategy = [n for n in retrieved_nodes if "strategy" in n.node_type]
        concession = [n for n in retrieved_nodes if "concession" in n.node_type]
        sessions = [n for n in retrieved_nodes if n.node_type == "session"]

        # Retrieve Top-K=8 experience rationales (paper Appendix C.4, Phase 2)
        if rationale_retriever is not None and rationale_retriever.available:
            experience_library_text = rationale_retriever.format_for_prompt(
                user_message
            )

        sections = [
            self._section_persona(persona_profile),
            self._section_latent(latent),
            self._section_strategy(strategy),
            self._section_negotiation(concession, sessions),
            self._section_memory(sessions, current_year=current_year),
            self._section_instructions(),
            experience_library_text,
            self._section_history(conversation_history),
            self._section_task(user_message),
        ]

        return "\n\n".join(s for s in sections if s)

    # ==================================================================
    # SECTION BUILDERS
    # ==================================================================

    def _section_instructions(self) -> str:
        return (
            "PERSUASION STRATEGIES\n"
            "Every response must invisibly embed one or two of these."
        )

    def _section_persona(self, profile: Dict) -> str:
        personality = profile.get("personality", "Unknown")
        dominant_trait = profile.get("dominant_trait", "Unknown")
        income_range = profile.get("income_range", "Unknown")

        return (
            f"PERSONA PROFILE\n"
            f"This customer has a {personality} personality with "
            f"{dominant_trait} as their dominant trait and falls in "
            f"the {income_range} income range."
        )

    def _section_latent(self, nodes: List[MemoryNode]) -> str:
        from inference.eval_metrics import map_scores_to_descriptions
        cum = [n for n in nodes if n.node_type == "latent_cumulative"]
        if cum:
            d = cum[0].data
            latent_keys = [
                ("trust", "trust"),
                ("price sensitivity", "price_sensitivity"),
                ("commitment", "commitment"),
                ("concession expectation", "concession_expectation"),
                ("logical responsiveness", "logical_resp"),
                ("emotional responsiveness", "emotional_resp"),
                ("credibility responsiveness", "credibility_resp"),
            ]
            scores = {label: d.get(key, 0.5) for label, key in latent_keys}
            desc_map = map_scores_to_descriptions(scores)
            parts = [f"{desc.lower()} {label}" for label, desc in desc_map.items()]
            return (
                f"LATENT PERSUASION STATE\n"
                f"This customer currently shows {', '.join(parts[:-1])}, "
                f"and {parts[-1]}."
            )

        if nodes:
            latest = sorted(nodes, key=lambda n: n.year or 0, reverse=True)[0]
            return f"LATENT PERSUASION STATE\n{latest.text}"

        return "LATENT PERSUASION STATE\nNo data available."

    def _section_strategy(self, nodes: List[MemoryNode]) -> str:
        cum = [n for n in nodes if n.node_type == "strategy_cumulative"]
        if cum:
            stats = cum[0].data.get("stats", {})
            parts = []
            for name, st in stats.items():
                uses = st.get("total_uses", 0)
                delta = st.get("total_delta_readiness", 0.0)
                if uses > 0:
                    avg_d = delta / uses
                    effect = "works well" if avg_d > 0 else "decreases readiness"
                    parts.append(f"{name} appeal strategy {effect}")
                else:
                    parts.append(f"{name} appeal strategy has no data yet")
            return (
                f"STRATEGY EFFECTIVENESS HISTORY\n"
                f"{', '.join(parts[:-1])}, and {parts[-1]}."
                if len(parts) > 1
                else f"STRATEGY EFFECTIVENESS HISTORY\n{parts[0]}."
            )

        if nodes:
            texts = " ".join(n.text for n in nodes)
            return f"STRATEGY EFFECTIVENESS HISTORY\n{texts}"

        return "STRATEGY EFFECTIVENESS HISTORY\nNo data available."

    def _section_negotiation(
        self,
        nodes: List[MemoryNode],
        session_nodes: Optional[List[MemoryNode]] = None,
    ) -> str:
        cum = [n for n in nodes if n.node_type == "concession_cumulative"]
        if cum:
            d = cum[0].data
            uh = d.get("user_requests_history", [])
            ah = d.get("agent_offers_history", [])

            if not uh and not ah:
                return "NEGOTIATION HISTORY\nNo prior negotiations."

            parts = []

            # Price ranges using IQR (robust to outliers)
            if uh and ah:
                u_p25, u_p75 = int(np.percentile(uh, 25)), int(np.percentile(uh, 75))
                a_p25, a_p75 = int(np.percentile(ah, 25)), int(np.percentile(ah, 75))

                if u_p25 == u_p75:
                    user_range = f"around ₹{u_p25:,}"
                else:
                    user_range = f"around ₹{u_p25:,}–₹{u_p75:,}"

                if a_p25 == a_p75:
                    agent_range = f"around ₹{a_p25:,}"
                else:
                    agent_range = f"around ₹{a_p25:,}–₹{a_p75:,}"

                parts.append(
                    f"In past interactions, the user has typically requested "
                    f"premiums {user_range} while the agent offered {agent_range}."
                )
            elif uh:
                u_p25, u_p75 = int(np.percentile(uh, 25)), int(np.percentile(uh, 75))
                if u_p25 == u_p75:
                    parts.append(f"In past interactions, the user has typically requested premiums around ₹{u_p25:,}.")
                else:
                    parts.append(f"In past interactions, the user has typically requested premiums around ₹{u_p25:,}–₹{u_p75:,}.")

            # Trend from last 10 entries
            if len(uh) >= 10:
                early = sum(uh[-10:-5]) / 5
                late = sum(uh[-5:]) / 5
                diff = late - early
                if diff < -500:
                    trend = "pushing the price down"
                elif diff > 500:
                    trend = "gradually accepting higher prices"
                else:
                    trend = "holding firm on their price"
            elif len(uh) >= 2:
                trend = "holding firm on their price"
            else:
                trend = None

            # Accept/reject ratio from session nodes
            accept_str = None
            if session_nodes:
                outcomes = [
                    n.data.get("final_outcome", "")
                    for n in session_nodes
                    if n.data.get("final_outcome")
                ]
                if outcomes:
                    accepts = sum(1 for o in outcomes if o.lower() == "accept")
                    total = len(outcomes)
                    accept_str = f"has accepted deals in {accepts} out of {total} years"

            if trend:
                trend_part = f"The user tends to be {trend}"
                if accept_str:
                    trend_part += f" and {accept_str}"
                parts.append(trend_part + ".")

            elif accept_str:
                parts.append(f"The user {accept_str}.")

            # Most recent settlement
            if uh and ah:
                last_u, last_a = uh[-1], ah[-1]
                if abs(last_u - last_a) <= 1000:
                    parts.append(f"Most recently the user and agent settled at ₹{last_u:,}.")
                else:
                    parts.append(
                        f"Most recently the user requested ₹{last_u:,} "
                        f"and the agent offered ₹{last_a:,}."
                    )
            elif uh:
                parts.append(f"The user's most recent request was ₹{uh[-1]:,}.")

            return "NEGOTIATION HISTORY\n" + " ".join(parts)

        # Fallback to per-year concession nodes
        if nodes:
            texts = " ".join(n.text for n in sorted(nodes, key=lambda x: x.year or 0, reverse=True)[:2])
            return f"NEGOTIATION HISTORY\n{texts}"

        return "NEGOTIATION HISTORY\nNo prior negotiations."

    def _section_memory(
        self,
        session_nodes: List[MemoryNode],
        current_year: Optional[int] = None,
    ) -> str:
        # Filter out sessions from the current year or later to
        # prevent future memory leakage (e.g. when building the
        # prompt for year 2015, only years before 2015 are shown).
        if current_year is not None:
            session_nodes = [
                n for n in session_nodes
                if n.year is not None and n.year < current_year
            ]

        if not session_nodes:
            return "RELEVANT PAST INTERACTIONS: No matching sessions found."

        lines = ["RELEVANT PAST INTERACTIONS:"]
        for n in sorted(session_nodes, key=lambda x: x.year or 0, reverse=True):
            summary = n.data.get("year_summary", n.text)
            outcome = n.data.get("final_outcome", "Unknown")
            lines.append(f"  [{n.year}] {summary}")
            lines.append(f"         Outcome: {outcome}")

        return "\n".join(lines)

    def _section_history(self, history: List[Dict]) -> str:
        if not history:
            return "Conversation so far:\n  (New conversation)"

        lines = ["Conversation so far:"]
        for turn in history[-16:]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if role == "summary":
                lines.append(f"  [Summary] {content}")
            else:
                lines.append(f"  {role.capitalize()}: {content}")
        return "\n".join(lines)

    def _section_task(self, user_message: str) -> str:
        return f'Current user message: "{user_message}"'

    # ==================================================================
    # HELPERS
    # ==================================================================

