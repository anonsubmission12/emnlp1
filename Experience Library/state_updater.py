"""
Online State Updater for Inference.

After each conversation turn, analyzes the user message + agent response
and updates the latent persuasion state, strategy stats, and concession
memory on the live memory graph — so the NEXT turn's prompt reflects the
evolved numerical values.

Update logic mirrors the training pipeline (dialogue_generator.py + memory.py):

    1. Classify the agent's strategy (Logical / Emotional / Credibility)
    2. Estimate the user's intent signals (trust, price_sensitivity,
       decision_readiness) from the user message
    3. Compute deltas against previous state
    4. Update latent state:
       - trust:             EMA toward observed trust signal
       - price_sensitivity: EMA toward observed price sensitivity
       - commitment:        set to current decision_readiness
       - responsiveness:    S_new = S_old + α*(signal − S_old)
    5. Update strategy memory (success count, total uses, delta readiness)
    6. Track negotiation prices (extract ₹ amounts from messages)

All updates are applied in-place on the PersonaMemoryGraph's cumulative
nodes, so the next call to PromptBuilder sees fresh numbers.

Experience Library Integration:
    When an experience library JSON exists for the current persona, the
    updater can look up similar past situations (by stage and latent-state
    proximity) and use the referee-scored reward to modulate the delta
    signal instead of relying solely on keyword heuristics.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Try importing the LLM client for strategy classification
# Falls back to keyword heuristics if unavailable
try:
    from llm_client import InferenceLLMClient
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

try:
    from config import EXPERIENCE_LIBRARY_DIR
except ImportError:
    EXPERIENCE_LIBRARY_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "experience_library"
    )


# ====================================================================
# PRICE EXTRACTOR  (mirrors training pipeline's extract_price)
# ====================================================================

def extract_price(text: str) -> Optional[float]:
    """
    Extract an Indian Rupee price from text.

    Matches patterns like:
        ₹7,000  |  Rs 7000  |  Rs. 7,000  |  INR 7000  |  7000 rupees
    Returns the numeric value or None.
    """
    patterns = [
        r'₹\s*([\d,]+(?:\.\d+)?)',
        r'[Rr][Ss]\.?\s*([\d,]+(?:\.\d+)?)',
        r'INR\s*([\d,]+(?:\.\d+)?)',
        r'([\d,]+(?:\.\d+)?)\s*[Rr]upees',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            val = match.group(1).replace(',', '')
            try:
                return float(val)
            except ValueError:
                continue
    return None


# ====================================================================
# INTENT SIGNAL ESTIMATOR  (keyword heuristic — no extra LLM call)
# ====================================================================

# Keyword lists for lightweight intent signal estimation
# NOTE: These are matched via substring ("kw in text_lower"), so phrases
#       like "get me" will match inside "get me one of those".
_TRUST_POSITIVE = {
    "trust", "agree", "okay", "ok", "sure", "sounds good", "makes sense",
    "perfect", "great", "thanks", "thank you", "confident", "reliable",
    "go ahead", "proceed", "deal", "accept", "yes", "alright", "fine",
    "cool", "awesome", "nice", "good", "understood", "fair enough",
    "that works", "i see", "right", "exactly", "absolutely", "definitely",
    "happy", "pleased", "glad", "appreciate", "love it", "like it",
    "no problem", "no worries",
}
_TRUST_NEGATIVE = {
    "scam", "fraud", "don't trust", "suspicious", "doubt", "worried",
    "concerned", "scared", "hidden", "catch", "trick", "expensive",
    "too much", "no way", "refuse", "reject", "not interested",
    "rip off", "overpriced", "waste", "lying", "cheat", "fake",
    "misleading", "shady", "not worth", "don't believe", "sketchy",
    "unreliable", "horrible", "terrible", "worst", "complain",
}
_PRICE_KEYWORDS = {
    "cheap", "expensive", "cost", "price", "afford", "budget",
    "discount", "offer", "less", "lower", "reduce", "negotiate",
    "premium", "pay", "rupees", "₹", "rs", "money", "rate",
    "how much", "what's the price", "total", "amount", "charge",
    "fee", "emi", "installment", "economical", "value for money",
    "save", "saving", "costly",
}
_READINESS_POSITIVE = {
    "buy", "purchase", "sign up", "go ahead", "proceed", "confirm",
    "take it", "deal", "accept", "finalize", "let's do it", "lock",
    "enroll", "register", "done", "yes", "get me", "get it",
    "i want", "i need", "i'll take", "give me", "book it", "book this",
    "let's go", "start", "begin", "do it", "send me", "set it up",
    "ready", "interested", "want it", "want this", "want one",
    "want that", "sounds good", "sure", "okay", "ok", "alright",
    "make it happen", "close the deal", "wrap it up", "sign me up",
    "count me in", "i'm in", "agreed", "sold", "that one", "one of those",
    "go for it", "approve", "seal", "activate", "issue the policy",
    "get the policy", "get a quote", "get quote", "fetch quote",
    "pull quote", "check quote",
}
_READINESS_NEGATIVE = {
    "think about it", "later", "not sure", "maybe", "wait",
    "reconsider", "pass", "no thanks", "not now", "hold on",
    "let me check", "need time", "not ready", "don't want",
    "not interested", "no", "nah", "nope", "cancel",
    "forget it", "never mind", "drop it", "some other time",
    "i'll pass", "not today", "too early", "just looking",
    "just browsing", "just asking", "exploring", "compare first",
}

_STRATEGY_KEYWORDS = {
    "Default": [
        "thanks", "thank you", "sure", "happy to", "glad", "assist",
        "help", "certainly", "of course", "please", "let me know",
    ],
    "Logical": [
        "data", "statistics", "fact", "compared", "analysis",
        "calculation", "logically", "number", "percentage", "ratio",
        "IDV", "coverage", "deductible", "clause", "terms",
        "breakdown", "detail", "explain", "how", "what", "feature",
        "include", "exclude", "condition", "specification", "plan",
    ],
    "Emotional": [
        "peace of mind", "family", "safe", "protect", "worry",
        "imagine", "feel", "care", "loved ones", "accident",
        "unfortunate", "story", "experience", "comfort",
        "happy", "exciting", "congratulations", "great choice",
        "love", "enjoy", "relief", "reassure", "heartfelt",
    ],
    "Credibility": [
        "years of experience", "settlement ratio", "claim ratio",
        "trusted", "certified", "authorized", "award", "rating",
        "reputation", "industry", "expert", "IRDAI", "ISO",
        "guarantee", "proven", "track record", "reliable", "top",
        "reputable", "insurer", "company", "brand", "New India",
        "HDFC", "ICICI", "Reliance", "Cholamandalam",
    ],
    "Personal": [
        "your car", "your family", "your situation", "based on your",
        "for you", "tailored", "custom", "specific to you",
    ],
    "Persona": [
        "i helped", "i've seen", "i remember", "a client", "a customer",
        "someone like you", "story", "anecdote",
    ],
}


# ====================================================================
# EXPERIENCE LIBRARY LOOKUP
# ====================================================================

class ExperienceLibraryLookup:
    """
    Reads a pre-built experience_library_{persona_id}.json and provides
    reward-modulated signals for the state updater.

    When an experience library exists, the updater can:
    1. Find the closest matching experience entry (by conversation stage
       and latent-state proximity)
    2. Use the referee reward (0–1) to modulate how much the agent's
       current response should shift the latent state — high-reward
       entries amplify positive deltas, low-reward entries dampen them.

    Falls back silently to keyword-only mode if no library exists.
    """

    def __init__(self, persona_id: str, library_dir: str = EXPERIENCE_LIBRARY_DIR):
        self.persona_id = persona_id
        self.entries: List[Dict] = []
        self._loaded = False

        lib_path = os.path.join(library_dir, f"experience_library_{persona_id}.json")
        if os.path.exists(lib_path):
            try:
                with open(lib_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                self._loaded = True
                print(f"  [ExperienceLibrary] Loaded {len(self.entries)} entries for {persona_id}")
            except Exception as e:
                print(f"  [ExperienceLibrary] Failed to load {lib_path}: {e}")

    @property
    def available(self) -> bool:
        return self._loaded and len(self.entries) > 0

    def lookup(
        self,
        current_latent: Dict[str, float],
        user_message: str,
    ) -> Optional[Dict]:
        """
        Find the closest experience entry to the current situation.

        Matching priority:
            1. Stage classification of the user message (opening / price /
               negotiation / closing)
            2. Euclidean distance in latent-state space

        Returns the best matching experience entry dict, or None if no
        library is loaded.
        """
        if not self.available:
            return None

        # Classify the current user message stage
        stage = self._classify_stage(user_message)

        # Filter entries by stage (if any match)
        stage_matches = [e for e in self.entries if e.get("stage") == stage]
        candidates = stage_matches if stage_matches else self.entries

        # Rank by latent-state proximity
        best_entry = None
        best_dist = float("inf")

        keys = ["trust", "price_sensitivity", "commitment",
                "logical_resp", "emotional_resp", "credibility_resp"]

        for entry in candidates:
            snap = entry.get("latent_state_snapshot", {})
            dist = sum(
                (current_latent.get(k, 0.5) - snap.get(k, 0.5)) ** 2
                for k in keys
            ) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        return best_entry

    def get_reward_modulation(
        self,
        current_latent: Dict[str, float],
        user_message: str,
    ) -> Tuple[float, Optional[Dict]]:
        """
        Get a reward-based modulation factor for the delta signals.

        Returns:
            (modulation_factor, matched_entry)

        modulation_factor:
            1.0 = neutral (no library or no match)
            >1.0 = amplify (high-reward experience → the current strategy
                    direction is historically good for this persona)
            <1.0 = dampen (low-reward experience → the agent should be
                    more conservative in state updates)

        The factor maps reward ∈ [0,1] → modulation ∈ [0.5, 1.5]:
            modulation = 0.5 + reward
        """
        entry = self.lookup(current_latent, user_message)
        if entry is None:
            return 1.0, None

        reward = entry.get("reward", 0.5)
        modulation = 0.5 + reward  # maps [0,1] → [0.5, 1.5]
        return modulation, entry

    @staticmethod
    def _classify_stage(user_message: str) -> str:
        """Classify user message into a conversation stage."""
        text = user_message.lower()

        # Closing signals
        closing_pos = ["go ahead", "deal", "accept", "finalize", "let's do",
                       "sign up", "confirm", "take it", "done", "okay let's",
                       "go with", "sold", "i'm in"]
        closing_neg = ["think about", "not sure", "maybe later", "let me check",
                       "get back to you", "not now", "pass", "no thanks"]
        if any(kw in text for kw in closing_pos):
            return "closing_accept"
        if any(kw in text for kw in closing_neg):
            return "closing_reject"

        # Negotiation signals
        nego = ["too much", "expensive", "discount", "lower", "reduce",
                "negotiate", "budget", "can you do", "high", "less"]
        if any(kw in text for kw in nego):
            return "negotiation"

        # Price inquiry
        price = ["cost", "price", "how much", "afford", "premium",
                 "rupees", "₹", "rs", "pay", "emi", "charge"]
        if any(kw in text for kw in price):
            return "price_inquiry"

        # Default: opening / information seeking
        return "opening"


def estimate_intent_signals(user_message: str) -> Dict[str, float]:
    """
    Estimate trust_level, price_sensitivity, and decision_readiness
    from user text using keyword matching.

    Returns dict with values in [0.0, 1.0].
    """
    words = set(user_message.lower().split())
    text_lower = user_message.lower()

    # --- Trust ---
    trust_pos = sum(1 for kw in _TRUST_POSITIVE if kw in text_lower)
    trust_neg = sum(1 for kw in _TRUST_NEGATIVE if kw in text_lower)
    trust_signal = 0.5 + 0.1 * trust_pos - 0.1 * trust_neg
    trust_signal = max(0.0, min(1.0, trust_signal))

    # --- Price sensitivity ---
    price_hits = sum(1 for kw in _PRICE_KEYWORDS if kw in text_lower)
    price_signal = 0.5 + 0.08 * price_hits
    # If user mentions a specific price, they're price-sensitive
    if extract_price(user_message) is not None:
        price_signal += 0.15
    price_signal = max(0.0, min(1.0, price_signal))

    # --- Decision readiness ---
    ready_pos = sum(1 for kw in _READINESS_POSITIVE if kw in text_lower)
    ready_neg = sum(1 for kw in _READINESS_NEGATIVE if kw in text_lower)
    ready_signal = 0.3 + 0.15 * ready_pos - 0.15 * ready_neg
    ready_signal = max(0.0, min(1.0, ready_signal))

    return {
        "trust": trust_signal,
        "price_sensitivity": price_signal,
        "decision_readiness": ready_signal,
    }


def classify_strategy(agent_response: str) -> str:
    """
    Classify which persuasion strategy the agent used, based on keywords.
    Returns one of: "Logical", "Emotional", "Credibility".
    """
    text_lower = agent_response.lower()
    scores = {}
    for strategy, keywords in _STRATEGY_KEYWORDS.items():
        scores[strategy] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Default"
    return best


# ====================================================================
# ONLINE STATE UPDATER
# ====================================================================

class OnlineStateUpdater:
    """
    Updates the PersonaMemoryGraph's cumulative nodes in-place
    after each conversation turn.

    Mirrors the training pipeline's update logic:
        - PersonaDriftModel:   gamma * old + (1-gamma) * observed
        - update_responsiveness: S + alpha * (signal - S)
        - strategy_memory.update()
        - concession_memory tracking

    When an experience library is available for the persona, the updater
    uses the referee reward to modulate how aggressively it shifts the
    latent state (high-reward experience entries amplify positive deltas,
    low-reward entries dampen them).

    Parameters:
        gamma:      EMA stability factor for trust / price_sensitivity (default 0.8)
        alpha:      Learning rate for responsiveness updates (default 0.1)
        persona_id: If provided, attempts to load the experience library for this persona.
    """

    def __init__(
        self,
        gamma: float = 0.8,
        alpha: float = 0.1,
        persona_id: Optional[str] = None,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.persona_id = persona_id

        # Running state for delta tracking within a session
        self._prev_readiness: float = 0.0
        self._readiness_trajectory: List[float] = []
        self._initialized: bool = False
        self._resp_history = {
            "Logical": [],
            "Emotional": [],
            "Credibility": [],
        }

        self._llm_client = None
        if HAS_LLM:
            try:
                self._llm_client = InferenceLLMClient(
                    model="gpt-5",
                    temperature=0.2,
                    max_tokens=120,
                    load_vlm=False,
                )
            except Exception:
                self._llm_client = None

        # Experience library lookup (optional)
        self.exp_library: Optional[ExperienceLibraryLookup] = None
        if persona_id:
            self.exp_library = ExperienceLibraryLookup(persona_id)

    @property
    def experience_library_active(self) -> bool:
        """Whether the experience library is loaded and providing modulation."""
        return self.exp_library is not None and self.exp_library.available

    def initialize_from_graph(self, graph) -> None:
        """
        Read the current cumulative latent state to seed the
        running trackers (call once when conversation starts).
        """
        latent_nodes = [
            n for n in graph.nodes.values()
            if n.node_type == "latent_cumulative"
        ]
        if latent_nodes:
            d = latent_nodes[0].data
            self._prev_readiness = d.get("decision_readiness",
                                          d.get("commitment", 0.0))
        else:
            self._prev_readiness = 0.0

        self._readiness_trajectory = [self._prev_readiness]
        self._initialized = True

    # ------------------------------------------------------------------
    # MAIN UPDATE ENTRY POINT
    # ------------------------------------------------------------------

    def update(
        self,
        graph,
        user_message: str,
        agent_response: str,
    ) -> Dict[str, any]:
        """
        Analyze the latest turn and update the graph's cumulative nodes.

        Args:
            graph:          PersonaMemoryGraph (modified in-place)
            user_message:   What the user just said
            agent_response: What the agent just replied

        Returns:
            Dict summarizing all changes made (for logging / trace).
        """
        if not self._initialized:
            self.initialize_from_graph(graph)

        changes: Dict = {}

        # 1. Estimate user intent signals
        signals = estimate_intent_signals(user_message)
        changes["observed_signals"] = signals

        # 2. Classify agent strategy
        strategy = classify_strategy(agent_response)
        changes["classified_strategy"] = strategy

        # 3. Compute decision readiness delta
        current_readiness = signals["decision_readiness"]
        delta_readiness = current_readiness - self._prev_readiness
        self._prev_readiness = current_readiness
        self._readiness_trajectory.append(current_readiness)
        changes["delta_readiness"] = delta_readiness

        # 3b. Experience Library reward modulation
        #     If the library is loaded, look up the closest experience
        #     entry and use its reward to modulate the delta signal.
        exp_modulation = 1.0
        exp_match = None
        if self.experience_library_active:
            # Get current latent state for lookup
            latent_nodes = [
                n for n in graph.nodes.values()
                if n.node_type == "latent_cumulative"
            ]
            current_latent = latent_nodes[0].data if latent_nodes else {}
            exp_modulation, exp_match = self.exp_library.get_reward_modulation(
                current_latent, user_message,
            )
            # Modulate the delta: amplify if high reward, dampen if low
            delta_readiness *= exp_modulation
            changes["experience_library"] = {
                "active": True,
                "modulation_factor": round(exp_modulation, 4),
                "matched_stage": exp_match.get("stage") if exp_match else None,
                "matched_reward": exp_match.get("reward") if exp_match else None,
                "matched_year": exp_match.get("year") if exp_match else None,
                "modulated_delta_readiness": round(delta_readiness, 4),
            }
        else:
            changes["experience_library"] = {"active": False}

        # 4. Update latent cumulative node
        latent_changes = self._update_latent(graph, signals, strategy, delta_readiness)
        changes["latent_updates"] = latent_changes

        # 5. Update strategy cumulative node
        strat_changes = self._update_strategy(graph, strategy, delta_readiness)
        changes["strategy_updates"] = strat_changes

        # 6. Update concession cumulative node
        conc_changes = self._update_concession(graph, user_message, agent_response)
        changes["concession_updates"] = conc_changes

        # 7. Re-build text for updated nodes (so embeddings / prompts reflect new data)
        self._rebuild_node_texts(graph)

        return changes

    # ------------------------------------------------------------------
    # LATENT STATE UPDATE
    # ------------------------------------------------------------------

    def _update_latent(
        self,
        graph,
        signals: Dict[str, float],
        strategy: str,
        delta_readiness: float,
    ) -> Dict:
        latent_nodes = [
            n for n in graph.nodes.values()
            if n.node_type == "latent_cumulative"
        ]
        if not latent_nodes:
            return {"error": "no latent_cumulative node found"}

        node = latent_nodes[0]
        d = node.data
        old = dict(d)

        # --- EMA updates (mirrors PersonaDriftModel) ---
        # trust: gamma * old + (1-gamma) * observed
        d["trust"] = self.gamma * d.get("trust", 0.5) + (1 - self.gamma) * signals["trust"]

        # price_sensitivity: gamma * old + (1-gamma) * observed
        d["price_sensitivity"] = (
            self.gamma * d.get("price_sensitivity", 0.5)
            + (1 - self.gamma) * signals["price_sensitivity"]
        )

        # commitment: mean of per-turn readiness (paper Eq. 8)
        if self._readiness_trajectory:
            mean_ready = sum(self._readiness_trajectory) / len(self._readiness_trajectory)
            d["commitment"] = max(0.0, min(1.0, mean_ready))

        # decision_readiness: track it
        d["decision_readiness"] = signals["decision_readiness"]

        # concession expectation: LLM-estimated per turn when available
        conc_est = self._estimate_concession_expectation(signals, delta_readiness)
        if conc_est is not None:
            d["concession_expectation"] = conc_est

        # --- Responsiveness update (mirrors update_responsiveness) ---
        # signal = clamp((delta_readiness * 5) + 0.5, 0, 1)
        # S_new = S_old + alpha * (signal - S_old)
        resp_signal = max(0.0, min(1.0, (delta_readiness * 5) + 0.5))

        attr_map = {
            "Logical": "logical_resp",
            "Emotional": "emotional_resp",
            "Credibility": "credibility_resp",
        }
        resp_attr = attr_map.get(strategy)
        if resp_attr and resp_attr in d:
            hist = self._resp_history.get(strategy, [])
            hist.append(resp_signal)
            if len(hist) > 5:
                hist = hist[-5:]
            self._resp_history[strategy] = hist
            d[resp_attr] = sum(hist) / len(hist)

        # Clamp all values
        for key in ["trust", "price_sensitivity", "commitment",
                     "logical_resp", "emotional_resp", "credibility_resp",
                     "concession_expectation", "decision_readiness"]:
            if key in d:
                d[key] = max(0.0, min(1.0, d[key]))

        # Compute what changed
        deltas = {k: round(d.get(k, 0) - old.get(k, 0), 4) for k in d if k in old}
        return {"old": {k: round(v, 4) for k, v in old.items()},
                "new": {k: round(v, 4) for k, v in d.items()},
                "deltas": deltas}

    def _estimate_concession_expectation(
        self,
        signals: Dict[str, float],
        delta_readiness: float,
    ) -> Optional[float]:
        """
        LLM-estimated concession expectation (paper Appendix C.1).
        Falls back to a heuristic if LLM is unavailable.
        """
        if self._llm_client is not None:
            prompt = (
                "Estimate concession expectation on a 0-1 scale based on: "
                f"trust={signals['trust']:.2f}, price_sensitivity={signals['price_sensitivity']:.2f}, "
                f"decision_readiness={signals['decision_readiness']:.2f}, delta_readiness={delta_readiness:.2f}. "
                "Return JSON: {\"concession_expectation\": <float>}"
            )
            text, _ = self._llm_client.generate(prompt)
            try:
                data = json.loads(text)
                val = float(data.get("concession_expectation", 0.5))
                return max(0.0, min(1.0, val))
            except Exception:
                return None

        # Heuristic fallback
        val = 0.5 + (signals.get("price_sensitivity", 0.5) - 0.5) * 0.4
        val += -0.2 if delta_readiness > 0.1 else 0.0
        return max(0.0, min(1.0, val))

    # ------------------------------------------------------------------
    # STRATEGY MEMORY UPDATE
    # ------------------------------------------------------------------

    def _update_strategy(
        self,
        graph,
        strategy: str,
        delta_readiness: float,
    ) -> Dict:
        """
        Update strategy-outcome memory weights using the paper's banded scheme
        (Appendix C.2, Eq. 12–13).

        After each turn t, compute M̄_t = mean decision_readiness over the 5
        most recent turns, then map to an update step Δ_t:

            Δ_t = +0.10  if M̄_t  > 0.8
            Δ_t = +0.05  if 0.6 < M̄_t ≤ 0.8
            Δ_t = -0.05  if 0.4 ≤ M̄_t ≤ 0.6
            Δ_t = -0.10  if M̄_t  < 0.4

        Weight update:  w_s ← w_s + α * Δ_t    (α = 0.05)
        """
        ALPHA = 0.05  # learning rate — paper Appendix C.2

        strat_nodes = [
            n for n in graph.nodes.values()
            if n.node_type == "strategy_cumulative"
        ]
        if not strat_nodes:
            return {"error": "no strategy_cumulative node found"}

        node = strat_nodes[0]
        stats = node.data.setdefault("stats", {})

        # Compute M̄_t — mean readiness over the last 5 turns in trajectory
        window = self._readiness_trajectory[-5:]
        m_bar = sum(window) / len(window) if window else 0.5

        # Banded Δ_t
        if m_bar > 0.8:
            delta_t = +0.10
        elif m_bar > 0.6:
            delta_t = +0.05
        elif m_bar >= 0.4:
            delta_t = -0.05
        else:
            delta_t = -0.10

        entry = stats.setdefault(strategy, {
            "weight": 1.0,   # w_s — initialized uniformly (paper: ws initialized uniformly)
            "success_count": 0,
            "total_uses": 0,
            "total_delta_readiness": 0.0,
        })

        # w_s ← w_s + α * Δ_t  (paper Eq. 13)
        entry["weight"] = max(0.0, entry["weight"] + ALPHA * delta_t)

        # Running stats (used for prompt context and logging)
        entry["total_uses"] += 1
        entry["total_delta_readiness"] += delta_readiness
        if delta_readiness > -0.05:
            entry["success_count"] += 1

        return {
            "strategy": strategy,
            "m_bar_recent_5": round(m_bar, 4),
            "delta_t_banded": delta_t,
            "alpha": ALPHA,
            "new_weight": round(entry["weight"], 4),
            "delta_readiness": round(delta_readiness, 4),
            "outcome_positive": delta_readiness > -0.05,
            "updated_stats": {k: dict(v) for k, v in stats.items()},
        }

    # ------------------------------------------------------------------
    # CONCESSION MEMORY UPDATE
    # ------------------------------------------------------------------

    def _update_concession(
        self,
        graph,
        user_message: str,
        agent_response: str,
    ) -> Dict:
        conc_nodes = [
            n for n in graph.nodes.values()
            if n.node_type == "concession_cumulative"
        ]
        if not conc_nodes:
            return {"error": "no concession_cumulative node found"}

        node = conc_nodes[0]
        d = node.data

        user_price = extract_price(user_message)
        agent_price = extract_price(agent_response)

        updates = {}

        if user_price is not None:
            d.setdefault("user_requests_history", []).append(user_price)
            updates["user_price"] = user_price

        if agent_price is not None:
            d.setdefault("agent_offers_history", []).append(agent_price)
            updates["agent_price"] = agent_price

        return updates

    # ------------------------------------------------------------------
    # REBUILD NODE TEXTS  (so prompt builder sees updated text)
    # ------------------------------------------------------------------

    def _rebuild_node_texts(self, graph) -> None:
        """Re-generate .text for all cumulative nodes from their updated .data."""
        for node in graph.nodes.values():
            if node.node_type == "latent_cumulative":
                d = node.data
                node.text = (
                    f"Current Latent Persuasion State: "
                    f"Trust={d.get('trust', 0.5):.2f}, "
                    f"Price Sensitivity={d.get('price_sensitivity', 0.5):.2f}, "
                    f"Commitment={d.get('commitment', 0.0):.2f}, "
                    f"Logical Resp={d.get('logical_resp', 0.5):.2f}, "
                    f"Emotional Resp={d.get('emotional_resp', 0.5):.2f}, "
                    f"Credibility Resp={d.get('credibility_resp', 0.5):.2f}, "
                    f"Concession Expectation={d.get('concession_expectation', 0.5):.2f}."
                )

            elif node.node_type == "strategy_cumulative":
                stats = node.data.get("stats", {})
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
                node.text = " | ".join(parts)

            elif node.node_type == "concession_cumulative":
                d = node.data
                uh = d.get("user_requests_history", [])
                ah = d.get("agent_offers_history", [])
                lu = uh[-1] if uh else 0
                la = ah[-1] if ah else 0
                ru = uh[-5:] if uh else []
                ra = ah[-5:] if ah else []
                node.text = (
                    f"Concession History: "
                    f"Last user request ₹{lu:,}. Last agent offer ₹{la:,}. "
                    f"Recent user asks: {['₹'+str(int(p)) for p in ru]}. "
                    f"Recent agent offers: {['₹'+str(int(p)) for p in ra]}. "
                    f"Total tracked: {len(uh)} user, {len(ah)} agent."
                )
