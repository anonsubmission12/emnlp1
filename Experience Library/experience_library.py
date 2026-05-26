"""
Experience Library Builder.

Pipeline:
    1. Load persona memory → extract per-year conversation context
    2. For each user utterance context, feed it to the Policy Model (Qwen,
       local via HuggingFace) which generates 3 rollouts (candidate agent
       responses)
    3. A Referee Model (GPT via OpenAI API) scores each rollout on
       persuasion alignment (0–1), picks the best, and writes a
       justification
    4. All rollouts + scores saved to  rollouts_{persona_id}.json
    5. Best rollout per utterance saved to  experience_library_{persona_id}.json

Usage:
    python experience_library.py --persona P_001
    python experience_library.py --persona P_001 P_004 --rollouts 3
    python experience_library.py --list-personas

Output files (in inference/experience_library/):
    rollouts_P_001.json            – all 3 rollouts per utterance + scores
    experience_library_P_001.json  – best rollout per utterance only
"""

import argparse
import copy
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np

# ── Path setup ──
INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(INFERENCE_DIR)
sys.path.insert(0, INFERENCE_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (
    MEMORY_DIR, TIMELINES_PATH, CONVERSATIONS_PATH, AGENT_MODEL,
    EMBEDDING_MODEL, TOKEN_BUDGET, SEMANTIC_WEIGHT, BM25_WEIGHT,
    GRAPH_WEIGHT, ACTIVATION_DECAY, TRAIN_SPLIT,
    get_big_five_scores,
)
from memory_graph import PersonaMemoryGraph
from retrieval import MemoryRetriever, EmbeddingEngine
from prompt_builder import PromptBuilder
from llm_client import InferenceLLMClient

# ====================================================================
# CONFIGURATION
# ====================================================================

# Regex to extract [Image: <url>] from user utterances
_IMAGE_TAG_RE = re.compile(r'\[Image:\s*(https?://[^\]]+)\]', re.IGNORECASE)

# Policy Model (local Qwen via HuggingFace transformers)
# Paper §4.1 backbone: Qwen3-3B-Instruct.
POLICY_MODEL_NAME = "Qwen/Qwen3-3B-Instruct"
POLICY_MAX_NEW_TOKENS = 300
POLICY_TEMPERATURE = 0.9       # higher for diverse rollouts
POLICY_TOP_P = 0.95
NUM_ROLLOUTS = 3

# Referee Model (OpenAI API) — paper §4.1 specifies GPT-5
REFEREE_MODEL = "gpt-5"
REFEREE_TEMPERATURE = 0.2      # low for consistent scoring
REFEREE_MAX_TOKENS = 800

# Output directory
OUTPUT_DIR = os.path.join(INFERENCE_DIR, "experience_library")

# Semantic similarity threshold for experience library dedup.
# Justifications above this cosine similarity are merged, not appended.
SIMILARITY_THRESHOLD = 0.8


# ====================================================================
# BASE SYSTEM PROMPT  (fed to Policy Model for rollout generation)
# ====================================================================

BASE_SYSTEM_PROMPT = """\
You are a professional motor insurance sales agent at an insurance company.

Your job is to guide and persuade customers to purchase the most suitable motor insurance policy for their vehicle.
Your responses should clearly feel like a skilled advisor who is trying to convince the customer in a helpful, trustworthy way.

You must gently influence the customer toward buying a policy, while ensuring they feel informed, respected, and confident in their decision.

CRITICAL CONSTRAINTS:
1. Always use Indian Rupees (₹). NEVER use dollars ($).
2. Be concise: 3-4 sentences max.
3. Be human-like: no corporate-speak.
4. Do NOT mention strategy names or persuasion techniques.
5. Ask only ONE question per response.
6. NEVER be pushy — be naturally convincing.

PERSUASION STRATEGIES (use naturally, not overtly):
- Default: Professional and informative baseline approach.
- Credibility: Emphasize company reputation, awards, claim settlement ratio.
- Emotional: Appeal to security, family safety, peace of mind.
- Logical: Use statistics, comparisons, cost-benefit analysis.
- Personal: Relate to the customer's specific situation and vehicle.
- Persona: Share a relevant anecdote or personal experience.
"""


def build_policy_prompt(
    persona_profile: Dict,
    latent_state: Dict,
    strategy_stats: Dict,
    negotiation_context: Dict,
    year_summary: str,
    conversation_turns: List[Dict],
    user_message: str,
) -> str:
    """
    Build the full prompt sent to the Policy Model (Qwen) for rollout
    generation.  Mirrors the inference prompt_builder structure.
    """
    sections = []

    # 1. System
    sections.append(BASE_SYSTEM_PROMPT)

    # 2. Persona profile
    from inference.eval_metrics import map_scores_to_descriptions
    big5 = persona_profile.get("big_five_scores", {})
    lines = ["PERSONA PROFILE:"]
    lines.append(f"  Personality type : {persona_profile.get('personality', 'Unknown')}")
    lines.append(f"  Dominant trait   : {persona_profile.get('dominant_trait', 'Unknown')}")
    lines.append(f"  Income range     : {persona_profile.get('income_range', 'Unknown')}")
    if big5:
        lines.append("  Big Five scores:")
        desc_map = map_scores_to_descriptions(big5)
        for trait, desc in desc_map.items():
            lines.append(f"    {trait.capitalize():20s}: {desc}")
    sections.append("\n".join(lines))

    # 3. Latent persuasion state
    if latent_state:
        lines = ["LATENT PERSUASION STATE:"]
        latent_keys = [
            ("Trust", "trust"),
            ("Price sensitivity", "price_sensitivity"),
            ("Commitment", "commitment"),
            ("Concession expect.", "concession_expectation"),
            ("Logical resp.", "logical_resp"),
            ("Emotional resp.", "emotional_resp"),
            ("Credibility resp.", "credibility_resp"),
        ]
        latent_scores = {label: latent_state.get(key, 0.5) for label, key in latent_keys}
        desc_map = map_scores_to_descriptions(latent_scores)
        for label, _ in latent_keys:
            lines.append(f"  {label:20s}: {desc_map[label]}")
        sections.append("\n".join(lines))

    # 4. Strategy effectiveness
    if strategy_stats:
        stats = strategy_stats.get("stats", {})
        lines = ["STRATEGY EFFECTIVENESS HISTORY:"]
        for name, st in stats.items():
            uses = st.get("total_uses", 0)
            succ = st.get("success_count", 0)
            delta = st.get("total_delta_readiness", 0.0)
            if uses > 0:
                avg_d = delta / uses
                label = "works well" if avg_d > 0 else "decreases readiness"
                lines.append(
                    f"  {name:12s}: {label} ({avg_d:+.3f} commitment/use, "
                    f"{succ}/{uses} success)"
                )
        sections.append("\n".join(lines))

    # 5. Negotiation context  (from the session being processed)
    if negotiation_context:
        lines = ["NEGOTIATION CONTEXT:"]
        moves = negotiation_context.get("negotiation_moves", [])
        if moves:
            last = moves[-1]
            lines.append(f"  Last user request : ₹{last.get('user_request', 0):,}")
            lines.append(f"  Last agent offer  : ₹{last.get('agent_offer', 0):,}")
            lines.append(f"  Total rounds      : {len(moves)}")
        sections.append("\n".join(lines))

    # 6. Year context
    if year_summary:
        sections.append(f"RELEVANT PAST INTERACTION:\n  {year_summary}")

    # 7. Conversation so far
    if conversation_turns:
        lines = ["Conversation so far:"]
        for turn in conversation_turns[-6:]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if role == "summary":
                lines.append(f"  [Summary] {content}")
            else:
                lines.append(f"  {role.capitalize()}: {content}")
        sections.append("\n".join(lines))

    # 8. Task
    sections.append(
        f'Current user message: "{user_message}"\n\n'
        "Generate the best next response. Be natural, use the memory "
        "context to personalise your approach, and guide toward a deal."
    )

    return "\n\n".join(sections)


# ====================================================================
# REFEREE SCORING PROMPT
# ====================================================================

REFEREE_SYSTEM_PROMPT = """\
You are an expert evaluator of persuasive motor insurance sales conversations.

You will be given:
- A persona profile (personality, traits, income)
- The latent persuasion state (trust, price sensitivity, commitment, etc.)
- A user message
- Three candidate agent responses (Rollout A, B, C)

Score EACH rollout on a scale of 0.0 to 1.0 based on how well it aligns with effective persuasion principles for THIS specific persona.

EVALUATION CRITERIA (weight each appropriately based on persona):

1. STRATEGY-PERSONA FIT (30%):
   - For price-sensitive and logically-responsive personas: structured underwriting explanation and trade-off framing performs better than emotional reassurance or rigid rejection.
   - For emotionally-responsive personas: empathy, rapport-building, and aspirational framing work better than dry facts.
   - For high-credibility-responsive personas: brand authority, settlement ratios, and expert positioning are most effective.
   - Personalization for the persona's dominant trait should always be evident.

2. PERSUASION QUALITY (25%):
   - Does it subtly use persuasion without being overt?
   - Does it advance the conversation toward a deal?
   - Does it build on prior context (memory, history)?

3. NATURALNESS & CONSTRAINTS (20%):
   - Indian Rupees (₹) used? (HARD FAIL if $ used)
   - Concise (3-4 sentences max)?
   - Human-like, no corporate jargon?
   - Only ONE question asked?

4. COMMITMENT PROGRESSION (15%):
   - Does it move the customer closer to a decision?
   - Does it build on existing commitment level?
   - Will it likely increase decision_readiness?

5. TACTICAL SOUNDNESS (10%):
   - Appropriate pricing anchoring?
   - Good handling of objections?
   - Follows up on customer needs?

OUTPUT FORMAT (strict JSON):
{
  "scores": {
    "A": <float 0.0-1.0>,
    "B": <float 0.0-1.0>,
    "C": <float 0.0-1.0>
  },
  "best": "<A|B|C>",
  "justification": "<2-3 sentences explaining WHY the best rollout scored highest, with specific reference to persuasion alignment and persona fit>",
  "per_rollout_reasoning": {
    "A": "<1 sentence: key strength or weakness>",
    "B": "<1 sentence: key strength or weakness>",
    "C": "<1 sentence: key strength or weakness>"
  }
}
"""


def build_referee_prompt(
    persona_profile: Dict,
    latent_state: Dict,
    user_message: str,
    rollouts: List[str],
) -> str:
    """Build the prompt for the Referee to score rollouts."""
    big5 = persona_profile.get("big_five_scores", {})

    ctx = []

    ctx.append("PERSONA:")
    ctx.append(f"  Personality: {persona_profile.get('personality', '?')}")
    ctx.append(f"  Dominant trait: {persona_profile.get('dominant_trait', '?')}")
    ctx.append(f"  Income: {persona_profile.get('income_range', '?')}")
    if big5:
        ctx.append(f"  Big Five: {', '.join(f'{k}={v:.2f}' for k, v in big5.items())}")

    if latent_state:
        ctx.append("\nLATENT STATE:")
        for k in ["trust", "price_sensitivity", "commitment",
                   "logical_resp", "emotional_resp", "credibility_resp"]:
            if k in latent_state:
                ctx.append(f"  {k}: {latent_state[k]:.2f}")

    ctx.append(f'\nUSER MESSAGE: "{user_message}"')

    labels = ["A", "B", "C"]
    ctx.append("\nCANDIDATE RESPONSES:")
    for i, (label, rollout) in enumerate(zip(labels, rollouts)):
        ctx.append(f"\n--- Rollout {label} ---")
        ctx.append(rollout)

    ctx.append("\nScore each rollout. Output ONLY valid JSON as specified.")

    return "\n".join(ctx)


# ====================================================================
# SEMANTIC EXPERIENCE LIBRARY  (Add or Merge via Cosine Similarity)
# ====================================================================

class SemanticExperienceLibrary:
    """
    Central per-persona experience library with semantic dedup.

    For each new entry the *justification* is compared against all
    existing justifications via cosine similarity.  If any existing
    entry exceeds SIMILARITY_THRESHOLD (0.8) the existing entry is
    *updated* (merged) and the new one is skipped.  Otherwise a fresh
    entry is inserted.

    Also provides snapshot() for per-utterance checkpoints so that
    inference at a given time-point can use only the library
    accumulated up to that moment (no future memory leakage).
    """

    def __init__(self, embedding_engine: "EmbeddingEngine",
                 threshold: float = SIMILARITY_THRESHOLD):
        self.entries: List[Dict] = []
        self._justification_embeddings: List[np.ndarray] = []
        self._engine = embedding_engine
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_or_merge(self, entry: Dict) -> str:
        """
        Semantically insert or update an entry.

        Returns 'inserted' or 'merged_with_<index>'.
        """
        justification = entry.get("justification", "")
        if not justification.strip():
            self._insert(entry)
            return "inserted"

        new_emb = self._engine.embed_text(justification)

        best_sim = -1.0
        best_idx = -1
        for idx, existing_emb in enumerate(self._justification_embeddings):
            sim = self._cosine_sim(new_emb, existing_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= self.threshold and best_idx >= 0:
            self._merge(best_idx, entry, new_emb)
            return f"merged_with_{best_idx}"
        else:
            self._insert(entry, new_emb)
            return "inserted"

    def snapshot(self) -> List[Dict]:
        """Return a deep copy of the current library state."""
        return copy.deepcopy(self.entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert(self, entry: Dict, embedding: Optional[np.ndarray] = None):
        """Append a genuinely new entry."""
        if embedding is None:
            embedding = self._engine.embed_text(
                entry.get("justification", "")
            )
        self.entries.append(entry)
        self._justification_embeddings.append(embedding)

    def _merge(self, idx: int, new_entry: Dict, new_emb: np.ndarray):
        """
        Merge *new_entry* into self.entries[idx].

        Strategy:
        - Track provenance of all merged turns.
        - Keep the higher-reward entry's best_response & justification.
        - Re-embed the justification if it changed.
        """
        existing = self.entries[idx]

        # Track provenance
        if "merged_from" not in existing:
            existing["merged_from"] = [{
                "year": existing["year"],
                "turn_no": existing["turn_no"],
                "conversation_id": existing.get("conversation_id"),
                "reward": existing["reward"],
            }]
        existing["merged_from"].append({
            "year": new_entry["year"],
            "turn_no": new_entry["turn_no"],
            "conversation_id": new_entry.get("conversation_id"),
            "reward": new_entry["reward"],
        })

        # Keep the version with the higher reward
        if new_entry["reward"] > existing["reward"]:
            existing["best_response"] = new_entry["best_response"]
            existing["reward"] = new_entry["reward"]
            existing["justification"] = new_entry["justification"]
            existing["user_message"] = new_entry["user_message"]
            existing["year"] = new_entry["year"]
            existing["turn_no"] = new_entry["turn_no"]
            existing["conversation_id"] = new_entry.get("conversation_id")
            existing["intent"] = new_entry.get("intent",
                                               existing.get("intent", ""))
            existing["latent_state_snapshot"] = new_entry.get(
                "latent_state_snapshot",
                existing.get("latent_state_snapshot"),
            )
            # Re-embed with updated justification
            self._justification_embeddings[idx] = self._engine.embed_text(
                existing["justification"]
            )

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        # TF-IDF fallback can produce vectors of different lengths
        # (vocabulary grows as new texts are embedded).  Pad the
        # shorter vector with zeros so np.dot works.
        if a.shape[0] != b.shape[0]:
            max_dim = max(a.shape[0], b.shape[0])
            if a.shape[0] < max_dim:
                a = np.pad(a, (0, max_dim - a.shape[0]))
            else:
                b = np.pad(b, (0, max_dim - b.shape[0]))
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ====================================================================
# HELPER: Format experience library for inclusion in the policy prompt
# ====================================================================

def _build_experience_snapshot(entries: List[Dict],
                               max_entries: int = 10) -> List[Dict]:
    """
    Build a structured list of summarised experience entries for JSON
    storage.  This is saved as 'experience_library_latest' — a compact,
    readable snapshot of the library at this point in time.
    """
    if not entries:
        return []

    snapshot: List[Dict] = []
    for entry in entries[-max_entries:]:
        yr = entry.get("year", "?")
        tn = entry.get("turn_no", "?")
        rw = entry.get("reward", 0.0)
        stage = entry.get("stage", "unknown")
        intent = entry.get("intent", "")

        # Full user message with intent tag
        um_raw = entry.get("user_message", "")
        um_summary = f"[{intent}] {um_raw}" if intent else um_raw

        # Full best response
        br_summary = entry.get("best_response", "")

        # Full justification
        jst = entry.get("justification", "")

        snapshot.append({
            "year": yr,
            "turn": tn,
            "reward": round(rw, 2),
            "stage": stage,
            "user_summary": um_summary,
            "agent_approach": br_summary,
            "why_it_worked": jst,
        })
    return snapshot


def _format_experience_for_prompt(entries: List[Dict],
                                  max_entries: int = 10) -> str:
    """
    Render the experience library as a compact prompt string for
    the Policy Model.  Uses the same summarisation logic.
    """
    snapshot = _build_experience_snapshot(entries, max_entries)
    if not snapshot:
        return ""

    lines = [
        "EXPERIENCE LIBRARY (proven effective responses from prior interactions):"
    ]
    for item in snapshot:
        lines.append(
            f"  [{item['year']}, Turn {item['turn']}] "
            f"(reward={item['reward']:.2f}, stage={item['stage']})"
        )
        lines.append(f"    User: {item['user_summary']}")
        lines.append(f"    Agent approach: {item['agent_approach']}")
        lines.append(f"    Why it worked: {item['why_it_worked']}")
    return "\n".join(lines)


# ====================================================================
# POLICY MODEL  (Local Qwen via HuggingFace)
# ====================================================================

class PolicyModel:
    """
    Wraps a local HuggingFace causal-LM (Qwen) for generating rollouts.

    Falls back to a placeholder if transformers/torch are not available.
    """

    def __init__(
        self,
        model_name: str = POLICY_MODEL_NAME,
        max_new_tokens: int = POLICY_MAX_NEW_TOKENS,
        temperature: float = POLICY_TEMPERATURE,
        top_p: float = POLICY_TOP_P,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self._load(device)

    def _load(self, device: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            print(f"  [PolicyModel] Loading {self.model_name} ...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=device,
                    trust_remote_code=True,
                )
            except Exception as e:
                fallback = "Qwen/Qwen2.5-3B-Instruct"
                print(f"  [PolicyModel] Failed to load {self.model_name}: {e}")
                print(f"  [PolicyModel] Falling back to {fallback}")
                self.model_name = fallback
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=device,
                    trust_remote_code=True,
                )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            print(f"  [PolicyModel] Ready on {self.model.device}")
        except ImportError as e:
            print(f"  [PolicyModel] WARNING: {e}")
            print("  [PolicyModel] Install: pip install transformers torch")
            print("  [PolicyModel] Running in DRY-RUN mode.")
        except Exception as e:
            print(f"  [PolicyModel] Error loading model: {e}")
            print("  [PolicyModel] Running in DRY-RUN mode.")

    def generate_rollouts(
        self,
        prompt: str,
        n: int = NUM_ROLLOUTS,
    ) -> List[str]:
        """Generate n diverse rollout responses for the given prompt."""

        if self.pipe is None:
            return [
                f"[DRY-RUN Rollout {i+1}] Placeholder response. "
                "Install transformers + torch to generate real rollouts."
                for i in range(n)
            ]

        rollouts = []
        messages = [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for i in range(n):
            # Vary temperature slightly per rollout for diversity
            temp = self.temperature + (i * 0.05)
            try:
                output = self.pipe(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temp,
                    top_p=self.top_p,
                    do_sample=True,
                    return_full_text=False,
                )
                text = output[0]["generated_text"]
                # If output is a list of dicts (chat format), extract content
                if isinstance(text, list):
                    text = text[-1].get("content", str(text))
                elif isinstance(text, dict):
                    text = text.get("content", str(text))
                rollouts.append(text.strip())
            except Exception as e:
                rollouts.append(f"[Generation error: {e}]")

        return rollouts


# ====================================================================
# REFEREE MODEL  (OpenAI API)
# ====================================================================

class RefereeModel:
    """
    Uses the OpenAI Responses API (GPT-5-mini) to score and rank rollouts.
    Matches the calling convention in openai_client.py.
    """

    def __init__(
        self,
        model: str = REFEREE_MODEL,
        temperature: float = REFEREE_TEMPERATURE,
        max_tokens: int = REFEREE_MAX_TOKENS,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        # API key: env var takes priority; no hardcoded fallback.
        # Set OPENAI_API_KEY in your environment or a .env file.
        key = api_key
        if not key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                key = os.environ.get("OPENAI_API_KEY")
            except ImportError:
                key = os.environ.get("OPENAI_API_KEY")

        try:
            from openai import OpenAI
            if key:
                self.client = OpenAI(api_key=key)
                print(f"  [Referee] OpenAI client ready (model={self.model})")
            else:
                print("  [Referee] WARNING: No API key → dry-run mode.")
        except ImportError:
            print("  [Referee] WARNING: openai not installed → dry-run mode.")

    def score_rollouts(
        self,
        persona_profile: Dict,
        latent_state: Dict,
        user_message: str,
        rollouts: List[str],
    ) -> Dict:
        """
        Score rollouts using the OpenAI Responses API (same as openai_client.py).

        Returns:
            {
                "scores": {"A": 0.85, "B": 0.72, "C": 0.91},
                "best": "C",
                "justification": "...",
                "per_rollout_reasoning": {"A": "...", "B": "...", "C": "..."}
            }
        """
        prompt = build_referee_prompt(
            persona_profile, latent_state, user_message, rollouts
        )

        if self.client is None:
            return self._dry_run(rollouts)

        # Build the full input exactly like openai_client.py does
        full_input = (
            f"{REFEREE_SYSTEM_PROMPT}\n\n"
            f"{prompt}\n\n"
            "Output ONLY valid JSON as specified above."
        )

        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Use Responses API (responses.create) — same as openai_client.py
                response = self.client.responses.create(
                    model=self.model,
                    input=full_input,
                )

                # Extract text — same as openai_client.py
                raw = getattr(response, "output_text", None)
                if raw is None:
                    # Fallback: try choices structure
                    try:
                        raw = response.choices[0].message.content
                    except (AttributeError, IndexError):
                        raw = None

                # Debug info
                content_len = len(raw) if raw else 0
                print(f"  [Referee] attempt {attempt}: content_length={content_len}")
                print(f"  [Referee] raw response: {raw[:500]}")

                if raw is None or raw.strip() == "":
                    raise ValueError(f"Empty response from Responses API on attempt {attempt}")

                raw = raw.strip()

                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```", 2)[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()

                result = json.loads(raw)

                # Validate structure
                if "scores" not in result:
                    result["scores"] = {"A": 0.5, "B": 0.5, "C": 0.5}
                if "best" not in result:
                    best_label = max(result["scores"], key=result["scores"].get)
                    result["best"] = best_label
                if "justification" not in result:
                    result["justification"] = "No justification provided."

                # Normalize scores to [0, 1]
                for label in result["scores"]:
                    result["scores"][label] = max(0.0, min(1.0,
                        float(result["scores"][label])
                    ))

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    import time
                    wait = 2 * attempt
                    print(f"  [Referee] Attempt {attempt}/{max_retries} failed: {e}  — retrying in {wait}s...")
                    time.sleep(wait)

        print(f"  [Referee] All {max_retries} attempts failed. Last error: {last_error}")
        return self._dry_run(rollouts)

    def _dry_run(self, rollouts: List[str]) -> Dict:
        labels = ["A", "B", "C"][:len(rollouts)]
        return {
            "scores": {l: 0.5 for l in labels},
            "best": labels[0],
            "justification": "[DRY-RUN] No API call made.",
            "per_rollout_reasoning": {l: "[DRY-RUN]" for l in labels},
        }


# ====================================================================
# EXPERIENCE LIBRARY BUILDER
# ====================================================================

class ExperienceLibraryBuilder:
    """
    Orchestrates the full pipeline:
        Memory JSON → extract contexts → Qwen rollouts → GPT referee → save
    """

    def __init__(
        self,
        policy_model: Optional[PolicyModel] = None,
        referee_model: Optional[RefereeModel] = None,
        num_rollouts: int = NUM_ROLLOUTS,
        output_dir: str = OUTPUT_DIR,
        train_split: float = TRAIN_SPLIT,
        load_vlm: bool = True,
    ):
        self.policy_model = policy_model or PolicyModel()
        self.referee_model = referee_model or RefereeModel()
        self.num_rollouts = num_rollouts
        self.output_dir = output_dir
        self.train_split = train_split
        os.makedirs(self.output_dir, exist_ok=True)

        # Load persona registry & conversations
        self.registry = self._load_registry()
        self.conversations = self._load_conversations()

        # ── Gemma 3 4B VLM for vehicle image analysis ──
        self.vlm_client: Optional[InferenceLLMClient] = None
        if load_vlm:
            try:
                self.vlm_client = InferenceLLMClient(
                    load_vlm=True,
                )
                print("  [ExpLib] Gemma VLM ready for image analysis")
            except Exception as e:
                print(f"  [ExpLib] VLM not available (images will be skipped): {e}")

        # Cache: (conversation_id) → vehicle description string
        self._image_analysis_cache: Dict[Any, str] = {}

    def _load_registry(self) -> Dict:
        registry = {}
        if os.path.exists(TIMELINES_PATH):
            with open(TIMELINES_PATH, "r", encoding="utf-8") as f:
                timelines = json.load(f)
            for entry in timelines:
                pid = entry.get("persona_id")
                if pid:
                    registry[pid] = {
                        "personality": entry.get("personality"),
                        "dominant_trait": entry.get("dominant_trait"),
                        "income_range": entry.get("income_range"),
                    }
        return registry

    def _load_conversations(self) -> Dict[str, List[Dict]]:
        """
        Load conversations.json and index by persona_id.

        Returns:
            { "P_001": [conv_year1, conv_year2, ...], ... }
        where each conv has 'conversation_id', 'year', 'turns', etc.
        """
        convs_by_persona: Dict[str, List[Dict]] = {}

        if not os.path.exists(CONVERSATIONS_PATH):
            print(f"  [WARN] conversations.json not found at {CONVERSATIONS_PATH}")
            return convs_by_persona

        with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
            all_convs = json.load(f)

        for conv in all_convs:
            pid = conv.get("persona_id")
            if pid:
                convs_by_persona.setdefault(pid, []).append(conv)

        # Sort each persona's conversations by year
        for pid in convs_by_persona:
            convs_by_persona[pid].sort(key=lambda c: c.get("year", 0))

        print(f"  [Conversations] Loaded {len(all_convs)} conversations "
              f"for {len(convs_by_persona)} personas")
        return convs_by_persona

    # ------------------------------------------------------------------
    # EXTRACT REAL CONVERSATION CONTEXTS
    # ------------------------------------------------------------------

    def extract_contexts(self, persona_id: str) -> List[Dict]:
        """
        For each year of this persona, load the REAL conversation turns
        from conversations.json and pair them with memory context from
        the persona memory JSON.

        Returns list of dicts, one per year:
            {
                "persona_id", "year", "persona_profile", "latent_state",
                "strategy_stats", "year_summary", "negotiation_context",
                "final_outcome", "conversation_id",
                "user_turns" — list of real user utterances with metadata
            }
        """
        mem_path = os.path.join(MEMORY_DIR, f"{persona_id}.json")
        if not os.path.exists(mem_path):
            print(f"  [SKIP] {persona_id}: memory not found at {mem_path}")
            return []

        with open(mem_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = self.registry.get(persona_id, {})
        profile = {
            "personality": meta.get("personality", "Unknown"),
            "dominant_trait": meta.get("dominant_trait", "Unknown"),
            "income_range": meta.get("income_range", "Unknown"),
            "big_five_scores": get_big_five_scores(
                meta.get("personality", "Unknown"),
                meta.get("dominant_trait", "Unknown"),
            ),
        }

        sessions = data.get("sessions", [])
        strategy_memory = data.get("strategy_memory", {})
        latent_state = data.get("latent_state", {})

        # Build a year→conversation lookup from conversations.json
        persona_convs = self.conversations.get(persona_id, [])
        conv_by_year: Dict[int, Dict] = {}
        for conv in persona_convs:
            conv_by_year[conv.get("year", 0)] = conv

        # Also build year→conversation_id from timelines
        timeline_conv_ids: Dict[int, int] = {}
        if os.path.exists(TIMELINES_PATH):
            with open(TIMELINES_PATH, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    if entry.get("persona_id") == persona_id:
                        for c in entry.get("conversations", []):
                            timeline_conv_ids[c["year"]] = c["conversation_id"]

        contexts = []

        for sess in sessions:
            year = sess.get("year", 0)
            year_summary = sess.get("year_summary", "")
            outcome = sess.get("final_outcome", "Unknown")

            # Find the real conversation for this year
            conv = conv_by_year.get(year)
            if conv is None:
                # Try matching by conversation_id from timeline
                cid = timeline_conv_ids.get(year)
                if cid is not None:
                    for c in persona_convs:
                        if c.get("conversation_id") == cid:
                            conv = c
                            break

            if conv is None:
                print(f"    [SKIP] Year {year}: no conversation found")
                continue

            # Extract real user turns from the conversation
            user_turns = []
            for turn in conv.get("turns", []):
                if turn.get("speaker") == "User":
                    user_turns.append({
                        "turn_no": turn.get("turn_no", 0),
                        "content": turn.get("utterance", ""),
                        "intent": turn.get("intent", ""),
                        "intent_vector": turn.get("intent_vector", {}),
                    })

            if not user_turns:
                print(f"    [SKIP] Year {year}: conversation has no user turns")
                continue

            # Also collect agent turns for building conversation context
            all_turns = []
            for turn in conv.get("turns", []):
                role = "user" if turn.get("speaker") == "User" else "agent"
                all_turns.append({
                    "turn_no": turn.get("turn_no", 0),
                    "role": role,
                    "content": turn.get("utterance", ""),
                })

            # Build per-year latent snapshot
            session_latent = {
                "trust": sess.get("trust_estimate", 0.5),
                "price_sensitivity": sess.get("price_sensitivity_estimate", 0.5),
                "commitment": sess.get("decision_readiness_end", 0.0),
                "logical_resp": latent_state.get("logical_resp", 0.5),
                "emotional_resp": latent_state.get("emotional_resp", 0.5),
                "credibility_resp": latent_state.get("credibility_resp", 0.5),
                "concession_expectation": latent_state.get("concession_expectation", 0.5),
            }

            negotiation_context = {
                "negotiation_moves": sess.get("negotiation_moves", []),
                "final_outcome": outcome,
            }

            contexts.append({
                "persona_id": persona_id,
                "year": year,
                "conversation_id": conv.get("conversation_id"),
                "persona_profile": profile,
                "latent_state": session_latent,
                "strategy_stats": strategy_memory,
                "year_summary": year_summary,
                "negotiation_context": negotiation_context,
                "final_outcome": outcome,
                "vehicle_category": conv.get("vehicle_category", "Unknown"),
                "user_turns": user_turns,
                "all_turns": all_turns,
            })

        # Enforce chronological 80/20 split (earliest 80% for library build)
        contexts.sort(key=lambda c: c.get("year", 0))
        if contexts and 0 < self.train_split < 1:
            split_idx = max(1, int(len(contexts) * self.train_split))
            train_contexts = contexts[:split_idx]
            if len(train_contexts) < len(contexts):
                print(
                    f"  [Split] Using {len(train_contexts)}/{len(contexts)} years "
                    f"for library build (train_split={self.train_split})"
                )
            return train_contexts

        return contexts

    # ------------------------------------------------------------------
    # PROCESS ONE PERSONA
    # ------------------------------------------------------------------

    def _build_graph_and_retriever(self, persona_id: str, meta: Dict) -> Tuple:
        """
        Build a PersonaMemoryGraph + MemoryRetriever for a persona.
        Same pipeline as used in the chat / run_inference path.
        """
        mem_path = os.path.join(MEMORY_DIR, f"{persona_id}.json")
        graph = PersonaMemoryGraph(persona_id)
        graph.build_from_json(mem_path, meta)

        retriever = MemoryRetriever(
            graph,
            embedding_model=EMBEDDING_MODEL,
            token_budget=TOKEN_BUDGET,
            semantic_weight=SEMANTIC_WEIGHT,
            bm25_weight=BM25_WEIGHT,
            graph_weight=GRAPH_WEIGHT,
            activation_decay=ACTIVATION_DECAY,
        )
        return graph, retriever

    def process_persona(self, persona_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Run the full pipeline for one persona using REAL conversation
        turns from conversations.json.

        For each year:
            - Loads the actual user utterances from the conversation
            - For each user turn, runs the same retrieval + prompt
              building pipeline as the live chat (MemoryRetriever +
              PromptBuilder) to build the full context prompt
            - Feeds that prompt to Qwen for rollout generation
            - GPT referee scores the rollouts
            - Best rollout saved to experience library

        Returns:
            (all_rollouts, experience_library)
        """
        print(f"\n{'='*60}")
        print(f"  Processing {persona_id}")
        print(f"{'='*60}")

        contexts = self.extract_contexts(persona_id)
        if not contexts:
            return [], []

        # Build graph + retriever (same as inference chat path)
        meta = self.registry.get(persona_id, {})
        print(f"  Building memory graph & retriever ...")
        graph, retriever = self._build_graph_and_retriever(persona_id, meta)
        prompt_builder = PromptBuilder(
            agent_name="Agent",
            company_name="the insurance company",
        )
        print(f"    {graph.summary()}")

        # ── Semantic experience library for this persona ──
        embedding_engine = EmbeddingEngine(EMBEDDING_MODEL)
        semantic_lib = SemanticExperienceLibrary(
            embedding_engine, SIMILARITY_THRESHOLD,
        )

        all_rollouts: List[Dict] = []
        processed_turns: set = set()

        # ── Resume from checkpoint ──
        rollouts_path = os.path.join(
            self.output_dir, f"rollouts_{persona_id}.json"
        )
        library_path = os.path.join(
            self.output_dir, f"experience_library_{persona_id}.json"
        )
        if os.path.exists(rollouts_path) and os.path.exists(library_path):
            try:
                with open(rollouts_path, "r", encoding="utf-8") as f:
                    all_rollouts = json.load(f)
                with open(library_path, "r", encoding="utf-8") as f:
                    saved_entries = json.load(f)
                # Re-populate semantic library (already deduplicated)
                for entry in saved_entries:
                    semantic_lib._insert(entry)
                for rec in all_rollouts:
                    processed_turns.add(
                        (rec["year"], rec.get("conversation_id"), rec["turn_no"])
                    )
                if processed_turns:
                    print(f"  Resuming from checkpoint: "
                          f"{len(processed_turns)} turns already processed, "
                          f"{len(semantic_lib.entries)} library entries")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Corrupt checkpoint, starting fresh: {e}")
                all_rollouts, processed_turns = [], set()
                semantic_lib = SemanticExperienceLibrary(
                    embedding_engine, SIMILARITY_THRESHOLD,
                )

        for ctx in contexts:
            year = ctx["year"]
            user_turns = ctx["user_turns"]
            all_turns = ctx["all_turns"]
            conv_id = ctx.get("conversation_id", "?")
            print(f"\n  Year {year} (conv #{conv_id}, {ctx['final_outcome']}) — "
                  f"{len(user_turns)} user turns")

            # Build conversation history progressively, replaying the
            # real conversation. For each user turn we:
            #   1. Use the conversation history up to this point
            #   2. Run retrieval on the user utterance
            #   3. Build the full prompt (same as inference chat)
            #   4. Feed to Qwen for rollouts
            # Older turn pairs are summarized to keep prompts compact.
            conversation_history: List[Dict] = []
            _RECENT_FULL_TURNS = 4  # keep last N entries (2 pairs) unsummarized

            for ut in user_turns:
                user_text = ut["content"]
                turn_no = ut["turn_no"]
                intent = ut.get("intent", "")

                # ── Skip if already processed (checkpoint resume) ──
                if (year, conv_id, turn_no) in processed_turns:
                    print(f"    [turn {turn_no}] (SKIP — already in checkpoint)")
                    real_agent_reply = self._find_next_agent_reply(
                        all_turns, turn_no
                    )
                    # Summarize and append as a single condensed entry
                    summary = self._summarize_turn_pair(user_text, real_agent_reply)
                    conversation_history.append({"role": "summary", "content": summary})
                    continue

                print(f"    [turn {turn_no}] ({intent}) {user_text[:70]}...")

                # ── Step 0: Image analysis via Gemma VLM ──
                # Extract [Image: <url>] tags from the utterance and run
                # the local Gemma vision model.  The result is cached per
                # conversation (same vehicle across all turns in one year).
                image_description = self._get_image_description(
                    user_text, conv_id, year,
                    vehicle_category=ctx.get("vehicle_category", "Unknown"),
                )

                # ── Step 1: Retrieve memory nodes (same as chat) ──
                history_texts = [h["content"] for h in conversation_history]
                nodes = retriever.retrieve(user_text, history_texts)

                # ── Step 2: Build full prompt via PromptBuilder ──
                #     This is the SAME prompt the live chat would build.
                #     Pass current year so Relevant Past Interactions
                #     excludes future years (prevents memory leakage).
                full_prompt = prompt_builder.build_system_prompt(
                    ctx["persona_profile"], nodes,
                    conversation_history, user_text,
                    current_year=year,
                )

                # ── Step 2b: Build experience library (separate field) ──
                #   (only entries built *up to* this point — no leakage)
                exp_snapshot = _build_experience_snapshot(
                    semantic_lib.entries,
                )

                # ── Step 2c: Inject vehicle image analysis ──
                # Keep a clean copy of the prompt (without image analysis)
                # for saving; use the augmented version for generation only.
                saved_prompt = full_prompt
                if image_description:
                    full_prompt = (
                        full_prompt + "\n\n"
                        "VEHICLE IMAGE ANALYSIS (from vehicle photo shared by customer):\n"
                        f"  {image_description}"
                    )

                # ── Step 3: Generate rollouts from Qwen ──
                rollouts = self.policy_model.generate_rollouts(
                    full_prompt, n=self.num_rollouts
                )
                print(f"      Generated {len(rollouts)} rollouts "
                      f"(retrieved {len(nodes)} memory nodes)")
                for ri, rtxt in enumerate(rollouts):
                    print(f"        Rollout {['A','B','C'][ri]}: {rtxt[:120]}...")

                # ── Step 4: Referee scores the rollouts ──
                referee_result = self.referee_model.score_rollouts(
                    persona_profile=ctx["persona_profile"],
                    latent_state=ctx["latent_state"],
                    user_message=user_text,
                    rollouts=rollouts,
                )
                print(f"      Scores: {referee_result.get('scores', {})}")
                print(f"      Best: Rollout {referee_result.get('best', '?')}")

                # Map labels to rollouts
                labels = ["A", "B", "C"][:len(rollouts)]
                best_label = referee_result.get("best", labels[0])
                best_idx = labels.index(best_label) if best_label in labels else 0
                best_rollout = rollouts[best_idx]
                best_score = referee_result.get("scores", {}).get(best_label, 0.5)

                # Classify the conversation stage from the real intent
                stage = self._classify_stage_from_intent(intent, user_text)

                # ── Step 5: Embed each rollout for DPO reuse ──
                rollout_embeddings = embedding_engine.embed_batch(rollouts)

                # Build the rollout record
                record = {
                    "persona_id": persona_id,
                    "year": year,
                    "conversation_id": conv_id,
                    "turn_no": turn_no,
                    "stage": stage,
                    "user_message": user_text,
                    "intent": intent,
                    "intent_vector": ut.get("intent_vector", {}),
                    "latent_state_snapshot": ctx["latent_state"],
                    "final_outcome": ctx["final_outcome"],
                    "full_prompt": saved_prompt,
                    "rollouts": {
                        label: {
                            "response": rollouts[i],
                            "score": referee_result.get("scores", {}).get(label, 0.5),
                            "reasoning": referee_result.get(
                                "per_rollout_reasoning", {}
                            ).get(label, ""),
                            "embedding": rollout_embeddings[i].tolist(),
                        }
                        for i, label in enumerate(labels)
                    },
                    "best_rollout_label": best_label,
                    "best_score": best_score,
                    "justification": referee_result.get("justification", ""),
                }
                all_rollouts.append(record)

                # Experience library entry (best only)
                experience_entry = {
                    "persona_id": persona_id,
                    "year": year,
                    "conversation_id": conv_id,
                    "turn_no": turn_no,
                    "stage": stage,
                    "user_message": user_text,
                    "intent": intent,
                    "intent_vector": ut.get("intent_vector", {}),
                    "latent_state_snapshot": ctx["latent_state"],
                    "final_outcome": ctx["final_outcome"],
                    "full_prompt": saved_prompt,
                    "experience_library_latest": exp_snapshot,
                    "best_response": best_rollout,
                    "reward": best_score,
                    "justification": referee_result.get("justification", ""),
                    "persona_profile": {
                        "personality": ctx["persona_profile"]["personality"],
                        "dominant_trait": ctx["persona_profile"]["dominant_trait"],
                        "income_range": ctx["persona_profile"]["income_range"],
                    },
                }
                # ── Semantic insert or merge into the central library ──
                action = semantic_lib.add_or_merge(experience_entry)
                print(f"      Experience library: {action} "
                      f"(library size: {len(semantic_lib.entries)})")

                # ── Per-utterance snapshot (prevents future memory leakage) ──
                snapshot_dir = os.path.join(
                    self.output_dir, f"snapshots_{persona_id}",
                )
                os.makedirs(snapshot_dir, exist_ok=True)
                snapshot_path = os.path.join(
                    snapshot_dir,
                    f"exp_lib_{persona_id}_y{year}_t{turn_no}.json",
                )
                with open(snapshot_path, "w", encoding="utf-8") as f:
                    json.dump(
                        semantic_lib.snapshot(), f,
                        indent=2, ensure_ascii=False,
                    )

                # Add to conversation history for next turn's context.
                # We use the REAL agent response (from the original
                # conversation) so the next user turn sees the correct
                # preceding context — not the Qwen rollout.
                # Older turn pairs are summarized to keep prompts compact;
                # only the most recent exchanges are kept in full.
                real_agent_reply = self._find_next_agent_reply(
                    all_turns, turn_no
                )

                # Before appending, summarize older entries if history is long
                if len(conversation_history) >= _RECENT_FULL_TURNS:
                    # Collapse the oldest full user+agent pair into a summary
                    oldest = conversation_history[0]
                    if oldest.get("role") != "summary":
                        # Find the first user+agent pair to collapse
                        if (len(conversation_history) >= 2
                                and conversation_history[0]["role"] == "user"
                                and conversation_history[1]["role"] == "agent"):
                            old_user = conversation_history.pop(0)["content"]
                            old_agent = conversation_history.pop(0)["content"]
                            summary = self._summarize_turn_pair(old_user, old_agent)
                            conversation_history.insert(
                                0, {"role": "summary", "content": summary}
                            )

                conversation_history.append({"role": "user", "content": user_text})
                if real_agent_reply:
                    conversation_history.append({
                        "role": "agent", "content": real_agent_reply,
                    })

                # ── Incremental save (checkpoint after every turn) ──
                self._save_results(
                    persona_id, all_rollouts, semantic_lib.entries,
                )

        # Final save (ensures last state is persisted)
        self._save_results(persona_id, all_rollouts, semantic_lib.entries)

        return all_rollouts, semantic_lib.entries

    @staticmethod
    def _find_next_agent_reply(all_turns: List[Dict], user_turn_no: int) -> Optional[str]:
        """Find the Agent response that immediately follows a user turn."""
        for turn in all_turns:
            if turn["turn_no"] == user_turn_no + 1 and turn["role"] == "agent":
                return turn["content"]
        return None

    @staticmethod
    def _summarize_turn_pair(user_text: str, agent_text: Optional[str]) -> str:
        """Condense a user+agent exchange into a single compact summary line."""
        user_short = user_text[:120].rstrip()
        if len(user_text) > 120:
            user_short += "..."
        if agent_text:
            agent_short = agent_text[:120].rstrip()
            if len(agent_text) > 120:
                agent_short += "..."
            return f"User asked: {user_short} | Agent replied: {agent_short}"
        return f"User asked: {user_short}"

    # ------------------------------------------------------------------
    # IMAGE ANALYSIS  (Gemma 3 4B, cached per conversation)
    # ------------------------------------------------------------------

    def _get_image_description(
        self,
        user_text: str,
        conv_id: Any,
        year: int,
        vehicle_category: str = "Unknown",
    ) -> Optional[str]:
        """
        If *user_text* contains an ``[Image: <url>]`` tag, run image
        analysis through the local Gemma VLM and return the description.

        Results are cached by *conv_id* because the same vehicle photo
        applies to all turns in one conversation/year.  Returns None
        when no image is found or the VLM is unavailable.
        """
        # Quick exit if no image tag
        match = _IMAGE_TAG_RE.search(user_text)
        if match is None:
            return self._image_analysis_cache.get(conv_id)

        image_url = match.group(1).strip()

        # Return cached result if we already analysed this conversation's image
        if conv_id in self._image_analysis_cache:
            return self._image_analysis_cache[conv_id]

        if self.vlm_client is None or self.vlm_client._vlm_model is None:
            # No VLM available — return a minimal fallback from metadata
            fallback = f"{vehicle_category} vehicle (image analysis unavailable)"
            self._image_analysis_cache[conv_id] = fallback
            print(f"      [VLM] No model loaded — using fallback: {fallback}")
            return fallback

        prompt = (
            f"This photo is from {year}. Look at this vehicle and determine:\n"
            "1. Category: Economy, Mid-range, or Premium vehicle?\n"
            "2. Type: SUV, Sedan, Hatchback, etc.?\n"
            "3. Approximate model year range based on design\n"
            "4. Notable features visible\n"
            f"Describe it as a vehicle available in {year}. Be specific and concise."
        )
        print(f"      [VLM] Analyzing vehicle image for conv #{conv_id} ...")
        try:
            desc, _usage = self.vlm_client.analyze_image(image_url, prompt)
            self._image_analysis_cache[conv_id] = desc
            print(f"      [VLM] Done ({len(desc)} chars)")
            return desc
        except Exception as e:
            fallback = f"{vehicle_category} vehicle (VLM error: {e})"
            self._image_analysis_cache[conv_id] = fallback
            print(f"      [VLM] Error: {e}")
            return fallback

    @staticmethod
    def _classify_stage_from_intent(intent: str, user_text: str) -> str:
        """
        Map the real conversation intent label to a stage tag.
        Falls back to keyword heuristics if intent is empty.
        """
        intent_lower = intent.lower().strip()

        # Direct intent mappings from the conversation data
        intent_map = {
            "i-1 session onset": "opening",
            "i-2 coverage inquiry": "information_seeking",
            "i-5 clarification request": "information_seeking",
            "i-3 quote solicitation": "price_inquiry",
            "i-6 premium negotiation": "negotiation",
            "i-4 concern disclosure": "objection",
            "i-7 renewal commitment": "closing_accept",
            "i-8 offer decline": "closing_reject",
            "compare options": "information_seeking",
            "accept offer": "closing_accept",
            "reject offer": "closing_reject",
            "final decision": "closing_accept",
        }

        for key, stage in intent_map.items():
            if key in intent_lower:
                return stage

        # Keyword fallback
        text = user_text.lower()
        if any(kw in text for kw in ["deal", "accept", "go ahead", "let's do"]):
            return "closing_accept"
        if any(kw in text for kw in ["think about", "not sure", "pass", "no thanks"]):
            return "closing_reject"
        if any(kw in text for kw in ["expensive", "discount", "lower", "negotiate", "budget"]):
            return "negotiation"
        if any(kw in text for kw in ["cost", "price", "how much", "premium", "quote"]):
            return "price_inquiry"

        return "information_seeking"

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def _save_results(
        self,
        persona_id: str,
        all_rollouts: List[Dict],
        experience_library: List[Dict],
    ):
        """Write both JSON files to the output directory."""
        rollouts_path = os.path.join(
            self.output_dir, f"rollouts_{persona_id}.json"
        )
        library_path = os.path.join(
            self.output_dir, f"experience_library_{persona_id}.json"
        )

        with open(rollouts_path, "w", encoding="utf-8") as f:
            json.dump(all_rollouts, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {len(all_rollouts)} rollout records → {rollouts_path}")

        with open(library_path, "w", encoding="utf-8") as f:
            json.dump(experience_library, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(experience_library)} experience entries → {library_path}")

    # ------------------------------------------------------------------
    # PROCESS MULTIPLE PERSONAS
    # ------------------------------------------------------------------

    def process_all(self, persona_ids: List[str]):
        """Run the pipeline for multiple personas."""
        for pid in persona_ids:
            try:
                self.process_persona(pid)
            except Exception as e:
                print(f"  [ERROR] {pid}: {e}")
                import traceback
                traceback.print_exc()


# ====================================================================
# CLI
# ====================================================================

def _gpu_status() -> str:
    """One-line GPU memory summary."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA not available"
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        free = total - alloc
        return (f"GPU: {alloc:.2f} / {total:.2f} GiB used "
                f"({free:.2f} GiB free)")
    except Exception:
        return "GPU status unavailable"


def _get_all_persona_ids() -> List[str]:
    """Return sorted list of all persona IDs found in the memory directory."""
    mem_dir = Path(MEMORY_DIR)
    if not mem_dir.exists():
        return []
    return sorted(f.stem for f in mem_dir.glob("P_*.json"))


def list_available_personas():
    mem_dir = Path(MEMORY_DIR)
    if not mem_dir.exists():
        print(f"Memory directory not found: {MEMORY_DIR}")
        return
    registry = {}
    if os.path.exists(TIMELINES_PATH):
        with open(TIMELINES_PATH, "r", encoding="utf-8") as f:
            for e in json.load(f):
                pid = e.get("persona_id")
                if pid:
                    registry[pid] = e

    print(f"\n{'ID':<8} {'Personality':<20} {'Trait':<18} {'Income':<8}")
    print("-" * 60)
    for f in sorted(mem_dir.glob("P_*.json")):
        pid = f.stem
        m = registry.get(pid, {})
        print(f"{pid:<8} {m.get('personality','?'):<20} "
              f"{m.get('dominant_trait','?'):<18} {m.get('income_range','?'):<8}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Experience Library: Qwen rollouts + GPT referee scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experience_library.py --persona P_001
  python experience_library.py --persona P_001 P_004 --rollouts 3
  python experience_library.py --list-personas
  python experience_library.py --persona P_001 --policy-model Qwen/Qwen2.5-3B-Instruct
  python experience_library.py --all
        """,
    )
    parser.add_argument(
        "--persona", "-p", nargs="+", default=None,
        help="Persona ID(s) to process.",
    )
    parser.add_argument(
        "--list-personas", "-l", action="store_true",
        help="List available personas and exit.",
    )
    parser.add_argument(
        "--rollouts", "-n", type=int, default=NUM_ROLLOUTS,
        help=f"Number of rollouts per utterance (default: {NUM_ROLLOUTS}).",
    )
    parser.add_argument(
        "--policy-model", type=str, default=POLICY_MODEL_NAME,
        help=f"HuggingFace model for rollout generation (default: {POLICY_MODEL_NAME}).",
    )
    parser.add_argument(
        "--referee-model", type=str, default=REFEREE_MODEL,
        help=f"OpenAI model for referee scoring (default: {REFEREE_MODEL}).",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="OpenAI API key for referee.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Skip loading Gemma VLM for image analysis.",
    )
    parser.add_argument(
        "--all", "-a", action="store_true",
        help="Process ALL personas. Skips any whose experience library already exists.",
    )

    args = parser.parse_args()

    if args.list_personas:
        list_available_personas()
        return

    if args.all:
        all_ids = _get_all_persona_ids()
        if not all_ids:
            print("No personas found in memory directory.")
            return
        # Filter out personas whose experience library already exists
        out_dir = args.output_dir
        pending = [
            pid for pid in all_ids
            if not os.path.exists(os.path.join(out_dir, f"experience_library_{pid}.json"))
        ]
        skipped = len(all_ids) - len(pending)
        if skipped:
            print(f"  Skipping {skipped} persona(s) with existing experience libraries.")
        if not pending:
            print("All personas already have experience libraries. Nothing to do.")
            return
        print(f"  Will process {len(pending)} / {len(all_ids)} personas.")
        args.persona = pending

    if not args.persona:
        print("Specify personas with --persona P_001 [P_002 ...] or use --all.")
        print("Or use --list-personas to see available ones.")
        return

    print("\n" + "=" * 60)
    print("  Experience Library Builder")
    print("=" * 60)
    print(f"  Policy model : {args.policy_model}")
    print(f"  Referee model: {args.referee_model}")
    print(f"  Rollouts/turn: {args.rollouts}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  VLM (image)  : {'disabled' if args.no_vlm else 'Gemma 3 4B'}")
    print("=" * 60)

    # ── Phase 1: Load ALL models upfront ──
    print("\n--- Phase 1: Loading all models ---")
    print(f"  {_gpu_status()}")

    print("\n  [1/4] Embedding model ...")
    _test_embedding = EmbeddingEngine(EMBEDDING_MODEL)
    print(f"  {_gpu_status()}")

    print("\n  [2/4] Policy model ...")
    policy = PolicyModel(model_name=args.policy_model)
    print(f"  {_gpu_status()}")

    print("\n  [3/4] Referee model (OpenAI API) ...")
    referee = RefereeModel(model=args.referee_model, api_key=args.api_key)
    print(f"  {_gpu_status()}")

    print("\n  [4/4] VLM (Gemma) ...")
    load_vlm = not args.no_vlm
    vlm_client = None
    if load_vlm:
        try:
            vlm_client = InferenceLLMClient(load_vlm=True)
            print(f"  {_gpu_status()}")
        except Exception as e:
            print(f"  [VLM] Failed to load: {e}")
            print(f"  {_gpu_status()}")
    else:
        print("  (skipped — --no-vlm)")

    # ── Verify all models are ready ──
    ready = []
    issues = []

    if _test_embedding.model is not None:
        ready.append("Embeddings (sentence-transformers)")
    else:
        issues.append("Embeddings: using TF-IDF fallback (install sentence-transformers for better quality)")
        ready.append("Embeddings (TF-IDF fallback)")

    if policy.pipe is not None:
        ready.append(f"Policy ({args.policy_model})")
    else:
        issues.append(f"Policy model: FAILED to load {args.policy_model} — will use DRY-RUN")

    if referee.client is not None:
        ready.append(f"Referee ({args.referee_model})")
    else:
        issues.append(f"Referee: No OpenAI client — will use DRY-RUN scoring")

    if load_vlm:
        if vlm_client and vlm_client._vlm_model is not None:
            ready.append("VLM (Gemma 3 4B)")
        else:
            issues.append("VLM: FAILED to load Gemma — images will use fallback")

    print("\n" + "=" * 60)
    print("  Model Loading Summary")
    print("=" * 60)
    for name in ready:
        print(f"    OK  {name}")
    for issue in issues:
        print(f"    !!  {issue}")
    print(f"\n  {_gpu_status()}")
    print("=" * 60)

    if issues:
        print("\n  WARNING: Some models have issues (see above).")
        print("  Proceeding anyway — affected steps will use fallbacks.")

    # ── Phase 2: Run generation ──
    print("\n--- Phase 2: Generating experience libraries ---")

    builder = ExperienceLibraryBuilder(
        policy_model=policy,
        referee_model=referee,
        num_rollouts=args.rollouts,
        output_dir=args.output_dir,
        load_vlm=False,  # already loaded above
    )
    # Attach the pre-loaded VLM client
    if vlm_client is not None:
        builder.vlm_client = vlm_client

    builder.process_all(args.persona)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
