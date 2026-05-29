"""
Configuration for the Memory-Augmented Inference System.
"""
import os

# ============================================================
# PATHS
# ============================================================
# Relative to the inference/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MEMORY_DIR = os.path.join(PROJECT_DIR, "memory", "personas")
TIMELINES_PATH = os.path.join(PROJECT_DIR, "output", "persona_timelines.json")
CONVERSATIONS_PATH = os.path.join(PROJECT_DIR, "output", "conversations.json")

# ============================================================
# EMBEDDING MODEL
# ============================================================
# Paper Appendix C.3: EmbeddingGemma (fallback handled in retrieval.EmbeddingEngine)
EMBEDDING_MODEL = "EmbeddingGemma"

# ============================================================
# RETRIEVAL SETTINGS
# ============================================================
TOP_K_SESSIONS = 2          # Top K sessions from semantic search (paper: k=2)
TOP_K_RATIONALES = 8        # Top K experience rationales at inference (paper: K=8)
HOP_EXPAND = 1              # Hops to expand from top sessions
TOKEN_BUDGET = 2000         # Max tokens for memory context in prompt

# Session node hybrid scoring weights (paper Appendix C.3):
#   score = λ * sim_sem + (1-λ) * sim_lex
# SEMANTIC_WEIGHT = λ = 0.8 ; BM25_WEIGHT = (1-λ) = 0.2
# GRAPH_WEIGHT applies only to Step-5 spreading-activation nodes (not
# part of the session-scoring formula — it scales the activation signal
# used to score 1-hop expanded neighbours).
SEMANTIC_WEIGHT = 0.8
BM25_WEIGHT = 0.2
GRAPH_WEIGHT = 0.5          # spreading-activation scale for expanded nodes

# Graph activation decay
ACTIVATION_DECAY = 0.5

# ============================================================
# LLM SETTINGS
# ============================================================
AGENT_MODEL = "gpt-4o-mini"    # Model for agent inference
AGENT_TEMPERATURE = 0.8
AGENT_MAX_TOKENS = 400

# ============================================================
# EXPERIENCE LIBRARY SETTINGS
# ============================================================
# Paper §4.1: policy backbone = Qwen3-3B-Instruct.
POLICY_MODEL_NAME = "Qwen/Qwen3-3B-Instruct"
POLICY_MAX_NEW_TOKENS = 300
POLICY_TEMPERATURE = 0.9
NUM_ROLLOUTS = 3
# Paper §4.1: referee = GPT-5 (OpenAI API).
REFEREE_MODEL = "gpt-5"
EXPERIENCE_LIBRARY_DIR = os.path.join(BASE_DIR, "experience_library")

# Paper §4.1: earliest 80% for memory/library, latest 20% for evaluation
TRAIN_SPLIT = 0.8

# ============================================================
# PERSONA DEFAULTS
# ============================================================
# Pre-configured 3 persona combos for testing inference.
# Edit these to match your generated personas.
DEFAULT_PERSONA_CONFIGS = [
    {
        "persona_id": "P_001",
        "personality": "budget-conscious",
        "dominant_trait": "openness",
        "income_range": "Low",
    },
    {
        "persona_id": "P_004",
        "personality": "budget-conscious",
        "dominant_trait": "conscientiousness",
        "income_range": "Low",
    },
    {
        "persona_id": "P_007",
        "personality": "budget-conscious",
        "dominant_trait": "extraversion",
        "income_range": "Low",
    },
]



# ============================================================
# FEW-SHOT EXAMPLES FOR INFERENCE
# ============================================================
INFERENCE_FEW_SHOT_EXAMPLES = """Here are examples of ideal agent responses. Match their tone, density, and structure:

Example 1 (Opening):
User: "Need motor insurance urgently. Just bought this beauty!"
Agent: "Congrats on the new ride — she's a beauty! I'd recommend National Insurance Company for your cover; they provide comprehensive protection for accidents, theft and natural calamities, accessories protection for added kit, and a wide cashless garage network with strong local support. I recently helped a similar owner get a cashless repair within 48 hours through National — would you like me to set that up with accessories and passenger cover?"

Example 2 (Budget negotiation):
User: "I need to stick to Premium ₹5,000 and really want to avoid a big bill after a repair. Can you add a zero-depreciation cover while keeping Premium ₹5,000?"
Agent: "I get wanting to avoid big bills — we can add zero-depreciation but it increases premium, so to hold it at ₹5,000 we'd need to offset that (usually by lowering the IDV further or raising your voluntary deductible). With zero-dep, a ₹50,000 repair typically leaves you with only the small voluntary deductible (≈₹1,000–2,000) instead of the ₹16,000–22,000 you'd face without it. Shall I run an exact quote that keeps the premium at ₹5,000 by adjusting IDV so you get zero-depreciation?"

Example 3 (Closing/Confirmation):
User: "Premium ₹6,000 works for me. Could you please outline exactly what's covered and excluded?"
Agent: "Got it — I'll prepare the formal quote for your car with premium ₹6,000 and a ₹5,000 voluntary deductible so you and your family can drive with peace of mind. I can send a signed, itemised quote and policy summary (inclusions, exclusions, claim steps and payment link) within 1 hour. Which do you prefer — email or WhatsApp?"

Adapt your style to match these examples. Never start with 'Absolutely'. Tailor each response to the specific conversation context."""


def get_big_five_scores(personality: str, dominant_trait: str) -> dict:
    """
    Get Big Five scores for a persona combo.
    Applies +0.3 boost to the dominant trait (capped at 1.0).
    """
    base = BIG_FIVE_PROFILES.get(personality, {
        "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
        "agreeableness": 0.5, "neuroticism": 0.5,
    })
    scores = base.copy()
    if dominant_trait in scores:
        scores[dominant_trait] = min(1.0, scores[dominant_trait] + 0.3)
    return scores
