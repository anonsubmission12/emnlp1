import json
import random
import re
import os
from typing import Dict, List, Tuple, Optional
from openai_client import OpenAIClient
from spectrum import IntentSpectrum, StrategySpectrum
from memory import MemoryManager
from judge import ConversationJudge
from graph_visualizer import GraphVisualizer
from graph import DenseGraph, create_intent_graph, create_strategy_graph
from few_shot_examples import get_persona_examples, get_strategy_examples, get_intent_examples
from verification import log_checkpoints


class ExpertAgents:
    """
    Multi-agent reasoning system for persuasion.
    Adapted from test10111.py.
    """
    
    def __init__(self, client: Optional[OpenAIClient] = None, use_llm: bool = True):
        self.client = client
        self.use_llm = use_llm
        self.experts = {
            "Engagement Expert": self._engagement_reasoning,
            "Keyterm Expert": self._keyterm_reasoning,
            "Motive Expert": self._intent_reasoning,
            "Sentiment Expert": self._sentiment_reasoning,
            "Knowledge Expert": self._knowledge_reasoning
        }

    def _llm_reasoning(
        self,
        expert_name: str,
        context: Dict,
        user_message: str,
        history: List,
    ) -> Optional[str]:
        if not self.client or not self.use_llm:
            return None

        history_text = "\n".join(
            [f"{h[2]}: {h[3]}" for h in history[-5:]]
        )
        kb = context.get("knowledge_base", {})
        kb_text = json.dumps(kb, indent=2)[:800]

        prompt = (
            f"You are the {expert_name} in a persuasion expert panel. "
            "Provide a 1-2 sentence rationale for the next agent response.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User message: {user_message}\n\n"
            f"Knowledge base excerpt:\n{kb_text}\n\n"
            "Return a concise rationale only."
        )

        try:
            text, _ = self.client.generate_text(prompt, temperature=0.3, max_tokens=120)
            return text.strip()
        except Exception:
            return None
    
    def _engagement_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Engagement Expert: Analyzes rapport-building opportunities."""
        turn_count = len(history)
        if turn_count < 3:
            return "Early conversation - focus on building rapport and trust"
        elif "thank" in user_message.lower() or "great" in user_message.lower():
            return "Positive sentiment detected - reinforce with enthusiasm"
        else:
            return "Maintain professional engagement"
    
    def _keyterm_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Keyterm Expert: Identifies key features to highlight."""
        keywords = ["coverage", "price", "premium", "deductible", "claim", "garage"]
        mentioned = [kw for kw in keywords if kw in user_message.lower()]
        if mentioned:
            return f"User interested in: {', '.join(mentioned)}"
        return "No specific keywords detected"
    
    def _intent_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Motive Expert: Identifies user intent."""
        msg_lower = user_message.lower()
        if "?" in user_message:
            return "User seeking information - provide clear answer"
        elif any(word in msg_lower for word in ["expensive", "discount", "reduce", "cheaper"]):
            return "Price concern - justify value or offer discount"
        elif any(word in msg_lower for word in ["worried", "concern", "risk"]):
            return "User has concerns - address with reassurance"
        else:
            return "General inquiry - provide helpful information"
    
    def _sentiment_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Sentiment Expert: Analyzes emotional tone."""
        positive_words = ["great", "good", "excellent", "perfect", "thanks"]
        negative_words = ["expensive", "worried", "concern", "not sure", "hesitant"]
        
        msg_lower = user_message.lower()
        if any(word in msg_lower for word in positive_words):
            return "Positive sentiment - user is receptive"
        elif any(word in msg_lower for word in negative_words):
            return "Negative/cautious sentiment - address concerns"
        else:
            return "Neutral sentiment"
    
    def _knowledge_reasoning(self, context: Dict, user_message: str, history: List) -> str:
        """Knowledge Expert: Determines what information to retrieve."""
        if "add-on" in user_message.lower() or "rider" in user_message.lower():
            return "Retrieve add-on information"
        elif "price" in user_message.lower() or "cost" in user_message.lower():
            return "Retrieve pricing information"
        else:
            return "Retrieve general coverage information"
    
    def orchestrate(self, context: Dict, user_message: str, history: List, 
                   active_experts: List[str] = None) -> Dict[str, str]:
        """
        Orchestrate multiple experts to produce reasoning.
        Returns dict of expert_name -> thinking.
        """
        if active_experts is None:
            active_experts = self._select_experts(context, user_message, len(history))
        
        thinking = {}
        for expert_name in active_experts:
            if expert_name in self.experts:
                llm_thought = self._llm_reasoning(expert_name, context, user_message, history)
                if llm_thought:
                    thinking[expert_name] = llm_thought
                else:
                    thinking[expert_name] = self.experts[expert_name](context, user_message, history)
        
        return thinking
    
    def _select_experts(self, context: Dict, user_message: str, turn_count: int) -> List[str]:
        """Dynamically select which experts to invoke based on context."""
        experts = ["Motive Expert", "Sentiment Expert"]
        
        if turn_count < 5:
            experts.append("Engagement Expert")
        
        if "?" in user_message:
            experts.append("Knowledge Expert")
        
        if any(word in user_message.lower() for word in ["coverage", "feature", "benefit"]):
            experts.append("Keyterm Expert")
        
        return experts
    
    def synthesize(self, thinking: Dict[str, str], context: Dict) -> str:
        """Synthesize expert thinking into orchestrator summary."""
        summaries = [f"{name}: {thought}" for name, thought in thinking.items()]
        if not self.client or not self.use_llm or not summaries:
            return " | ".join(summaries[:3])

        prompt = (
            "You are the Persuasion Synthesizer. Combine expert rationales "
            "into a single 1-2 sentence guidance note.\n\n"
            + "\n".join(summaries[:5])
            + "\n\nReturn the synthesized guidance only."
        )
        try:
            text, _ = self.client.generate_text(prompt, temperature=0.3, max_tokens=120)
            return text.strip()
        except Exception:
            return " | ".join(summaries[:3])


class DialogueGenerator:
    """
    Enhanced dialogue generator with spectrum-based intents/strategies,
    Memory, LLM-as-judge, and Local Models (Mistral v0.3 7B & Gemma 3 4B).
    """
    
    def __init__(self, insurance_data: Dict, openai_client: Optional[OpenAIClient] = None, token_tracker: Optional['TokenTracker'] = None):
        self.insurance_data = insurance_data
        self.client = openai_client or OpenAIClient()
        self.token_tracker = token_tracker
        
        # Initialize components
        self.experts = ExpertAgents(self.client, use_llm=True)
        self.intent_graph = create_intent_graph()
        self.strategy_graph = create_strategy_graph()
        self.judge = ConversationJudge(OpenAIClient(model="gemini-2.5-pro"))
        self.graph_visualizer = GraphVisualizer()  # Initialize graph visualizer
        self.graph_visualizer.start_server(port=8000)  # Start live server
        
        # Conversation state
        self.conversation_context = {}
        self.conversation_history = []
        self.intent_history = []
        self.strategy_history = []
        self.thinking_log = []
        
        # Pre-process insurer names
        self.insurers = {name.lower(): name for name in self.insurance_data["Motor Insurance"]["description"].keys()}
        
        # User personalities (Reduced to 7)
        self.user_personalities = [
            "budget-conscious",
            "detail-oriented",
            "skeptical",
            "enthusiastic",
            "cautious",
            "tech-savvy",
            "first-time buyer"
        ]
        
        # BIG FIVE personality dimensions (OCEAN model)
        # Scores: 0.0 (low) to 1.0 (high)
        self.big_five_profiles = {
            "budget-conscious": {
                "openness": 0.4,          # Moderate - focused on practical value
                "conscientiousness": 0.7, # High - careful with money
                "extraversion": 0.5,      # Moderate
                "agreeableness": 0.5,     # Moderate - will negotiate
                "neuroticism": 0.6        # Moderate-high - financial anxiety
            },
            "detail-oriented": {
                "openness": 0.6,          # Moderate-high - curious about details
                "conscientiousness": 0.9, # Very high - meticulous
                "extraversion": 0.4,      # Low-moderate - analytical
                "agreeableness": 0.5,     # Moderate
                "neuroticism": 0.5        # Moderate - wants certainty
            },
            "skeptical": {
                "openness": 0.5,          # Moderate
                "conscientiousness": 0.7, # High - does research
                "extraversion": 0.4,      # Low-moderate - reserved
                "agreeableness": 0.3,     # Low - questions everything
                "neuroticism": 0.6        # Moderate-high - trust issues
            },
            "enthusiastic": {
                "openness": 0.8,          # High - open to new ideas
                "conscientiousness": 0.5, # Moderate
                "extraversion": 0.9,      # Very high - energetic
                "agreeableness": 0.7,     # High - cooperative
                "neuroticism": 0.3        # Low - optimistic
            },
            "cautious": {
                "openness": 0.4,          # Low-moderate - prefers safe options
                "conscientiousness": 0.8, # High - risk-averse
                "extraversion": 0.3,      # Low - reserved
                "agreeableness": 0.6,     # Moderate-high
                "neuroticism": 0.7        # High - worries about risks
            },
            "tech-savvy": {
                "openness": 0.9,          # Very high - loves innovation
                "conscientiousness": 0.6, # Moderate-high
                "extraversion": 0.6,      # Moderate-high
                "agreeableness": 0.5,     # Moderate
                "neuroticism": 0.4        # Low-moderate
            },
            "first-time buyer": {
                "openness": 0.6,          # Moderate-high - learning
                "conscientiousness": 0.6, # Moderate-high - wants to do it right
                "extraversion": 0.5,      # Moderate
                "agreeableness": 0.7,     # High - trusting
                "neuroticism": 0.6        # Moderate-high - uncertain
            }
        }
        
        # User openings
        self.user_openings = [
            "Hi, I just bought a new car. Can you help me find a good insurance policy?",
            "Hello, I need to get insurance for my new vehicle. What do you recommend?",
            "I'm looking for an insurance plan for my brand new car.",
            "Hey, I need to sort out insurance for my new car. Can you help?",
            "Just got my first car! Need help with insurance.",
            "I bought a car yesterday and need to insure it quickly.",
            "Looking for comprehensive coverage for my new vehicle.",
            "Can you suggest insurance options for my newly purchased car?",
            "Need motor insurance urgently. Just bought this beauty!",
            "Hi! New car owner here. What insurance should I get?"
        ]
        
        # Persona backstories for mid-conversation injection
        self.persona_stories = {
            "budget-conscious": "You're a young professional saving for a house. Every rupee counts.",
            "detail-oriented": "You're an engineer who reads every contract thoroughly. You've been burned by fine print before.",
            "skeptical": "You had a bad experience with an insurance claim being delayed. You need proof now.",
            "enthusiastic": "You just got promoted and bought your dream car. You want the best protection for it.",
            "cautious": "You have a family with young kids. Their safety is your top priority.",
            "tech-savvy": "You work in IT and appreciate digital solutions. You hate paperwork.",
            "first-time buyer": "This is your first car and first insurance policy. You're learning as you go."
        }
        
        # Deductible ranges
        self.deductible_ranges = {
            "Economy": {"min": 1000, "standard": 2500, "max": 5000},
            "Mid-range": {"min": 2500, "standard": 5000, "max": 10000},
            "Premium": {"min": 5000, "standard": 10000, "max": 25000}
        }
    
    def _extract_price(self, text: str) -> Optional[int]:
        """Extracts the first reasonable premium price from a string."""
        # Enhanced regex to catch more currency formats
        # We find ALL matches and pick the most reasonable one for a premium
        matches = re.findall(r'(?:₹|Rs\.?|INR)\s*([\d,]+)|([\d,]+)\s*(?:rs|inr)', text, re.IGNORECASE)
        
        candidates = []
        for m in matches:
            val_str = (m[0] or m[1]).replace(',', '')
            try:
                val = int(val_str)
                # Filter out likely IDV values (e.g. > 2L) or tiny values (e.g. < 1000)
                # Unless it's a super luxury car, premiums rarely exceed 2L in this context.
                if 1000 <= val <= 300000: 
                    candidates.append(val)
            except ValueError:
                continue
                
        if candidates:
            # If multiple, pick the one closest to a "standard" premium range? 
            # Or just the first reasonable one.
            return candidates[0]
            
        return None
    
    def _retrieve_relevant_info(self, query: str, recommended_provider: str = None) -> Dict:
        """Retrieves specific, relevant information from the knowledge base."""
        query_lower = query.lower()
        data = self.insurance_data["Motor Insurance"]
        
        # Prioritize recommended provider if specified
        if recommended_provider:
            for name_lower, original_name in self.insurers.items():
                if recommended_provider.lower() in name_lower or name_lower in recommended_provider.lower():
                    return {
                        f"Info for {original_name}": {
                            "description": data["description"].get(original_name),
                            "pricing": data["pricing_information"]["comprehensive_premium_ranges"].get(original_name),
                            "unique_features": data["insurer_specific_features"].get(original_name)
                        }
                    }
        
        # Check for specific insurers
        for name_lower, original_name in self.insurers.items():
            if name_lower in query_lower or original_name.split(' ')[0].lower() in query_lower.split(' '):
                return {
                    f"Info for {original_name}": {
                        "description": data["description"].get(original_name),
                        "pricing": data["pricing_information"]["comprehensive_premium_ranges"].get(original_name),
                        "unique_features": data["insurer_specific_features"].get(original_name)
                    }
                }
        
        # Check for add-ons
        addon_keywords = ['add-on', 'addon', 'rider', 'zero depreciation', 'engine protection']
        if any(keyword in query_lower for keyword in addon_keywords):
            return {"Relevant Add-Ons": data["add_ons_detailed"]}
        
        # Check for pricing
        price_keywords = ['price', 'premium', 'cost', 'quote', 'rate', 'how much']
        if any(keyword in query_lower for keyword in price_keywords):
            return {"Pricing Information": data["pricing_information"]["comprehensive_premium_ranges"]}
        
        # Check for features
        feature_keywords = ['feature', 'coverage', 'cover', 'benefit', 'protect']
        if any(keyword in query_lower for keyword in feature_keywords):
            return {"Coverage Features": data["features_classification"]}
        
        # Fallback
        return {
            "Core Coverage": data["features_classification"]["core_coverage_features"],
            "Service Features": data["features_classification"]["service_features"][:3]
        }
    
    def analyze_vehicle_with_vlm(self, image_url: str, conversation_id: Optional[int] = None, year: Optional[int] = None) -> Dict:
        """Analyze vehicle using local Gemma 3 4B vision with year-aware descriptions."""
        try:
            # Year-aware prompt to prevent temporal inconsistencies
            year_context = f"This photo is from {year}." if year else "This is a recent photo."
            
            prompt = f"""{year_context} Look at this vehicle and determine:
1. Category: Is it an Economy (basic/budget), Mid-range (standard family car), or Premium (luxury/sports) vehicle?
2. Type: What kind of vehicle is it? (SUV, Sedan, Hatchback, etc.)?
3. Model year: Based on the design, what approximate year range does this vehicle appear to be from?
4. Notable features visible

IMPORTANT: If this is from {year if year else 'the present'}, describe it as a vehicle that would be available in that year. Do NOT mention future model years.

Be specific and concise."""
            
            print(f"Analyzing vehicle with Gemma vision (year: {year if year else 'current'})...")
            analysis, token_usage = self.client.analyze_image(image_url, prompt)
            
            # Track tokens
            if self.token_tracker:
                self.token_tracker.track_usage(
                    token_usage.get('prompt_tokens', 0),
                    token_usage.get('completion_tokens', 0),
                    conversation_id=conversation_id,
                    call_type="vision"
                )
                print(f"  → Tokens: {token_usage.get('total_tokens', 0):,} (prompt: {token_usage.get('prompt_tokens', 0):,}, completion: {token_usage.get('completion_tokens', 0):,})")
            
            # Parse category
            analysis_lower = analysis.lower()
            if "premium" in analysis_lower or "luxury" in analysis_lower or "sports" in analysis_lower:
                category = "Premium"
            elif "economy" in analysis_lower or "budget" in analysis_lower or "basic" in analysis_lower:
                category = "Economy"
            else:
                category = "Mid-range"
            
            return {
                "vehicle_category": category,
                "description": analysis
            }
        
        except Exception as e:
            print(f"VLM analysis error: {e}")
            # Fallback with year-appropriate description
            fallback_year = year if year else 2025
            return {
                "vehicle_category": random.choice(['Economy', 'Mid-range', 'Premium']),
                "description": f"Standard {fallback_year} model vehicle requiring comprehensive coverage."
            }

    def _format_life_events(self, life_events: Optional[List]) -> str:
        if not life_events:
            return ""
        parts = []
        for ev in life_events:
            desc = ev.get("description", "") if isinstance(ev, dict) else str(ev)
            impact = ev.get("impact", "") if isinstance(ev, dict) else ""
            year = ev.get("year") if isinstance(ev, dict) else None
            label = f"{year}: " if year else ""
            if impact:
                parts.append(f"{label}{desc} (impact: {impact})")
            else:
                parts.append(f"{label}{desc}")
        return " | ".join(parts)

    def _estimate_strategy_spectrum(
        self,
        user_message: str,
        context: Dict,
        conversation_history: List,
        memory_context: Dict,
    ) -> Optional[StrategySpectrum]:
        """
        Use the LLM to estimate the per-turn strategy spectrum multipliers.
        Returns a StrategySpectrum or None on failure.
        """
        history_text = "\n".join([f"{h[2]}: {h[3]}" for h in conversation_history[-5:]])
        memory_text = json.dumps(memory_context, indent=2)[:600]
        life_events = context.get("life_events_text", "")

        prompt = (
            "Estimate the best-fit persuasion spectrum for the next agent reply. "
            "Return JSON with keys: credibility, emotional_appeal, logical_argument, personalization. "
            "Values must be between 0 and 1.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"User message: {user_message}\n\n"
            f"Memory context: {memory_text}\n\n"
            f"Life events: {life_events}\n\n"
            "Return JSON only."
        )

        try:
            text, _ = self.client.generate_text(prompt, temperature=0.2, max_tokens=120)
            data = json.loads(text)
            vector = {
                "credibility": float(data.get("credibility", 0.5)),
                "emotional_appeal": float(data.get("emotional_appeal", 0.5)),
                "logical_argument": float(data.get("logical_argument", 0.5)),
                "personalization": float(data.get("personalization", 0.5)),
            }
            # Clamp
            vector = {k: max(0.0, min(1.0, v)) for k, v in vector.items()}
            return StrategySpectrum(vector)
        except Exception:
            return None

    def _estimate_concession_expectation(
        self,
        user_message: str,
        negotiation_context: Dict,
    ) -> Optional[float]:
        """
        LLM estimate of concession expectation for this turn.
        Returns float in [0,1] or None on failure.
        """
        prompt = (
            "Estimate the user's concession expectation (0-1) for this turn. "
            "Consider the message and recent negotiation context. "
            "Return JSON: {\"concession_expectation\": <float>}\n\n"
            f"User message: {user_message}\n\n"
            f"Negotiation context: {json.dumps(negotiation_context, indent=2)[:500]}"
        )

        try:
            text, _ = self.client.generate_text(prompt, temperature=0.2, max_tokens=80)
            data = json.loads(text)
            val = float(data.get("concession_expectation", 0.5))
            return max(0.0, min(1.0, val))
        except Exception:
            return None

    def _estimate_decision_readiness(self, user_message: str) -> float:
        """Lexical readiness scoring: 0.3 + 0.15*n_pos - 0.15*n_neg (Eq. 7)."""
        pos_words = [
            "buy", "purchase", "go ahead", "proceed", "confirm", "deal",
            "accept", "finalize", "ready", "sign up", "take it",
        ]
        neg_words = [
            "think about", "not sure", "maybe", "later", "not now",
            "pass", "no thanks", "not interested", "wait",
        ]
        text = user_message.lower()
        n_pos = sum(1 for w in pos_words if w in text)
        n_neg = sum(1 for w in neg_words if w in text)
        score = 0.3 + 0.15 * n_pos - 0.15 * n_neg
        return max(0.0, min(1.0, score))

    def _load_curated_personas(self, filepath: str = "curated_personas.json") -> List[Dict]:
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.path.dirname(__file__), filepath)
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []
    
    def generate_agent_response(
        self,
        user_message: str,
        context: Dict,
        conversation_history: List,
        conversation_id: int,
        turn_no: int,
        memory: MemoryManager,
        image_url: str = None,
        recommended_provider: str = None
    ) -> Tuple[str, str, StrategySpectrum]:
        """
        Generate agent response with spectrum-based strategy selection.
        
        Returns:
            (strategy_name, response, strategy_spectrum)
        """
        # Retrieve memory context
        memory_context = memory.retrieve_context(user_message, session_id=f"session_{conversation_id}")

        # Attach knowledge base excerpt for expert modules
        context["knowledge_base"] = self._retrieve_relevant_info(user_message, recommended_provider)

        # Multi-agent reasoning
        expert_thinking = self.experts.orchestrate(
            context=context,
            user_message=user_message,
            history=conversation_history
        )
        
        # Log thinking
        for expert_name, thinking in expert_thinking.items():
            self.thinking_log.append({
                "conversation_id": conversation_id,
                "turn_no": turn_no,
                "agent_name": expert_name,
                "thinking": thinking
            })
        
        # Synthesize expert insights
        orchestrator_summary = self.experts.synthesize(expert_thinking, context)
        self.thinking_log.append({
            "conversation_id": conversation_id,
            "turn_no": turn_no,
            "agent_name": "Orchestrator",
            "thinking": orchestrator_summary
        })
        
        # Get strategy using spectrum-based graph sampling
        last_strategy = self.strategy_history[-1] if self.strategy_history else "Default"
        
        # Create strategy spectrums for candidates
        strategy_spectrums = {
            node: StrategySpectrum.from_strategy(node)
            for node in self.strategy_graph.nodes
        }
        # LLM-estimated per-turn spectrum (paper Appendix B.3)
        current_spectrum = self._estimate_strategy_spectrum(
            user_message=user_message,
            context=context,
            conversation_history=conversation_history,
            memory_context=memory_context,
        )
        if current_spectrum is None:
            current_spectrum = StrategySpectrum.from_strategy(last_strategy)
        
        # Sample next strategy
        chosen_strategy = self.strategy_graph.sample_next(
            last_strategy,
            spectrum_current=current_spectrum,
            spectrum_candidates=strategy_spectrums,
            exclude_nodes=[last_strategy] if len(self.strategy_history) > 0 else None
        )
        
        self.strategy_history.append(chosen_strategy)
        chosen_spectrum = strategy_spectrums[chosen_strategy]
        
        # Build conversation history
        history_context = "\n".join([f"{speaker}: {msg}" for _, _, speaker, msg in conversation_history[-6:]])
        
        memory_str = json.dumps({
            "background": memory_context.get("background", {}),
            "recent_situations": memory_context.get("situations", "")[:100]
        }, indent=2)
        
        # Strategy descriptions
        strategy_descriptions = {
            "Default": "Be professional and informative.",
            "Credibility": "Emphasize company reputation, awards, claim settlement ratio.",
            "Emotional": "Appeal to security, family safety, peace of mind.",
            "Logical": "Use statistics, comparisons, cost-benefit analysis.",
            "Personal": "Relate to their specific situation and vehicle.",
            "Persona": "Share a relevant anecdote or personal experience."
        }
        
        is_first_agent_turn = len(conversation_history) == 1
        vehicle_category = context.get('vehicle_value_category', 'Mid-range')
        deductibles = self.deductible_ranges[vehicle_category]
        
        if is_first_agent_turn:
            provider_info = self._retrieve_relevant_info("", recommended_provider)
            insurance_context_str = json.dumps(provider_info, indent=2)
            
            action_instruction = f"""
            This is your first response. You must:
            1. Acknowledge their new car purchase enthusiastically
            2. Recommend {recommended_provider} specifically
            3. Mention 2-3 key benefits relevant to their vehicle type
            4. Keep it conversational and under 4 sentences
            """
        else:
            relevant_kb = self._retrieve_relevant_info(user_message, recommended_provider)
            insurance_context_str = json.dumps(relevant_kb, indent=2)
            
            # Handle price queries
            if "price" in user_message.lower() and not context.get('premium_offered'):
                base_premium = {'Economy': 15000, 'Mid-range': 25000, 'Premium': 40000}[context['vehicle_value_category']]
                offer_price = int(base_premium * random.uniform(1.05, 1.25))
                self.conversation_context['premium_offered'] = offer_price
                action_instruction = f"Quote an annual premium of ₹{offer_price:,}. Briefly mention what's included. Use RUPEES (₹), not dollars."
            
            # Handle negotiation - Enhanced 3-stage flow
            elif any(word in user_message.lower() for word in ["negotiate", "discount", "expensive", "reduce", "cheaper", "lower"]):
                premium = context.get('premium_offered') or 25000
                
                # Stage 0: First discount request - Justify with add-ons
                if context.get('negotiation_stage', 0) == 0:
                    self.conversation_context['negotiation_stage'] = 1
                    self.conversation_context['negotiation_started'] = True
                    
                    # Get add-on details for justification
                    addon_info = self._retrieve_relevant_info("add-on", recommended_provider)
                    addon_details = json.dumps(addon_info, indent=2)[:400]
                    
                    action_instruction = f"""User wants a discount on ₹{premium:,}. Stage 1 - JUSTIFY PRICE FIRST:
                    1. Acknowledge their budget concern empathetically
                    2. Break down what the ₹{premium:,} includes (list 3-4 specific add-ons/features)
                    3. Mention the value of these add-ons (e.g., "Zero Depreciation alone saves ₹X in claims")
                    4. DO NOT offer discount yet - emphasize value
                    
                    Add-on reference: {addon_details}
                    Use RUPEES (₹), not dollars."""
                
                # Stage 1: User persists - Offer customization
                elif context.get('negotiation_stage') == 1:
                    self.conversation_context['negotiation_stage'] = 2
                    
                    action_instruction = f"""User still wants discount. Stage 2 - OFFER CUSTOMIZATION:
                    1. Suggest removing optional add-ons to reduce price
                    2. Mention 2-3 add-ons they could skip (e.g., roadside assistance, engine protection)
                    3. Estimate savings: "Removing X and Y could bring it down to ₹{int(premium * 0.90):,}"
                    4. Ask what they'd like to keep vs. remove
                    Use RUPEES (₹), not dollars."""
                
                # Stage 2: User still stubborn - Finally offer discount
                else:
                    current = context['premium_offered']
                    new_price = int(current * 0.92)  # 8% discount
                    self.conversation_context['premium_offered'] = new_price
                    self.conversation_context['negotiation_stage'] += 1
                    
                    action_instruction = f"""User insists on discount. Stage 3 - FINAL OFFER:
                    1. Acknowledge their persistence positively
                    2. Offer special discount: ₹{new_price:,} (down from ₹{current:,})
                    3. Emphasize this is a limited-time offer
                    4. Confirm all features are still included
                    Use RUPEES (₹), not dollars."""
            else:
                action_instruction = f"Respond using {chosen_strategy} strategy. Address their specific question/concern. Always use RUPEES (₹) for monetary values."
        
        # Get strategy-specific few-shot examples
        agent_few_shots = get_strategy_examples(chosen_strategy, max_examples=3)
        if not agent_few_shots:
            # Fallback to generic examples
            agent_few_shots = """
Example 1 (Credibility):
User: "Why should I trust you?"
Agent: "That's a fair question. At Aspire, we've settled over 98% of our motor claims last year alone. We've been recognized as the 'Most Trusted Insurer' in 2024, and our 24/7 support ensures you're never stranded in case of an accident."

Example 2 (Emotional):
User: "Insurance feels like a waste of money."
Agent: "I understand. Think of it as buying peace of mind for your family. If something unexpected happens to your new car, you shouldn't have to worry about the financial burden while also dealing with the stress of the situation. We're here to protect what you value most."

Example 3 (Personal):
User: "I'm worried about the price for my SUV."
Agent: "I see you've got a premium SUV there; it's a great choice for family trips! Given its high value, a small scratch could be expensive to fix. Our specific SUV-tier plan covers those minor cosmetic issues that standard plans often miss, keeping your beauty in showroom condition."
"""

        # Build prompt
        prompt = f"""You are a senior insurance consultant at Aspire Insurance. You are professional, empathetic, and highly knowledgeable. Your goal is to guide the customer to the best policy for their new vehicle.

CRITICAL CONSTRAINTS:
1. Always use Indian Rupees (₹) for ALL monetary values. NEVER use dollars ($).
2. Be concise: 3-4 sentences max.
3. Be human-like: Avoid robotic or overly formal corporate-speak.
4. Do NOT mention your strategy name (e.g., don't say "I am using a Logical strategy").
5. Ask only ONE question per response if asking questions. Never ask multiple questions in a single turn.

PERSUASION FRAMEWORK (Use naturally, don't be obvious):
- Reciprocity: Offer valuable information/insights first to create goodwill
- Commitment: Build on customer's stated needs and previous agreements
- Social Proof: Reference what similar customers chose or industry standards
- Authority: Cite expertise, awards, settlement ratios when relevant
- Liking: Build rapport through empathy and understanding their situation
- Scarcity: Mention limited-time offers or exclusive benefits when appropriate

Expert Reasoning Context:
{orchestrator_summary[:150]}

Memory Context (Known info about customer):
{memory_str}

Life Events (if any):
{context.get('life_events_text', 'None')}

Conversation History:
{history_context}

Strategic Guidance:
- Strategy to use: {chosen_strategy}
- Strategy Goal: {strategy_descriptions[chosen_strategy]}

Few-Shot Examples for Style:
{agent_few_shots}

Specific Task: {action_instruction}

Relevant Product Knowledge:
{insurance_context_str[:600]}

Current Message to respond to: "{user_message}"

Respond naturally, using the guidance and context above. Remember: ALWAYS use ₹ for prices."""
        
        try:
            response, token_usage = self.client.generate_text(
                prompt,
                temperature=0.8,
                max_tokens=350
            )
            
            # Track tokens
            if self.token_tracker:
                self.token_tracker.track_usage(
                    token_usage.get('prompt_tokens', 0),
                    token_usage.get('completion_tokens', 0),
                    conversation_id=conversation_id,
                    call_type="text_generation"
                )
                print(f"    → Tokens: {token_usage.get('total_tokens', 0):,} (Agent response)")
            
            # Log synthesizer thinking
            self.thinking_log.append({
                "conversation_id": conversation_id,
                "turn_no": turn_no,
                "agent_name": "Synthesizer",
                "thinking": f"Generated response using {chosen_strategy} based on expert insights."
            })
            
            # Extract price if mentioned
            price_in_response = self._extract_price(response)
            if price_in_response:
                self.conversation_context['premium_offered'] = price_in_response
                # Log agent offer immediately
                memory.concession_memory.log_agent_offer(price_in_response)
            
            return chosen_strategy, response, chosen_spectrum
        
        except Exception as e:
            print(f"Error generating agent response: {e}")
            import traceback
            traceback.print_exc()
            
            # Context-aware fallback instead of generic opening
            if "price" in user_message.lower() or "cost" in user_message.lower():
                fallback_response = "I'd be happy to discuss pricing options. Could you tell me more about your budget expectations?"
            elif "?" in user_message:
                fallback_response = "That's a great question. Let me get you the specific information you need."
            elif any(word in user_message.lower() for word in ["discount", "negotiate", "reduce"]):
                fallback_response = "I understand you're looking for the best value. Let me see what options we have available."
            else:
                fallback_response = "I appreciate your interest. Could you tell me more about what you're looking for?"
            
            return "Default", fallback_response, StrategySpectrum.from_strategy("Default")
    
    def generate_user_utterance(
        self,
        agent_message: str,
        context: Dict,
        conversation_history: List,
        personality: str
    ) -> Tuple[str, str, IntentSpectrum]:
        """
        Generate user utterance with spectrum-based intent selection.
        
        Returns:
            (intent_name, response, intent_spectrum)
        """
        # Get intent using spectrum-based graph sampling
        last_intent = self.intent_history[-1] if self.intent_history else "I-1 Session Onset"
        
        # Determine current personality parameters
        current_personality = context.get('personality', personality)
        current_income = context.get('income_range', 'Mid')
        current_dominant_trait = context.get('dominant_trait', 'openness')
        
        # Create intent spectrums for candidates
        intent_spectrums = {
            node: IntentSpectrum.from_intent(node, personality)
            for node in self.intent_graph.nodes
        }
        current_spectrum = IntentSpectrum.from_intent(last_intent, personality)
        
        # Determine if we should finalize
        turn_count = context.get('turn_count', 0)
        should_finalize = (
            turn_count >= 27 and  # After ~27 turns (targeting ~30 total)
            context.get('premium_offered') and
            context.get('negotiation_stage', 0) >= 1
        )
        
        if should_finalize:
            if context['outcome'] == 'Accept':
                chosen_intent = "I-7 Renewal Commitment"
            else:
                chosen_intent = "I-8 Offer Decline"
        else:
            # Exclude finalization intents before ready
            exclude_intents = ["I-7 Renewal Commitment", "I-8 Offer Decline"] if not should_finalize else []
            if last_intent in exclude_intents:
                exclude_intents.remove(last_intent)
            
            # Sample next intent
            chosen_intent = self.intent_graph.sample_next(
                last_intent,
                spectrum_current=current_spectrum,
                spectrum_candidates=intent_spectrums,
                exclude_nodes=exclude_intents + ([last_intent] if len(self.intent_history) > 0 else [])
            )
        
        self.intent_history.append(chosen_intent)
        chosen_spectrum = intent_spectrums[chosen_intent]
        
        # Build conversation history
        history_context = "\n".join([f"{speaker}: {msg}" for _, _, speaker, msg in conversation_history[-5:]])
        
        # Get BIG FIVE profile for this personality
        base_big_five = self.big_five_profiles.get(current_personality, {
            "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
            "agreeableness": 0.5, "neuroticism": 0.5
        })
        
        # Adjust the base big five using the 105 net persona variation logic
        big_five = base_big_five.copy()
        if current_dominant_trait in big_five:
            # Boost the dominant trait by 0.3 (cap at 1.0) to reflect the 105 persona matrix
            big_five[current_dominant_trait] = min(1.0, big_five[current_dominant_trait] + 0.3)
        
        # Personality-specific instructions
        personality_traits = {
            "budget-conscious": "You're price-sensitive. You want good value without compromising essentials.",
            "detail-oriented": "You're meticulous and want to understand every clause cleanly.",
            "skeptical": "You're cautious and need evidence, settlement ratios, and clear justifications.",
            "enthusiastic": "You're thrilled about your new car and open to premium features that enhance ownership.",
            "cautious": "Safety and security are your top priorities. You want maximum coverage to avoid future risks.",
            "tech-savvy": "You value modern conveniences like app-based claims and digital support.",
            "first-time buyer": "You're new to car insurance, so you need things explained simply and clearly."
        }
        
        # Income Range adjustments
        income_traits = {
            "Low": "You have tight budget constraints and prioritize keeping the premium as low as possible. Use humble, straightforward language.",
            "Mid": "You want a balance of quality coverage and value. You are pragmatic about expenses.",
            "High": "You are less price-sensitive and more focused on premium services, zero hassle, and top-tier comprehensive protection."
        }
        
        # BIG FIVE behavioral modifiers
        big_five_modifiers = []
        if big_five["openness"] > 0.7:
            big_five_modifiers.append("You're curious about innovative features and new coverage options.")
        if big_five["conscientiousness"] > 0.7:
            big_five_modifiers.append("You want detailed documentation and ask thorough questions.")
        if big_five["extraversion"] > 0.7:
            big_five_modifiers.append("You're very chatty and share personal anecdotes.")
        elif big_five["extraversion"] < 0.4:
            big_five_modifiers.append("You're reserved, keeping responses very brief.")
        if big_five["agreeableness"] > 0.6:
            big_five_modifiers.append("You're extremely polite, cooperative, and never rude. Ask questions rather than making demands.")
        elif big_five["agreeableness"] < 0.4:
            big_five_modifiers.append("You're assertive but still professional, directly stating what you want. Do NOT be rude, just firm.")
        if big_five["neuroticism"] > 0.6:
            big_five_modifiers.append("You express slight anxiety and worries about potential risks.")
        
        big_five_context = " ".join(big_five_modifiers) if big_five_modifiers else ""
        
        # Build instruction
        if chosen_intent == "I-7 Renewal Commitment":
            instruction = f"Accept the offer of ₹{context['premium_offered']:,}. Ask about next steps."
        elif chosen_intent == "I-8 Offer Decline":
            instruction = f"Politely decline. The ₹{context['premium_offered']:,} is over your budget."
        else:
            instruction = f"Respond with intent '{chosen_intent}'. Be {personality}."
        
        # Get persona-specific few-shot examples
        user_few_shots = get_persona_examples(personality, max_examples=3)
        if not user_few_shots:
            # Fallback to generic examples
            user_few_shots = """
Example 1 (Budget-conscious):
Agent: "Our premium plan is ₹25,000."
User: "That's a bit steep for my budget. Are there any discounts available or perhaps a more basic plan that covers just the essentials?"

Example 2 (Detail-oriented):
Agent: "The policy includes zero depreciation."
User: "Could you explain exactly what that covers? Does it include plastic and fiber parts as well, or just the metal components of the engine?"

Example 3 (Skeptical):
Agent: "We offer the fastest claim settlement."
User: "I've heard that before from other agents. Do you have any data or customer reviews that back up that claim? I want to be sure I won't be stuck waiting for weeks."
"""

        prompt = f"""You are a {current_personality} car owner in a conversation with an insurance agent. You just bought a new vehicle and need to secure the best possible insurance for it.

CRITICAL CONSTRAINTS:
1. Always use Indian Rupees (₹) for ALL monetary values.
2. Respond naturally in 1-2 sentences. 
3. Be conversational and authentic to your personality.
4. Don't repeat phrases verbatim from the agent.
5. Ask only ONE question if you have questions. Never ask multiple questions in a single turn.
6. NEVER BE RUDE. Even if you are skeptical or assertive, maintain a polite, respectful demeanor.

Recent Conversation Context:
{history_context}

Your Profile:
- Overall Personality: {personality_traits.get(current_personality, "Be yourself")}
- Income Range: {current_income}. {income_traits.get(current_income, "")}
- Behavioral Traits: {big_five_context if big_five_context else "Stay true to your personality"}
- Specific Intent for this turn: {chosen_intent}

Life Events (if any):
{context.get('life_events_text', 'None')}

Style Examples (How different people speak):
{user_few_shots}

Your Current Task:
{instruction} 

Respond as the {current_personality} owner. Be curious, engaged, and stay in character while staying extremely polite. Use ₹ for any price mentions.
IMPORTANT: If you are negotiating, clearly state "Premium ₹X" or "Price ₹X" to be specific. Avoid confusing premium with IDV or coverage amounts."""
        
        try:
            response, token_usage = self.client.generate_text(
                prompt,
                temperature=0.9,
                max_tokens=150
            )
            
            # Track tokens (no conversation_id for user utterances in this context)
            if self.token_tracker:
                self.token_tracker.track_usage(
                    token_usage.get('prompt_tokens', 0),
                    token_usage.get('completion_tokens', 0),
                    conversation_id=None,
                    call_type="text_generation"
                )
                print(f"    → Tokens: {token_usage.get('total_tokens', 0):,} (User utterance)")
            
            return chosen_intent, response, chosen_spectrum
        
        except Exception as e:
            print(f"Error generating user response: {e}")
            fallback_responses = {
                "I-3 Quote Solicitation": "What's the price for that?",
                "I-2 Coverage Inquiry": "What does this cover exactly?",
                "I-4 Concern Disclosure": "I'm not sure about this.",
                "I-6 Premium Negotiation": "Can you do better on the price?"
            }
            return chosen_intent, fallback_responses.get(chosen_intent, "Tell me more."), chosen_spectrum
    
    def generate_conversation(
        self,
        image_url: str,
        conversation_id: int,
        target_turns: int = 30,
        persona_id: Optional[str] = None,
        year: Optional[int] = None,
        life_events: Optional[List] = None,
        **context_kwargs
    ) -> Tuple[Dict, List[Dict], List[Dict]]:
        """
        Generate a complete conversation with optional persona memory and life events.
        
        Args:
            image_url: URL of vehicle image
            conversation_id: Unique conversation ID
            target_turns: Target number of turns
            persona_id: Persona identifier for memory persistence
            year: Year of conversation (for 10-year timeline)
            life_events: List of life events for this year
            dominant_trait: The specific Big Five trait being emphasized
            income_range: Income range key ('Low', 'Mid', 'High')
            
        Returns:
            (conversation_dict, thinking_log, memory_snapshots)
        """
        # Reset state
        self.conversation_history = []
        self.intent_history = []
        self.strategy_history = []
        conversation_thinking = []
        memory_snapshots = []
        if self.judge:
            self.judge._qt_history = []
        
        # Initialize Memory Manager
        # Note: client is passed if needed, but MemoryManager mainly needs persona_id
        memory = MemoryManager(persona_id=persona_id if persona_id else f"P_Unknown_{conversation_id}")
        
        # Load persona memory if exists
        if persona_id:
            persona_memory_path = f"memory/personas/{persona_id}.json"
            if memory.load(persona_memory_path):
                print(f"  Loaded persona memory from {persona_memory_path}")
            else:
                print(f"  Created new persona memory for {persona_id}")
        
        # Start Level 1 Session Memory
        current_year = year if year else 2025
        
        # Analyze vehicle with year context
        vehicle_analysis = self.analyze_vehicle_with_vlm(image_url, conversation_id=conversation_id, year=year)

        # Select persona (curated 60 if available)
        curated_personas = self._load_curated_personas()
        if context_kwargs.get("personality"):
            personality = context_kwargs.get("personality")
            dominant_trait = context_kwargs.get("dominant_trait", "openness")
            income_range = context_kwargs.get("income_range", "Mid")
        elif curated_personas:
            chosen = random.choice(curated_personas)
            personality = chosen.get("personality", random.choice(self.user_personalities))
            dominant_trait = chosen.get("dominant_trait", "openness")
            income_range = chosen.get("income_range", "Mid")
        else:
            personality = random.choice(self.user_personalities)
            dominant_trait = context_kwargs.get("dominant_trait", "openness")
            income_range = context_kwargs.get("income_range", "Mid")
        
        # Select random insurer
        all_providers = list(self.insurance_data["Motor Insurance"]["description"].keys())
        recommended_provider = random.choice(all_providers)
        
        # Initialize context
        base_premiums = {'Economy': 15000, 'Mid-range': 25000, 'Premium': 40000}
        base_premium = base_premiums[vehicle_analysis['vehicle_category']]
        
        # Determine outcome - 75% accept, 25% reject
        if random.random() < 0.25:
            outcome = 'Reject'
            user_budget = int(base_premium * random.uniform(0.75, 0.90))
        else:
            outcome = 'Accept'
            user_budget = int(base_premium * random.uniform(0.95, 1.20))
        
        self.conversation_context = {
            "vehicle_info": vehicle_analysis["description"],
            "vehicle_value_category": vehicle_analysis["vehicle_category"],
            "premium_offered": None,
            "negotiation_stage": 0,
            "negotiation_started": False,
            "outcome": outcome,
            "user_budget": user_budget,
            "personality": personality,
            "income_range": income_range,
            "dominant_trait": dominant_trait,
            "recommended_provider": recommended_provider,
            "turn_count": 0,
            "finalized": False,
            "life_events": life_events or [],
            "life_events_text": self._format_life_events(life_events),
        }

        # Start Level 1 Session Memory with persona profile
        persona_profile = {
            "personality": personality,
            "dominant_trait": dominant_trait,
            "income_range": income_range,
        }
        memory.start_new_session(current_year, persona_profile=persona_profile)

        # Store life events in memory tracker
        if life_events:
            for ev in life_events:
                if isinstance(ev, dict):
                    memory.life_events.add_event(
                        event_type=ev.get("event_type", "other"),
                        description=ev.get("description", ""),
                        year=ev.get("year"),
                        impact=ev.get("impact", ""),
                        metadata=ev.get("metadata", {}),
                    )
                else:
                    memory.life_events.add_event(
                        event_type="other",
                        description=str(ev),
                        year=year,
                    )
        
        print(f"\\n=== Conversation {conversation_id} ===")
        print(f"Personality: {personality} | Dominant Trait: {self.conversation_context['dominant_trait'].capitalize()} | Income: {self.conversation_context['income_range']}")
        print(f"Outcome: {outcome} | Provider: {recommended_provider}")
        print(f"Vehicle: {vehicle_analysis['vehicle_category']} | Budget: ₹{user_budget:,}")
        
        # Save exact scores for output
        base_scores = self.big_five_profiles.get(personality, {
            "openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5,
            "agreeableness": 0.5, "neuroticism": 0.5
        })
        bf_scores = base_scores.copy()
        dom_trait = self.conversation_context['dominant_trait']
        if dom_trait in bf_scores:
            bf_scores[dom_trait] = min(1.0, bf_scores[dom_trait] + 0.3)

        # Initialize conversation structure
        conversation_data = {
            "conversation_id": conversation_id,
            "personality": personality,
            "income_range": self.conversation_context["income_range"],
            "dominant_trait": self.conversation_context["dominant_trait"],
            "big_five_scores": bf_scores,
            "vehicle_category": vehicle_analysis["vehicle_category"],
            "outcome": outcome,
            "recommended_provider": recommended_provider,
            "life_events": self.conversation_context.get("life_events", []),
            "turns": []
        }
        
        # Opening
        turn_no = 1
        opening = random.choice(self.user_openings)
        user_msg = f"{opening} [Image: {image_url}]"
        
        # Add opening turn
        opening_spectrum = IntentSpectrum.from_intent("I-1 Session Onset", personality)
        conversation_data["turns"].append({
            "turn_no": turn_no,
            "speaker": "User",
            "utterance": user_msg,
            "intent": "I-1 Session Onset",
            "intent_vector": opening_spectrum.to_dict()
        })
        
        self.conversation_history.append((conversation_id, turn_no, "User", user_msg))
        
        # Track decision readiness trajectory
        decision_readiness_trajectory = [opening_spectrum.vector["decision_readiness"]]
        cumulative_trust = [opening_spectrum.vector["trust_level"]]
        cumulative_price_sensitivity = [opening_spectrum.vector["price_sensitivity"]]
        
        # Track vectors for averaging
        all_intent_vectors = [opening_spectrum.vector]
        all_strategy_vectors = []
        
        # Track last user intent for judge
        last_user_intent_vector = opening_spectrum.vector

        turn_no += 1
        
        # Helper for price extraction (defined once, outside loop)
        def extract_price(text, base_val=base_premium):
            matches = re.findall(r'(?:₹|Rs\.?|INR)\s*([\d,]+)|([\d,]+)\s*(?:rs|inr)', text, re.IGNORECASE)
            candidates = []
            for m in matches:
                val_str = (m[0] or m[1]).replace(',', '')
                try:
                    val = int(val_str)
                    if 1000 <= val <= 300000:
                        if base_val and val > 4 * base_val:
                            continue
                        candidates.append(val)
                except ValueError:
                    continue
            return candidates[0] if candidates else None

        # Generate conversation - target ~15 turns
        max_turns = target_turns + 5  # Allow some flexibility
        
        while turn_no <= max_turns and not self.conversation_context.get('finalized'):
            self.conversation_context['turn_count'] = turn_no
            self.thinking_log = []
            
            # Agent turn
            if turn_no == 2:
                strategy, response, strategy_spectrum = self.generate_agent_response(
                    self.conversation_history[-1][3],
                    self.conversation_context,
                    self.conversation_history,
                    conversation_id,
                    turn_no,
                    memory, # Passing real MemoryManager
                    image_url=image_url,
                    recommended_provider=recommended_provider
                )
            else:
                strategy, response, strategy_spectrum = self.generate_agent_response(
                    self.conversation_history[-1][3],
                    self.conversation_context,
                    self.conversation_history,
                    conversation_id,
                    turn_no,
                    memory, # Passing real MemoryManager
                    recommended_provider=recommended_provider
                )
            
            # Add agent turn
            conversation_data["turns"].append({
                "turn_no": turn_no,
                "speaker": "Agent",
                "utterance": response,
                "strategy": strategy,
                "strategy_vector": strategy_spectrum.to_dict()
            })
            
            self.conversation_history.append((conversation_id, turn_no, "Agent", response))
            
            conversation_thinking.extend(self.thinking_log)
            print(f"Turn {turn_no}: Agent ({strategy})")

            # Human-in-the-loop verification checkpoints (3 per turn)
            expert_thinking = {
                t.get("agent_name"): t.get("thinking")
                for t in self.thinking_log
                if t.get("agent_name")
            }
            try:
                log_checkpoints(
                    conversation_id=conversation_id,
                    turn_no=turn_no,
                    user_msg=self.conversation_history[-2][3],
                    agent_msg=response,
                    expert_thinking=expert_thinking,
                )
            except Exception as e:
                print(f"      [Verification] Logging failed: {e}")
            
            # Capture memory snapshot (Using new structure)
            memory_snapshots.append({
                "conversation_id": conversation_id,
                "turn_no": turn_no,
                "speaker": "Agent",
                "memory_state": {
                    "latent_state": memory.latent_state.to_dict(),
                }
            })
            
            # Track strategy vector
            all_strategy_vectors.append(strategy_spectrum.vector)
            
            # Judge Evaluation & Graph Update
            # We evaluate the PAIR: (Last User Message) -> (Agent Response)
            # The 'last_user_intent_vector' corresponds to that Last User Message ("before this turn" or "start of this turn")
            if self.judge and len(self.conversation_history) >= 2:
                # User msg is at index -2 (since we just appended Agent msg at -1)
                user_msg_text = self.conversation_history[-2][3]
                
                try:
                    # Retrieve memory context again or reuse? We need what the agent SAW.
                    # We can re-retrieve or assume 'memory_context' variable inside generate_agent_response stays consistent 
                    # but we don't have access to local vars of that function here.
                    # We'll re-retrieve lightweight context or just pass empty if acceptable.
                    # Better: The judge needs to know if Agent used memory. 
                    # For now, we'll pass the memory object's current state which is close enough.
                    
                    judge_scores, judge_tokens = self.judge.evaluate_turn(
                        history=conversation_data["turns"][:-2], # Pass previous turns as history
                        user_msg=user_msg_text,
                        agent_msg=response,
                        strategy_vector=strategy_spectrum.vector,
                        intent_vector=last_user_intent_vector if last_user_intent_vector else {},
                        memory_context={} 
                    )
                    
                    # Update Strategy Graph
                    if len(self.strategy_history) >= 2:
                        prev_strat = self.strategy_history[-2]
                        curr_strat = self.strategy_history[-1]
                        
                        # Get adjustment from judge
                        adj = self.judge.suggest_adjustment(judge_scores)
                        if adj != 0:
                            print(f"      [Judge] {judge_scores.get('reasoning', 'No reasoning')} (Adj: {adj})")
                            self.strategy_graph.update_weight(prev_strat, curr_strat, adj)
                    
                    # Update Intent Graph based on user intent transitions
                    if len(self.intent_history) >= 2:
                        prev_intent = self.intent_history[-2]
                        curr_intent = self.intent_history[-1]
                        
                        # Use intent_accuracy from judge as signal for intent transition quality
                        intent_score = judge_scores.get('intent_accuracy', 0.7)
                        # Map score to adjustment: good intent read -> reinforce transition
                        if intent_score > 0.8:
                            intent_adj = 0.1
                        elif intent_score > 0.6:
                            intent_adj = 0.05
                        elif intent_score < 0.4:
                            intent_adj = -0.1
                        elif intent_score < 0.6:
                            intent_adj = -0.05
                        else:
                            intent_adj = 0.0
                        
                        if intent_adj != 0:
                            self.intent_graph.update_weight(prev_intent, curr_intent, intent_adj)
                    
                    # Persist global weights upon each graph update
                    self.intent_graph.save_global_weights("memory/global_intent_graph.json")
                    self.strategy_graph.save_global_weights("memory/global_strategy_graph.json")
                            
                except Exception as e:
                    print(f"      [Judge Error] {e}")
            
            # Track negotiation moves (Level 5)
            # Check for agent offer
            agent_price = extract_price(response, base_val=base_premium)
            if agent_price:
                # We need to know user request to complete the tuple.
                # We can store pending agent offer.
                self.conversation_context['last_agent_price'] = agent_price

            turn_no += 1
            
            if turn_no > max_turns:
                break
            
            # User turn
            intent, response, intent_spectrum = self.generate_user_utterance(
                self.conversation_history[-1][3],
                self.conversation_context,
                self.conversation_history,
                personality
            )
            
            # Add user turn
            conversation_data["turns"].append({
                "turn_no": turn_no,
                "speaker": "User",
                "utterance": response,
                "intent": intent,
                "intent_vector": intent_spectrum.to_dict()
            })
            
            # Track intent vector
            all_intent_vectors.append(intent_spectrum.vector)
            last_user_intent_vector = intent_spectrum.vector
            
            self.conversation_history.append((conversation_id, turn_no, "User", response))
            
            
            # 1. Decision Readiness
            # Use judge score for decision readiness if available, else fallback to spectrum
            if self.judge and 'judge_scores' in locals() and "decision_readiness" in judge_scores:
                current_readiness = judge_scores["decision_readiness"]
                print(f"      [Judge] Overriding Decision Readiness with LLM score: {current_readiness}")
            else:
                current_readiness = self._estimate_decision_readiness(response)
            
            prev_readiness = decision_readiness_trajectory[-1]
            delta_readiness = current_readiness - prev_readiness
            decision_readiness_trajectory.append(current_readiness)
            
            # 2. Track estimation data
            cumulative_trust.append(intent_spectrum.vector["trust_level"])
            cumulative_price_sensitivity.append(intent_spectrum.vector["price_sensitivity"])

            # 3. Strategy Outcome Memory Update (Level 2)
            # Update the strategy used by agent in previous turn
            window = decision_readiness_trajectory[-5:]
            m_bar = sum(window) / len(window) if window else None
            memory.strategy_memory.update(
                strategy=strategy,
                outcome_positive=(delta_readiness > -0.05), # Loose definition of positive
                delta_readiness=delta_readiness,
                m_bar=m_bar,
            )
            
            # 4. Agent Responsiveness Update (Level 3)
            memory.latent_state.update_responsiveness(strategy, delta_readiness)

            # 5. Negotiation Tracking (Level 5) - Enhanced
            user_price = extract_price(response, base_val=base_premium)
            
            # Independent logging
            if user_price:
                memory.concession_memory.log_user_request(user_price)
            
            # Paired logging if we have context
            if user_price and self.conversation_context.get('last_agent_price'):
                move = {
                    "user_request": user_price, 
                    "agent_offer": self.conversation_context['last_agent_price']
                }
                memory.current_session.add_negotiation_move(move)
                # We already logged individual moves, but can update graph state if needed
                # The independent logs handle the history now.

            # 6. Concession expectation (Level 3) - LLM estimate per turn
            negotiation_context = {
                "user_requests_history": memory.concession_memory.user_requests_history[-5:],
                "agent_offers_history": memory.concession_memory.agent_offers_history[-5:],
                "last_agent_price": self.conversation_context.get("last_agent_price"),
            }
            conc_est = self._estimate_concession_expectation(response, negotiation_context)
            if conc_est is not None:
                memory.latent_state.concession_expectation = conc_est
            
            print(f"Turn {turn_no}: User ({intent})")
            
            # Capture memory snapshot
            memory_snapshots.append({
                "conversation_id": conversation_id,
                "turn_no": turn_no,
                "speaker": "User",
                "memory_state": {
                    "latent_state": memory.latent_state.to_dict(),
                }
            })
            
            turn_no += 1
            
            # Check for finalization
            if intent in ["I-7 Renewal Commitment", "I-8 Offer Decline"]:
                self.conversation_context['finalized'] = True
                
                # Add final agent response
                if intent == "I-7 Renewal Commitment":
                    final_response = random.choice([
                        "Excellent choice! I'll send you the policy documents right away. Welcome to our family!",
                        "Perfect! Let me process this for you. You'll receive everything via email shortly.",
                        "Great decision! Your car is now in safe hands. Documents coming your way!"
                    ])
                else:
                    final_response = random.choice([
                        "I understand. Feel free to reach out if you change your mind. Drive safely!",
                        "No problem at all. We're here whenever you're ready. Best of luck!",
                        "That's okay. Take your time to decide. Our offer stands when you need us."
                    ])
                
                conversation_data["turns"].append({
                    "turn_no": turn_no,
                    "speaker": "Agent",
                    "utterance": final_response,
                    "strategy": "Default",
                    "strategy_vector": StrategySpectrum.from_strategy("Default").to_dict()
                })
                
                conversation_thinking.append({
                    "conversation_id": conversation_id,
                    "turn_no": turn_no,
                    "agent_name": "Synthesizer",
                    "thinking": f"Conversation concluded with {intent}. Providing closing statement."
                })
                
                print(f"Turn {turn_no}: Agent (Default) - Final closing")
                break
        
        # End Session and Update Memory (Level 3 & 4)
        estimates = {
            "trust": sum(cumulative_trust) / len(cumulative_trust),
            "price_sensitivity": sum(cumulative_price_sensitivity) / len(cumulative_price_sensitivity),
            "decision_readiness": decision_readiness_trajectory[-1]
        }
        
        # Calculate Average Vectors for Memory
        if all_intent_vectors:
            avg_intent = {}
            for k in all_intent_vectors[0].keys():
                avg_intent[k] = sum(v.get(k, 0) for v in all_intent_vectors) / len(all_intent_vectors)
            memory.current_session.avg_intent_vector = avg_intent

        if all_strategy_vectors:
            avg_strategy = {}
            for k in all_strategy_vectors[0].keys():
                avg_strategy[k] = sum(v.get(k, 0) for v in all_strategy_vectors) / len(all_strategy_vectors)
            memory.current_session.avg_strategy_vector = avg_strategy
            
        # Generate <80 word year summary at end of session
        full_transcript = "\\n".join([f"{t['speaker']}: {t['utterance']}" for t in conversation_data["turns"]])
        summary_prompt = f"Summarize the following customer service chat about motor insurance in under 80 words. Focus on the user's main concerns, chosen features, and final outcome.\\n\\n{full_transcript}"
        try:
            summary_text, __ = self.client.generate_text(summary_prompt, max_tokens=100)
            if len(summary_text.split()) > 80:
                summary_text = " ".join(summary_text.split()[:78]) + "..." # forcefully truncate if model disobeys
        except Exception:
            summary_text = "Summary unavailable due to generation error."
        
        memory.current_session.year_summary = summary_text
        conversation_data["year_summary"] = summary_text

        # Concession memory summary (LLM)
        conc_prompt = (
            "Summarize how the negotiation unfolded and which concessions worked best "
            "in 1-2 sentences.\n\n"
            f"Outcome: {outcome}\n"
            f"User requests: {memory.concession_memory.user_requests_history}\n"
            f"Agent offers: {memory.concession_memory.agent_offers_history}\n"
            f"Negotiation moves: {memory.current_session.negotiation_moves}\n"
        )
        try:
            conc_summary, __ = self.client.generate_text(conc_prompt, max_tokens=120)
            memory.concession_memory.summary = conc_summary.strip()
        except Exception:
            memory.concession_memory.summary = "Summary unavailable due to generation error."

        memory.end_session(outcome, estimates)
        
        # Calculate commitment (Level 3)
        memory.latent_state.calculate_commitment(decision_readiness_trajectory)
        
        # Save memory
        memory.save(f"memory/{conversation_id}.json")
        if persona_id:
             memory.save(f"memory/personas/{persona_id}.json")
             print(f"  Saved updated persona memory to memory/personas/{persona_id}.json")
        
        return conversation_data, conversation_thinking, memory_snapshots


if __name__ == "__main__":
    print("DialogueGenerator module loaded successfully")
