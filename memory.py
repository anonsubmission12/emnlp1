import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# ==========================================
# PRESERVED EXISTING CLASS
# ==========================================

class LifeEventsTracker:
    """
    Tracks significant life events that impact insurance needs and persona evolution.
    Events: vehicle_purchase, family_change, relocation, career_change, financial_change
    """
    
    EVENT_TYPES = [
        "vehicle_purchase",
        "family_change",
        "relocation",
        "career_change",
        "financial_change",
        "accident_history",
        "policy_change"
    ]
    
    def __init__(self):
        self.events = []
    
    def add_event(
        self,
        event_type: str,
        description: str,
        year: Optional[int] = None,
        impact: str = "",
        metadata: Optional[Dict] = None
    ):
        """
        Add a life event.
        
        Args:
            event_type: Type of event (from EVENT_TYPES)
            description: Description of the event
            year: Year the event occurred
            impact: Impact on insurance needs
            metadata: Additional event metadata
        """
        if event_type not in self.EVENT_TYPES:
            event_type = "other"
        
        event = {
            "event_type": event_type,
            "description": description,
            "year": year,
            "impact": impact,
            "metadata": metadata or {}
        }
        
        self.events.append(event)
        return len(self.events) - 1
    
    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """Get all events of a specific type."""
        return [e for e in self.events if e["event_type"] == event_type]
    
    def get_events_by_year(self, year: int) -> List[Dict]:
        """Get all events from a specific year."""
        return [e for e in self.events if e.get("year") == year]
    
    def get_recent_events(self, n: int = 3) -> List[Dict]:
        """Get n most recent events."""
        return self.events[-n:] if len(self.events) >= n else self.events
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {"events": self.events}
    
    def from_dict(self, data: Dict):
        """Load from dictionary."""
        self.events = data.get("events", [])


# ==========================================
# NEW MEMORY ARCHITECTURE (LEVELS 1-4)
# ==========================================

class SessionMemory:
    """
    Level 1: Session Memory - Episodic memory of specific interactions.
    Stores raw interaction summaries, vectors, and outcomes.
    """
    def __init__(self, persona_id: str, year: int, persona_profile: Optional[Dict] = None):
        self.persona_id = persona_id
        self.year = year

        self.persona_profile = persona_profile or {}
        
        # Vectors
        self.avg_intent_vector = None
        self.avg_strategy_vector = None
        
        # Outcomes
        self.final_outcome = None  # "Reject", "Deal", etc.
        
        # Estimates (End of Session)
        self.trust_estimate = 0.5
        self.price_sensitivity_estimate = 0.5
        self.decision_readiness_end = 0.0
        self.year_summary = ""
        
        # Negotiation Log
        self.negotiation_moves = []  # List of {"user_request": -3000, "agent_offer": ...}

    def add_negotiation_move(self, move: Dict):
        """Add a negotiation move to the log."""
        self.negotiation_moves.append(move)

    def to_dict(self) -> Dict:
        return {
            "persona_id": self.persona_id,
            "year": self.year,
            "persona_profile": self.persona_profile,
            "avg_intent_vector": self.avg_intent_vector,
            "avg_strategy_vector": self.avg_strategy_vector,
            "final_outcome": self.final_outcome,
            "trust_estimate": self.trust_estimate,
            "price_sensitivity_estimate": self.price_sensitivity_estimate,
            "decision_readiness_end": self.decision_readiness_end,
            "year_summary": self.year_summary,
            "negotiation_moves": self.negotiation_moves
        }

    def from_dict(self, data: Dict):
        self.persona_id = data.get("persona_id")
        self.year = data.get("year")
        self.persona_profile = data.get("persona_profile", {})
        self.avg_intent_vector = data.get("avg_intent_vector")
        self.avg_strategy_vector = data.get("avg_strategy_vector")
        self.final_outcome = data.get("final_outcome")
        self.trust_estimate = data.get("trust_estimate", 0.5)
        self.price_sensitivity_estimate = data.get("price_sensitivity_estimate", 0.5)
        self.decision_readiness_end = data.get("decision_readiness_end", 0.0)
        self.year_summary = data.get("year_summary", "")
        self.negotiation_moves = data.get("negotiation_moves", [])


class StrategyOutcomeMemory:
    """
    Level 2: Strategy Outcome Memory - Tracks effectiveness of persuasion strategies.
    Strategies: Default, Credibility, Emotional, Logical, Personal, Persona
    """
    STRATEGY_TYPES = [
        "Default",
        "Credibility",
        "Emotional",
        "Logical",
        "Personal",
        "Persona",
    ]

    def __init__(self):
        # Dictionary to store metrics for each strategy
        # Structure: { "Logical": { "success_count": 0, "total_uses": 0, "total_delta_readiness": 0.0 } }
        self.stats = {st: {"success_count": 0, "total_uses": 0, "total_delta_readiness": 0.0} for st in self.STRATEGY_TYPES}

    def update(
        self,
        strategy: str,
        outcome_positive: bool,
        delta_readiness: float,
        m_bar: Optional[float] = None,
    ):
        """
        Update stats for a specific strategy application.
        """
        if strategy not in self.stats:
            # Track previously unseen strategies to avoid silent drops
            self.stats[strategy] = {
                "success_count": 0,
                "total_uses": 0,
                "total_delta_readiness": 0.0,
                "weight": 1.0,
            }

        # Optional banded update (paper Appendix C.2)
        if m_bar is not None:
            alpha = 0.05
            if m_bar > 0.8:
                delta_t = +0.10
            elif m_bar > 0.6:
                delta_t = +0.05
            elif m_bar >= 0.4:
                delta_t = -0.05
            else:
                delta_t = -0.10
            self.stats[strategy]["weight"] = max(
                0.0, self.stats[strategy].get("weight", 1.0) + alpha * delta_t
            )

        self.stats[strategy]["total_uses"] += 1
        if outcome_positive:
            self.stats[strategy]["success_count"] += 1
        self.stats[strategy]["total_delta_readiness"] += delta_readiness

    def get_effectiveness(self, strategy: str) -> Dict:
        """
        Get calculated effectiveness metrics for a strategy.
        Returns: { "success_rate": float, "avg_delta_decision_readiness": float }
        """
        if strategy not in self.stats or self.stats[strategy]["total_uses"] == 0:
            return {"success_rate": 0.0, "avg_delta_decision_readiness": 0.0}
        
        s = self.stats[strategy]
        return {
            "success_rate": s["success_count"] / s["total_uses"],
            "avg_delta_decision_readiness": s["total_delta_readiness"] / s["total_uses"]
        }

    def to_dict(self) -> Dict:
        return {"stats": self.stats}

    def from_dict(self, data: Dict):
        self.stats = data.get("stats", self.stats)


class LatentPersuasionState:
    """
    Level 3: Latent Persuasion State - Controllable memory summarizing persuadability.
    Normalized 0.0 to 1.0.
    """
    def __init__(self):
        self.trust = 0.5
        self.price_sensitivity = 0.5
        self.decision_readiness = 0.0 # Added missing attribute
        self.commitment = 0.0
        
        # Responsiveness to strategies
        self.logical_resp = 0.5
        self.emotional_resp = 0.5
        self.credibility_resp = 0.5
        
        self.concession_expectation = 0.5

        # Rolling responsiveness history (not persisted)
        self._resp_history = {
            "Logical": [],
            "Emotional": [],
            "Credibility": [],
        }

    def calculate_commitment(self, readiness_trajectory: List[float]):
        """
        Logic: Derive commitment from trajectory of decision_readiness.
        Mapping: If moves consistently positive, high commitment.
        """
        if not readiness_trajectory:
            self.commitment = 0.0
            return

        # Paper Eq. 8: commitment is the mean of per-turn readiness scores.
        mean_readiness = sum(readiness_trajectory) / len(readiness_trajectory)
        self.commitment = max(0.0, min(1.0, mean_readiness))

    def update_responsiveness(self, strategy_type: str, delta_readiness: float, current_weight: float = 0.1):
        """
        Update responsiveness to a strategy.
        Formula 1 (Immediate): Implied from delta.
        Formula 2 (Historical): Moving average update.
        """
        # Map strategy string to attribute
        attr_map = {
            "Logical": "logical_resp",
            "Emotional": "emotional_resp",
            "Credibility": "credibility_resp"
        }
        
        if strategy_type not in attr_map:
            return

        attr = attr_map[strategy_type]
        current_val = getattr(self, attr)
        
        # Formula: S_new = S_old + alpha * (Observation - S_old)
        # Observation is normalized delta? 
        # Assuming delta_readiness is roughly -0.2 to +0.2. 
        # We need to map it to 0-1 responsiveness scale?
        # Let's assume responsiveness is "potential to be swayed". 
        # If delta is high, responsiveness is high.
        
        # Normalize delta roughly: 0.2 delta -> 1.0 responsiveness signal?
        signal = max(0.0, min(1.0, (delta_readiness * 5) + 0.5)) # Scaling heuristic
        
        hist = self._resp_history.get(strategy_type, [])
        hist.append(signal)
        if len(hist) > 5:
            hist = hist[-5:]
        self._resp_history[strategy_type] = hist

        # 5-turn moving average (paper Eq. 9-10)
        new_val = sum(hist) / len(hist)
        setattr(self, attr, max(0.0, min(1.0, new_val)))

    def to_dict(self) -> Dict:
        return {
            "trust": self.trust,
            "price_sensitivity": self.price_sensitivity,
            "decision_readiness": self.decision_readiness,
            "commitment": self.commitment,
            "logical_resp": self.logical_resp,
            "emotional_resp": self.emotional_resp,
            "credibility_resp": self.credibility_resp,
            "concession_expectation": self.concession_expectation
        }

    def from_dict(self, data: Dict):
        self.trust = data.get("trust", 0.5)
        self.price_sensitivity = data.get("price_sensitivity", 0.5)
        self.decision_readiness = data.get("decision_readiness", 0.0)
        self.commitment = data.get("commitment", 0.0)
        self.logical_resp = data.get("logical_resp", 0.5)
        self.emotional_resp = data.get("emotional_resp", 0.5)
        self.credibility_resp = data.get("credibility_resp", 0.5)
        self.concession_expectation = data.get("concession_expectation", 0.5)


class PersonaDriftModel:
    """
    Level 4: Persona Drift Model - Updates personality traits over time.
    Formula: S_{t+1} = gamma * S_t + (1 - gamma) * O_t
    """
    def __init__(self, gamma: float = 0.8):
        self.gamma = gamma  # Stability factor (0.0 to 1.0)

    def update_state(self, current_state: LatentPersuasionState, new_observations: Dict):
        """
        Update the latent state based on new session observations.
        new_observations: Dict of { 'trust': 0.8, 'price_sensitivity': 0.3, ... }
        """
        for attr, observed_val in new_observations.items():
            if hasattr(current_state, attr):
                current_val = getattr(current_state, attr)
                # Formula: S_{t+1} = gamma * S_t + (1 - gamma) * O_t
                updated_val = (self.gamma * current_val) + ((1 - self.gamma) * observed_val)
                setattr(current_state, attr, updated_val)


class ConcessionMemory:
    """
    Level 5: Concession Memory - Tracks financial/negotiation boundaries.
    """
    def __init__(self):
        self.negotiation_graph = {
            "initial_price": 0,
            "user_request": 0,
            "agent_offer": 0,
            "user_final_request": 0
        }
        self.user_requests_history = []
        self.agent_offers_history = []
        self.summary = ""

    def log_negotiation(self, initial: float, user_req: float, agent_off: float, user_final: float):
        self.negotiation_graph = {
            "initial_price": initial,
            "user_request": user_req,
            "agent_offer": agent_off,
            "user_final_request": user_final
        }
        # Only append if non-zero/not None to avoid cluttering history with zeros
        if user_req: self.user_requests_history.append(user_req)
        if agent_off: self.agent_offers_history.append(agent_off)

    def log_user_request(self, price: float):
        if price:
            self.user_requests_history.append(price)
            self.negotiation_graph["user_request"] = price

    def log_agent_offer(self, price: float):
        if price:
            self.agent_offers_history.append(price)
            self.negotiation_graph["agent_offer"] = price

    def to_dict(self) -> Dict:
        return {
            "negotiation_graph": self.negotiation_graph,
            "user_requests_history": self.user_requests_history,
            "agent_offers_history": self.agent_offers_history,
            "summary": self.summary,
        }

    def from_dict(self, data: Dict):
        self.negotiation_graph = data.get("negotiation_graph", {})
        self.user_requests_history = data.get("user_requests_history", [])
        self.agent_offers_history = data.get("agent_offers_history", [])
        self.summary = data.get("summary", "")


class MemoryManager:
    """
    Orchestrator for the new Memory Architecture.
    """
    def __init__(self, persona_id: str = "P_Unknown"):
        self.persona_id = persona_id
        
        # Modules
        self.life_events = LifeEventsTracker()
        self.session_memory = [] # List of SessionMemory objects
        self.strategy_memory = StrategyOutcomeMemory()
        self.latent_state = LatentPersuasionState()
        # Paper Appendix C.1: gamma = 0.8 for trust/price EMA
        self.drift_model = PersonaDriftModel(gamma=0.8)
        self.concession_memory = ConcessionMemory()
        
        # Current active session
        self.current_session = None

    def start_new_session(self, year: int, persona_profile: Optional[Dict] = None):
        self.current_session = SessionMemory(self.persona_id, year, persona_profile=persona_profile)
        self.session_memory.append(self.current_session)

    def retrieve_context(self, query: str, session_id: str) -> Dict:
        """
        Retrieve relevant context for the agent.
        Returns Latent State and Recent Session Summaries.
        """
        # 1. Background from Latent State
        background = {
            "trust_level": self.latent_state.trust,
            "price_sensitivity": self.latent_state.price_sensitivity,
            "decision_readiness": self.latent_state.decision_readiness,
            "commitment_level": self.latent_state.commitment,
            "emotional_responsiveness": self.latent_state.emotional_resp,
            "logical_responsiveness": self.latent_state.logical_resp,
            "credibility_responsiveness": self.latent_state.credibility_resp
        }

        # 2. Recent Situations (Summaries of last 3 sessions)
        # Filter out current session if it's in the list
        recent_sessions = [s for s in self.session_memory if s != self.current_session][-3:]
        situations_text = ""
        for sess in recent_sessions:
            if sess.year_summary:
                situations_text += f"Year {sess.year} Summary: {sess.year_summary}. "
            else:
                situations_text += f"Year {sess.year}: Outcome {sess.final_outcome}. "
            if sess.negotiation_moves:
                situations_text += f"Negotiation moves: {len(sess.negotiation_moves)}. "
        
        return {
            "background": background,
            "situations": situations_text
        }

    def end_session(self, final_outcome: str, estimates: Dict):
        """
        Finalize session, update latent state with drift.
        estimates: { 'trust_estimate': 0.X, ... }
        """
        if not self.current_session:
            return

        # 1. Update session details
        self.current_session.final_outcome = final_outcome
        self.current_session.trust_estimate = estimates.get('trust', 0.5)
        self.current_session.price_sensitivity_estimate = estimates.get('price_sensitivity', 0.5)
        self.current_session.decision_readiness_end = estimates.get('decision_readiness', 0.0)

        # 2. Update Latent State using Drift Model
        observations = {
            'trust': self.current_session.trust_estimate,
            'price_sensitivity': self.current_session.price_sensitivity_estimate,
            'decision_readiness': self.current_session.decision_readiness_end,
        }
        self.drift_model.update_state(self.latent_state, observations)
        
        # 3. Calculate Commitment (based on readiness trajectory if we tracked it)
        # Note: Trajectory tracking needs to happen during the session.
        # self.latent_state.calculate_commitment(...)

    def save(self, filepath: str):
        data = {
            "persona_id": self.persona_id,
            "life_events": self.life_events.to_dict(),
            "sessions": [s.to_dict() for s in self.session_memory],
            "strategy_memory": self.strategy_memory.to_dict(),
            "latent_state": self.latent_state.to_dict(),
            "concession_memory": self.concession_memory.to_dict()
        }
        
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.persona_id = data.get("persona_id", self.persona_id)
        
        if "life_events" in data:
            self.life_events.from_dict(data["life_events"])
            
        if "sessions" in data:
            self.session_memory = []
            for s_data in data["sessions"]:
                sess = SessionMemory(s_data.get("persona_id"), s_data.get("year"))
                sess.from_dict(s_data)
                self.session_memory.append(sess)
                
        if "strategy_memory" in data:
            self.strategy_memory.from_dict(data["strategy_memory"])
            
        if "latent_state" in data:
            self.latent_state.from_dict(data["latent_state"])
            
        if "concession_memory" in data:
            self.concession_memory.from_dict(data["concession_memory"])
            
        return True


if __name__ == "__main__":
    print("Testing New Memory Architecture...")
    
    manager = MemoryManager("P_001")
    manager.start_new_session(2015)
    
    # Simulate updates
    manager.strategy_memory.update("Logical", True, 0.15)
    manager.concession_memory.log_negotiation(25000, -5000, -2000, -3000)
    
    estimates = {"trust": 0.7, "price_sensitivity": 0.4, "decision_readiness": 0.8}
    manager.end_session("Deal", estimates)
    
    print(f"Latent State Trust: {manager.latent_state.trust}")
    print(f"Strategy Effectiveness: {manager.strategy_memory.get_effectiveness('Logical')}")
    
    manager.save("test_new_memory.json")
    print("Saved test memory.")
