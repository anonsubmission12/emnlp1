import numpy as np
from typing import Dict, List
import re


class IntentSpectrum:
    """
    5-dimensional intent spectrum for user intents.
    Dimensions: information_seeking, price_sensitivity, trust_level, 
                decision_readiness, emotional_valence
    """
    
    DIMENSIONS = [
        "information_seeking",
        "price_sensitivity", 
        "trust_level",
        "decision_readiness",
        "emotional_valence"
    ]
    
    def __init__(self, vector: Dict[str, float] = None):
        if vector is None:
            # Default neutral spectrum
            self.vector = {dim: 0.5 for dim in self.DIMENSIONS}
        else:
            self.vector = {dim: max(0.0, min(1.0, vector.get(dim, 0.5))) 
                          for dim in self.DIMENSIONS}
    
    @classmethod
    def from_intent(cls, intent: str, personality: str = None) -> 'IntentSpectrum':
        """
        Create spectrum from intent name and personality.
        """
        # Base vectors for each intent type
        intent_vectors = {
            "I-1 Session Onset": {
                "information_seeking": 0.8,
                "price_sensitivity": 0.5,
                "trust_level": 0.5,
                "decision_readiness": 0.2,
                "emotional_valence": 0.6
            },
            "I-2 Coverage Inquiry": {
                "information_seeking": 0.9,
                "price_sensitivity": 0.4,
                "trust_level": 0.5,
                "decision_readiness": 0.3,
                "emotional_valence": 0.5
            },
            "I-3 Quote Solicitation": {
                "information_seeking": 0.7,
                "price_sensitivity": 0.8,
                "trust_level": 0.5,
                "decision_readiness": 0.5,
                "emotional_valence": 0.5
            },
            "I-4 Concern Disclosure": {
                "information_seeking": 0.6,
                "price_sensitivity": 0.5,
                "trust_level": 0.3,
                "decision_readiness": 0.3,
                "emotional_valence": 0.3
            },
            "I-5 Clarification Request": {
                "information_seeking": 0.9,
                "price_sensitivity": 0.4,
                "trust_level": 0.4,
                "decision_readiness": 0.3,
                "emotional_valence": 0.5
            },
            "I-6 Premium Negotiation": {
                "information_seeking": 0.4,
                "price_sensitivity": 0.9,
                "trust_level": 0.4,
                "decision_readiness": 0.6,
                "emotional_valence": 0.4
            },
            "I-7 Renewal Commitment": {
                "information_seeking": 0.3,
                "price_sensitivity": 0.5,
                "trust_level": 0.8,
                "decision_readiness": 1.0,
                "emotional_valence": 0.8
            },
            "I-8 Offer Decline": {
                "information_seeking": 0.2,
                "price_sensitivity": 0.8,
                "trust_level": 0.3,
                "decision_readiness": 1.0,
                "emotional_valence": 0.3
            }
        }
        
        base_vector = intent_vectors.get(intent, {dim: 0.5 for dim in cls.DIMENSIONS})
        
        # Adjust based on personality
        if personality:
            adjustments = {
                "budget-conscious": {"price_sensitivity": 0.2, "decision_readiness": -0.1},
                "detail-oriented": {"information_seeking": 0.2, "decision_readiness": -0.1},
                "skeptical": {"trust_level": -0.2, "emotional_valence": -0.1},
                "enthusiastic": {"emotional_valence": 0.2, "decision_readiness": 0.1},
                "cautious": {"trust_level": -0.1, "decision_readiness": -0.2},
                "tech-savvy": {"information_seeking": 0.1},
                "first-time buyer": {"information_seeking": 0.2, "trust_level": -0.1},
                "experienced buyer": {"decision_readiness": 0.2, "information_seeking": -0.1}
            }
            
            if personality in adjustments:
                for dim, delta in adjustments[personality].items():
                    base_vector[dim] = max(0.0, min(1.0, base_vector[dim] + delta))
        
        return cls(base_vector)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.vector[dim] for dim in self.DIMENSIONS])
    
    def distance(self, other: 'IntentSpectrum') -> float:
        """Compute Euclidean distance to another spectrum."""
        return np.linalg.norm(self.to_vector() - other.to_vector())
    
    def similarity(self, other: 'IntentSpectrum') -> float:
        """Compute similarity (1 - normalized distance)."""
        max_distance = np.sqrt(len(self.DIMENSIONS))
        return 1.0 - (self.distance(other) / max_distance)
    
    def blend(self, other: 'IntentSpectrum', weight: float = 0.5) -> 'IntentSpectrum':
        """Blend with another spectrum."""
        blended = {
            dim: weight * self.vector[dim] + (1 - weight) * other.vector[dim]
            for dim in self.DIMENSIONS
        }
        return IntentSpectrum(blended)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.vector.copy()


class StrategySpectrum:
    """
    4-dimensional strategy spectrum for agent strategies.
    Dimensions: credibility, emotional_appeal, logical_argument,
                personalization
    """
    
    DIMENSIONS = [
        "credibility",
        "emotional_appeal",
        "logical_argument",
        "personalization"
    ]
    
    def __init__(self, vector: Dict[str, float] = None):
        if vector is None:
            # Default neutral spectrum
            self.vector = {dim: 0.5 for dim in self.DIMENSIONS}
        else:
            self.vector = {dim: max(0.0, min(1.0, vector.get(dim, 0.5))) 
                          for dim in self.DIMENSIONS}
    
    @classmethod
    def from_strategy(cls, strategy: str) -> 'StrategySpectrum':
        """
        Create spectrum from strategy name.
        """
        strategy_vectors = {
            "Default": {
                "credibility": 0.5,
                "emotional_appeal": 0.4,
                "logical_argument": 0.5,
                "personalization": 0.5
            },
            "Credibility": {
                "credibility": 0.9,
                "emotional_appeal": 0.3,
                "logical_argument": 0.7,
                "personalization": 0.4
            },
            "Emotional": {
                "credibility": 0.4,
                "emotional_appeal": 0.9,
                "logical_argument": 0.3,
                "personalization": 0.7
            },
            "Logical": {
                "credibility": 0.6,
                "emotional_appeal": 0.2,
                "logical_argument": 0.9,
                "personalization": 0.4
            },
            "Personal": {
                "credibility": 0.5,
                "emotional_appeal": 0.6,
                "logical_argument": 0.5,
                "personalization": 0.9
            },
            "Persona": {
                "credibility": 0.7,
                "emotional_appeal": 0.5,
                "logical_argument": 0.6,
                "personalization": 0.8
            }
        }
        
        vector = strategy_vectors.get(strategy, {dim: 0.5 for dim in cls.DIMENSIONS})
        return cls(vector)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.vector[dim] for dim in self.DIMENSIONS])
    
    def distance(self, other: 'StrategySpectrum') -> float:
        """Compute Euclidean distance to another spectrum."""
        return np.linalg.norm(self.to_vector() - other.to_vector())
    
    def similarity(self, other: 'StrategySpectrum') -> float:
        """Compute similarity (1 - normalized distance)."""
        max_distance = np.sqrt(len(self.DIMENSIONS))
        return 1.0 - (self.distance(other) / max_distance)
    
    def blend(self, other: 'StrategySpectrum', weight: float = 0.5) -> 'StrategySpectrum':
        """Blend with another spectrum."""
        blended = {
            dim: weight * self.vector[dim] + (1 - weight) * other.vector[dim]
            for dim in self.DIMENSIONS
        }
        return StrategySpectrum(blended)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.vector.copy()


if __name__ == "__main__":
    print("Testing Spectrum Classes...")
    
    # Test IntentSpectrum
    print("\n1. Testing IntentSpectrum...")
    intent1 = IntentSpectrum.from_intent("I-3 Quote Solicitation", "budget-conscious")
    intent2 = IntentSpectrum.from_intent("I-7 Renewal Commitment", "enthusiastic")
    
    print(f"I-3 Quote Solicitation (budget-conscious): {intent1.to_dict()}")
    print(f"I-7 Renewal Commitment (enthusiastic): {intent2.to_dict()}")
    print(f"Distance: {intent1.distance(intent2):.3f}")
    print(f"Similarity: {intent1.similarity(intent2):.3f}")
    
    blended = intent1.blend(intent2, weight=0.7)
    print(f"Blended (70/30): {blended.to_dict()}")
    
    # Test StrategySpectrum
    print("\n2. Testing StrategySpectrum...")
    strat1 = StrategySpectrum.from_strategy("Credibility")
    strat2 = StrategySpectrum.from_strategy("Emotional")
    
    print(f"Credibility: {strat1.to_dict()}")
    print(f"Emotional: {strat2.to_dict()}")
    print(f"Distance: {strat1.distance(strat2):.3f}")
    print(f"Similarity: {strat1.similarity(strat2):.3f}")
    
    print("\n✓ All tests passed!")
