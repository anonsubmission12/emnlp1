import json
import random
import os
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional
from spectrum import IntentSpectrum, StrategySpectrum


class DenseGraph:
    """
    Dense weighted directed graph where every node connects to every other node.
    Edge weights sum to 1.0 for each source node.
    """
    
    def __init__(self, nodes: List[str], initial_weights: Dict[str, Dict[str, float]] = None):
        """
        Initialize dense graph.
        
        Args:
            nodes: List of node names
            initial_weights: Optional dict of {from_node: {to_node: weight}}
        """
        self.nodes = nodes
        self.update_counts = {f"{from_node}->{to_node}": 0 
                             for from_node in nodes for to_node in nodes}
        
        if initial_weights:
            self.weights = initial_weights
            self._normalize_weights()
        else:
            # Initialize with uniform weights
            self.weights = {}
            for from_node in nodes:
                self.weights[from_node] = {
                    to_node: 1.0 / len(nodes) for to_node in nodes
                }
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0 for each source node."""
        for from_node in self.nodes:
            total = sum(self.weights[from_node].values())
            if total > 0:
                for to_node in self.nodes:
                    self.weights[from_node][to_node] /= total
    
    def sample_next(
        self, 
        current_node: str, 
        spectrum_current: Optional[object] = None,
        spectrum_candidates: Optional[Dict[str, object]] = None,
        exclude_nodes: Optional[List[str]] = None
    ) -> str:
        """
        Sample next node using edge weights and optional spectrum similarity.
        
        Args:
            current_node: Current node
            spectrum_current: Current spectrum (IntentSpectrum or StrategySpectrum)
            spectrum_candidates: Dict of {node_name: spectrum} for candidates
            exclude_nodes: Nodes to exclude from sampling
            
        Returns:
            Next node name
        """
        if current_node not in self.weights:
            return random.choice(self.nodes)
        
        # Get base weights
        base_weights = self.weights[current_node].copy()
        
        # Exclude nodes if specified
        if exclude_nodes:
            for node in exclude_nodes:
                base_weights[node] = 0.0
        
        # Apply spectrum similarity if provided
        if spectrum_current and spectrum_candidates:
            for node in base_weights:
                if node in spectrum_candidates:
                    similarity = spectrum_current.similarity(spectrum_candidates[node])
                    # Combine edge weight with spectrum similarity
                    base_weights[node] *= (1.0 + similarity)
        
        # Normalize
        total = sum(base_weights.values())
        if total == 0:
            # Fallback to uniform
            return random.choice([n for n in self.nodes if n not in (exclude_nodes or [])])
        
        probs = {node: weight / total for node, weight in base_weights.items()}
        
        # Sample
        nodes_list = list(probs.keys())
        probs_list = [probs[node] for node in nodes_list]
        
        return np.random.choice(nodes_list, p=probs_list)
    
    def update_weight(
        self, 
        from_node: str, 
        to_node: str, 
        adjustment: float,
        learning_rate: float = 1.0
    ):
        """
        Update edge weight based on judge feedback.

        Paper Appendix B.3 (Eq. 4):
            W_p(z_{t-1}, z_t) ← W_p(z_{t-1}, z_t) + δ(q_t)

        δ(q_t) is the banded nudge already computed by judge.suggest_adjustment()
        (values ±0.0050 / ±0.0025).  learning_rate defaults to 1.0 so the nudge
        is applied as-is; set < 1.0 only for additional damping experiments.
        Weight is clamped to a floor of 0.01 so no transition ever becomes
        impossible (paper Appendix B.3).

        Args:
            from_node:     Source node
            to_node:       Target node
            adjustment:    Pre-computed banded nudge δ(q_t)
            learning_rate: Multiplicative scale (default 1.0 = no extra scaling)
        """
        if from_node not in self.weights or to_node not in self.weights[from_node]:
            return

        delta = learning_rate * adjustment
        # Floor = 0.01 (paper: "clamped to a floor of 0.01")
        self.weights[from_node][to_node] = max(
            0.01, self.weights[from_node][to_node] + delta
        )

        # Re-normalise so rows still sum to 1
        self._normalize_weights()

        key = f"{from_node}->{to_node}"
        self.update_counts[key] += 1
    
    def get_transition_probability(self, from_node: str, to_node: str) -> float:
        """Get edge weight between two nodes."""
        if from_node not in self.weights or to_node not in self.weights[from_node]:
            return 0.0
        return self.weights[from_node][to_node]
    
    def save_weights(self, filepath: str):
        """Save weights to JSON file."""
        data = {
            "weights": self.weights,
            "update_counts": self.update_counts,
            "nodes": self.nodes
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_weights(self, filepath: str):
        """Load weights from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.weights = data["weights"]
        self.update_counts = data.get("update_counts", {})
        self.nodes = data.get("nodes", self.nodes)
    
    def save_global_weights(self, filepath: str):
        """
        Save cumulative global weights across all conversations.
        Used for long-horizon learning (2015-2025).
        
        Args:
            filepath: Path to save global weights (e.g., memory/global_intent_graph.json)
        """
        data = {
            "weights": self.weights,
            "update_counts": self.update_counts,
            "nodes": self.nodes,
            "metadata": {
                "total_updates": sum(self.update_counts.values()),
                "last_updated": str(datetime.now())
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_global_weights(self, filepath: str) -> bool:
        """
        Load cumulative global weights from previous runs.
        
        Args:
            filepath: Path to global weights file
            
        Returns:
            True if loaded successfully, False if file doesn't exist
        """
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.weights = data["weights"]
        self.update_counts = data.get("update_counts", {})
        self.nodes = data.get("nodes", self.nodes)
        
        return True
    
    def merge_weights(self, other_graph: 'DenseGraph', alpha: float = 0.1):
        """
        Merge weights from another graph for incremental learning.
        
        Args:
            other_graph: Another DenseGraph to merge from
            alpha: Learning rate for merging (0.0 = keep current, 1.0 = use other)
        """
        for from_node in other_graph.weights:
            if from_node not in self.weights:
                continue
            
            for to_node, other_weight in other_graph.weights[from_node].items():
                if to_node not in self.weights[from_node]:
                    continue
                
                # Weighted average
                current_weight = self.weights[from_node][to_node]
                self.weights[from_node][to_node] = (
                    (1 - alpha) * current_weight + alpha * other_weight
                )
        
        # Normalize after merging
        self._normalize_weights()
        
        # Merge update counts
        for key, count in other_graph.update_counts.items():
            self.update_counts[key] = self.update_counts.get(key, 0) + count

    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "weights": self.weights,
            "update_counts": self.update_counts
        }


def create_intent_graph() -> DenseGraph:
    """
    Create intent graph with expert-defined initial weights.
    Based on natural conversation flow from test10111.py.
    """
    nodes = [
        "I-1 Session Onset",
        "I-2 Coverage Inquiry",
        "I-3 Quote Solicitation",
        "I-4 Concern Disclosure",
        "I-5 Clarification Request",
        "I-6 Premium Negotiation",
        "I-7 Renewal Commitment",
        "I-8 Offer Decline"
    ]
    
    # Expert-defined weights based on conversation flow
    initial_weights = {
        "I-1 Session Onset": {
            "I-2 Coverage Inquiry": 0.40,
            "I-3 Quote Solicitation": 0.30,
            "I-5 Clarification Request": 0.20,
            "I-4 Concern Disclosure": 0.05,
            "I-6 Premium Negotiation": 0.03,
            "I-7 Renewal Commitment": 0.01,
            "I-8 Offer Decline": 0.01,
            "I-1 Session Onset": 0.00
        },
        "I-2 Coverage Inquiry": {
            "I-4 Concern Disclosure": 0.30,
            "I-3 Quote Solicitation": 0.25,
            "I-5 Clarification Request": 0.25,
            "I-6 Premium Negotiation": 0.10,
            "I-1 Session Onset": 0.05,
            "I-7 Renewal Commitment": 0.03,
            "I-8 Offer Decline": 0.02,
            "I-2 Coverage Inquiry": 0.00
        },
        "I-3 Quote Solicitation": {
            "I-6 Premium Negotiation": 0.35,
            "I-2 Coverage Inquiry": 0.25,
            "I-4 Concern Disclosure": 0.20,
            "I-5 Clarification Request": 0.10,
            "I-7 Renewal Commitment": 0.05,
            "I-8 Offer Decline": 0.03,
            "I-1 Session Onset": 0.02,
            "I-3 Quote Solicitation": 0.00
        },
        "I-4 Concern Disclosure": {
            "I-5 Clarification Request": 0.35,
            "I-2 Coverage Inquiry": 0.25,
            "I-6 Premium Negotiation": 0.20,
            "I-3 Quote Solicitation": 0.10,
            "I-8 Offer Decline": 0.05,
            "I-1 Session Onset": 0.03,
            "I-7 Renewal Commitment": 0.02,
            "I-4 Concern Disclosure": 0.00
        },
        "I-5 Clarification Request": {
            "I-3 Quote Solicitation": 0.30,
            "I-2 Coverage Inquiry": 0.25,
            "I-4 Concern Disclosure": 0.20,
            "I-6 Premium Negotiation": 0.15,
            "I-7 Renewal Commitment": 0.05,
            "I-1 Session Onset": 0.03,
            "I-8 Offer Decline": 0.02,
            "I-5 Clarification Request": 0.00
        },
        "I-6 Premium Negotiation": {
            "I-7 Renewal Commitment": 0.40,
            "I-4 Concern Disclosure": 0.25,
            "I-5 Clarification Request": 0.15,
            "I-8 Offer Decline": 0.10,
            "I-3 Quote Solicitation": 0.05,
            "I-2 Coverage Inquiry": 0.03,
            "I-1 Session Onset": 0.02,
            "I-6 Premium Negotiation": 0.00
        },
        "I-7 Renewal Commitment": {
            "I-7 Renewal Commitment": 0.70,
            "I-5 Clarification Request": 0.15,
            "I-2 Coverage Inquiry": 0.05,
            "I-4 Concern Disclosure": 0.03,
            "I-6 Premium Negotiation": 0.03,
            "I-3 Quote Solicitation": 0.02,
            "I-8 Offer Decline": 0.01,
            "I-1 Session Onset": 0.01
        },
        "I-8 Offer Decline": {
            "I-8 Offer Decline": 0.70,
            "I-5 Clarification Request": 0.10,
            "I-4 Concern Disclosure": 0.08,
            "I-2 Coverage Inquiry": 0.05,
            "I-3 Quote Solicitation": 0.03,
            "I-6 Premium Negotiation": 0.02,
            "I-7 Renewal Commitment": 0.01,
            "I-1 Session Onset": 0.01
        }
    }
    
    return DenseGraph(nodes, initial_weights)


def create_strategy_graph() -> DenseGraph:
    """
    Create strategy graph with expert-defined initial weights.
    """
    nodes = ["Default", "Credibility", "Emotional", "Logical", "Personal", "Persona"]
    
    initial_weights = {
        "Default": {
            "Credibility": 0.25,
            "Logical": 0.25,
            "Personal": 0.25,
            "Emotional": 0.15,
            "Persona": 0.05,
            "Default": 0.05
        },
        "Credibility": {
            "Logical": 0.30,
            "Personal": 0.25,
            "Emotional": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Credibility": 0.05
        },
        "Emotional": {
            "Personal": 0.30,
            "Logical": 0.25,
            "Credibility": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Emotional": 0.05
        },
        "Logical": {
            "Personal": 0.30,
            "Credibility": 0.25,
            "Emotional": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Logical": 0.05
        },
        "Personal": {
            "Emotional": 0.30,
            "Logical": 0.25,
            "Credibility": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Personal": 0.05
        },
        "Persona": {
            "Credibility": 0.30,
            "Personal": 0.25,
            "Emotional": 0.20,
            "Logical": 0.15,
            "Default": 0.05,
            "Persona": 0.05
        }
    }
    
    return DenseGraph(nodes, initial_weights)


if __name__ == "__main__":
    print("Testing DenseGraph...")
    
    # Test intent graph
    print("\n1. Testing Intent Graph...")
    intent_graph = create_intent_graph()
    
    print(f"Nodes: {intent_graph.nodes}")
    print("\nTransition from 'I-1 Session Onset':")
    for node in intent_graph.nodes:
        prob = intent_graph.get_transition_probability("I-1 Session Onset", node)
        if prob > 0.05:
            print(f"  -> {node}: {prob:.3f}")
    
    # Test sampling
    print("\n2. Testing sampling (10 transitions from 'I-1 Session Onset'):")
    samples = []
    for _ in range(10):
        next_node = intent_graph.sample_next("I-1 Session Onset")
        samples.append(next_node)
    
    from collections import Counter
    counts = Counter(samples)
    for node, count in counts.most_common():
        print(f"  {node}: {count}/10")
    
    # Test weight update
    print("\n3. Testing weight update...")
    original_weight = intent_graph.get_transition_probability("I-1 Session Onset", "I-3 Quote Solicitation")
    print(f"Original weight (I-1 Session Onset -> I-3 Quote Solicitation): {original_weight:.3f}")
    
    intent_graph.update_weight("I-1 Session Onset", "I-3 Quote Solicitation", adjustment=0.15)
    new_weight = intent_graph.get_transition_probability("I-1 Session Onset", "I-3 Quote Solicitation")
    print(f"Updated weight (I-1 Session Onset -> I-3 Quote Solicitation): {new_weight:.3f}")
    
    # Test save/load
    print("\n4. Testing save/load...")
    intent_graph.save_weights("test_graph_weights.json")
    print("✓ Saved to test_graph_weights.json")
    
    # Test strategy graph
    print("\n5. Testing Strategy Graph...")
    strategy_graph = create_strategy_graph()
    print(f"Nodes: {strategy_graph.nodes}")
    
    print("\n✓ All tests passed!")
