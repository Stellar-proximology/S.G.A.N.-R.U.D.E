"""
Resonance Decider - Master decision engine for the YOU-N-I-VERSE system
Implements the Resonant Utility (RU) equation for least-friction, most-progression action selection
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class NodeType(Enum):
    """Five primary consciousness nodes"""
    MOVEMENT = "Movement"
    EVOLUTION = "Evolution"
    BEING = "Being"
    DESIGN = "Design"
    SPACE = "Space"


class ElementType(Enum):
    """Five elements"""
    FIRE = "Fire"
    WATER = "Water"
    EARTH = "Earth"
    AIR = "Air"
    AETHER = "Aether"
    VOID = "Void"


@dataclass
class ActionCandidate:
    """
    Single action candidate with all RU components.
    
    All scores should be in range [0, 1] for proper RU calculation.
    """
    label: str
    progress: float  # How much this action advances toward goal
    friction: float  # Resistance (time + cost + emotion + context switch)
    coherence: float  # Body-Mind-Heart alignment from Sentence System
    feasibility: float  # Can we actually do this now?
    risk: float  # Probability × severity of downside
    synergy: float  # Future compounding / option value
    
    # Semantic geometry (optional but recommended)
    dimension: Optional[int] = None  # 0-11 for 12D space
    shape: Optional[str] = None
    element: Optional[ElementType] = None
    operator: Optional[str] = None
    color_rgb: Optional[Tuple[float, float, float]] = None
    
    # Active node context
    active_node: Optional[NodeType] = None
    
    # Computed fields
    ru_score: Optional[float] = None
    ru_breakdown: Optional[Dict[str, float]] = None


class ResonanceDecider:
    """
    Core decision engine implementing Resonant Utility maximization.
    
    RU(a) = w_P·Progress(a) - w_F·Friction(a) + w_C·Coherence(a) 
            + w_R·Feasibility(a) - w_K·Risk(a) + w_S·Synergy(a)
    
    Integrates with:
    - Sentence System for Coherence(a)
    - Semantic GAN for action proposal + element/operator matching
    - LLM for P/F/R/K/S estimation
    - Consciousness math for dimensional scoring
    """
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 normalize_weights: bool = False,
                 consciousness_math=None,
                 semantic_engine=None):
        """
        Initialize the Resonance Decider.
        
        Args:
            weights: Custom weight parameters (P, F, C, R, K, S)
            normalize_weights: If True, normalize weights to sum to 1.0
            consciousness_math: Optional ConsciousnessMath instance
            semantic_engine: Optional SemanticEngine instance
        """
        # Default weights (unnormalized - as proposed in your spec)
        self.default_weights = {
            'P': 1.0,  # Progress
            'F': 0.8,  # Friction
            'C': 0.6,  # Coherence
            'R': 0.5,  # Feasibility (Realizability)
            'K': 0.7,  # Risk
            'S': 0.4   # Synergy
        }
        
        self.weights = weights if weights is not None else self.default_weights.copy()
        
        if normalize_weights:
            total = sum(abs(v) for v in self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Integration with existing system
        self.consciousness_math = consciousness_math
        self.semantic_engine = semantic_engine
        
        # Element compatibility matrix for friction penalties
        self.element_friction_matrix = self._build_element_friction_matrix()
        
        # Per-node weight modulation (optional advanced feature)
        self.node_weight_modulation = {
            NodeType.MOVEMENT: {'P': 1.2, 'F': 0.9, 'S': 1.1},  # Favor progress
            NodeType.EVOLUTION: {'C': 1.2, 'S': 1.3, 'K': 0.8},  # Favor coherence + synergy
            NodeType.BEING: {'C': 1.3, 'F': 0.7, 'K': 0.7},      # Favor coherence, reduce friction
            NodeType.DESIGN: {'R': 1.2, 'P': 1.1, 'F': 0.9},     # Favor feasibility + progress
            NodeType.SPACE: {'S': 1.3, 'K': 0.8, 'C': 1.1}       # Favor synergy + coherence
        }
    
    def _build_element_friction_matrix(self) -> Dict[Tuple[ElementType, ElementType], float]:
        """
        Build element compatibility matrix.
        Returns friction penalty multiplier (1.0 = no penalty, >1.0 = higher friction).
        """
        # Simplified - you can expand this based on your element philosophy
        matrix = {}
        elements = list(ElementType)
        
        # Default: all compatible (1.0)
        for e1 in elements:
            for e2 in elements:
                matrix[(e1, e2)] = 1.0
        
        # Define incompatibilities (increase friction)
        incompatible = [
            (ElementType.FIRE, ElementType.WATER),   # Classic opposition
            (ElementType.WATER, ElementType.FIRE),
            (ElementType.EARTH, ElementType.AIR),    # Grounded vs fluid
            (ElementType.AIR, ElementType.EARTH),
        ]
        
        for e1, e2 in incompatible:
            matrix[(e1, e2)] = 1.4  # 40% friction increase
        
        # Define synergies (reduce friction)
        synergies = [
            (ElementType.FIRE, ElementType.AIR),     # Fire feeds on air
            (ElementType.AIR, ElementType.FIRE),
            (ElementType.WATER, ElementType.EARTH),  # Water nourishes earth
            (ElementType.EARTH, ElementType.WATER),
            (ElementType.AETHER, ElementType.VOID),  # Transcendent pair
            (ElementType.VOID, ElementType.AETHER),
        ]
        
        for e1, e2 in synergies:
            matrix[(e1, e2)] = 0.8  # 20% friction reduction
        
        return matrix
    
    def calculate_resonant_utility(self, 
                                   action: ActionCandidate,
                                   current_field_element: Optional[ElementType] = None,
                                   active_node: Optional[NodeType] = None) -> float:
        """
        Calculate Resonant Utility score for a single action.
        
        Args:
            action: ActionCandidate with all components
            current_field_element: Current field's element for friction adjustment
            active_node: Active consciousness node for weight modulation
            
        Returns:
            RU score (can be negative)
        """
        # Get working weights (may be modulated by active node)
        w = self._get_modulated_weights(active_node)
        
        # Extract components
        P = self._clamp(action.progress)
        F = self._clamp(action.friction)
        C = self._clamp(action.coherence)
        R = self._clamp(action.feasibility)
        K = self._clamp(action.risk)
        S = self._clamp(action.synergy)
        
        # Apply element friction penalty if applicable
        if current_field_element and action.element:
            friction_penalty = self.element_friction_matrix.get(
                (current_field_element, action.element), 1.0
            )
            F = F * friction_penalty
        
        # Calculate RU
        ru = (
            w['P'] * P
            - w['F'] * F
            + w['C'] * C
            + w['R'] * R
            - w['K'] * K
            + w['S'] * S
        )
        
        # Store breakdown for analysis
        action.ru_score = ru
        action.ru_breakdown = {
            'progress_contribution': w['P'] * P,
            'friction_penalty': -w['F'] * F,
            'coherence_contribution': w['C'] * C,
            'feasibility_contribution': w['R'] * R,
            'risk_penalty': -w['K'] * K,
            'synergy_contribution': w['S'] * S,
            'element_friction_multiplier': friction_penalty if current_field_element and action.element else 1.0
        }
        
        return ru
    
    def pick_best_action(self, 
                        candidates: List[ActionCandidate],
                        current_field_element: Optional[ElementType] = None,
                        active_node: Optional[NodeType] = None,
                        top_k: int = 1) -> List[ActionCandidate]:
        """
        Select the best action(s) from candidates based on RU.
        
        Args:
            candidates: List of action candidates
            current_field_element: Current field element for friction adjustment
            active_node: Active node for weight modulation
            top_k: Number of top actions to return
            
        Returns:
            List of top-k actions sorted by RU (best first)
        """
        if not candidates:
            return []
        
        # Calculate RU for all candidates
        for action in candidates:
            self.calculate_resonant_utility(
                action, 
                current_field_element=current_field_element,
                active_node=active_node
            )
        
        # Sort by RU score (descending)
        sorted_candidates = sorted(
            candidates, 
            key=lambda a: a.ru_score if a.ru_score is not None else float('-inf'),
            reverse=True
        )
        
        return sorted_candidates[:top_k]
    
    def calculate_multi_node_ru(self,
                               action: ActionCandidate,
                               node_distribution: Dict[NodeType, float]) -> float:
        """
        Calculate weighted RU across multiple active nodes.
        
        Use when multiple nodes are active simultaneously (e.g., Movement + Evolution).
        
        Args:
            action: ActionCandidate
            node_distribution: Dict mapping NodeType -> activation weight (should sum to 1.0)
            
        Returns:
            Weighted RU score
        """
        total_ru = 0.0
        
        for node, weight in node_distribution.items():
            ru = self.calculate_resonant_utility(action, active_node=node)
            total_ru += ru * weight
        
        return total_ru
    
    def estimate_coherence_from_semantic(self,
                                        semantic_latent: Dict[str, Any]) -> float:
        """
        Estimate Coherence(a) from semantic latent via your Discriminator/GAN.
        
        This is where your GAN (D) plugs in to compute Body-Mind-Heart coherence
        from the semantic geometry.
        
        Args:
            semantic_latent: Dict with dimension, shape, element, operator, color
            
        Returns:
            Coherence score [0, 1]
        """
        if self.consciousness_math is None:
            # Fallback: basic estimation from element
            element = semantic_latent.get('element')
            if element == ElementType.EARTH:
                return 0.7  # Earth = grounded = moderate coherence
            elif element == ElementType.AETHER:
                return 0.85  # Aether = high coherence
            elif element == ElementType.FIRE:
                return 0.6  # Fire = dynamic but less stable
            return 0.5
        
        # Use existing consciousness math to compute coherence
        # In full implementation, you'd extract BMH from the latent
        # For now, approximate based on dimensional mapping
        dimension = semantic_latent.get('dimension', 0)
        
        # Map dimension to BMH (simplified - you'd use your full mapping)
        body = 0.5 + 0.3 * np.sin(dimension * np.pi / 12)
        mind = 0.5 + 0.3 * np.cos(dimension * np.pi / 12)
        heart = 0.5 + 0.2 * np.sin(dimension * np.pi / 6)
        
        coherence = self.consciousness_math.calculate_coherence(body, mind, heart)
        
        return coherence
    
    def _get_modulated_weights(self, active_node: Optional[NodeType]) -> Dict[str, float]:
        """
        Get weights modulated by active node.
        
        Args:
            active_node: Currently active consciousness node
            
        Returns:
            Modulated weight dict
        """
        if active_node is None or active_node not in self.node_weight_modulation:
            return self.weights
        
        modulated = self.weights.copy()
        node_mods = self.node_weight_modulation[active_node]
        
        for key in modulated:
            if key in node_mods:
                modulated[key] *= node_mods[key]
        
        return modulated
    
    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to valid range."""
        return max(min_val, min(max_val, value))
    
    def update_weights_from_feedback(self, 
                                    chosen_action: ActionCandidate,
                                    satisfaction_score: float):
        """
        Update weights based on user feedback (simple online learning).
        
        This implements inverse RL / bandit learning:
        If user picks low-RU actions but reports high satisfaction,
        adjust weights to match their preferences.
        
        Args:
            chosen_action: The action the user actually chose
            satisfaction_score: User satisfaction [0, 1]
        """
        # Simple gradient-like update
        learning_rate = 0.05
        
        if chosen_action.ru_breakdown is None:
            return
        
        # If satisfaction is high, increase weights of positive contributors
        # If satisfaction is low, decrease them
        delta = satisfaction_score - 0.5  # Centered at 0.5
        
        breakdown = chosen_action.ru_breakdown
        
        # Update weights based on contribution sign and satisfaction
        if breakdown['progress_contribution'] > 0:
            self.weights['P'] += learning_rate * delta
        
        if breakdown['coherence_contribution'] > 0:
            self.weights['C'] += learning_rate * delta
        
        if breakdown['synergy_contribution'] > 0:
            self.weights['S'] += learning_rate * delta
        
        # Ensure weights stay positive
        for key in self.weights:
            self.weights[key] = max(0.1, self.weights[key])
    
    def get_status(self) -> Dict[str, Any]:
        """Get decider status."""
        return {
            'weights': self.weights,
            'node_modulation_active': True,
            'element_friction_matrix_size': len(self.element_friction_matrix),
            'consciousness_math_connected': self.consciousness_math is not None,
            'semantic_engine_connected': self.semantic_engine is not None
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration for persistence."""
        return {
            'weights': self.weights,
            'node_weight_modulation': {
                k.value: v for k, v in self.node_weight_modulation.items()
            }
        }
    
    def import_config(self, config: Dict[str, Any]):
        """Import configuration from saved state."""
        if 'weights' in config:
            self.weights = config['weights']
        
        if 'node_weight_modulation' in config:
            self.node_weight_modulation = {
                NodeType(k): v for k, v in config['node_weight_modulation'].items()
            }


# ============================================================================
# Helper functions for quick usage
# ============================================================================

def create_action_from_dict(data: Dict[str, Any]) -> ActionCandidate:
    """
    Create ActionCandidate from dictionary.
    
    Expected keys: label, progress, friction, coherence, feasibility, risk, synergy
    Optional: dimension, shape, element, operator, color_rgb, active_node
    """
    element = None
    if 'element' in data and data['element'] is not None:
        if isinstance(data['element'], ElementType):
            element = data['element']
        elif isinstance(data['element'], str):
            element = ElementType[data['element'].upper()]
    
    active_node = None
    if 'active_node' in data and data['active_node'] is not None:
        if isinstance(data['active_node'], NodeType):
            active_node = data['active_node']
        elif isinstance(data['active_node'], str):
            active_node = NodeType[data['active_node'].upper()]
    
    return ActionCandidate(
        label=data['label'],
        progress=data['progress'],
        friction=data['friction'],
        coherence=data['coherence'],
        feasibility=data['feasibility'],
        risk=data['risk'],
        synergy=data['synergy'],
        dimension=data.get('dimension'),
        shape=data.get('shape'),
        element=element,
        operator=data.get('operator'),
        color_rgb=data.get('color_rgb'),
        active_node=active_node
    )


def quick_decide(candidates: List[Dict[str, Any]], 
                weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Quick decision function - returns best action.
    
    Args:
        candidates: List of dicts with action data
        weights: Optional custom weights
        
    Returns:
        Dict with 'best_action' and 'all_results'
    """
    decider = ResonanceDecider(weights=weights)
    
    action_candidates = [create_action_from_dict(c) for c in candidates]
    
    best = decider.pick_best_action(action_candidates, top_k=1)[0]
    
    return {
        'best_action': {
            'label': best.label,
            'ru_score': best.ru_score,
            'breakdown': best.ru_breakdown
        },
        'all_results': [
            {
                'label': a.label,
                'ru_score': a.ru_score,
                'breakdown': a.ru_breakdown
            }
            for a in sorted(action_candidates, key=lambda x: x.ru_score or 0, reverse=True)
        ]
    }
