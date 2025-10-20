"""
Unified Resonance Engine with S-GAN Integration
Merges: RU decision engine + S-GAN generator/discriminator + elemental vector algebra

This bridges:
- Waveform substrate (ontology) → Semantic latents (S-GAN) → Actions (RU) → Reality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# ELEMENTAL VECTOR ALGEBRA (Your Friend's Contribution)
# ============================================================================

class ElementType(Enum):
    """5 Elements mapped to consciousness dimensions"""
    EARTH = 0    # Design dimension - structure, manifestation
    WATER = 1    # Evolution dimension - flow, integration
    AIR = 2      # Space dimension - influence, communication
    FIRE = 3     # Movement dimension - drive, action, transformation
    AETHER = 4   # Being dimension - unity, transcendence


# Compatibility Matrix (0-1 scale: 1=harmonious, 0=destructive)
# Based on traditional element theory + your dimensional mappings
ELEMENT_COMPATIBILITY_MATRIX = torch.tensor([
    # E    W    A    F    Ae
    [1.0, 0.8, 0.7, 0.5, 1.0],  # Earth: nourishes w/ water, eroded by air, melted by fire
    [0.8, 1.0, 0.6, 0.2, 1.0],  # Water: dampens fire, evaporates in air
    [0.7, 0.6, 1.0, 0.9, 1.0],  # Air: shapes earth, fans fire
    [0.5, 0.2, 0.9, 1.0, 1.0],  # Fire: melts earth, extinguished by water, amplified by air
    [1.0, 1.0, 1.0, 1.0, 1.0]   # Aether: harmonizes all (unity principle)
])


def elemental_mismatch_penalty(action_element: ElementType, 
                               state_element: ElementType) -> float:
    """
    Compute friction penalty from elemental mismatch.
    Returns 0-1, where 0=perfect match, 1=maximum friction.
    """
    compatibility = ELEMENT_COMPATIBILITY_MATRIX[
        action_element.value, 
        state_element.value
    ].item()
    
    return 1.0 - compatibility


def elemental_operator(elem1: ElementType, elem2: ElementType, 
                      operation: str = 'blend') -> Tuple[float, str]:
    """
    Elemental algebra: combine elements with operators.
    
    Operations:
    - 'blend' (+): Merge energies (e.g., Fire + Earth = Magma)
    - 'amplify' (*): Strengthen (e.g., Air * Fire = Wildfire)
    - 'dampen' (-): Reduce (e.g., Water - Fire = Steam/Extinguish)
    
    Returns:
        (resonance_score, symbolic_result)
    """
    compat = ELEMENT_COMPATIBILITY_MATRIX[elem1.value, elem2.value].item()
    
    if operation == 'blend':
        # Addition: average compatibility as resonance
        resonance = compat
        result = f"{elem1.name}+{elem2.name}"
        
    elif operation == 'amplify':
        # Multiplication: amplify if compatible, destabilize if not
        resonance = compat ** 2  # Square emphasizes harmony/disharmony
        result = f"{elem1.name}×{elem2.name}"
        
    elif operation == 'dampen':
        # Subtraction: reduce opposing element
        resonance = abs(compat - 0.5) * 2  # Peak at 0.5 (balanced), low at extremes
        result = f"{elem1.name}−{elem2.name}"
        
    else:
        resonance = compat
        result = f"{elem1.name}{operation}{elem2.name}"
    
    return resonance, result


# ============================================================================
# S-GAN ARCHITECTURE (Your Friend's Generator/Discriminator)
# ============================================================================

class SemanticGenerator(nn.Module):
    """
    Generator: Noise → Semantic latent vector
    Proposes action candidates as 5D vectors (one per dimension)
    """
    def __init__(self, latent_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate semantic latent from noise."""
        return self.model(z)
    
    def generate_action_latent(self, batch_size: int = 1) -> torch.Tensor:
        """Generate action latents from random noise."""
        z = torch.randn(batch_size, self.latent_dim)
        return self.forward(z)


class SemanticDiscriminator(nn.Module):
    """
    Discriminator: Semantic latent → Coherence score
    Estimates Body-Mind-Heart alignment from semantic geometry
    """
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output coherence in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate coherence from semantic latent."""
        return self.model(x)
    
    def estimate_coherence(self, latent: torch.Tensor) -> float:
        """Get coherence score as float."""
        with torch.no_grad():
            return self.forward(latent).item()


# ============================================================================
# SEMANTIC LATENT TO ACTION MAPPING
# ============================================================================

def latent_to_element(latent_vector: torch.Tensor) -> ElementType:
    """
    Map 5D semantic latent to element.
    Uses argmax of latent components (after normalization).
    """
    # Normalize to positive values
    normalized = (latent_vector + 1) / 2  # From [-1,1] to [0,1]
    element_idx = torch.argmax(normalized).item()
    return ElementType(element_idx)


def latent_to_action_properties(latent_vector: torch.Tensor,
                                current_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Extract action properties from semantic latent.
    
    Args:
        latent_vector: 5D semantic vector from generator
        current_state: Current state vector for friction estimation
        
    Returns:
        Dict with element, friction, dimension, etc.
    """
    latent_np = latent_vector.detach().cpu().numpy()
    
    # Element from dominant dimension
    element = latent_to_element(latent_vector)
    
    # Dimension (1-12 in your system, but we'll use 0-4 for 5 nodes)
    # Map latent[0] from [-1,1] to dimension index
    dimension = int((latent_np[0] + 1) * 2.5)  # Maps to 0-4
    
    # Estimate friction from state mismatch if state provided
    friction = 0.5  # Default
    if current_state is not None:
        # Cosine similarity as alignment measure
        similarity = F.cosine_similarity(
            latent_vector.unsqueeze(0), 
            current_state.unsqueeze(0),
            dim=1
        ).item()
        # Convert similarity [-1,1] to friction [0,1]
        # High similarity = low friction
        friction = (1 - similarity) / 2
    
    return {
        'element': element,
        'element_idx': element.value,
        'dimension': dimension,
        'latent_vector': latent_np.tolist(),
        'friction_base': friction
    }


# ============================================================================
# UNIFIED RESONANT UTILITY ENGINE
# ============================================================================

@dataclass
class UnifiedActionCandidate:
    """
    Action candidate with both RU components and semantic latent.
    """
    label: str
    
    # RU Components (0-1)
    progress: float
    friction: float
    coherence: float
    feasibility: float
    risk: float
    synergy: float
    
    # Semantic latent
    latent: Dict[str, Any]
    
    # Computed
    ru_score: Optional[float] = None
    ru_breakdown: Optional[Dict[str, float]] = None


class UnifiedResonanceEngine:
    """
    Complete resonance system:
    Generator → Discriminator → RU → Best Action
    
    Integrates:
    - S-GAN for action proposal + coherence estimation
    - Elemental algebra for friction adjustment
    - RU formula for multi-objective optimization
    - Node modulation for consciousness state awareness
    """
    
    def __init__(self,
                 generator: Optional[SemanticGenerator] = None,
                 discriminator: Optional[SemanticDiscriminator] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize unified engine.
        
        Args:
            generator: S-GAN generator (creates if None)
            discriminator: S-GAN discriminator (creates if None)
            weights: RU weight parameters
        """
        # S-GAN components
        self.generator = generator if generator else SemanticGenerator()
        self.discriminator = discriminator if discriminator else SemanticDiscriminator()
        
        # RU weights (default from your spec)
        self.weights = weights if weights else {
            'P': 1.0,  # Progress
            'F': 0.8,  # Friction
            'C': 0.6,  # Coherence
            'R': 0.5,  # Feasibility
            'K': 0.7,  # Risk
            'S': 0.4   # Synergy
        }
        
        # Elemental friction penalty weight
        self.lambda_m = 0.3  # Mismatch penalty multiplier
        
        # Current system state
        self.current_state_vector: Optional[torch.Tensor] = None
        self.current_state_element: ElementType = ElementType.EARTH
        
        # Node activation weights (for 5-node modulation)
        self.node_pi = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform by default
    
    def set_current_state(self, 
                         state_vector: Optional[torch.Tensor] = None,
                         state_element: ElementType = ElementType.EARTH):
        """Update current consciousness state."""
        self.current_state_vector = state_vector
        self.current_state_element = state_element
    
    def set_node_distribution(self, node_pi: List[float]):
        """
        Set node activation distribution.
        
        Args:
            node_pi: List of 5 weights for [Earth/Design, Water/Evolution, 
                     Air/Space, Fire/Movement, Aether/Being]
        """
        assert len(node_pi) == 5, "Must provide 5 node weights"
        assert abs(sum(node_pi) - 1.0) < 0.01, "Node weights must sum to 1.0"
        self.node_pi = node_pi
    
    def generate_action_candidates(self, 
                                   num_candidates: int = 5,
                                   temperature: float = 1.0) -> List[torch.Tensor]:
        """
        Generate action candidate latents using generator.
        
        Args:
            num_candidates: Number of candidates to generate
            temperature: Sampling temperature (>1 = more diverse)
            
        Returns:
            List of semantic latent vectors
        """
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_candidates, self.generator.latent_dim) * temperature
            latents = self.generator(z)
        
        return [latents[i] for i in range(num_candidates)]
    
    def calculate_resonant_utility(self,
                                   action: UnifiedActionCandidate,
                                   use_elemental_friction: bool = True) -> float:
        """
        Calculate RU with elemental friction adjustment.
        
        Args:
            action: Action candidate
            use_elemental_friction: Whether to apply element mismatch penalty
            
        Returns:
            RU score
        """
        P = action.progress
        F = action.friction
        C = action.coherence
        R = action.feasibility
        K = action.risk
        S = action.synergy
        
        # Apply elemental friction penalty
        if use_elemental_friction and 'element' in action.latent:
            action_element = action.latent['element']
            mismatch = elemental_mismatch_penalty(action_element, self.current_state_element)
            F = F + self.lambda_m * mismatch
            F = min(F, 1.0)  # Cap at 1.0
        
        # RU formula
        w = self.weights
        ru = (
            w['P'] * P
            - w['F'] * F
            + w['C'] * C
            + w['R'] * R
            - w['K'] * K
            + w['S'] * S
        )
        
        # Store breakdown
        action.ru_score = ru
        action.ru_breakdown = {
            'progress_contribution': w['P'] * P,
            'friction_penalty': -w['F'] * F,
            'coherence_contribution': w['C'] * C,
            'feasibility_contribution': w['R'] * R,
            'risk_penalty': -w['K'] * K,
            'synergy_contribution': w['S'] * S,
            'elemental_friction': self.lambda_m * elemental_mismatch_penalty(
                action.latent['element'], self.current_state_element
            ) if use_elemental_friction and 'element' in action.latent else 0.0
        }
        
        return ru
    
    def pick_best_action(self,
                        candidates: List[UnifiedActionCandidate],
                        top_k: int = 1) -> List[UnifiedActionCandidate]:
        """
        Select best action(s) by RU score.
        
        Args:
            candidates: List of action candidates
            top_k: Number of top actions to return
            
        Returns:
            Top-k actions sorted by RU (best first)
        """
        # Calculate RU for all
        for action in candidates:
            self.calculate_resonant_utility(action)
        
        # Sort by RU
        sorted_candidates = sorted(
            candidates,
            key=lambda a: a.ru_score if a.ru_score else float('-inf'),
            reverse=True
        )
        
        return sorted_candidates[:top_k]
    
    def generate_and_select(self,
                           num_candidates: int = 10,
                           top_k: int = 1,
                           use_llm_for_estimates: bool = False) -> List[UnifiedActionCandidate]:
        """
        Full pipeline: Generate → Estimate → Select
        
        Args:
            num_candidates: How many to generate
            top_k: How many to return
            use_llm_for_estimates: If True, would call LLM for P/F/R/K/S
                                   (Currently uses heuristics)
            
        Returns:
            Top-k best actions
        """
        # Step 1: Generate latents
        latents = self.generate_action_candidates(num_candidates)
        
        # Step 2: Create action candidates
        candidates = []
        
        for i, latent in enumerate(latents):
            # Extract properties from latent
            props = latent_to_action_properties(latent, self.current_state_vector)
            
            # Estimate coherence from discriminator
            coherence = self.discriminator.estimate_coherence(latent.unsqueeze(0))
            
            # Heuristic estimates for other components
            # In production, use LLM with prompt templates
            friction = props['friction_base']
            progress = 0.5 + 0.3 * (latent[0].item() + 1) / 2  # Map latent to progress
            feasibility = 0.7  # Default assumption
            risk = 0.3 - 0.2 * coherence  # High coherence = low risk
            synergy = 0.5 + 0.3 * abs(latent[4].item())  # Aether dimension → synergy
            
            action = UnifiedActionCandidate(
                label=f"Generated Action {i+1}",
                progress=np.clip(progress, 0, 1),
                friction=np.clip(friction, 0, 1),
                coherence=coherence,
                feasibility=np.clip(feasibility, 0, 1),
                risk=np.clip(risk, 0, 1),
                synergy=np.clip(synergy, 0, 1),
                latent=props
            )
            
            candidates.append(action)
        
        # Step 3: Select best
        return self.pick_best_action(candidates, top_k=top_k)
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'generator_ready': self.generator is not None,
            'discriminator_ready': self.discriminator is not None,
            'current_state_element': self.current_state_element.name,
            'node_distribution': self.node_pi,
            'weights': self.weights,
            'elemental_mismatch_penalty': self.lambda_m
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def visualize_element_interactions():
    """Print elemental compatibility matrix for reference."""
    print("Elemental Compatibility Matrix")
    print("=" * 50)
    print("       Earth  Water   Air   Fire  Aether")
    print("-" * 50)
    
    elements = [e.name for e in ElementType]
    for i, elem in enumerate(elements):
        row = ELEMENT_COMPATIBILITY_MATRIX[i].tolist()
        row_str = "  ".join([f"{x:.1f}" for x in row])
        print(f"{elem:7s}  {row_str}")
    print()


def test_unified_engine():
    """Quick test of unified engine."""
    print("=" * 60)
    print("UNIFIED RESONANCE ENGINE TEST")
    print("=" * 60)
    
    # Initialize
    engine = UnifiedResonanceEngine()
    
    # Set state: currently in Water/Evolution mode
    engine.set_current_state(
        state_vector=torch.randn(5),
        state_element=ElementType.WATER
    )
    
    # Set node distribution (heavy on Evolution)
    engine.set_node_distribution([0.1, 0.4, 0.2, 0.2, 0.1])
    
    print("\nEngine Status:")
    print(engine.get_status())
    
    # Generate and select
    print("\n" + "-" * 60)
    print("GENERATING ACTION CANDIDATES")
    print("-" * 60)
    
    best_actions = engine.generate_and_select(num_candidates=5, top_k=3)
    
    print("\nTop 3 Actions:\n")
    for rank, action in enumerate(best_actions, 1):
        print(f"{rank}. {action.label}")
        print(f"   Element: {action.latent['element'].name}")
        print(f"   RU Score: {action.ru_score:.4f}")
        print(f"   Coherence: {action.coherence:.3f}")
        print(f"   Progress: {action.progress:.3f}")
        print(f"   Friction: {action.friction:.3f}")
        print(f"   Elemental Friction: {action.ru_breakdown['elemental_friction']:.3f}")
        print()
    
    print("=" * 60)
    print("✓ Test Complete")
    print("=" * 60)


if __name__ == '__main__':
    visualize_element_interactions()
    test_unified_engine()
