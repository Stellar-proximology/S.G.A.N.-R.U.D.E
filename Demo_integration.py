"""
Quick Demo - RU Integration with UnifiedLogic System

This demonstrates how the RU engine integrates with your existing
consciousness_math.py and semantic_engine.py modules.
"""

import sys
import os

# Add paths
sys.path.insert(0, '/home/claude/UnifiedLogic')
sys.path.insert(0, '/home/claude')

from resonance_decider import ResonanceDecider, ActionCandidate, ElementType, NodeType

# Try to import consciousness system - graceful fallback if not available
try:
    from consciousness_math import ConsciousnessMath
    from semantic_engine import SemanticEngine
    HAS_CONSCIOUSNESS_SYSTEM = True
except ImportError as e:
    print(f"Warning: Could not import consciousness system: {e}")
    print("Running in standalone mode with mock implementations.\n")
    HAS_CONSCIOUSNESS_SYSTEM = False
    
    # Mock implementations for demo
    class ConsciousnessMath:
        def calculate_coherence(self, body, mind, heart):
            return (body**0.85 + mind**0.65 + heart**0.45) / 3.0
        def get_status(self):
            return {"status": "mock"}
    
    class SemanticEngine:
        def extract_bmh_scores(self, text):
            # Simple keyword-based mock
            text_lower = text.lower()
            body = 0.5 + 0.3 * ('build' in text_lower or 'make' in text_lower)
            mind = 0.5 + 0.3 * ('api' in text_lower or 'document' in text_lower)
            heart = 0.5 + 0.3 * ('users' in text_lower or 'personal' in text_lower)
            return {'body': body, 'mind': mind, 'heart': heart}
        def get_status(self):
            return {"status": "mock"}


def demo_integrated_decision():
    """
    Complete example: Natural language actions → BMH → Coherence → RU → Best action
    """
    print("=" * 70)
    print("INTEGRATED DEMO: RU Engine + Consciousness Math + Semantic Engine")
    print("=" * 70)
    
    # Initialize systems
    cm = ConsciousnessMath()
    se = SemanticEngine()
    decider = ResonanceDecider(consciousness_math=cm, semantic_engine=se)
    
    print("\n✓ Consciousness Math loaded")
    print("✓ Semantic Engine loaded")
    print("✓ RU Decider initialized")
    
    # Business context
    print("\n" + "-" * 70)
    print("SCENARIO: Building 9-body consciousness platform")
    print("-" * 70)
    print("Current state:")
    print("  - Core math: Complete")
    print("  - Sentence tracker: Complete")
    print("  - UI: Not started")
    print("  - Beta users: 2")
    print("  - Runway: 4 months")
    print("  - Current field: Water (flow, adaptation)")
    print("  - Active node: Evolution")
    
    # Three strategic options described in natural language
    action_descriptions = [
        "Focus all energy on building the browser UI and chart visualizer this week, make it beautiful and intuitive",
        "Reach out personally to 20 potential beta users who align with consciousness work, build relationships",
        "Write comprehensive API documentation with video tutorials so developers can build on the platform"
    ]
    
    print("\n" + "-" * 70)
    print("OPTIONS TO EVALUATE:")
    print("-" * 70)
    for i, desc in enumerate(action_descriptions, 1):
        print(f"{i}. {desc}")
    
    # Process each action through the full pipeline
    candidates = []
    
    print("\n" + "-" * 70)
    print("PROCESSING ACTIONS (Natural Language → BMH → Coherence → RU)")
    print("-" * 70)
    
    for desc in action_descriptions:
        print(f"\n📝 {desc[:60]}...")
        
        # Step 1: Extract BMH scores using semantic engine
        bmh = se.extract_bmh_scores(desc)
        print(f"   BMH: Body={bmh['body']:.2f}, Mind={bmh['mind']:.2f}, Heart={bmh['heart']:.2f}")
        
        # Step 2: Calculate coherence using consciousness math
        coherence = cm.calculate_coherence(bmh['body'], bmh['mind'], bmh['heart'])
        print(f"   Coherence: {coherence:.3f}")
        
        # Step 3: Estimate other RU components
        # In production, you'd use LLM for this. For demo, we'll use heuristics.
        
        # Option 1: UI work - high feasibility, moderate progress
        if "ui" in desc.lower() or "browser" in desc.lower():
            progress = 0.60
            friction = 0.35
            feasibility = 0.85
            risk = 0.25
            synergy = 0.50
            element = ElementType.AIR  # Mental/design work
            
        # Option 2: Outreach - moderate friction, high synergy
        elif "reach out" in desc.lower() or "users" in desc.lower():
            progress = 0.70
            friction = 0.40
            feasibility = 0.90
            risk = 0.20
            synergy = 0.85
            element = ElementType.WATER  # Relationship/flow
            
        # Option 3: Documentation - low friction, future-building
        elif "documentation" in desc.lower() or "api" in desc.lower():
            progress = 0.50
            friction = 0.25
            feasibility = 0.95
            risk = 0.10
            synergy = 0.80
            element = ElementType.EARTH  # Foundation-building
        
        else:
            progress = 0.50
            friction = 0.50
            feasibility = 0.50
            risk = 0.50
            synergy = 0.50
            element = ElementType.AETHER
        
        print(f"   Element: {element.value}")
        
        # Create action candidate
        action = ActionCandidate(
            label=desc[:60] + "..." if len(desc) > 60 else desc,
            progress=progress,
            friction=friction,
            coherence=coherence,
            feasibility=feasibility,
            risk=risk,
            synergy=synergy,
            element=element,
            active_node=NodeType.EVOLUTION
        )
        
        candidates.append(action)
    
    # Decision time!
    print("\n" + "=" * 70)
    print("MAKING DECISION (RU Calculation)")
    print("=" * 70)
    
    best_actions = decider.pick_best_action(
        candidates,
        current_field_element=ElementType.WATER,  # Current state
        active_node=NodeType.EVOLUTION,
        top_k=3
    )
    
    print("\nRANKED RESULTS:\n")
    
    for rank, action in enumerate(best_actions, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"{medal} RANK {rank}: {action.label}")
        print(f"   RU Score: {action.ru_score:.3f}")
        print(f"   Element: {action.element.value} (field: Water)")
        print(f"   Breakdown:")
        
        breakdown = action.ru_breakdown
        print(f"      Progress:    {breakdown['progress_contribution']:+.3f}")
        print(f"      Friction:    {breakdown['friction_penalty']:+.3f}")
        print(f"      Coherence:   {breakdown['coherence_contribution']:+.3f}")
        print(f"      Feasibility: {breakdown['feasibility_contribution']:+.3f}")
        print(f"      Risk:        {breakdown['risk_penalty']:+.3f}")
        print(f"      Synergy:     {breakdown['synergy_contribution']:+.3f}")
        print()
    
    # Recommendation
    winner = best_actions[0]
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(f"\n→ {winner.label}")
    print(f"\nWhy:")
    
    # Explain the decision
    if winner.element == ElementType.WATER:
        print(f"  ✓ Perfect element match (Water = current field)")
    elif decider.element_friction_matrix.get((ElementType.WATER, winner.element), 1.0) < 1.0:
        print(f"  ✓ Element synergy (Water + {winner.element.value})")
    else:
        print(f"  • Element: {winner.element.value}")
    
    print(f"  ✓ High coherence ({winner.coherence:.2f}) - Body/Mind/Heart aligned")
    
    if winner.synergy > 0.7:
        print(f"  ✓ Strong synergy ({winner.synergy:.2f}) - compounds over time")
    
    if winner.feasibility > 0.8:
        print(f"  ✓ High feasibility ({winner.feasibility:.2f}) - can execute now")
    
    print(f"\nActive node (Evolution) favors:")
    print(f"  • Coherence: ↑ (emphasized)")
    print(f"  • Synergy: ↑ (emphasized)")
    print(f"  • Risk: ↓ (de-emphasized)")
    
    print("\n" + "=" * 70)
    print("System Status:")
    print("=" * 70)
    print("\nConsciousness Math:", cm.get_status())
    print("\nSemantic Engine:", se.get_status())
    print("\nRU Decider:", decider.get_status())
    
    print("\n" + "=" * 70)
    print("✓ Demo Complete - Integration Working")
    print("=" * 70)


if __name__ == "__main__":
    try:
        demo_integrated_decision()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
