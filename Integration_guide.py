"""
Resonance Decision Engine - Integration Guide & Examples

This guide shows how to integrate the RU engine with your existing YOU-N-I-VERSE system.
"""

# ============================================================================
# QUICK START - Three Ways to Use
# ============================================================================

"""
1. STANDALONE - Quick decisions without full consciousness system
2. INTEGRATED - Full integration with ConsciousnessMath + SemanticEngine
3. LLM-POWERED - Natural language input → automated RU estimation
"""


# ============================================================================
# Example 1: Standalone Usage (No Dependencies)
# ============================================================================

def example_standalone():
    """Use RU engine without consciousness system."""
    from resonance_decider import ResonanceDecider, ActionCandidate, ElementType, NodeType
    
    # Create decider with custom weights
    decider = ResonanceDecider(weights={
        'P': 1.2,  # Emphasize progress
        'F': 0.6,  # De-emphasize friction
        'C': 0.8,
        'R': 0.5,
        'K': 0.7,
        'S': 0.9   # Emphasize synergy
    })
    
    # Define action candidates manually
    candidates = [
        ActionCandidate(
            label="Launch on Product Hunt",
            progress=0.75,
            friction=0.40,
            coherence=0.82,
            feasibility=0.65,
            risk=0.35,
            synergy=0.85,
            element=ElementType.FIRE,
            active_node=NodeType.MOVEMENT
        ),
        ActionCandidate(
            label="Build email automation",
            progress=0.55,
            friction=0.25,
            coherence=0.90,
            feasibility=0.85,
            risk=0.15,
            synergy=0.75,
            element=ElementType.WATER,
            active_node=NodeType.EVOLUTION
        ),
        ActionCandidate(
            label="Hire first engineer",
            progress=0.65,
            friction=0.70,
            coherence=0.78,
            feasibility=0.45,
            risk=0.50,
            synergy=0.95,
            element=ElementType.EARTH,
            active_node=NodeType.DESIGN
        )
    ]
    
    # Pick best action
    best = decider.pick_best_action(
        candidates,
        current_field_element=ElementType.WATER,  # Current state
        active_node=NodeType.EVOLUTION,
        top_k=1
    )[0]
    
    print("=" * 60)
    print("STANDALONE EXAMPLE")
    print("=" * 60)
    print(f"\nBest Action: {best.label}")
    print(f"RU Score: {best.ru_score:.3f}")
    print(f"\nBreakdown:")
    for key, value in best.ru_breakdown.items():
        print(f"  {key}: {value:.3f}")
    print()


# ============================================================================
# Example 2: Integrated with Consciousness System
# ============================================================================

def example_integrated():
    """Full integration with ConsciousnessMath and SemanticEngine."""
    from resonance_decider import ResonanceDecider, ActionCandidate, ElementType, NodeType
    from consciousness_math import ConsciousnessMath
    from semantic_engine import SemanticEngine
    
    # Initialize consciousness system
    cm = ConsciousnessMath()
    se = SemanticEngine()
    
    # Create decider with consciousness integration
    decider = ResonanceDecider(
        consciousness_math=cm,
        semantic_engine=se
    )
    
    # Action descriptions in natural language
    action_texts = [
        "Focus on building the core chart decoder this week",
        "Reach out to 10 potential beta users via email",
        "Write comprehensive documentation for the API"
    ]
    
    # Extract BMH and compute coherence for each action
    candidates = []
    for text in action_texts:
        # Extract BMH scores
        bmh = se.extract_bmh_scores(text)
        
        # Calculate coherence
        coherence = cm.calculate_coherence(bmh['body'], bmh['mind'], bmh['heart'])
        
        # For this example, manually set other RU components
        # In production, you'd use LLM estimation (see Example 3)
        candidates.append(ActionCandidate(
            label=text[:50],
            progress=0.7,  # Would be estimated
            friction=0.3,  # Would be estimated
            coherence=coherence,  # Computed from consciousness system
            feasibility=0.8,  # Would be estimated
            risk=0.2,  # Would be estimated
            synergy=0.6  # Would be estimated
        ))
    
    # Decide
    best = decider.pick_best_action(candidates, top_k=1)[0]
    
    print("=" * 60)
    print("INTEGRATED EXAMPLE (with Consciousness System)")
    print("=" * 60)
    print(f"\nBest Action: {best.label}")
    print(f"RU Score: {best.ru_score:.3f}")
    print(f"Coherence (from BMH): {best.coherence:.3f}")
    print(f"\nBreakdown:")
    for key, value in best.ru_breakdown.items():
        print(f"  {key}: {value:.3f}")
    print()


# ============================================================================
# Example 3: LLM-Powered Decision (Natural Language → RU)
# ============================================================================

def example_llm_powered():
    """Use LLM to estimate all RU components from natural language."""
    from llm_prompts import format_master_prompt
    import json
    
    # Define decision context
    context = """
    Building a 9-body consciousness system. Current status:
    - Core math engine: Complete
    - Sentence tracker: Complete  
    - API: Partial
    - UI: Not started
    - Beta users: 2 interested
    - Runway: 4 months
    """
    
    goal = "Get to 10 beta users testing the MVP by end of next month"
    
    actions = [
        "Focus entirely on building the browser UI this week",
        "Do 20 cold outreach emails to potential beta users",
        "Polish the API documentation and create video tutorials"
    ]
    
    print("=" * 60)
    print("LLM-POWERED EXAMPLE")
    print("=" * 60)
    print("\nContext:", context.strip())
    print("\nGoal:", goal)
    print("\nActions to evaluate:")
    for i, action in enumerate(actions, 1):
        print(f"  {i}. {action}")
    
    # Generate prompts for each action
    print("\n" + "-" * 60)
    print("SAMPLE PROMPT (for first action):")
    print("-" * 60)
    
    sample_prompt = format_master_prompt(
        context=context,
        goal=goal,
        action_description=actions[0],
        current_element="Water",
        active_node="Evolution"
    )
    
    print(sample_prompt[:500] + "...")
    
    # In production, you'd:
    # 1. Send each prompt to Claude API
    # 2. Parse JSON responses to get P/F/C/R/K/S
    # 3. Create ActionCandidates with those scores
    # 4. Run decider.pick_best_action()
    
    print("\n" + "-" * 60)
    print("Next steps:")
    print("  1. Call Claude API with above prompt")
    print("  2. Parse JSON response")
    print("  3. Create ActionCandidate with scores")
    print("  4. Run RU calculation")
    print()


# ============================================================================
# Example 4: Element-Aware Friction Adjustment
# ============================================================================

def example_element_friction():
    """Demonstrate element-based friction penalties."""
    from resonance_decider import ResonanceDecider, ActionCandidate, ElementType
    
    decider = ResonanceDecider()
    
    # Same action, different elements
    base_action = ActionCandidate(
        label="Rapid product launch",
        progress=0.8,
        friction=0.4,  # Base friction
        coherence=0.7,
        feasibility=0.6,
        risk=0.3,
        synergy=0.7
    )
    
    # Test with different element combinations
    scenarios = [
        ("Fire action in Fire field", ElementType.FIRE, ElementType.FIRE),
        ("Fire action in Water field", ElementType.FIRE, ElementType.WATER),
        ("Fire action in Air field", ElementType.FIRE, ElementType.AIR),
        ("Fire action in Earth field", ElementType.FIRE, ElementType.EARTH),
    ]
    
    print("=" * 60)
    print("ELEMENT FRICTION EXAMPLE")
    print("=" * 60)
    print("\nBase action: Rapid product launch (Fire energy)")
    print("Base friction: 0.4")
    print()
    
    for scenario_name, action_element, field_element in scenarios:
        action = ActionCandidate(
            label=base_action.label,
            progress=base_action.progress,
            friction=base_action.friction,
            coherence=base_action.coherence,
            feasibility=base_action.feasibility,
            risk=base_action.risk,
            synergy=base_action.synergy,
            element=action_element
        )
        
        ru = decider.calculate_resonant_utility(
            action,
            current_field_element=field_element
        )
        
        friction_multiplier = action.ru_breakdown['element_friction_multiplier']
        effective_friction = base_action.friction * friction_multiplier
        
        print(f"{scenario_name}:")
        print(f"  Friction multiplier: {friction_multiplier:.2f}")
        print(f"  Effective friction: {effective_friction:.2f}")
        print(f"  RU score: {ru:.3f}")
        print()


# ============================================================================
# Example 5: Multi-Node Weighted Decision
# ============================================================================

def example_multi_node():
    """Decide with multiple active nodes (e.g., Movement + Evolution)."""
    from resonance_decider import ResonanceDecider, ActionCandidate, NodeType
    
    decider = ResonanceDecider()
    
    action = ActionCandidate(
        label="Launch beta program with 10 users",
        progress=0.75,
        friction=0.35,
        coherence=0.82,
        feasibility=0.70,
        risk=0.30,
        synergy=0.85
    )
    
    # User is in transition between Movement and Evolution
    node_distribution = {
        NodeType.MOVEMENT: 0.6,  # Still moving fast
        NodeType.EVOLUTION: 0.4  # Starting to mature
    }
    
    ru_multi = decider.calculate_multi_node_ru(action, node_distribution)
    
    # Compare to single-node
    ru_movement = decider.calculate_resonant_utility(action, active_node=NodeType.MOVEMENT)
    ru_evolution = decider.calculate_resonant_utility(action, active_node=NodeType.EVOLUTION)
    
    print("=" * 60)
    print("MULTI-NODE EXAMPLE")
    print("=" * 60)
    print(f"\nAction: {action.label}")
    print(f"\nNode distribution:")
    print(f"  Movement: 60%")
    print(f"  Evolution: 40%")
    print(f"\nRU Scores:")
    print(f"  Pure Movement: {ru_movement:.3f}")
    print(f"  Pure Evolution: {ru_evolution:.3f}")
    print(f"  Weighted (60/40): {ru_multi:.3f}")
    print()


# ============================================================================
# Example 6: Online Learning from User Feedback
# ============================================================================

def example_feedback_learning():
    """Demonstrate weight adaptation from user feedback."""
    from resonance_decider import ResonanceDecider, ActionCandidate
    
    decider = ResonanceDecider()
    
    print("=" * 60)
    print("FEEDBACK LEARNING EXAMPLE")
    print("=" * 60)
    print("\nInitial weights:")
    print(decider.weights)
    
    # User picks a high-synergy action
    chosen_action = ActionCandidate(
        label="Build reusable component library",
        progress=0.50,
        friction=0.30,
        coherence=0.75,
        feasibility=0.80,
        risk=0.20,
        synergy=0.95  # Very high synergy
    )
    
    # Calculate RU to populate breakdown
    decider.calculate_resonant_utility(chosen_action)
    
    # User reports high satisfaction
    print("\n→ User picks high-synergy action and reports 90% satisfaction")
    decider.update_weights_from_feedback(chosen_action, satisfaction_score=0.9)
    
    print("\nUpdated weights (synergy should increase):")
    print(decider.weights)
    print()


# ============================================================================
# Example 7: Real Business Decision (Full Pipeline)
# ============================================================================

def example_real_business():
    """
    Complete example: Real business decision with full pipeline.
    
    Scenario: Early-stage SaaS founder deciding next move
    """
    from resonance_decider import ResonanceDecider, ActionCandidate, ElementType, NodeType
    
    print("=" * 60)
    print("REAL BUSINESS DECISION")
    print("=" * 60)
    
    print("\nScenario:")
    print("  - Early-stage SaaS: consciousness coaching platform")
    print("  - Current MRR: $2,500")
    print("  - Team: 2 people")
    print("  - Runway: 6 months")
    print("  - Current focus: User acquisition")
    print("  - Current element: Water (flow, adaptation)")
    print("  - Active node: Evolution")
    
    # Define three strategic options
    candidates = [
        ActionCandidate(
            label="Launch paid ads ($2k/month budget)",
            progress=0.65,  # Could get users quickly
            friction=0.55,  # Costly, requires learning ads
            coherence=0.45,  # Feels forced, not natural
            feasibility=0.80,  # Can start immediately
            risk=0.60,  # Might not work, burns cash
            synergy=0.30,  # Doesn't build long-term assets
            element=ElementType.FIRE,  # Aggressive, quick
            active_node=NodeType.MOVEMENT
        ),
        ActionCandidate(
            label="Build content machine (blog + SEO)",
            progress=0.40,  # Slow initial progress
            friction=0.30,  # Time-consuming but natural
            coherence=0.85,  # Aligns with teaching/coaching
            feasibility=0.90,  # Can start writing today
            risk=0.20,  # Low financial risk
            synergy=0.90,  # Compounds forever, builds authority
            element=ElementType.EARTH,  # Grounded, patient
            active_node=NodeType.BEING
        ),
        ActionCandidate(
            label="Partner with 3 established coaches",
            progress=0.75,  # Could unlock their audiences
            friction=0.40,  # Relationship building takes time
            coherence=0.90,  # Perfect fit for mission
            feasibility=0.60,  # Need to find right partners
            risk=0.35,  # Might not find good matches
            synergy=0.85,  # Network effects, referrals
            element=ElementType.WATER,  # Flow, connection
            active_node=NodeType.EVOLUTION
        ),
    ]
    
    # Current state: Water field + Evolution node
    decider = ResonanceDecider()
    
    best = decider.pick_best_action(
        candidates,
        current_field_element=ElementType.WATER,
        active_node=NodeType.EVOLUTION,
        top_k=3
    )
    
    print("\n" + "=" * 60)
    print("RESULTS (ranked by RU)")
    print("=" * 60)
    
    for i, action in enumerate(best, 1):
        print(f"\n{i}. {action.label}")
        print(f"   RU Score: {action.ru_score:.3f}")
        print(f"   Element: {action.element.value}")
        print(f"   Breakdown:")
        for key, val in action.ru_breakdown.items():
            if 'contribution' in key or 'penalty' in key:
                print(f"     {key}: {val:+.3f}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    winner = best[0]
    print(f"\n→ {winner.label}")
    print(f"\nWhy: Element match ({winner.element.value} = current field),")
    print(f"     high coherence, strong synergy,")
    print(f"     aligns with Evolution node priorities.")
    print()


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        '1': ('Standalone', example_standalone),
        '2': ('Integrated', example_integrated),
        '3': ('LLM-Powered', example_llm_powered),
        '4': ('Element Friction', example_element_friction),
        '5': ('Multi-Node', example_multi_node),
        '6': ('Feedback Learning', example_feedback_learning),
        '7': ('Real Business', example_real_business),
    }
    
    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]
        if example_num in examples:
            name, func = examples[example_num]
            print(f"\nRunning Example {example_num}: {name}\n")
            func()
        else:
            print(f"Unknown example: {example_num}")
            print("Available: 1-7")
    else:
        # Run all examples
        print("\n" + "=" * 60)
        print("RUNNING ALL EXAMPLES")
        print("=" * 60 + "\n")
        
        for num, (name, func) in examples.items():
            try:
                func()
                print()
            except Exception as e:
                print(f"Example {num} ({name}) failed: {e}\n")
        
        print("=" * 60)
        print("To run individual example: python integration_guide.py <number>")
        print("=" * 60)
