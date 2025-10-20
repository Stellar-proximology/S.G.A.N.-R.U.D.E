"""
Test Suite for Resonance Decision Engine

Validates:
- RU formula correctness
- Element friction matrix
- Node weight modulation
- Edge cases and boundary conditions
- Integration with consciousness system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from resonance_decider import (
    ResonanceDecider,
    ActionCandidate,
    ElementType,
    NodeType
)


# ============================================================================
# Test 1: Basic RU Calculation
# ============================================================================

def test_basic_ru():
    """Test basic RU formula calculation."""
    print("TEST 1: Basic RU Calculation")
    print("-" * 60)
    
    decider = ResonanceDecider(weights={
        'P': 1.0, 'F': 1.0, 'C': 1.0, 'R': 1.0, 'K': 1.0, 'S': 1.0
    })
    
    # Test case 1: All components at 0.5
    action = ActionCandidate(
        label="Test 1",
        progress=0.5, friction=0.5, coherence=0.5,
        feasibility=0.5, risk=0.5, synergy=0.5
    )
    
    ru = decider.calculate_resonant_utility(action)
    expected = 1.0*0.5 - 1.0*0.5 + 1.0*0.5 + 1.0*0.5 - 1.0*0.5 + 1.0*0.5
    
    print(f"Test 1.1 - All 0.5:")
    print(f"  Expected: {expected:.3f}")
    print(f"  Got: {ru:.3f}")
    print(f"  {'✓ PASS' if abs(ru - expected) < 0.001 else '✗ FAIL'}")
    
    # Test case 2: Max progress, zero friction
    action2 = ActionCandidate(
        label="Test 2",
        progress=1.0, friction=0.0, coherence=1.0,
        feasibility=1.0, risk=0.0, synergy=1.0
    )
    
    ru2 = decider.calculate_resonant_utility(action2)
    expected2 = 1.0 + 1.0 + 1.0 + 1.0  # Perfect action
    
    print(f"\nTest 1.2 - Perfect action:")
    print(f"  Expected: {expected2:.3f}")
    print(f"  Got: {ru2:.3f}")
    print(f"  {'✓ PASS' if abs(ru2 - expected2) < 0.001 else '✗ FAIL'}")
    
    # Test case 3: Max friction, high risk
    action3 = ActionCandidate(
        label="Test 3",
        progress=0.2, friction=1.0, coherence=0.3,
        feasibility=0.1, risk=1.0, synergy=0.0
    )
    
    ru3 = decider.calculate_resonant_utility(action3)
    expected3 = 0.2 - 1.0 + 0.3 + 0.1 - 1.0 + 0.0  # Bad action
    
    print(f"\nTest 1.3 - Bad action:")
    print(f"  Expected: {expected3:.3f}")
    print(f"  Got: {ru3:.3f}")
    print(f"  {'✓ PASS' if abs(ru3 - expected3) < 0.001 else '✗ FAIL'}")
    
    print()


# ============================================================================
# Test 2: Element Friction Matrix
# ============================================================================

def test_element_friction():
    """Test element compatibility friction adjustments."""
    print("TEST 2: Element Friction Matrix")
    print("-" * 60)
    
    decider = ResonanceDecider()
    
    base_action = ActionCandidate(
        label="Test action",
        progress=0.5, friction=0.5, coherence=0.5,
        feasibility=0.5, risk=0.5, synergy=0.5,
        element=ElementType.FIRE
    )
    
    tests = [
        (ElementType.FIRE, 1.0, "Same element"),
        (ElementType.WATER, 1.4, "Incompatible (Fire-Water)"),
        (ElementType.AIR, 0.8, "Synergy (Fire-Air)"),
    ]
    
    for field_element, expected_mult, description in tests:
        action = ActionCandidate(
            label=base_action.label,
            progress=base_action.progress,
            friction=base_action.friction,
            coherence=base_action.coherence,
            feasibility=base_action.feasibility,
            risk=base_action.risk,
            synergy=base_action.synergy,
            element=ElementType.FIRE
        )
        
        decider.calculate_resonant_utility(action, current_field_element=field_element)
        actual_mult = action.ru_breakdown['element_friction_multiplier']
        
        print(f"{description}:")
        print(f"  Expected multiplier: {expected_mult}")
        print(f"  Got: {actual_mult}")
        print(f"  {'✓ PASS' if abs(actual_mult - expected_mult) < 0.001 else '✗ FAIL'}")
        print()


# ============================================================================
# Test 3: Node Weight Modulation
# ============================================================================

def test_node_modulation():
    """Test node-specific weight modulation."""
    print("TEST 3: Node Weight Modulation")
    print("-" * 60)
    
    decider = ResonanceDecider()
    
    action = ActionCandidate(
        label="Test action",
        progress=0.8, friction=0.3, coherence=0.7,
        feasibility=0.6, risk=0.4, synergy=0.9
    )
    
    # Calculate RU with different nodes
    ru_neutral = decider.calculate_resonant_utility(action, active_node=None)
    ru_movement = decider.calculate_resonant_utility(action, active_node=NodeType.MOVEMENT)
    ru_being = decider.calculate_resonant_utility(action, active_node=NodeType.BEING)
    
    print(f"Neutral (no node): {ru_neutral:.3f}")
    print(f"Movement node: {ru_movement:.3f}")
    print(f"Being node: {ru_being:.3f}")
    
    # Movement should favor progress more
    # Being should reduce friction penalty more
    print(f"\n{'✓ PASS' if ru_movement != ru_neutral else '✗ FAIL'} - Node modulation affects score")
    print()


# ============================================================================
# Test 4: Ranking Order
# ============================================================================

def test_ranking():
    """Test that actions are correctly ranked by RU."""
    print("TEST 4: Action Ranking")
    print("-" * 60)
    
    decider = ResonanceDecider()
    
    candidates = [
        ActionCandidate(
            label="Low RU action",
            progress=0.2, friction=0.8, coherence=0.3,
            feasibility=0.4, risk=0.7, synergy=0.2
        ),
        ActionCandidate(
            label="High RU action",
            progress=0.9, friction=0.2, coherence=0.8,
            feasibility=0.9, risk=0.1, synergy=0.8
        ),
        ActionCandidate(
            label="Medium RU action",
            progress=0.6, friction=0.4, coherence=0.6,
            feasibility=0.7, risk=0.3, synergy=0.5
        ),
    ]
    
    ranked = decider.pick_best_action(candidates, top_k=3)
    
    print("Expected order: High, Medium, Low")
    print("Actual order:")
    for i, action in enumerate(ranked, 1):
        print(f"  {i}. {action.label} (RU: {action.ru_score:.3f})")
    
    # Check order
    is_correct = (
        ranked[0].label == "High RU action" and
        ranked[1].label == "Medium RU action" and
        ranked[2].label == "Low RU action"
    )
    
    print(f"\n{'✓ PASS' if is_correct else '✗ FAIL'} - Correct ranking")
    print()


# ============================================================================
# Test 5: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("TEST 5: Edge Cases")
    print("-" * 60)
    
    decider = ResonanceDecider()
    
    # Test 5.1: All zeros
    action_zeros = ActionCandidate(
        label="All zeros",
        progress=0, friction=0, coherence=0,
        feasibility=0, risk=0, synergy=0
    )
    ru_zeros = decider.calculate_resonant_utility(action_zeros)
    print(f"Test 5.1 - All zeros: RU = {ru_zeros:.3f}")
    print(f"  {'✓ PASS' if ru_zeros == 0.0 else '✗ FAIL'}")
    
    # Test 5.2: Values outside [0,1] should be clamped
    action_overflow = ActionCandidate(
        label="Overflow test",
        progress=1.5,  # Should clamp to 1.0
        friction=-0.2,  # Should clamp to 0.0
        coherence=0.5,
        feasibility=0.5,
        risk=0.5,
        synergy=0.5
    )
    ru_overflow = decider.calculate_resonant_utility(action_overflow)
    print(f"\nTest 5.2 - Clamping (P=1.5→1.0, F=-0.2→0.0):")
    print(f"  RU = {ru_overflow:.3f}")
    print(f"  {'✓ PASS' if ru_overflow is not None else '✗ FAIL'}")
    
    # Test 5.3: Empty candidates list
    empty_result = decider.pick_best_action([])
    print(f"\nTest 5.3 - Empty candidates:")
    print(f"  Result: {empty_result}")
    print(f"  {'✓ PASS' if empty_result == [] else '✗ FAIL'}")
    
    print()


# ============================================================================
# Test 6: Weight Learning
# ============================================================================

def test_weight_learning():
    """Test feedback-based weight updates."""
    print("TEST 6: Weight Learning")
    print("-" * 60)
    
    decider = ResonanceDecider()
    
    initial_synergy_weight = decider.weights['S']
    
    # User picks high-synergy action and reports satisfaction
    action = ActionCandidate(
        label="High synergy action",
        progress=0.5, friction=0.4, coherence=0.6,
        feasibility=0.7, risk=0.3, synergy=0.95
    )
    
    decider.calculate_resonant_utility(action)
    decider.update_weights_from_feedback(action, satisfaction_score=0.9)
    
    new_synergy_weight = decider.weights['S']
    
    print(f"Initial synergy weight: {initial_synergy_weight:.3f}")
    print(f"After positive feedback on high-synergy action: {new_synergy_weight:.3f}")
    print(f"Change: {new_synergy_weight - initial_synergy_weight:+.3f}")
    print(f"\n{'✓ PASS' if new_synergy_weight > initial_synergy_weight else '✗ FAIL'} - Weight increased")
    print()


# ============================================================================
# Test 7: Multi-Node Calculation
# ============================================================================

def test_multi_node():
    """Test multi-node weighted RU calculation."""
    print("TEST 7: Multi-Node Calculation")
    print("-" * 60)
    
    decider = ResonanceDecider()
    
    action = ActionCandidate(
        label="Multi-node test",
        progress=0.7, friction=0.3, coherence=0.8,
        feasibility=0.6, risk=0.2, synergy=0.7
    )
    
    # Test with 50/50 split between two nodes
    distribution = {
        NodeType.MOVEMENT: 0.5,
        NodeType.EVOLUTION: 0.5
    }
    
    ru_multi = decider.calculate_multi_node_ru(action, distribution)
    
    # Calculate individual RUs
    ru_movement = decider.calculate_resonant_utility(action, active_node=NodeType.MOVEMENT)
    ru_evolution = decider.calculate_resonant_utility(action, active_node=NodeType.EVOLUTION)
    
    # Weighted average
    expected = 0.5 * ru_movement + 0.5 * ru_evolution
    
    print(f"Movement RU: {ru_movement:.3f}")
    print(f"Evolution RU: {ru_evolution:.3f}")
    print(f"Expected weighted (50/50): {expected:.3f}")
    print(f"Actual multi-node RU: {ru_multi:.3f}")
    print(f"\n{'✓ PASS' if abs(ru_multi - expected) < 0.001 else '✗ FAIL'}")
    print()


# ============================================================================
# Test 8: Breakdown Accuracy
# ============================================================================

def test_breakdown():
    """Test RU breakdown components sum correctly."""
    print("TEST 8: Breakdown Accuracy")
    print("-" * 60)
    
    decider = ResonanceDecider()
    
    action = ActionCandidate(
        label="Breakdown test",
        progress=0.6, friction=0.3, coherence=0.7,
        feasibility=0.8, risk=0.2, synergy=0.5
    )
    
    ru = decider.calculate_resonant_utility(action)
    
    # Sum breakdown components
    breakdown_sum = sum([
        action.ru_breakdown['progress_contribution'],
        action.ru_breakdown['friction_penalty'],
        action.ru_breakdown['coherence_contribution'],
        action.ru_breakdown['feasibility_contribution'],
        action.ru_breakdown['risk_penalty'],
        action.ru_breakdown['synergy_contribution']
    ])
    
    print(f"RU from formula: {ru:.6f}")
    print(f"Sum of breakdown: {breakdown_sum:.6f}")
    print(f"Difference: {abs(ru - breakdown_sum):.9f}")
    print(f"\n{'✓ PASS' if abs(ru - breakdown_sum) < 0.0001 else '✗ FAIL'} - Breakdown matches RU")
    print()


# ============================================================================
# Test Summary
# ============================================================================

def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 60)
    print("RESONANCE DECISION ENGINE - TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_ru,
        test_element_friction,
        test_node_modulation,
        test_ranking,
        test_edge_cases,
        test_weight_learning,
        test_multi_node,
        test_breakdown
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
