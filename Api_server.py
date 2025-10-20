"""
Flask API for Resonance Decision Engine
Integrates: RU calculation + LLM estimation + Consciousness Math + Semantic Engine

Example endpoints:
- POST /api/decide - Get best action from candidates
- POST /api/estimate - Estimate RU components from text using LLM
- GET /api/status - Get system status
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from typing import Dict, List, Any, Optional

# Import your existing modules
import sys
sys.path.insert(0, os.path.dirname(__file__))

from resonance_decider import (
    ResonanceDecider, 
    ActionCandidate, 
    ElementType, 
    NodeType,
    create_action_from_dict,
    quick_decide
)
from llm_prompts import (
    format_master_prompt,
    format_business_prompt,
    format_personal_prompt,
    format_multi_option_prompt
)

# Import existing consciousness system
try:
    from consciousness_math import ConsciousnessMath
    from semantic_engine import SemanticEngine
    HAS_CONSCIOUSNESS_SYSTEM = True
except ImportError:
    HAS_CONSCIOUSNESS_SYSTEM = False
    print("Warning: Could not import consciousness system. Running in standalone mode.")


# ============================================================================
# LLM Integration (Claude API via fetch - as per your Claudeception spec)
# ============================================================================

def call_claude_for_estimation(prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
    """
    Call Claude API to estimate RU components from natural language.
    
    In browser/artifact context, this would use fetch to https://api.anthropic.com/v1/messages
    For backend, you'd use requests or the Anthropic SDK.
    
    Args:
        prompt: Formatted prompt for RU estimation
        max_tokens: Max tokens for response
        
    Returns:
        Parsed JSON response with RU components
    """
    # This is a placeholder - in production you'd call the actual API
    # For now, return a mock response
    
    # In real implementation:
    # import anthropic
    # client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # message = client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     max_tokens=max_tokens,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # response_text = message.content[0].text
    
    # Mock response for testing
    mock_response = {
        "progress": 0.72,
        "friction": 0.25,
        "coherence": 0.81,
        "feasibility": 0.88,
        "risk": 0.18,
        "synergy": 0.65,
        "reasoning": {
            "progress_reason": "Advances goal significantly",
            "friction_reason": "Moderate time investment",
            "coherence_reason": "Strong alignment across BMH",
            "feasibility_reason": "Resources available",
            "risk_reason": "Low downside",
            "synergy_reason": "Creates future options"
        }
    }
    
    return mock_response


# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for browser access

# Initialize systems
if HAS_CONSCIOUSNESS_SYSTEM:
    consciousness_math = ConsciousnessMath()
    semantic_engine = SemanticEngine()
else:
    consciousness_math = None
    semantic_engine = None

# Initialize Resonance Decider
decider = ResonanceDecider(
    consciousness_math=consciousness_math,
    semantic_engine=semantic_engine
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/decide', methods=['POST'])
def decide():
    """
    Main decision endpoint - accepts action candidates and returns best action.
    
    Request body:
    {
        "candidates": [
            {
                "label": "Action 1",
                "progress": 0.7,
                "friction": 0.3,
                "coherence": 0.8,
                "feasibility": 0.9,
                "risk": 0.2,
                "synergy": 0.6,
                "element": "Fire",  // optional
                "active_node": "Movement"  // optional
            },
            ...
        ],
        "current_element": "Earth",  // optional
        "active_node": "Design",  // optional
        "top_k": 3  // optional, default 1
    }
    
    Response:
    {
        "best_action": {...},
        "all_results": [...],
        "ru_formula": "..."
    }
    """
    data = request.json
    
    if 'candidates' not in data:
        return jsonify({"error": "Missing 'candidates' field"}), 400
    
    try:
        # Parse candidates
        candidates = [create_action_from_dict(c) for c in data['candidates']]
        
        # Parse current field element
        current_element = None
        if 'current_element' in data:
            try:
                current_element = ElementType[data['current_element'].upper()]
            except (KeyError, AttributeError):
                pass
        
        # Parse active node
        active_node = None
        if 'active_node' in data:
            try:
                active_node = NodeType[data['active_node'].upper()]
            except (KeyError, AttributeError):
                pass
        
        # Get top_k
        top_k = data.get('top_k', 1)
        
        # Pick best action(s)
        best_actions = decider.pick_best_action(
            candidates,
            current_field_element=current_element,
            active_node=active_node,
            top_k=top_k
        )
        
        # Format response
        response = {
            "best_action": {
                "label": best_actions[0].label,
                "ru_score": best_actions[0].ru_score,
                "breakdown": best_actions[0].ru_breakdown,
                "element": best_actions[0].element.value if best_actions[0].element else None,
                "active_node": best_actions[0].active_node.value if best_actions[0].active_node else None
            },
            "all_results": [
                {
                    "label": a.label,
                    "ru_score": a.ru_score,
                    "breakdown": a.ru_breakdown,
                    "element": a.element.value if a.element else None
                }
                for a in candidates
            ],
            "ru_formula": "RU = w_P·P - w_F·F + w_C·C + w_R·R - w_K·K + w_S·S",
            "weights_used": decider._get_modulated_weights(active_node),
            "context": {
                "current_element": current_element.value if current_element else None,
                "active_node": active_node.value if active_node else None
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/estimate', methods=['POST'])
def estimate():
    """
    Estimate RU components from natural language using LLM.
    
    Request body:
    {
        "context": "Building a product",
        "goal": "Launch MVP",
        "action_description": "Focus on UI polish this week",
        "current_element": "Earth",  // optional
        "active_node": "Design",  // optional
        "prompt_type": "master"  // or "business", "personal", "creative"
    }
    
    Response:
    {
        "estimated_scores": {
            "progress": 0.72,
            "friction": 0.25,
            ...
        },
        "action_candidate": {...},
        "prompt_used": "..."
    }
    """
    data = request.json
    
    required_fields = ['action_description']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    try:
        # Format prompt based on type
        prompt_type = data.get('prompt_type', 'master')
        
        if prompt_type == 'master':
            prompt = format_master_prompt(
                context=data.get('context', 'General context'),
                goal=data.get('goal', 'Achieve optimal outcome'),
                action_description=data['action_description'],
                current_element=data.get('current_element', 'Unknown'),
                active_node=data.get('active_node', 'Unknown')
            )
        elif prompt_type == 'business':
            prompt = format_business_prompt(
                mrr=data.get('mrr', 'Unknown'),
                team_size=data.get('team_size', 1),
                runway_months=data.get('runway_months', 12),
                focus_area=data.get('focus_area', 'Growth'),
                action_description=data['action_description']
            )
        else:
            return jsonify({"error": f"Unknown prompt_type: {prompt_type}"}), 400
        
        # Call LLM for estimation
        estimated_scores = call_claude_for_estimation(prompt)
        
        # Create action candidate
        action_data = {
            "label": data.get('label', data['action_description'][:50]),
            "progress": estimated_scores['progress'],
            "friction": estimated_scores['friction'],
            "coherence": estimated_scores['coherence'],
            "feasibility": estimated_scores['feasibility'],
            "risk": estimated_scores['risk'],
            "synergy": estimated_scores['synergy']
        }
        
        if 'element' in data:
            action_data['element'] = data['element']
        if 'active_node' in data:
            action_data['active_node'] = data['active_node']
        
        action_candidate = create_action_from_dict(action_data)
        
        # Calculate RU
        current_element = None
        if 'current_element' in data:
            try:
                current_element = ElementType[data['current_element'].upper()]
            except (KeyError, AttributeError):
                pass
        
        active_node = None
        if 'active_node' in data:
            try:
                active_node = NodeType[data['active_node'].upper()]
            except (KeyError, AttributeError):
                pass
        
        ru_score = decider.calculate_resonant_utility(
            action_candidate,
            current_field_element=current_element,
            active_node=active_node
        )
        
        response = {
            "estimated_scores": estimated_scores,
            "action_candidate": {
                "label": action_candidate.label,
                "ru_score": action_candidate.ru_score,
                "breakdown": action_candidate.ru_breakdown
            },
            "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/decide/multi', methods=['POST'])
def decide_multi_text():
    """
    Decide between multiple text action descriptions using LLM estimation.
    
    Request body:
    {
        "context": "Current situation",
        "goal": "What to achieve",
        "actions": [
            "Action 1 description",
            "Action 2 description",
            "Action 3 description"
        ],
        "current_element": "Earth",  // optional
        "active_node": "Movement"  // optional
    }
    
    Response:
    {
        "best_action": {...},
        "all_candidates": [...]
    }
    """
    data = request.json
    
    required = ['actions']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    try:
        # Format multi-option prompt
        prompt = format_multi_option_prompt(
            context=data.get('context', 'General context'),
            goal=data.get('goal', 'Optimal outcome'),
            action_list=data['actions']
        )
        
        # Call LLM (would return array of estimates)
        # For now, mock it
        estimated_candidates = []
        for action_text in data['actions']:
            scores = call_claude_for_estimation(format_master_prompt(
                context=data.get('context', ''),
                goal=data.get('goal', ''),
                action_description=action_text
            ))
            estimated_candidates.append({
                "label": action_text[:50],
                **scores
            })
        
        # Convert to ActionCandidates
        candidates = [create_action_from_dict(c) for c in estimated_candidates]
        
        # Parse element/node
        current_element = None
        if 'current_element' in data:
            try:
                current_element = ElementType[data['current_element'].upper()]
            except (KeyError, AttributeError):
                pass
        
        active_node = None
        if 'active_node' in data:
            try:
                active_node = NodeType[data['active_node'].upper()]
            except (KeyError, AttributeError):
                pass
        
        # Pick best
        best = decider.pick_best_action(
            candidates,
            current_field_element=current_element,
            active_node=active_node,
            top_k=1
        )[0]
        
        response = {
            "best_action": {
                "label": best.label,
                "ru_score": best.ru_score,
                "breakdown": best.ru_breakdown
            },
            "all_candidates": [
                {
                    "label": c.label,
                    "ru_score": c.ru_score,
                    "breakdown": c.ru_breakdown
                }
                for c in sorted(candidates, key=lambda x: x.ru_score or 0, reverse=True)
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    try:
        status = {
            "decider": decider.get_status(),
            "consciousness_system_available": HAS_CONSCIOUSNESS_SYSTEM
        }
        
        if HAS_CONSCIOUSNESS_SYSTEM:
            status["consciousness_math"] = consciousness_math.get_status()
            status["semantic_engine"] = semantic_engine.get_status()
        
        return jsonify(status)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/weights', methods=['GET', 'POST'])
def manage_weights():
    """Get or update RU weights."""
    if request.method == 'GET':
        return jsonify({
            "weights": decider.weights,
            "default_weights": decider.default_weights
        })
    
    elif request.method == 'POST':
        data = request.json
        
        if 'weights' in data:
            decider.weights = data['weights']
            return jsonify({
                "message": "Weights updated",
                "new_weights": decider.weights
            })
        
        return jsonify({"error": "Missing 'weights' field"}), 400


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit user feedback to update weights (online learning).
    
    Request body:
    {
        "action": {...ActionCandidate...},
        "satisfaction": 0.8  // 0.0 to 1.0
    }
    """
    data = request.json
    
    required = ['action', 'satisfaction']
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    try:
        action = create_action_from_dict(data['action'])
        satisfaction = float(data['satisfaction'])
        
        # Update weights based on feedback
        decider.update_weights_from_feedback(action, satisfaction)
        
        return jsonify({
            "message": "Feedback processed, weights updated",
            "new_weights": decider.weights
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Quick Test Endpoint
# ============================================================================

@app.route('/api/test', methods=['GET'])
def test():
    """Quick test endpoint with sample data."""
    sample_candidates = [
        {
            "label": "Email 5 beta users",
            "progress": 0.55,
            "friction": 0.20,
            "coherence": 0.78,
            "feasibility": 0.85,
            "risk": 0.15,
            "synergy": 0.60
        },
        {
            "label": "Rewrite landing page",
            "progress": 0.40,
            "friction": 0.10,
            "coherence": 0.74,
            "feasibility": 0.90,
            "risk": 0.10,
            "synergy": 0.50
        },
        {
            "label": "Build complex feature",
            "progress": 0.85,
            "friction": 0.70,
            "coherence": 0.62,
            "feasibility": 0.40,
            "risk": 0.45,
            "synergy": 0.80
        }
    ]
    
    result = quick_decide(sample_candidates)
    
    return jsonify({
        "test_data": "Sample decision with 3 candidates",
        "result": result
    })


# ============================================================================
# Run App
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Resonance Decision Engine API")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /api/decide          - Decide from scored candidates")
    print("  POST /api/estimate        - Estimate RU scores via LLM")
    print("  POST /api/decide/multi    - Decide from text descriptions")
    print("  GET  /api/status          - System status")
    print("  GET  /api/test            - Quick test")
    print("  GET/POST /api/weights     - Manage weights")
    print("  POST /api/feedback        - Submit feedback")
    print("\n" + "=" * 60)
    print()
    
    # Run on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
