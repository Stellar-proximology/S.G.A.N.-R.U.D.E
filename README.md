# Resonance Decision Engine

**Least-friction, most-progression action selection for the YOU-N-I-VERSE consciousness system.**

---

## Overview

The Resonance Decision Engine implements a master utility function (RU) that scores any action candidate across six dimensions, enabling optimal decision-making for individuals, teams, and AI agents.

### Core Formula

```
RU(a) = w_P·Progress(a) - w_F·Friction(a) + w_C·Coherence(a) 
        + w_R·Feasibility(a) - w_K·Risk(a) + w_S·Synergy(a)
```

Where:
- **Progress** (P): How much this action advances toward the goal
- **Friction** (F): Total resistance (time + cost + emotion + context switch)
- **Coherence** (C): Body-Mind-Heart alignment from your Sentence System
- **Feasibility** (R): Can we actually execute this now?
- **Risk** (K): Probability × severity of downside
- **Synergy** (S): Future compounding / option value

All components in [0, 1]. Weights default to: P=1.0, F=0.8, C=0.6, R=0.5, K=0.7, S=0.4

---

## Quick Start

### Installation

```bash
# Copy files to your project
cp resonance_decider.py your_project/
cp llm_prompts.py your_project/
cp api_server.py your_project/  # Optional: if using API

# Install dependencies
pip install numpy flask flask-cors torch transformers
```

### Basic Usage

```python
from resonance_decider import ResonanceDecider, ActionCandidate, ElementType

# Create decider
decider = ResonanceDecider()

# Define action candidates
candidates = [
    ActionCandidate(
        label="Email 5 beta users",
        progress=0.55, friction=0.20, coherence=0.78,
        feasibility=0.85, risk=0.15, synergy=0.60
    ),
    ActionCandidate(
        label="Rewrite landing page",
        progress=0.40, friction=0.10, coherence=0.74,
        feasibility=0.90, risk=0.10, synergy=0.50
    )
]

# Pick best action
best = decider.pick_best_action(candidates, top_k=1)[0]

print(f"Best action: {best.label}")
print(f"RU score: {best.ru_score:.3f}")
print(f"Breakdown: {best.ru_breakdown}")
```

---

## Features

### 1. **Element-Aware Friction Adjustment**

Automatically adjusts friction based on elemental compatibility:

```python
action = ActionCandidate(
    ...,
    element=ElementType.FIRE  # Action element
)

best = decider.pick_best_action(
    [action],
    current_field_element=ElementType.WATER  # Current state
)

# Fire-Water incompatibility increases friction by 40%
```

**Element Matrix:**
- Incompatible: Fire↔Water, Earth↔Air (friction +40%)
- Synergies: Fire+Air, Water+Earth, Aether+Void (friction -20%)

### 2. **Node Weight Modulation**

Automatically adjust weights based on active consciousness node:

```python
best = decider.pick_best_action(
    candidates,
    active_node=NodeType.MOVEMENT  # Emphasizes progress
)
```

**Node Preferences:**
- **Movement**: Favors progress (P↑), reduces friction penalty (F↓)
- **Evolution**: Emphasizes coherence (C↑) and synergy (S↑)
- **Being**: Maximizes coherence (C↑), minimizes friction (F↓)
- **Design**: Favors feasibility (R↑) and progress (P↑)
- **Space**: Emphasizes synergy (S↑) and openness

### 3. **Multi-Node Support**

Handle transitions between nodes:

```python
node_distribution = {
    NodeType.MOVEMENT: 0.6,
    NodeType.EVOLUTION: 0.4
}

ru = decider.calculate_multi_node_ru(action, node_distribution)
```

### 4. **Online Learning from Feedback**

Weights adapt based on user satisfaction:

```python
decider.update_weights_from_feedback(
    chosen_action=action,
    satisfaction_score=0.9  # High satisfaction
)

# Weights now favor patterns similar to chosen_action
```

### 5. **LLM Integration**

Estimate RU components from natural language:

```python
from llm_prompts import format_master_prompt

prompt = format_master_prompt(
    context="Building MVP with limited time",
    goal="Get 10 beta users",
    action_description="Polish UI this week"
)

# Send prompt to Claude API → get P/F/C/R/K/S estimates
# Then create ActionCandidate with those scores
```

---

## Integration with Consciousness System

### Full Integration

```python
from consciousness_math import ConsciousnessMath
from semantic_engine import SemanticEngine
from resonance_decider import ResonanceDecider

# Initialize systems
cm = ConsciousnessMath()
se = SemanticEngine()

# Create integrated decider
decider = ResonanceDecider(
    consciousness_math=cm,
    semantic_engine=se
)

# Extract BMH scores from text
action_text = "Focus on building the chart decoder"
bmh = se.extract_bmh_scores(action_text)

# Calculate coherence
coherence = cm.calculate_coherence(bmh['body'], bmh['mind'], bmh['heart'])

# Create action with consciousness-derived coherence
action = ActionCandidate(
    label=action_text,
    progress=0.7,  # Estimate or use LLM
    friction=0.3,  # Estimate or use LLM
    coherence=coherence,  # From consciousness system
    feasibility=0.8,
    risk=0.2,
    synergy=0.6
)
```

---

## API Server

Run the Flask API for browser/external access:

```bash
python api_server.py
```

### Endpoints

#### POST `/api/decide`
Decide from pre-scored candidates.

**Request:**
```json
{
  "candidates": [
    {
      "label": "Action 1",
      "progress": 0.7,
      "friction": 0.3,
      "coherence": 0.8,
      "feasibility": 0.9,
      "risk": 0.2,
      "synergy": 0.6
    }
  ],
  "current_element": "Earth",
  "active_node": "Design",
  "top_k": 3
}
```

**Response:**
```json
{
  "best_action": {
    "label": "Action 1",
    "ru_score": 0.847,
    "breakdown": {
      "progress_contribution": 0.700,
      "friction_penalty": -0.240,
      ...
    }
  },
  "all_results": [...]
}
```

#### POST `/api/estimate`
Estimate RU components from text using LLM.

**Request:**
```json
{
  "context": "Building a product",
  "goal": "Launch MVP",
  "action_description": "Focus on UI polish this week",
  "current_element": "Earth",
  "prompt_type": "master"
}
```

#### POST `/api/decide/multi`
Decide between multiple text descriptions.

**Request:**
```json
{
  "context": "Need to increase revenue",
  "goal": "Get to $10k MRR",
  "actions": [
    "Cold email 100 leads",
    "Build new feature",
    "Launch paid ads"
  ]
}
```

---

## Examples

See `integration_guide.py` for 7 complete examples:

```bash
# Run all examples
python integration_guide.py

# Run specific example
python integration_guide.py 7  # Real business decision
```

**Examples:**
1. Standalone usage (no dependencies)
2. Integration with consciousness system
3. LLM-powered estimation
4. Element-aware friction
5. Multi-node weighted decision
6. Feedback learning
7. Real business decision (full pipeline)

---

## Testing

Run the test suite:

```bash
python test_ru_engine.py
```

**Tests cover:**
- Basic RU calculation
- Element friction matrix
- Node weight modulation
- Action ranking
- Edge cases & boundary conditions
- Weight learning
- Multi-node calculation
- Breakdown accuracy

---

## File Structure

```
resonance_decider.py      # Core RU engine + ActionCandidate classes
llm_prompts.py            # LLM prompt templates for estimation
api_server.py             # Flask API (optional)
integration_guide.py      # 7 complete examples
test_ru_engine.py         # Test suite
README.md                 # This file
```

---

## Advanced Usage

### Custom Weights

```python
decider = ResonanceDecider(weights={
    'P': 1.5,  # Emphasize progress
    'F': 0.4,  # De-emphasize friction
    'C': 0.8,
    'R': 0.5,
    'K': 0.9,  # Emphasize risk avoidance
    'S': 1.2   # Emphasize synergy
})
```

### Per-Node Weight Profiles

Customize how each node modulates weights:

```python
decider.node_weight_modulation[NodeType.MOVEMENT] = {
    'P': 1.5,  # Movement really favors progress
    'F': 0.6,  # Cares less about friction
    'S': 1.2   # Values momentum
}
```

### Element Friction Tuning

Adjust element incompatibility penalties:

```python
# Make Fire-Water even more incompatible
decider.element_friction_matrix[(ElementType.FIRE, ElementType.WATER)] = 1.6

# Create new synergy
decider.element_friction_matrix[(ElementType.EARTH, ElementType.VOID)] = 0.7
```

### Coherence from Semantic Latent

If you have a GAN discriminator outputting semantic latents:

```python
semantic_latent = {
    'dimension': 3,
    'shape': 'Triangle',
    'element': ElementType.FIRE,
    'operator': '×',
    'color_rgb': (0.9, 0.5, 0.2)
}

coherence = decider.estimate_coherence_from_semantic(semantic_latent)
```

---

## Mathematical Properties

### 1. **Additive Components**
Each term contributes independently, allowing clear attribution.

### 2. **Sign-Aware**
Progress/Coherence/Feasibility/Synergy are positive contributors.  
Friction/Risk are negative (penalties).

### 3. **Bounded Inputs**
All components in [0, 1] ensure comparable magnitudes.

### 4. **Unbounded Output**
RU can be negative (bad actions) or >1 (exceptional actions).

### 5. **Differentiable**
Can be used with gradient-based optimization if needed.

### 6. **Interpretable**
Breakdown shows exactly which components drove the decision.

---

## Design Philosophy

### Why RU Works

1. **Multi-Objective**: Balances speed (low F) with quality (high C, S)
2. **Context-Aware**: Element/node matching reduces misalignment
3. **Adaptive**: Weights learn from feedback
4. **Semantic**: Coherence ties to Body-Mind-Heart consciousness
5. **Explainable**: Breakdown shows reasoning

### When to Use

- **Business**: Prioritize features, allocate resources, strategic pivots
- **Personal**: Life decisions, habit formation, goal pursuit
- **Creative**: Project direction, collaboration choices
- **AI Agents**: Action selection in multi-step planning
- **Teams**: Consensus-building around priorities

---

## Roadmap

### Current (v1.0)
- ✅ Core RU formula
- ✅ Element friction matrix
- ✅ Node weight modulation
- ✅ Online learning
- ✅ Flask API
- ✅ LLM prompt templates

### Planned (v1.1)
- [ ] GAN integration for action proposal
- [ ] Multi-user weight profiles
- [ ] Historical decision tracking + analytics
- [ ] Reinforcement learning for weight optimization
- [ ] React component for browser UI

### Future
- [ ] Multi-agent consensus (team decisions)
- [ ] Time-series prediction for synergy
- [ ] Causal inference for breakdown validation

---

## Contributing

This is part of the YOU-N-I-VERSE consciousness system. 

To extend:
1. Add new element types in `ElementType` enum
2. Define friction matrix entries
3. Update node modulation profiles
4. Add domain-specific prompt templates in `llm_prompts.py`

---

## License

Part of the YOU-N-I-VERSE project. Use freely in your consciousness systems.

---

## Citation

If using in research or production:

```
Resonance Decision Engine (2025)
Part of the YOU-N-I-VERSE 9-Body Consciousness System
Based on waveform resonance, Human Design, and stellar proximology
```

---

## Support

For questions about integration with your consciousness system:
- Check `integration_guide.py` examples
- Run `test_ru_engine.py` to validate setup
- Review existing code in `consciousness_math.py` and `semantic_engine.py`

---

**Stand on it. This is your solver kernel.**
