"""
LLM Prompt Templates for Resonant Utility Estimation

These prompts guide an LLM to estimate P/F/C/R/K/S scores for action candidates
from natural language descriptions.
"""

# ============================================================================
# Master Prompt Template
# ============================================================================

MASTER_RU_ESTIMATION_PROMPT = """You are an expert decision analyst for a consciousness-based decision system. Your task is to estimate Resonant Utility (RU) components for a proposed action.

Given an action description, current context, and goal, estimate these six scores (all 0.0 to 1.0):

**Progress (P)**: How much does this action advance toward the goal?
- 0.0 = No progress or moves backward
- 0.5 = Moderate progress
- 1.0 = Completes the goal or major breakthrough

**Friction (F)**: Total resistance (time + cost + emotion + context switch)
- 0.0 = Effortless, instant, free, natural
- 0.5 = Moderate effort/cost
- 1.0 = Extremely difficult, expensive, draining, requires major context shift

**Coherence (C)**: Body-Mind-Heart alignment (intuition + logic + emotion agree)
- 0.0 = Strong internal conflict, feels wrong
- 0.5 = Neutral or mixed signals
- 1.0 = Complete alignment, feels deeply right

**Feasibility (R)**: Can we actually do this now with current resources?
- 0.0 = Impossible with current resources/skills/time
- 0.5 = Possible but challenging
- 1.0 = Easy to execute right now

**Risk (K)**: Probability × severity of downside
- 0.0 = No risk, safe
- 0.5 = Moderate risk
- 1.0 = High probability of severe negative outcome

**Synergy (S)**: Future compounding / option value
- 0.0 = No future benefit, one-time only
- 0.5 = Some future benefit
- 1.0 = Unlocks major future options, compounds over time, creates assets

---

**Current Context:**
{context}

**Goal:**
{goal}

**Proposed Action:**
{action_description}

**Current Field Element:** {current_element}
**Active Node:** {active_node}

---

Respond ONLY with valid JSON in this exact format:
{{
  "progress": <float 0.0-1.0>,
  "friction": <float 0.0-1.0>,
  "coherence": <float 0.0-1.0>,
  "feasibility": <float 0.0-1.0>,
  "risk": <float 0.0-1.0>,
  "synergy": <float 0.0-1.0>,
  "reasoning": {{
    "progress_reason": "<brief explanation>",
    "friction_reason": "<brief explanation>",
    "coherence_reason": "<brief explanation>",
    "feasibility_reason": "<brief explanation>",
    "risk_reason": "<brief explanation>",
    "synergy_reason": "<brief explanation>"
  }}
}}

DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON. No markdown, no backticks, just pure JSON.
"""


# ============================================================================
# Specialized Templates for Different Domains
# ============================================================================

BUSINESS_ACTION_PROMPT = """You are analyzing a business decision for Resonant Utility scoring.

**Business Context:**
- Current MRR: {mrr}
- Team Size: {team_size}
- Runway: {runway_months} months
- Current Focus: {focus_area}

**Proposed Action:** {action_description}

Estimate P/F/C/R/K/S scores (0.0-1.0) considering:
- Progress: Revenue impact, customer acquisition, market position
- Friction: Time to implement, cost, opportunity cost, team strain
- Coherence: Alignment with company vision, team buy-in, founder conviction
- Feasibility: Skills available, resources at hand, time to MVP
- Risk: Market risk, execution risk, financial risk
- Synergy: Platform effects, compounding growth, option value

Respond with ONLY valid JSON:
{{
  "progress": <float>,
  "friction": <float>,
  "coherence": <float>,
  "feasibility": <float>,
  "risk": <float>,
  "synergy": <float>,
  "reasoning": {{...}}
}}
"""


PERSONAL_DECISION_PROMPT = """You are analyzing a personal life decision for Resonant Utility scoring.

**Personal Context:**
- Current Life Phase: {life_phase}
- Energy Level: {energy_level}/10
- Top Values: {values}
- Current Challenge: {challenge}

**Proposed Action:** {action_description}

Estimate P/F/C/R/K/S scores (0.0-1.0) considering:
- Progress: How much this moves you toward fulfillment, growth, or resolution
- Friction: Emotional labor, time, money, social friction, habit disruption
- Coherence: Does this feel right in your body, mind, and heart? Authentic?
- Feasibility: Can you actually start this today with what you have?
- Risk: Emotional risk, relationship risk, financial risk, health risk
- Synergy: Does this create positive ripples? Build capacity? Open doors?

Respond with ONLY valid JSON:
{{
  "progress": <float>,
  "friction": <float>,
  "coherence": <float>,
  "feasibility": <float>,
  "risk": <float>,
  "synergy": <float>,
  "reasoning": {{...}}
}}
"""


CREATIVE_PROJECT_PROMPT = """You are analyzing a creative/artistic decision for Resonant Utility scoring.

**Creative Context:**
- Project: {project_name}
- Current Phase: {phase}
- Vision: {vision}
- Constraint: {constraint}

**Proposed Action:** {action_description}

Estimate P/F/C/R/K/S scores (0.0-1.0) considering:
- Progress: Does this advance the creative vision? Bring clarity? Add depth?
- Friction: Creative blocks, technical barriers, time, perfectionism drag
- Coherence: Does this resonate with your artistic truth? Feel inspired?
- Feasibility: Skills, tools, time available? Can you prototype quickly?
- Risk: Will this dilute the vision? Lose audience? Technical failure?
- Synergy: Does this unlock new creative paths? Build skills? Attract collaborators?

Respond with ONLY valid JSON:
{{
  "progress": <float>,
  "friction": <float>,
  "coherence": <float>,
  "feasibility": <float>,
  "risk": <float>,
  "synergy": <float>,
  "reasoning": {{...}}
}}
"""


# ============================================================================
# Element-Aware Prompt (integrates with your semantic geometry)
# ============================================================================

ELEMENT_AWARE_PROMPT = """You are analyzing an action through the lens of elemental consciousness.

**Current Field Element:** {current_element}
- Fire: Action, drive, transformation, quick bursts
- Water: Flow, emotion, adaptation, depth
- Earth: Grounding, structure, patience, manifestation
- Air: Thought, communication, clarity, movement
- Aether: Spirit, connection, transcendence, unity
- Void: Emptiness, potential, mystery, receptivity

**Proposed Action Element:** {action_element}

**Action Description:** {action_description}

Estimate P/F/C/R/K/S scores (0.0-1.0), considering:
- Friction: INCREASE friction score by 0.2-0.4 if current element and action element are incompatible
  - Incompatible pairs: Fire↔Water, Earth↔Air
- Coherence: INCREASE coherence if elements are complementary
  - Synergies: Fire+Air, Water+Earth, Aether+Void
- Adjust other scores based on elemental wisdom

Respond with ONLY valid JSON:
{{
  "progress": <float>,
  "friction": <float>,
  "coherence": <float>,
  "feasibility": <float>,
  "risk": <float>,
  "synergy": <float>,
  "element_friction_penalty": <float 0.0-0.5>,
  "reasoning": {{...}}
}}
"""


# ============================================================================
# Multi-Option Comparison Prompt
# ============================================================================

MULTI_OPTION_COMPARISON_PROMPT = """You are comparing multiple action options for Resonant Utility scoring.

**Context:** {context}
**Goal:** {goal}

**Action Options:**
{action_list}

For EACH action, estimate P/F/C/R/K/S scores (0.0-1.0).

Respond with ONLY valid JSON (array of action scores):
[
  {{
    "action_label": "<action 1 label>",
    "progress": <float>,
    "friction": <float>,
    "coherence": <float>,
    "feasibility": <float>,
    "risk": <float>,
    "synergy": <float>,
    "reasoning_summary": "<1-sentence summary>"
  }},
  {{
    "action_label": "<action 2 label>",
    ...
  }},
  ...
]

DO NOT output anything other than valid JSON array.
"""


# ============================================================================
# Python Helper Functions
# ============================================================================

def format_master_prompt(context: str, 
                         goal: str, 
                         action_description: str,
                         current_element: str = "Unknown",
                         active_node: str = "Unknown") -> str:
    """
    Format the master RU estimation prompt.
    
    Args:
        context: Current situation context
        goal: What we're trying to achieve
        action_description: Description of the proposed action
        current_element: Current field element (Fire/Water/Earth/Air/Aether/Void)
        active_node: Active consciousness node (Movement/Evolution/Being/Design/Space)
        
    Returns:
        Formatted prompt string
    """
    return MASTER_RU_ESTIMATION_PROMPT.format(
        context=context,
        goal=goal,
        action_description=action_description,
        current_element=current_element,
        active_node=active_node
    )


def format_business_prompt(mrr: str,
                           team_size: int,
                           runway_months: int,
                           focus_area: str,
                           action_description: str) -> str:
    """Format business decision prompt."""
    return BUSINESS_ACTION_PROMPT.format(
        mrr=mrr,
        team_size=team_size,
        runway_months=runway_months,
        focus_area=focus_area,
        action_description=action_description
    )


def format_personal_prompt(life_phase: str,
                           energy_level: int,
                           values: str,
                           challenge: str,
                           action_description: str) -> str:
    """Format personal decision prompt."""
    return PERSONAL_DECISION_PROMPT.format(
        life_phase=life_phase,
        energy_level=energy_level,
        values=values,
        challenge=challenge,
        action_description=action_description
    )


def format_creative_prompt(project_name: str,
                           phase: str,
                           vision: str,
                           constraint: str,
                           action_description: str) -> str:
    """Format creative project prompt."""
    return CREATIVE_PROJECT_PROMPT.format(
        project_name=project_name,
        phase=phase,
        vision=vision,
        constraint=constraint,
        action_description=action_description
    )


def format_element_aware_prompt(current_element: str,
                               action_element: str,
                               action_description: str) -> str:
    """Format element-aware prompt."""
    return ELEMENT_AWARE_PROMPT.format(
        current_element=current_element,
        action_element=action_element,
        action_description=action_description
    )


def format_multi_option_prompt(context: str,
                               goal: str,
                               action_list: list) -> str:
    """
    Format multi-option comparison prompt.
    
    Args:
        context: Current context
        goal: Goal to achieve
        action_list: List of action descriptions
        
    Returns:
        Formatted prompt
    """
    formatted_actions = "\n".join([f"{i+1}. {action}" for i, action in enumerate(action_list)])
    
    return MULTI_OPTION_COMPARISON_PROMPT.format(
        context=context,
        goal=goal,
        action_list=formatted_actions
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Master prompt
    prompt = format_master_prompt(
        context="Building a 9-body consciousness system with limited time",
        goal="Ship MVP to 10 beta users by end of month",
        action_description="Focus on polishing the chart decoder UI this week",
        current_element="Earth",
        active_node="Design"
    )
    print("=== MASTER PROMPT ===")
    print(prompt)
    print()
    
    # Example 2: Business prompt
    prompt = format_business_prompt(
        mrr="$2,500",
        team_size=2,
        runway_months=6,
        focus_area="User acquisition",
        action_description="Launch a podcast to build audience before product launch"
    )
    print("=== BUSINESS PROMPT ===")
    print(prompt)
    print()
    
    # Example 3: Multi-option
    prompt = format_multi_option_prompt(
        context="Need to increase revenue, 3 months runway",
        goal="Get to $10k MRR",
        action_list=[
            "Cold email 100 leads",
            "Build new feature customers requested",
            "Launch paid ads campaign"
        ]
    )
    print("=== MULTI-OPTION PROMPT ===")
    print(prompt)
