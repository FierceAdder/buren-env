"""Independent reward functions and composite RewardCalculator (rule-based only)."""

from __future__ import annotations

import re
from statistics import variance

from environment.state import BurenAction, BurenState


def survival_reward(state: BurenState) -> float:
    if min(state.health, state.wealth, state.happiness) < 10:
        return -5.0
    if min(state.health, state.wealth, state.happiness) < 25:
        return -1.0
    return min(state.health, state.wealth, state.happiness) / 100.0


def balance_reward(state: BurenState) -> float:
    vals = [state.health, state.wealth, state.happiness]
    var = variance(vals) if len(vals) > 1 else 0.0
    score = 1.0 - (var / 2500.0)
    return max(0.0, min(1.0, score))


def foresight_reward(state: BurenState) -> float:
    age_weight = state.age / 70.0
    return age_weight * (state.health + state.wealth + state.happiness) / 300.0


def _combined_text(action: BurenAction) -> str:
    return f"{action.reasoning}\n{action.decision}".lower()


def reasoning_reward(action: BurenAction, state: BurenState) -> float:
    """Rubric score for CoT quality (no LLM). Max 0.7."""
    text = _combined_text(action)
    score = 0.0
    categories = 0

    if any(k in text for k in ("health", "wealth", "happiness")):
        score += 0.15
        categories += 1
    if any(k in text for k in ("long term", "long-term", "future", "later", "years", "decades")):
        score += 0.15
        categories += 1
    if any(k in text for k in ("but ", "however", "sacrifice", "cost", "risk", "tradeoff", "trade-off")):
        score += 0.15
        categories += 1
    phase_bits = ("at my age", "given that i'm", "with retirement", "at this stage", "midlife", "in my twenties", "in my thirties", "in my forties", "in my fifties", "in my sixties")
    if any(k in text for k in phase_bits) or (state.age >= 55 and "retir" in text):
        score += 0.15
        categories += 1

    if categories >= 4:
        score += 0.1
    return min(0.7, score)


def _word_count(action: BurenAction) -> int:
    blob = f"{action.reasoning} {action.decision} {action.raw_response}".strip()
    if not blob:
        return 0
    return len(re.findall(r"\b\w+\b", blob))


def _consistency_penalty(action: BurenAction) -> float:
    """Simple keyword-level contradiction check between reasoning and decision."""
    r = action.reasoning.lower()
    d = action.decision.lower()
    if not r or not d:
        return 0.0

    neg = ("won't", "will not", "not going to", "decline", "refuse", "say no", "reject", "quit", "leave", "avoid")
    pos = ("will accept", "i'll take", "i will take", "accept", "yes", "agree", "commit", "stay", "join")

    r_neg = any(x in r for x in neg)
    r_pos = any(x in r for x in pos)
    d_neg = any(x in d for x in neg)
    d_pos = any(x in d for x in pos)

    if r_neg and d_pos and not d_neg:
        return -0.3
    if r_pos and d_neg and not d_pos:
        return -0.3
    return 0.0


class RewardCalculator:
    """Weighted composite reward with anti-hack cap and penalties."""

    def __init__(self, max_total: float = 3.0):
        self.max_total = max_total

    def compute(self, action: BurenAction, state: BurenState) -> float:
        r1 = survival_reward(state)
        r2 = balance_reward(state)
        r3 = foresight_reward(state)
        r4 = reasoning_reward(action, state)

        total = 0.35 * r1 + 0.30 * r2 + 0.20 * r3 + 0.15 * r4

        if _word_count(action) < 20:
            total -= 0.5

        total += _consistency_penalty(action)

        return max(-10.0, min(self.max_total, total))
