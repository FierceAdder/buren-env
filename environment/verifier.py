"""Rule-based consequence derivation from agent text (no LLM)."""

from __future__ import annotations

import re

from environment.rewards import reasoning_reward
from environment.state import BurenAction, BurenState, phase_scale

KEYWORD_MAP: dict[str, dict[str, list[str]]] = {
    "health_positive": {
        "kw": [
            "exercise",
            "sleep",
            "doctor",
            "rest",
            "vacation",
            "mental health",
            "boundaries",
            "therapy",
        ]
    },
    "health_negative": {
        "kw": [
            "overwork",
            "stress",
            "skip",
            "ignore",
            "push through",
            "all-nighter",
            "sacrifice health",
        ]
    },
    "wealth_positive": {
        "kw": [
            "accept",
            "negotiate",
            "invest",
            "save",
            "opportunity",
            "promotion",
            "side income",
        ]
    },
    "wealth_negative": {
        "kw": [
            "decline",
            "quit",
            "risk",
            "spend",
            "charity",
            "sabbatical",
        ]
    },
    "happiness_positive": {
        "kw": [
            "family",
            "friends",
            "passion",
            "purpose",
            "meaning",
            "balance",
            "present",
            "relationship",
        ]
    },
    "happiness_negative": {
        "kw": [
            "alone",
            "regret",
            "miss",
            "compromise",
            "resentment",
            "drift",
            "disconnect",
        ]
    },
}


def _count_matches(text: str, keywords: list[str]) -> int:
    n = 0
    for kw in keywords:
        if kw in text:
            n += 1
    return n


def _clamp_delta(x: float, lo: float = -20.0, hi: float = 20.0) -> float:
    return max(lo, min(hi, x))


class RubricVerifier:
    """Maps keywords in reasoning+decision to stat deltas."""

    def derive_consequences(self, scenario: str, action: BurenAction, state: BurenState) -> dict[str, float]:
        _ = scenario  # reserved for future context-aware rules
        text = f"{action.reasoning}\n{action.decision}".lower()

        hp = _count_matches(text, KEYWORD_MAP["health_positive"]["kw"])
        hn = _count_matches(text, KEYWORD_MAP["health_negative"]["kw"])
        wp = _count_matches(text, KEYWORD_MAP["wealth_positive"]["kw"])
        wn = _count_matches(text, KEYWORD_MAP["wealth_negative"]["kw"])
        yp = _count_matches(text, KEYWORD_MAP["happiness_positive"]["kw"])
        yn = _count_matches(text, KEYWORD_MAP["happiness_negative"]["kw"])

        raw_h = 8.0 * (hp - hn)
        raw_w = 8.0 * (wp - wn)
        raw_y = 8.0 * (yp - yn)

        raw_h = _clamp_delta(raw_h)
        raw_w = _clamp_delta(raw_w)
        raw_y = _clamp_delta(raw_y)

        r4 = reasoning_reward(action, state)
        scale = 1.0 + r4

        dh = raw_h * scale
        dw = raw_w * scale
        dy = raw_y * scale

        dh, dw, dy = phase_scale(state.phase, dh, dw, dy)
        return {
            "health_delta": _clamp_delta(dh),
            "wealth_delta": _clamp_delta(dw),
            "happiness_delta": _clamp_delta(dy),
        }
