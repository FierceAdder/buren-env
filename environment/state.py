"""Pydantic models for Buren session state, observations, and actions."""

from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State as OpenEnvState
from pydantic import Field, model_validator

Phase = Literal["early", "mid", "late"]


def phase_from_age(age: int) -> Phase:
    """Life phase from age: early <35, mid <55, else late."""
    if age < 35:
        return "early"
    if age < 55:
        return "mid"
    return "late"


def phase_scale(phase: Phase, health_delta: float, wealth_delta: float, happiness_delta: float) -> tuple[float, float, float]:
    """Scale stat deltas by life phase (applied once in verifier)."""
    if phase == "early":
        return health_delta * 0.7, wealth_delta * 1.3, happiness_delta * 1.0
    if phase == "mid":
        return health_delta, wealth_delta, happiness_delta
    # late
    return health_delta * 1.4, wealth_delta * 1.0, happiness_delta * 1.3


class BurenState(OpenEnvState):
    """Full game state + OpenEnv episode fields."""

    age: int = Field(default=25, ge=20, le=70)
    health: float = Field(default=60.0, ge=0, le=100)
    wealth: float = Field(default=60.0, ge=0, le=100)
    happiness: float = Field(default=60.0, ge=0, le=100)
    phase: Phase = "early"
    turn: int = Field(default=0, ge=0)
    max_turns: int = Field(default=15, ge=1)
    history: list[str] = Field(default_factory=list)
    done: bool = False

    @model_validator(mode="after")
    def _sync_phase_and_step_count(self) -> BurenState:
        object.__setattr__(self, "phase", phase_from_age(self.age))
        if self.step_count != self.turn:
            object.__setattr__(self, "step_count", self.turn)
        return self


class BurenObservation(Observation):
    """What the agent sees (structured + LLM-ready prompt)."""

    state: BurenState
    scenario_text: str
    prompt: str


class BurenAction(Action):
    """Free-text reasoning + decision from the LLM."""

    reasoning: str = ""
    decision: str = ""
    raw_response: str = ""
