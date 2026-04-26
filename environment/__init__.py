"""Project Buren — life-stage RL environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.env import BurenEnvironment
    from environment.state import BurenAction, BurenObservation, BurenState

__all__ = ["BurenEnvironment", "BurenAction", "BurenObservation", "BurenState"]


def __getattr__(name: str):
    if name == "BurenEnvironment":
        from environment.env import BurenEnvironment

        return BurenEnvironment
    if name in ("BurenAction", "BurenObservation", "BurenState"):
        from environment import state as _state

        return getattr(_state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
