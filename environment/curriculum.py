"""Adaptive curriculum: weakness tracking and starting-age schedule."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from environment.rewards import reasoning_reward
from environment.state import BurenAction, BurenState


@dataclass
class CurriculumManager:
    """Tracks episodes, detects weaknesses, biases scenarios after confidence streak."""

    window: int = 20
    _episode_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _states_hist: deque[list[BurenState]] = field(default_factory=lambda: deque(maxlen=20))
    _actions_hist: deque[list[BurenAction]] = field(default_factory=lambda: deque(maxlen=20))
    _weakness_streak: dict[str, int] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=random.Random)

    def track_episode(
        self,
        reward: float,
        state_history: list[BurenState],
        action_history: list[BurenAction],
    ) -> None:
        self._episode_rewards.append(reward)
        self._states_hist.append(state_history)
        self._actions_hist.append(action_history)

        weakness = self._detect_episode_weakness(state_history, action_history, reward)
        if weakness:
            for k in list(self._weakness_streak.keys()):
                if k != weakness:
                    self._weakness_streak[k] = 0
            self._weakness_streak[weakness] = self._weakness_streak.get(weakness, 0) + 1
        else:
            self._weakness_streak.clear()

    def _detect_episode_weakness(
        self,
        states: list[BurenState],
        actions: list[BurenAction],
        reward: float,
    ) -> str | None:
        if not states:
            return None

        last = states[-1]
        # survival_failure: ended before age 50 with done (episode cut short badly)
        if last.done and last.age < 50 and last.turn < last.max_turns:
            if min(last.health, last.wealth, last.happiness) <= 0 or last.age < 45:
                return "survival_failure"

        # wealth_bias: wealth consistently highest across trajectory
        if len(states) >= 3:
            highs = 0
            for s in states[-3:]:
                if s.wealth >= s.health and s.wealth >= s.happiness:
                    highs += 1
            if highs >= 3:
                return "wealth_bias"

        # health_neglect: health lowest most of the time
        if len(states) >= 3:
            lows = 0
            for s in states[-3:]:
                if s.health <= s.wealth and s.health <= s.happiness:
                    lows += 1
            if lows >= 3:
                return "health_neglect"

        # shallow_reasoning: low r4 on average for last few actions
        if actions:
            scores = [reasoning_reward(a, states[min(i, len(states) - 1)]) for i, a in enumerate(actions[-5:])]
            if scores and sum(scores) / len(scores) < 0.15:
                return "shallow_reasoning"

        if reward < 0 and last.turn < 5:
            return "survival_failure"

        return None

    def get_scenario_bias(self) -> str | None:
        for w, streak in self._weakness_streak.items():
            if streak >= 3:
                return w
        return None

    def _avg_reward(self) -> float:
        if not self._episode_rewards:
            return 0.0
        return sum(self._episode_rewards) / len(self._episode_rewards)

    def get_starting_age(self) -> int:
        avg = self._avg_reward()
        if avg < 0.3:
            return self._rng.randint(20, 25)
        if avg < 0.6:
            return self._rng.randint(20, 35)
        return self._rng.randint(20, 50)
