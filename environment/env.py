"""Core OpenEnv-compatible Buren environment."""

from __future__ import annotations

import random
import traceback
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from pydantic import ValidationError

from environment.curriculum import CurriculumManager
from environment.rewards import RewardCalculator
from environment.scenarios import ScenarioEngine
from environment.state import BurenAction, BurenObservation, BurenState, phase_from_age
from environment.verifier import RubricVerifier
from training.prompt_utils import format_prompt


class BurenEnvironment(Environment[BurenAction, BurenObservation, BurenState]):
    """Life-stage simulator with rubric-derived consequences (RLVR)."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        curriculum: CurriculumManager | None = None,
        rng: random.Random | None = None,
    ):
        super().__init__(transform=None, rubric=None)
        self._rng = rng or random.Random()
        self._curriculum = curriculum or CurriculumManager()
        self._engine = ScenarioEngine(
            rng=self._rng,
            bias_fn=self._curriculum.get_scenario_bias,
        )
        self._verifier = RubricVerifier()
        self._calculator = RewardCalculator()
        self._state: BurenState = BurenState()
        self._scenario_text: str = ""

    @property
    def state(self) -> BurenState:
        return self._state

    def _coerce_action(self, action: Any) -> tuple[BurenAction, bool]:
        """Return (action, ok). ok=False on validation failure."""
        try:
            if isinstance(action, BurenAction):
                return action, True
            if isinstance(action, dict):
                return BurenAction.model_validate(action), True
        except ValidationError:
            return BurenAction(reasoning="", decision="", raw_response=str(action)), False
        except Exception:
            return BurenAction(reasoning="", decision="", raw_response=str(action)), False
        return BurenAction(reasoning="", decision="", raw_response=str(action)), False

    def _build_observation(
        self,
        st: BurenState,
        scenario_text: str,
        reward: float | None = None,
        done: bool | None = None,
    ) -> BurenObservation:
        d = done if done is not None else st.done
        stub = BurenObservation(
            state=st,
            scenario_text=scenario_text,
            prompt=".",
            reward=None,
            done=d,
        )
        prompt = format_prompt(stub)
        return BurenObservation(
            state=st,
            scenario_text=scenario_text,
            prompt=prompt,
            reward=reward,
            done=d,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        starting_age: Optional[int] = None,
        **kwargs: Any,
    ) -> BurenObservation:
        if seed is not None:
            self._rng.seed(seed)
        if starting_age is not None:
            start_age = max(20, min(70, int(starting_age)))
        else:
            start_age = self._curriculum.get_starting_age()
        self._state = BurenState(
            age=start_age,
            health=60.0,
            wealth=60.0,
            happiness=60.0,
            phase=phase_from_age(start_age),
            turn=0,
            max_turns=15,
            history=[],
            done=False,
            episode_id=episode_id,
            step_count=0,
        )
        self._scenario_text = self._engine.causal_sample(self._state)
        return self._build_observation(self._state, self._scenario_text, reward=None, done=False)

    def _simulate_step(
        self,
        state: BurenState,
        scenario_text: str,
        action: Any,
    ) -> tuple[BurenObservation, float, bool]:
        act, ok = self._coerce_action(action)
        if not ok:
            st = state.model_copy(deep=True)
            obs = self._build_observation(st, scenario_text, reward=-0.5, done=st.done)
            obs.reward = -0.5
            return obs, -0.5, st.done

        try:
            deltas = self._verifier.derive_consequences(scenario_text, act, state)
            new_state = state.model_copy(deep=True)

            new_state.health = max(0.0, min(100.0, new_state.health + deltas["health_delta"]))
            new_state.wealth = max(0.0, min(100.0, new_state.wealth + deltas["wealth_delta"]))
            new_state.happiness = max(0.0, min(100.0, new_state.happiness + deltas["happiness_delta"]))

            dy = self._rng.randint(2, 4)
            new_state.age = min(70, new_state.age + dy)
            new_state.phase = phase_from_age(new_state.age)
            new_state.turn = state.turn + 1
            new_state.step_count = new_state.turn

            done = (
                new_state.age >= 70
                or new_state.health <= 0
                or new_state.wealth <= 0
                or new_state.happiness <= 0
                or new_state.turn >= new_state.max_turns
            )
            new_state.done = done

            reward = self._calculator.compute(act, new_state)

            snippet = f"{scenario_text[:160]} | Decision: {act.decision[:120]}"
            new_state.history = list(new_state.history) + [snippet]

            if done:
                next_scenario = "Episode ended."
            else:
                next_scenario = self._engine.causal_sample(new_state)

            obs = self._build_observation(new_state, next_scenario, reward=reward, done=done)
            return obs, reward, done
        except Exception:
            traceback.print_exc()
            st = state.model_copy(deep=True)
            obs = self._build_observation(st, scenario_text, reward=-0.5, done=st.done)
            obs.reward = -0.5
            return obs, -0.5, st.done

    def step(
        self,
        action: BurenAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BurenObservation:
        _ = timeout_s
        obs, _, _ = self._simulate_step(self._state, self._scenario_text, action)
        self._state = obs.state
        self._scenario_text = obs.scenario_text
        return obs

    def step_from_state(
        self,
        state: BurenState,
        scenario_text: str,
        action: Any,
    ) -> tuple[BurenObservation, float, bool]:
        """Pure transition for GRPO branching (does not mutate live env)."""
        obs, r, d = self._simulate_step(state, scenario_text, action)
        return obs, r, d

    def attach_curriculum(self, cm: CurriculumManager) -> None:
        """Replace curriculum and rewire scenario engine bias."""
        self._curriculum = cm
        self._engine = ScenarioEngine(rng=self._rng, bias_fn=self._curriculum.get_scenario_bias)
