"""FastAPI server for Project Buren (OpenEnv-compatible REST API)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from environment.env import BurenEnvironment
from environment.state import BurenAction, BurenObservation, BurenState

logger = logging.getLogger("buren.server")
logging.basicConfig(level=logging.INFO)

_env: BurenEnvironment | None = None


def get_environment() -> BurenEnvironment:
    global _env
    if _env is None:
        _env = BurenEnvironment()
    return _env


app = FastAPI(title="Project Buren", version="0.1.0")


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        return JSONResponse({"detail": "Request timeout (30s)"}, status_code=504)


class StepPayload(BaseModel):
    reasoning: str = ""
    decision: str = ""
    raw_response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResetBody(BaseModel):
    seed: int | None = None
    episode_id: str | None = None
    starting_age: int | None = Field(default=None, ge=20, le=70)


class StepFromStatePayload(BaseModel):
    state: dict[str, Any]
    scenario_text: str
    action: StepPayload


def _action_from_payload(p: StepPayload) -> BurenAction:
    return BurenAction(
        reasoning=p.reasoning,
        decision=p.decision,
        raw_response=p.raw_response,
        metadata=p.metadata,
    )


@app.post("/reset")
def post_reset(body: ResetBody | None = None):
    env = get_environment()
    b = body or ResetBody()
    obs = env.reset(seed=b.seed, episode_id=b.episode_id, starting_age=b.starting_age)
    return JSONResponse(content=obs.model_dump(mode="json"))


@app.post("/step")
def post_step(body: StepPayload):
    env = get_environment()
    action = _action_from_payload(body)
    obs = env.step(action)
    st = obs.state
    logger.info(
        "step turn=%s reward=%s health=%.2f wealth=%.2f happiness=%.2f done=%s",
        st.turn,
        obs.reward,
        st.health,
        st.wealth,
        st.happiness,
        obs.done,
    )
    return JSONResponse(
        content={
            "observation": obs.model_dump(mode="json"),
            "reward": float(obs.reward) if obs.reward is not None else 0.0,
            "done": bool(obs.done),
        }
    )


@app.post("/step_from_state")
def post_step_from_state(body: StepFromStatePayload):
    env = get_environment()
    st = BurenState.model_validate(body.state)
    action = _action_from_payload(body.action)
    obs, reward, done = env.step_from_state(st, body.scenario_text, action)
    logger.info(
        "step_from_state turn=%s reward=%s health=%.2f wealth=%.2f happiness=%.2f done=%s",
        obs.state.turn,
        reward,
        obs.state.health,
        obs.state.wealth,
        obs.state.happiness,
        done,
    )
    return JSONResponse(
        content={
            "observation": obs.model_dump(mode="json"),
            "reward": float(reward),
            "done": bool(done),
        }
    )


@app.get("/state")
def get_state():
    env = get_environment()
    return JSONResponse(content=env.state.model_dump(mode="json"))


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
