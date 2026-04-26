"""REST client for Buren environment (no server imports; not OpenEnv WebSocket)."""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from environment.state import BurenAction, BurenObservation, BurenState


class BurenClient:
    """HTTP client with retries and exponential backoff on connection failures."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = self.session.request(method, url, timeout=30, **kwargs)
                r.raise_for_status()
                return r
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_err = e
                time.sleep(0.5 * (2**attempt))
        assert last_err is not None
        raise last_err

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        starting_age: int | None = None,
    ) -> BurenObservation:
        payload: dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if starting_age is not None:
            payload["starting_age"] = starting_age
        r = self._request("POST", "/reset", json=payload)
        return BurenObservation.model_validate(r.json())

    def step(self, action: BurenAction) -> tuple[BurenObservation, float, bool]:
        r = self._request("POST", "/step", json=action.model_dump(mode="json"))
        data = r.json()
        obs = BurenObservation.model_validate(data["observation"])
        return obs, float(data["reward"]), bool(data["done"])

    def step_from_state(
        self,
        state: BurenState,
        scenario_text: str,
        action: BurenAction,
    ) -> tuple[BurenObservation, float, bool]:
        body = {
            "state": state.model_dump(mode="json"),
            "scenario_text": scenario_text,
            "action": action.model_dump(mode="json"),
        }
        r = self._request("POST", "/step_from_state", json=body)
        data = r.json()
        obs = BurenObservation.model_validate(data["observation"])
        return obs, float(data["reward"]), bool(data["done"])

    def state(self) -> BurenState:
        r = self._request("GET", "/state")
        return BurenState.model_validate(r.json())

    def health(self) -> dict[str, str]:
        r = self._request("GET", "/health")
        return r.json()
