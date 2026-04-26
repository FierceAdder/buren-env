#!/usr/bin/env python3
"""Fast smoke test for Google Colab (or any machine): Buren server, client, TRL, HF tokenizer.

Exits 0 if checks pass, 1 otherwise.

Examples::

    cd buren-env && PYTHONPATH=. python training/smoke_colab.py --base-url http://127.0.0.1:7860

    PYTHONPATH=. python training/smoke_colab.py --launch-server
"""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time
from pathlib import Path


def _silence_multiprocess_resource_tracker() -> None:
    """Suppress the harmless Python 3.12 + multiprocess ResourceTracker.__del__ traceback."""
    try:
        from multiprocess.resource_tracker import ResourceTracker  # type: ignore[import-untyped]

        _orig_del = ResourceTracker.__del__

        def _quiet_del(self: object) -> None:
            try:
                _orig_del(self)
            except Exception:
                pass

        ResourceTracker.__del__ = _quiet_del  # type: ignore[method-assign]
    except Exception:
        pass


atexit.register(_silence_multiprocess_resource_tracker)

ROOT = Path(__file__).resolve().parent.parent


def _ok(msg: str) -> None:
    print(f"[smoke OK] {msg}")


def _fail(msg: str) -> None:
    print(f"[smoke FAIL] {msg}", file=sys.stderr)


def _patch_transformers_wandb_probe_for_trl() -> None:
    if os.environ.get("BUREN_ALLOW_WANDB", "").lower() in ("1", "true", "yes"):
        return
    try:
        import transformers.integrations.integration_utils as _iu
    except Exception:
        return

    def _wandb_unavailable() -> bool:
        return False

    _iu.is_wandb_available = _wandb_unavailable  # type: ignore[method-assign]
    try:
        import transformers.utils as _tu

        if callable(getattr(_tu, "is_wandb_available", None)):
            _tu.is_wandb_available = _wandb_unavailable  # type: ignore[method-assign]
    except Exception:
        pass


def _wait_health(base_url: str, seconds: float = 30.0) -> None:
    from client.client import BurenClient

    c = BurenClient(base_url)
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            if c.health().get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server health check failed for {base_url} within {seconds}s")


def main() -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Buren Colab / CI smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Spawn uvicorn on 127.0.0.1:7860 (use if server is not already running)",
    )
    parser.add_argument("--no-trl", action="store_true", help="Skip TRL GRPO import")
    parser.add_argument("--no-hf", action="store_true", help="Skip HuggingFace tokenizer download")
    parser.add_argument(
        "--hf-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Public model id for tokenizer-only check",
    )
    args = parser.parse_args()
    proc: subprocess.Popen | None = None

    print("--- torch ---")
    import torch

    _ok(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _ok(str(torch.cuda.get_device_name(0)))

    print("--- unsloth (optional; import before transformers when present) ---")
    try:
        import unsloth  # noqa: F401

        _ok("unsloth importable")
    except Exception as _e:
        print(f"[smoke SKIP] unsloth not usable ({type(_e).__name__}: {_e}) — HF+PEFT fallback in train.py")

    if args.launch_server:
        print("--- uvicorn ---")
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "7860"],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )

    try:
        print("--- HTTP /health ---")
        _wait_health(args.base_url)
        _ok(args.base_url)

        print("--- client: reset, step, step_from_state ---")
        from client.client import BurenClient
        from environment.state import BurenAction

        client = BurenClient(args.base_url)
        obs = client.reset(seed=42, starting_age=30)
        if not obs.prompt.strip() or not obs.scenario_text.strip():
            _fail("reset returned empty prompt or scenario")
            return 1
        _ok("POST /reset")

        action = BurenAction(
            reasoning="Smoke test: balance short-term and long-term.",
            decision="Keep steady savings and avoid big risks.",
            raw_response="",
        )
        obs2, reward, done = client.step(action)
        if obs2.state.turn < 1:
            _fail("expected state.turn >= 1 after one step")
            return 1
        _ok(f"POST /step reward={reward:.4f} done={done}")

        obs0 = client.reset(seed=99, starting_age=25)
        obs3, r3, d3 = client.step_from_state(obs0.state, obs0.scenario_text, action)
        _ok(f"POST /step_from_state reward={r3:.4f} done={d3}")

        print("--- training.prompt_utils ---")
        from training import prompt_utils

        parsed = prompt_utils.parse_response(
            "<reasoning>thinking</reasoning><decision>proceed with caution</decision>"
        )
        if "proceed" not in parsed.decision.lower():
            _fail("parse_response did not extract decision")
            return 1
        _ok("parse_response")

        if not args.no_hf:
            print("--- HuggingFace tokenizer + chat template ---")
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
            msgs = [{"role": "user", "content": "Buren smoke: reply with tags."}]
            ids = prompt_utils.chat_prompt_token_ids(tok, msgs)
            if len(ids) < 4:
                _fail("chat_prompt_token_ids unexpectedly short")
                return 1
            _ok(f"AutoTokenizer + chat_prompt_token_ids ({len(ids)} token ids)")

        if not args.no_trl:
            print("--- TRL (GRPO) ---")
            _patch_transformers_wandb_probe_for_trl()
            from trl import GRPOConfig, GRPOTrainer  # noqa: F401

            _ok("import GRPOConfig, GRPOTrainer")

        print("--- all smoke checks passed ---")
        return 0
    except Exception as e:
        _fail(str(e))
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
