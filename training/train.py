#!/usr/bin/env python3
"""TRL + Unsloth GRPO training for Project Buren (requires env server on localhost:7860)."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

# Unsloth patches transformers/peft; must run before those imports (and before trl in main).
_UNSLOTH_IMPORT_OK = False
try:
    import unsloth  # noqa: F401, E402

    _UNSLOTH_IMPORT_OK = True
except Exception:
    pass

from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.client import BurenClient  # noqa: E402
from environment.curriculum import CurriculumManager  # noqa: E402
from environment.state import BurenAction  # noqa: E402
from training import prompt_utils  # noqa: E402

ROLLOUT_DEBUG: list[str] = []


def _unsloth_import_ok() -> bool:
    return _UNSLOTH_IMPORT_OK


def _load_unsloth_bundle(model_id: str):
    from unsloth import FastLanguageModel  # type: ignore  # noqa: WPS433

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    return model, tokenizer, device, "unsloth"


def _load_hf_peft_bundle(model_id: str):
    """Transformers + PEFT only (no Unsloth). For Mac CPU/MPS or when Unsloth is missing."""
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("mps")
        device = torch.device("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        device = torch.device("cpu")

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.eval()
    return model, tokenizer, device, "hf_peft"


def _prompt_key(prompt: list | Any) -> str:
    return json.dumps(prompt, sort_keys=True)


def _compute_token_logprobs(model, device, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
    if not completion_ids:
        return [0.0]
    full = prompt_ids + completion_ids
    input_ids = torch.tensor([full], device=device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
    lp: list[float] = []
    lp_len = len(prompt_ids)
    for t, tid in enumerate(completion_ids):
        pos = lp_len - 1 + t
        if pos < 0:
            lp.append(0.0)
            continue
        logp = torch.log_softmax(logits[pos], dim=-1)[tid].item()
        lp.append(float(logp))
    return lp


def env_reward_fn(
    prompts,
    completions,
    completion_ids,
    env_reward=None,
    env_health=None,
    env_wealth=None,
    env_happiness=None,
    env_age=None,
    log_metric=None,
    **kwargs: Any,
):
    del prompts, completions, completion_ids, kwargs
    if env_reward is None:
        return [0.0]
    if log_metric is not None and env_reward:
        log_metric("episode_reward", float(sum(env_reward) / len(env_reward)))
        if env_health:
            log_metric("rollout/mean_health", float(sum(env_health) / len(env_health)))
        if env_wealth:
            log_metric("rollout/mean_wealth", float(sum(env_wealth) / len(env_wealth)))
        if env_happiness:
            log_metric("rollout/mean_happiness", float(sum(env_happiness) / len(env_happiness)))
        if env_age:
            log_metric("rollout/mean_age", float(sum(env_age) / len(env_age)))
    return env_reward


def make_rollout_func(base_url: str, curriculum: CurriculumManager):
    """GRPO rollout: chunk prompts into groups of num_generations, each group gets a fresh env reset."""

    def rollout_func(prompts: list, trainer):
        tokenizer = trainer.processing_class
        model = trainer.model
        device = trainer.accelerator.device
        num_gen = trainer.num_generations
        max_new = min(trainer.args.max_completion_length, 1024)

        prompt_ids_out: list[list[int]] = []
        completion_ids_out: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        env_reward_out: list[float] = []
        env_health_out: list[float] = []
        env_wealth_out: list[float] = []
        env_happiness_out: list[float] = []
        env_age_out: list[float] = []

        client = BurenClient(base_url)
        model.eval()
        ROLLOUT_DEBUG.clear()

        for grp_idx, chunk_start in enumerate(range(0, len(prompts), num_gen)):
            chunk = prompts[chunk_start : chunk_start + num_gen]

            seed = (trainer.state.global_step * 10007 + grp_idx * 30011) % (2**31)
            start_age = curriculum.get_starting_age()
            obs = client.reset(seed=seed, starting_age=start_age)
            scenario = obs.scenario_text
            state0 = obs.state
            user_text = obs.prompt
            messages = [{"role": "user", "content": user_text}]
            pid_list = prompt_utils.chat_prompt_token_ids(tokenizer, messages)

            for j in range(len(chunk)):
                prompt_ids_out.append(pid_list)
                input_ids = torch.tensor([pid_list], device=device, dtype=torch.long)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_new,
                        do_sample=True,
                        temperature=0.85,
                        top_p=0.95,
                        pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                gen = out[0].tolist()
                comp = gen[len(pid_list) :]
                text = tokenizer.decode(comp, skip_special_tokens=True)
                action = prompt_utils.parse_response(text)
                next_obs, r, _ = client.step_from_state(state0, scenario, action)
                env_reward_out.append(float(r))
                env_health_out.append(float(next_obs.state.health))
                env_wealth_out.append(float(next_obs.state.wealth))
                env_happiness_out.append(float(next_obs.state.happiness))
                env_age_out.append(float(next_obs.state.age))
                logprobs_out.append(_compute_token_logprobs(model, device, pid_list, comp))
                completion_ids_out.append(comp)
                if random.random() < 0.05:
                    ROLLOUT_DEBUG.append(
                        f"reward={r:.3f} age={next_obs.state.age} "
                        f"H={next_obs.state.health:.1f} W={next_obs.state.wealth:.1f} "
                        f"J={next_obs.state.happiness:.1f}\n{text[:1200]}"
                    )

        return {
            "prompt_ids": prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
            "env_reward": env_reward_out,
            "env_health": env_health_out,
            "env_wealth": env_wealth_out,
            "env_happiness": env_happiness_out,
            "env_age": env_age_out,
        }

    return rollout_func


class BurenTrainCallback(TrainerCallback):
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.reward_curve: list[float] = []
        self.health_curve: list[float] = []
        self.wealth_curve: list[float] = []
        self.happiness_curve: list[float] = []
        self.age_curve: list[float] = []
        self._csv_path = assets_dir / "training_log.csv"
        self._csv_initialized = False

    def _init_csv(self) -> None:
        if not self._csv_initialized:
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "reward", "health", "wealth", "happiness", "age"])
            self._csv_initialized = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        if not logs:
            return
        reward = float(logs.get("episode_reward", logs.get("reward", float("nan"))))
        health = float(logs.get("rollout/mean_health", float("nan")))
        wealth = float(logs.get("rollout/mean_wealth", float("nan")))
        happiness = float(logs.get("rollout/mean_happiness", float("nan")))
        age = float(logs.get("rollout/mean_age", float("nan")))

        if not (reward != reward):  # not NaN
            self.reward_curve.append(reward)
        if not (health != health):
            self.health_curve.append(health)
        if not (wealth != wealth):
            self.wealth_curve.append(wealth)
        if not (happiness != happiness):
            self.happiness_curve.append(happiness)
        if not (age != age):
            self.age_curve.append(age)

        step = state.global_step
        h_str = f"{health:.1f}" if not (health != health) else "—"
        w_str = f"{wealth:.1f}" if not (wealth != wealth) else "—"
        j_str = f"{happiness:.1f}" if not (happiness != happiness) else "—"
        a_str = f"{age:.0f}" if not (age != age) else "—"
        r_str = f"{reward:.4f}" if not (reward != reward) else "—"
        print(
            f"[Buren] Step {step}: reward={r_str}  age={a_str}  "
            f"H={h_str}  W={w_str}  J={j_str}"
        )

        self._init_csv()
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([step, r_str, h_str, w_str, j_str, a_str])

    def on_step_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        if state.global_step > 0 and state.global_step % 50 == 0 and ROLLOUT_DEBUG:
            k = min(3, len(ROLLOUT_DEBUG))
            print(f"[Buren] Step {state.global_step} — {k} sample rollouts:")
            for s in random.sample(ROLLOUT_DEBUG, k):
                print("---\n", s[:2000])
        if state.global_step > 0 and state.global_step % 100 == 0 and self.reward_curve:
            self._save_reward_curve()

    def _save_reward_curve(self) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(self.reward_curve)
        plt.xlabel("Training Step")
        plt.ylabel("Episode Reward")
        plt.tight_layout()
        p = self.assets_dir / "reward_curve.png"
        plt.savefig(p)
        plt.close()
        print(f"[Buren] Saved {p}")

    def save_stats_curve(self) -> None:
        """Save a 4-subplot figure of H/W/J/Age over training steps."""
        curves = [
            (self.health_curve, "Health", "tab:red"),
            (self.wealth_curve, "Wealth", "tab:green"),
            (self.happiness_curve, "Happiness", "tab:blue"),
            (self.age_curve, "Mean Age", "tab:orange"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (data, label, color) in zip(axes.flat, curves):
            if data:
                ax.plot(data, color=color)
            ax.set_title(label)
            ax.set_xlabel("Step")
            ax.set_ylabel(label)
        fig.suptitle("Rollout Stats over Training", fontsize=13)
        fig.tight_layout()
        p = self.assets_dir / "stats_curve.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"[Buren] Saved {p}")


def run_episode_with_model(
    model, tokenizer, client: BurenClient, curriculum: CurriculumManager, device, seed: int
) -> dict:
    """Run one full episode and return a rich stats dict."""
    start_age = curriculum.get_starting_age()
    obs = client.reset(seed=seed, starting_age=start_age)
    actual_start_age = obs.state.age
    total = 0.0
    steps = 0
    transcript: list[str] = []
    while not obs.done and steps < 25:
        messages = [{"role": "user", "content": obs.prompt}]
        pids = prompt_utils.chat_prompt_token_ids(tokenizer, messages)
        input_ids = torch.tensor([pids], device=device, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = out[0].tolist()[len(pids) :]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        if len(transcript) < 3:
            transcript.append(
                f"  [age {obs.state.age}] scenario: {obs.scenario_text[:200]!r}\n"
                f"  response: {text[:400]!r}"
            )
        act = prompt_utils.parse_response(text)
        obs, r, d = client.step(act)
        total += float(r)
        steps += 1
        if d:
            break
    final_state = obs.state
    return {
        "total_reward": total,
        "start_age": actual_start_age,
        "final_age": final_state.age,
        "steps": steps,
        "survived": final_state.age >= 70,
        "final_health": final_state.health,
        "final_wealth": final_state.wealth,
        "final_happiness": final_state.happiness,
        "transcript": transcript,
    }


def _baseline_summary(label: str, episodes: list[dict]) -> None:
    """Print per-episode lines and a summary block to stdout."""
    n = len(episodes)
    for i, ep in enumerate(episodes):
        survived_str = "YES" if ep["survived"] else "no"
        print(
            f"  Ep {i + 1:>2}/{n}: age {ep['start_age']}→{ep['final_age']} "
            f"({ep['steps']} steps)  reward={ep['total_reward']:.3f}  "
            f"H={ep['final_health']:.1f} W={ep['final_wealth']:.1f} J={ep['final_happiness']:.1f}  "
            f"survived={survived_str}"
        )
    if not episodes:
        return
    avg_reward = sum(e["total_reward"] for e in episodes) / n
    avg_lifetime = sum(e["final_age"] for e in episodes) / n
    min_age = min(e["final_age"] for e in episodes)
    max_age = max(e["final_age"] for e in episodes)
    survival_pct = 100.0 * sum(1 for e in episodes if e["survived"]) / n
    avg_h = sum(e["final_health"] for e in episodes) / n
    avg_w = sum(e["final_wealth"] for e in episodes) / n
    avg_j = sum(e["final_happiness"] for e in episodes) / n
    print(
        f"[Buren] {label} summary ({n} eps):\n"
        f"  Avg reward:      {avg_reward:.3f}\n"
        f"  Avg lifetime:    {avg_lifetime:.1f} yrs  (range {min_age}–{max_age})\n"
        f"  Survival rate:   {survival_pct:.1f}%\n"
        f"  Avg final H/W/J: {avg_h:.1f} / {avg_w:.1f} / {avg_j:.1f}"
    )
    if episodes:
        sample = random.choice(episodes)
        print(
            f"  — Sample transcript (ep with age {sample['start_age']}→{sample['final_age']}):"
        )
        for line in sample["transcript"]:
            print(line)


def run_baseline(model, tokenizer, client, curriculum, device, n: int = 20) -> list[dict]:
    """Run n evaluation episodes; print per-episode + summary stats. Returns list of episode dicts."""
    episodes: list[dict] = []
    for ep in range(n):
        result = run_episode_with_model(model, tokenizer, client, curriculum, device, seed=ep + 17)
        episodes.append(result)
    return episodes


def _ep_mean(episodes: list[dict], key: str) -> float:
    vals = [e[key] for e in episodes]
    return sum(vals) / max(1, len(vals))


def plot_before_after(before_eps: list[dict], after_eps: list[dict], path: Path) -> None:
    """Grouped bar chart: reward, H, W, J, lifetime, survival — before vs after."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _surv(eps: list[dict]) -> float:
        return 100.0 * sum(1 for e in eps if e["survived"]) / max(1, len(eps))

    metrics = [
        ("Reward", "total_reward", None),
        ("Health", "final_health", None),
        ("Wealth", "final_wealth", None),
        ("Happiness", "final_happiness", None),
        ("Lifetime (yrs)", "final_age", None),
    ]

    fig, axes = plt.subplots(1, len(metrics) + 1, figsize=(16, 4))
    colors = ["#4C72B0", "#55A868"]

    for ax, (title, key, _) in zip(axes[:-1], metrics):
        bv = _ep_mean(before_eps, key)
        av = _ep_mean(after_eps, key)
        ax.bar(["Before", "After"], [bv, av], color=colors, width=0.5)
        ax.set_title(title)
        ax.set_ylim(0, max(bv, av) * 1.25 + 1e-6)
        for rect, val in zip(ax.patches, [bv, av]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax_surv = axes[-1]
    bsurv = _surv(before_eps)
    asurv = _surv(after_eps)
    ax_surv.bar(["Before", "After"], [bsurv, asurv], color=colors, width=0.5)
    ax_surv.set_title("Survival %")
    ax_surv.set_ylim(0, 105)
    for rect, val in zip(ax_surv.patches, [bsurv, asurv]):
        ax_surv.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Before vs After Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_baseline_stats(before_eps: list[dict], after_eps: list[dict], path: Path) -> None:
    """Grouped bar chart of avg final H/W/J before vs after, saved to assets/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Health", "Wealth", "Happiness"]
    keys = ["final_health", "final_wealth", "final_happiness"]
    before_vals = [_ep_mean(before_eps, k) for k in keys]
    after_vals = [_ep_mean(after_eps, k) for k in keys]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_b = ax.bar([i - width / 2 for i in x], before_vals, width, label="Before", color="#4C72B0")
    bars_a = ax.bar([i + width / 2 for i in x], after_vals, width, label="After", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Mean final value")
    ax.set_title("Avg final Health / Wealth / Happiness — Before vs After")
    ax.legend()
    for bar in (*bars_b, *bars_a):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def launch_server() -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "7860"],
        cwd=str(ROOT),
        env=env,
    )


def _patch_transformers_wandb_probe_for_trl() -> None:
    """TRL's GRPO stack imports profiling, which calls transformers' `is_wandb_available()`.

    Colab images sometimes ship a broken `wandb` (proto mismatch); the probe then raises
    while importing `trl`. Training uses ``report_to=\"none\"``, so skipping the probe is safe.
    Set ``BUREN_ALLOW_WANDB=1`` to restore default behavior (after ``pip install -U wandb``).
    """
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


def wait_for_health(base_url: str, seconds: float = 30.0):
    c = BurenClient(base_url)
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            if c.health().get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Server failed health check")


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-server", action="store_true", help="Spawn uvicorn subprocess")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--skip-train", action="store_true", help="Only baselines / plots smoke test")
    parser.add_argument(
        "--require-unsloth",
        action="store_true",
        help="Fail if Unsloth is not installed (default: auto-fallback to HF+PEFT on Mac / without GPU stack)",
    )
    parser.add_argument(
        "--unsloth-model",
        default="unsloth/Qwen2.5-7B-Instruct",
        help="Model id when using Unsloth (Linux + NVIDIA GPU typical)",
    )
    parser.add_argument(
        "--hf-fallback-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Smaller HF model when Unsloth is unavailable (dev / Mac CPU-MPS)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast verification run: 3 baseline eps, 16 train rows, 1 epoch, 4 gens, 128 max tokens",
    )
    parser.add_argument("--train-rows", type=int, default=None, help="Override number of training rows")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--num-generations", type=int, default=None, help="Override num_generations")
    parser.add_argument("--max-completion-length", type=int, default=None, help="Override max_completion_length")
    parser.add_argument("--baseline-eps", type=int, default=None, help="Override baseline episode count")
    args_ns = parser.parse_args()

    proc: subprocess.Popen | None = None
    if args_ns.launch_server:
        proc = launch_server()
        wait_for_health(args_ns.base_url)

    _patch_transformers_wandb_probe_for_trl()
    try:
        from trl import GRPOConfig, GRPOTrainer  # noqa: WPS433
    except ImportError as e:
        raise SystemExit("Install trl: pip install trl") from e

    curriculum = CurriculumManager()
    client = BurenClient(args_ns.base_url)
    assets_dir = ROOT / "assets"

    if args_ns.require_unsloth:
        if not _unsloth_import_ok():
            raise SystemExit(
                "Unsloth is required (--require-unsloth) but not installed.\n"
                "  pip install unsloth\n"
                "Full 7B 4-bit training needs Linux + NVIDIA GPU; on macOS use Colab or --hf-fallback-model."
            )
        model, tokenizer, device, backend = _load_unsloth_bundle(args_ns.unsloth_model)
    elif _unsloth_import_ok():
        model, tokenizer, device, backend = _load_unsloth_bundle(args_ns.unsloth_model)
    else:
        print(
            "[Buren] Unsloth not installed — using Transformers + PEFT fallback "
            f"({args_ns.hf_fallback_model}).\n"
            "  For the hackathon 7B run: Linux/CUDA + `pip install unsloth` or use Colab.\n"
            "  To force Unsloth only: pass --require-unsloth (will error if missing).\n"
        )
        model, tokenizer, device, backend = _load_hf_peft_bundle(args_ns.hf_fallback_model)

    # --- Resolve hyperparams (--quick overrides defaults) ---
    if args_ns.quick:
        baseline_n = args_ns.baseline_eps or 3
        train_rows = args_ns.train_rows or 16
        n_epochs = args_ns.num_epochs or 1
        n_gens = args_ns.num_generations or 4
        max_comp = args_ns.max_completion_length or 128
        print(f"[Buren] --quick mode: {baseline_n} baseline eps, {train_rows} rows, "
              f"{n_epochs} epoch(s), {n_gens} gens, {max_comp} max tokens")
    else:
        baseline_n = args_ns.baseline_eps or 20
        train_rows = args_ns.train_rows or 256
        n_epochs = args_ns.num_epochs or 3
        n_gens = args_ns.num_generations or 8
        max_comp = args_ns.max_completion_length or 512

    # --- Baseline before ---
    print(f"[Buren] Baseline (before training), {baseline_n} episodes...")
    before_eps = run_baseline(model, tokenizer, client, curriculum, device, n=baseline_n)
    _baseline_summary("Before-training", before_eps)
    for ep in before_eps:
        curriculum.track_episode(ep["total_reward"], [], [])

    if args_ns.skip_train:
        print("skip-train set; exiting after before baseline")
        return

    placeholder = [{"role": "user", "content": "Buren RLVR: use the scenario in the user message and answer with tags."}]
    train_dataset = Dataset.from_list([{"prompt": placeholder} for _ in range(train_rows)])

    cb = BurenTrainCallback(assets_dir)
    rollout_fn = make_rollout_func(args_ns.base_url, curriculum)

    training_args = GRPOConfig(
        output_dir=str(ROOT / "buren-trained"),
        num_train_epochs=n_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4 if args_ns.quick else 8,
        learning_rate=5e-6,
        logging_steps=1 if args_ns.quick else 5,
        save_steps=50,
        num_generations=n_gens,
        max_completion_length=max_comp,
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=env_reward_fn,
        train_dataset=train_dataset,
        args=training_args,
        rollout_func=rollout_fn,
        callbacks=[cb],
    )

    try:
        trainer.train()

        merged_dir = ROOT / "buren-trained-merged"
        if backend == "unsloth":
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        else:
            model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            print(f"[Buren] Saved PEFT adapter + tokenizer to {merged_dir} (use merge_and_unload locally if needed)")

        # Reward curve from callback
        if cb.reward_curve:
            cb._save_reward_curve()

        # Stats curve (H/W/J/Age over training steps)
        if cb.health_curve or cb.age_curve:
            cb.save_stats_curve()

        # training_log.csv already written incrementally by callback

        # --- Baseline after ---
        print(f"[Buren] Baseline (after training), {baseline_n} episodes...")
        after_eps = run_baseline(model, tokenizer, client, curriculum, device, n=baseline_n)
        _baseline_summary("After-training", after_eps)

        # Plots
        plot_before_after(before_eps, after_eps, assets_dir / "before_after.png")
        print(f"[Buren] Saved {assets_dir / 'before_after.png'}")
        plot_baseline_stats(before_eps, after_eps, assets_dir / "baseline_stats.png")
        print(f"[Buren] Saved {assets_dir / 'baseline_stats.png'}")

        # Final summary table
        def _pct(eps: list[dict]) -> float:
            return 100.0 * sum(1 for e in eps if e["survived"]) / max(1, len(eps))

        def _mean(eps: list[dict], k: str) -> float:
            return sum(e[k] for e in eps) / max(1, len(eps))

        b_r = _mean(before_eps, "total_reward")
        a_r = _mean(after_eps, "total_reward")
        b_lt = _mean(before_eps, "final_age")
        a_lt = _mean(after_eps, "final_age")
        b_sv = _pct(before_eps)
        a_sv = _pct(after_eps)
        b_h = _mean(before_eps, "final_health")
        a_h = _mean(after_eps, "final_health")
        b_w = _mean(before_eps, "final_wealth")
        a_w = _mean(after_eps, "final_wealth")
        b_j = _mean(before_eps, "final_happiness")
        a_j = _mean(after_eps, "final_happiness")

        print(
            "\n" + "=" * 50 + "\n"
            "  Project Buren — Training Complete\n" +
            "=" * 50 + "\n"
            f"  {'':12s} {'Reward':>8}  {'Lifetime':>10}  {'Survival':>10}  {'H/W/J':>15}\n"
            f"  {'Before':12s} {b_r:>8.3f}  {b_lt:>8.1f}yr  {b_sv:>8.1f}%   "
            f"{b_h:.1f}/{b_w:.1f}/{b_j:.1f}\n"
            f"  {'After':12s} {a_r:>8.3f}  {a_lt:>8.1f}yr  {a_sv:>8.1f}%   "
            f"{a_h:.1f}/{a_w:.1f}/{a_j:.1f}\n"
            f"  {'Delta':12s} {a_r - b_r:>+8.3f}  {a_lt - b_lt:>+8.1f}yr  "
            f"{a_sv - b_sv:>+8.1f}%   "
            f"{a_h - b_h:+.1f}/{a_w - b_w:+.1f}/{a_j - b_j:+.1f}\n" +
            "=" * 50
        )
    finally:
        if proc is not None:
            proc.terminate()

    print("[Buren] Done.")


if __name__ == "__main__":
    main()
