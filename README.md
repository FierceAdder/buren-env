---
title: Project Buren
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Project Buren — Life-Stage RL Environment
## Blog / Video

[Blog.md](https://github.com/FierceAdder/buren-env/blob/b017435f5613b045b924b5cf8c807c29416402f6/Blog.md)

## Problem

Most LLMs answer isolated multiple-choice questions well but struggle with **messy, longitudinal tradeoffs** where consequences unfold over decades. Project Buren targets that gap with a multi-turn simulator that forces **health, wealth, and happiness** reasoning without clean options, scored by **verifiable rubrics** instead of lookup tables.

## Environment

The agent receives a **structured observation**: current age and stats (health / wealth / happiness), optional **history** (last turns), a **messy first-person scenario** (no listed choices), and a **ready-to-use prompt** string. Each turn it outputs **free text**: chain-of-thought inside `<reasoning>` and a concise choice in `<decision>`. An episode ends when **age ≥ 70**, **any stat hits 0**, or **turns reach the horizon** (15).

## Reward Design

Four **independent** signals are combined: **r1** survival (catastrophic penalties below 10/25, else min-stat normalized), **r2** balance (penalize variance across the three stats), **r3** foresight (later-life outcomes weighted higher), **r4** reasoning quality (keyword rubric over CoT; capped below 1 so outcomes dominate). Weights: **0.35 r1 + 0.30 r2 + 0.20 r3 + 0.15 r4**, with a **3.0 per-turn cap**, **short-response penalty**, and a **simple consistency check** between reasoning and decision. Multi-objective structure makes naive reward hacking harder than optimizing a single proxy.

## Results

![Training reward curve](assets/reward_curve.png)  
*Caption: Episode reward logged during GRPO training (placeholder until you train).*

![Before vs after baseline](assets/before_after.png)  
*Caption: Mean return over 20 evaluation episodes before and after training (placeholder until you train).*

[Training Logs](https://github.com/FierceAdder/buren-env/blob/fccfdb7f02861c9cdc05d66cea65af65e421845c/assets/training_log.csv)

## Try It

[Project Buren on Hugging Face Spaces](https://huggingface.co/spaces/Ashmit0110/project-buren/)

## Training

[Open `training/colab_train.ipynb` in Google Colab](https://colab.research.google.com/github/FierceAdder/buren-env/blob/main/training/colab_train.ipynb) 
### Google Colab checklist

1. **GPU runtime** (T4 or better): Runtime → Change runtime type → Hardware accelerator → GPU.
2. **Hugging Face token (optional):** For gated models or higher Hub rate limits, add a Colab secret `HF_TOKEN`; cell 1 loads it into the environment.
3. **Project files:** Run cell 3 once after **git clone** or **upload** of `buren-env` to `/content/buren-env`. The notebook **does not** delete that folder every run (avoid `rm -rf` unless you want a clean clone).
4. **Do not `pip install torch` on Colab** in the middle of the stack: the notebook relies on Colab’s prebuilt CUDA PyTorch; reinstalling `torch` from pip often breaks GPU.
5. **Order:** install (cell 1) → optional Drive (2) → path check (3) → **server** (4) → optional **smoke** (4b) → **train** (6).
6. **Unsloth:** If `import unsloth` fails, `train.py` still runs with **Transformers + PEFT** and `--hf-fallback-model` (default 0.5B). Demo cell 8 skips 7B inference if Unsloth is missing.
7. **Protobuf:** Cell 1 pins `protobuf>=5.29.1,<6` to avoid Colab’s old `protobuf` conflicting with `wandb` / `grpc` / Google client libraries.


## How to Run Locally

```bash
git clone https://huggingface.co/spaces/Ashmit0110/project-buren/
cd buren-env && pip install -r requirements.txt   # or: pip install -r requirements-frozen.txt
PYTHONPATH=. python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
# equivalent: uv run server
```

**Colab smoke test:** After dependencies and `PYTHONPATH=.` from the repo root, run `python training/smoke_colab.py --base-url http://127.0.0.1:7860` (with the server already up) or add `--launch-server` to spawn uvicorn. Exit code **0** means HTTP env, tokenizer, and TRL GRPO imports are healthy.

**Training:** On **Linux + NVIDIA GPU**, install Unsloth (`pip install unsloth`) for the full **7B 4-bit** run. On **macOS**, Unsloth is often unavailable or not useful; `training/train.py` will **automatically fall back** to **Transformers + PEFT** with a small model (`Qwen/Qwen2.5-0.5B-Instruct` by default) so you can exercise the GRPO loop locally. For the real hackathon training, use **Colab** or a **cloud GPU** with Unsloth. The harmless `multiprocess` `ResourceTracker` warning on exit is a known Python 3.12 quirk and can be ignored.

**Protobuf on Colab:** If `pip` warns about `protobuf 3.20.3` vs `wandb` / `google-*` / `grpcio-status`, reinstall: `pip install "protobuf>=5.29.1,<6"` (already included in `requirements.txt` and the Colab notebook install cell).

**HF token:** Run **`hf auth login`** (install/upgrade `huggingface_hub`; the executable may live under `python3 -m pip install --user`’s bin path). Or set **`HF_TOKEN`** / **`HUGGING_FACE_HUB_TOKEN`** in the environment — no file edit needed. Do not paste `pip install ... # comment` on one line — the **`#` breaks pip** (run installs without inline comments).

## Deployment (OpenEnv)

```bash
huggingface-cli login
openenv validate --verbose
openenv push --repo-id ashmit0110/project-buren/
curl -f https://ashmit0110-project-buren.hf.space/health
```

