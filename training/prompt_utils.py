"""Prompt formatting and response parsing for the LLM agent."""

from __future__ import annotations

import re

from environment.state import BurenAction, BurenObservation


def chat_prompt_token_ids(tokenizer, messages: list) -> list[int]:
    """Stable int token ids across Transformers 4.x / 5.x (avoids str/mixed chat-template outputs)."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return [int(t) for t in ids]


def format_prompt(obs: BurenObservation) -> str:
    st = obs.state
    lines = [
        "You are playing Project Buren — a life simulator. Make one decision per turn.",
        "",
        f"Current age: {st.age} | Phase: {st.phase}",
        f"Health: {st.health:.1f}/100 | Wealth: {st.wealth:.1f}/100 | Happiness: {st.happiness:.1f}/100",
        f"Turn: {st.turn + 1}/{st.max_turns} (0-based turn index after this choice advances time)",
        "",
    ]
    hist = st.history[-3:] if st.history else []
    if hist:
        lines.append("Recent context (last up to 3 scenario snippets + your past decisions):")
        for i, h in enumerate(hist, 1):
            lines.append(f"  {i}. {h}")
        lines.append("")

    lines.extend(
        [
            "Current scenario (messy real life — no multiple choice):",
            obs.scenario_text,
            "",
            "Instructions:",
            "1) Think through tradeoffs across health, wealth, and happiness (chain-of-thought).",
            "2) State a clear decision in plain English.",
            "3) Explicitly connect your reasoning to your age and current life phase.",
            "",
            "Respond in exactly this format:",
            "<reasoning>",
            "[Think through health, wealth, happiness tradeoffs here. Consider your age "
            "and life phase. Acknowledge what you're sacrificing.]",
            "</reasoning>",
            "<decision>",
            "[State your decision clearly in 1–2 sentences]",
            "</decision>",
        ]
    )
    return "\n".join(lines)


def parse_response(response: str) -> BurenAction:
    """Extract reasoning/decision tags; never raises."""
    raw = response or ""
    reasoning = ""
    decision = ""
    try:
        m_r = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", raw, re.DOTALL | re.IGNORECASE)
        if m_r:
            reasoning = (m_r.group(1) or "").strip()
        m_d = re.search(r"<decision>\s*(.*?)\s*</decision>", raw, re.DOTALL | re.IGNORECASE)
        if m_d:
            decision = (m_d.group(1) or "").strip()
    except Exception:
        pass
    return BurenAction(reasoning=reasoning, decision=decision, raw_response=raw)
