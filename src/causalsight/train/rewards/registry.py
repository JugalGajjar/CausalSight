"""Named reward functions for the trainer: (completion_text, record) -> float in [0, 1]."""

from __future__ import annotations

import re

from causalsight.eval.benchmarks.answers import normalize_short, parse_letters
from causalsight.train.format import ANSWER_RE, parse_chain

FORMAT_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None


def outcome(text: str, rec: dict) -> float:
    ans = extract_answer(text)
    if ans is None:
        return 0.0
    if rec.get("problem_type") == "multiple choice":
        return float(parse_letters(ans, len(rec.get("options", []))) == rec["gold"])
    pred = normalize_short(ans)
    if (rec.get("subtype") == "exist" or rec["gold"] in ("yes", "no")) and pred.isdigit():
        pred = "no" if pred == "0" else "yes"
    return float(pred == rec["gold"])


def outcome_lenient(text: str, rec: dict) -> float:
    """Outcome scored on the <answer> content if present, else on the raw text: gives a learning signal
    before the model has learned the tags (the format reward then pulls it toward tags)."""
    ans = extract_answer(text)
    if ans is None:
        ans = text.strip().splitlines()[-1] if text.strip() else ""
    return outcome(f"<answer>{ans}</answer>", rec)


def format_tags(text: str, rec: dict) -> float:
    return float(bool(FORMAT_RE.match(text)))


def format_chain(text: str, rec: dict) -> float:
    """Chain format: parseable steps inside <think> and an <answer>; partial credit for tag structure."""
    p = parse_chain(text)
    if p.chain is not None:
        return 1.0
    return 0.2 if p.has_tags and p.answer is not None else 0.0


from causalsight.train.rewards.chain_rewards import grounding, grounding_obj, process

REWARDS = {
    "outcome": outcome,
    "outcome_lenient": outcome_lenient,
    "format": format_tags,
    "format_chain": format_chain,
    "grounding": grounding,
    "grounding_obj": grounding_obj,
    "process": process,
    "necessity": None,  # computed inside the trainer (needs the reference model); see GRPOTrainer._necessity
}
