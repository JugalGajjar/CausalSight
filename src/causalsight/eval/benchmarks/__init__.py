"""Benchmark loaders. Each yields `Item`s and provides `score(item, prediction)`."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    id: str
    question_type: str
    video_path: Path
    prompt: str  # full prompt text shown to the model
    gold: str  # normalized gold: descriptive answer, or letters of correct options e.g. "AC"
    options: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Benchmark:
    name: str = "base"

    def items(self, split: str, limit: int | None = None) -> Iterator[Item]:
        raise NotImplementedError

    def score(self, item: Item, prediction: str) -> dict:
        """Returns per-item scores; must include 'correct' (0/1)."""
        raise NotImplementedError


def load_benchmark(name: str, root: Path) -> Benchmark:
    if name == "clevrer":
        from causalsight.eval.benchmarks.clevrer import ClevrerBenchmark

        return ClevrerBenchmark(root)
    if name == "causalvqa":
        from causalsight.eval.benchmarks.causalvqa import CausalVQABenchmark

        return CausalVQABenchmark(root)
    raise NotImplementedError(
        f"{name}: MMVU, Video-MME, TempCompass and Video-Holmes are run through lmms-eval for now; see notes/PLAN.md Phase B"
    )
