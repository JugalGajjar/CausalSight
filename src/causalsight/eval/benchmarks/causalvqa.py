"""CausalVQA loader (Foss et al., NeurIPS 2025 D&B). Fill in once the release format has been
inspected: data/raw/causalvqa/CausalVQA/ (clone) plus the video segments per its README."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from causalsight.eval.benchmarks import Benchmark, Item


class CausalVQABenchmark(Benchmark):
    name = "causalvqa"

    def __init__(self, root: Path) -> None:
        self.root = root

    def items(self, split: str = "test", limit: int | None = None) -> Iterator[Item]:
        raise NotImplementedError("inspect data/raw/causalvqa release format first (PLAN Phase B step 1)")

    def score(self, item: Item, prediction: str) -> dict:
        raise NotImplementedError
