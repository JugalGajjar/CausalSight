"""Deterministic stand-in backend so the harness can be tested without weights or GPUs."""

from __future__ import annotations

from PIL import Image

from causalsight.models.base import VLMBackend


class DummyBackend(VLMBackend):
    name = "dummy"

    def __init__(self, reply: str = "A", **_) -> None:
        self.reply = reply
        self.calls: list[tuple[int, str]] = []

    def generate(self, frames: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> str:
        self.calls.append((len(frames), prompt))
        return self.reply
