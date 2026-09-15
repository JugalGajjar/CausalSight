from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class VLMBackend(ABC):
    """A video-language model: frames + prompt -> text."""

    name: str = "base"

    @abstractmethod
    def generate(self, frames: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> str: ...

    def generate_batch(self, frames_list: list[list[Image.Image]], prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        """Default: sequential. Backends override with true batched decoding."""
        return [self.generate(f, p, max_new_tokens) for f, p in zip(frames_list, prompts)]
