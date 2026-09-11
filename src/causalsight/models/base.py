from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class VLMBackend(ABC):
    """A video-language model: frames + prompt -> text."""

    name: str = "base"

    @abstractmethod
    def generate(self, frames: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> str: ...
