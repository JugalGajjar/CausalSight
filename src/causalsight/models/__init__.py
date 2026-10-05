"""Model backends for evaluation and data pipelines. `load_backend(name)` picks one by id."""

from __future__ import annotations

from causalsight.models.base import VLMBackend


def load_backend(model: str, **kw) -> VLMBackend:
    if model == "dummy":
        from causalsight.models.dummy import DummyBackend

        return DummyBackend(**kw)

    # Every real model we evaluate is a Qwen2.5-VL checkpoint (ours, or published ones built on it:
    # VideoRFT-3B, VideoThinker-R1-3B, Video-R1-7B), or one of our LoRA adapters on top of one.
    from causalsight.models.qwen_vl import QwenVLBackend

    return QwenVLBackend(model, **kw)
