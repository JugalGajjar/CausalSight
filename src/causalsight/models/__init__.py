"""Model backends for evaluation and data pipelines. `load_backend(name)` picks one by id."""

from __future__ import annotations

from causalsight.models.base import VLMBackend


def load_backend(model: str, **kw) -> VLMBackend:
    if model == "dummy":
        from causalsight.models.dummy import DummyBackend

        return DummyBackend(**kw)
    from pathlib import Path

    if "qwen" in model.lower() or (Path(model) / "adapter_config.json").exists():
        from causalsight.models.qwen_vl import QwenVLBackend

        return QwenVLBackend(model, **kw)
    raise ValueError(f"no backend for model id {model!r}")
