"""Model backends for evaluation and data pipelines. `load_backend(name)` picks one by id."""

from __future__ import annotations

from causalsight.models.base import VLMBackend


def load_backend(model: str, **kw) -> VLMBackend:
    if model == "dummy":
        from causalsight.models.dummy import DummyBackend

        return DummyBackend(**kw)

    import json
    from pathlib import Path

    from transformers import AutoConfig

    adapter = Path(model) / "adapter_config.json"
    base_id = json.loads(adapter.read_text())["base_model_name_or_path"] if adapter.exists() else model
    model_type = AutoConfig.from_pretrained(base_id).model_type
    if model_type == "internvl":
        from causalsight.models.internvl import InternVLBackend

        return InternVLBackend(model, **kw)
    # Qwen2.5-VL checkpoints: ours, published ones built on it (VideoRFT-3B, VideoThinker-R1-3B,
    # Video-R1-7B), or our LoRA adapters on top of one.
    from causalsight.models.qwen_vl import QwenVLBackend

    return QwenVLBackend(model, **kw)
