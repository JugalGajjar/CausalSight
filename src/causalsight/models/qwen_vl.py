"""Qwen2.5-VL backend via transformers. Works on CUDA, and on Apple MPS for small smoke tests.

`model_id` may be a Hugging Face id or a local LoRA adapter directory (containing adapter_config.json,
as saved by cs-grpo / cs-sft); the base model named in the adapter config is loaded and the adapter is
merged in, so evaluation of a trained checkpoint is a drop-in `--model runs/stage0/adapter`.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from causalsight.models.base import VLMBackend


def resolve_init_adapters(adapter_dir: Path, explicit: str | None = None) -> list[Path]:
    """Adapters that must be merged before `adapter_dir`. Reads `init_adapter` from the run's config.yaml
    (written by cs-grpo next to the adapter directory). If that path does not exist on this machine (runs
    are trained on Colab and evaluated elsewhere), falls back to a sibling run directory of the same name."""
    if explicit:
        p = Path(explicit)
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(f"--init-adapter {p} has no adapter_config.json")
        return [p]
    cfg = adapter_dir.parent / "config.yaml"
    if not cfg.exists():
        return []
    import yaml

    init = (yaml.safe_load(cfg.read_text()) or {}).get("init_adapter")
    if not init:
        return []
    cands = [Path(init), adapter_dir.parent.parent / Path(init).parent.name / "adapter"]
    for c in cands:
        if (c / "adapter_config.json").exists():
            return [c]
    raise FileNotFoundError(f"{adapter_dir} was trained on top of {init}, which was not found (tried {[str(c) for c in cands]}); pass --init-adapter")


class QwenVLBackend(VLMBackend):
    def __init__(self, model_id: str, device: str | None = None, dtype: str = "bfloat16", max_pixels: int = 360 * 420, init_adapter: str | None = None, **_) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.name = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        adapter = Path(model_id) / "adapter_config.json"
        base_id = json.loads(adapter.read_text())["base_model_name_or_path"] if adapter.exists() else model_id
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch_dtype)
        if adapter.exists():
            from peft import PeftModel

            # A Stage 2 adapter was trained on top of merged Stage 1 weights: merge that one first.
            for init in resolve_init_adapters(Path(model_id), init_adapter):
                model = PeftModel.from_pretrained(model, str(init)).merge_and_unload()
                print(f"merged init adapter {init}")
            model = PeftModel.from_pretrained(model, model_id).merge_and_unload()
            print(f"loaded adapter {model_id} on {base_id} (merged)")
        self.model = model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(base_id)
        self.processor.tokenizer.padding_side = "left"  # required for batched generation
        self.max_pixels = max_pixels

    def _messages(self, frames: list[Image.Image], prompt: str) -> list[dict]:
        return [{"role": "user", "content": [{"type": "video", "video": frames, "max_pixels": self.max_pixels}, {"type": "text", "text": prompt}]}]

    def generate(self, frames: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> str:
        return self.generate_batch([frames], [prompt], max_new_tokens)[0]

    def generate_batch(self, frames_list: list[list[Image.Image]], prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        """Greedy decoding for a batch of (video, prompt) pairs in one generate call (left-padded)."""
        import torch
        from qwen_vl_utils import process_vision_info

        texts, videos = [], []
        for frames, prompt in zip(frames_list, prompts):
            msgs = self._messages(frames, prompt)
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            _imgs, vids = process_vision_info(msgs)
            videos.extend(vids)
        inputs = self.processor(text=texts, videos=videos, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.processor.tokenizer.pad_token_id)
        out = out[:, inputs["input_ids"].shape[1] :]
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]
