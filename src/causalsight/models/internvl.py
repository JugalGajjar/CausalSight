"""InternVL3 (HF-native) evaluation backend, built on the trainer encoder so evaluation inputs match
training inputs exactly. Accepts a base id or one of our LoRA adapter directories (merging any init
adapter first, as the Qwen backend does)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from causalsight.models.base import VLMBackend
from causalsight.models.qwen_vl import resolve_init_adapters


class InternVLBackend(VLMBackend):
    def __init__(self, model_id: str, device: str | None = None, dtype: str = "bfloat16", init_adapter: str | None = None, **_) -> None:
        import torch

        from causalsight.train.encoders import InternVLEncoder

        self.name = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        adapter = Path(model_id) / "adapter_config.json"
        base_id = json.loads(adapter.read_text())["base_model_name_or_path"] if adapter.exists() else model_id
        self.enc = InternVLEncoder(base_id)
        model = self.enc.load_model(base_id, torch_dtype, "sdpa")
        if adapter.exists():
            from peft import PeftModel

            for init in resolve_init_adapters(Path(model_id), init_adapter):
                model = PeftModel.from_pretrained(model, str(init)).merge_and_unload()
                print(f"merged init adapter {init}")
            model = PeftModel.from_pretrained(model, model_id).merge_and_unload()
            print(f"loaded adapter {model_id} on {base_id} (merged)")
        self.model = model.to(self.device).eval()
        self.processor = self.enc.processor

    def generate(self, frames: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> str:
        return self.generate_batch([frames], [prompt], max_new_tokens)[0]

    def generate_batch(self, frames_list, prompts, max_new_tokens: int = 64, temperature: float = 0.0) -> list[str]:
        import torch

        texts = [self.enc.prompt_text(self.enc.messages(f, p)) for f, p in zip(frames_list, prompts)]
        inputs = self.enc.encode(texts, frames_list, self.device)
        kw = {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0} if temperature > 0 else {"do_sample": False}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, pad_token_id=self.enc.tokenizer.pad_token_id,
                num_beams=1, repetition_penalty=1.0, length_penalty=1.0, use_cache=True, **kw,
            )
        out = out[:, inputs["input_ids"].shape[1] :]
        return [t.strip() for t in self.enc.tokenizer.batch_decode(out, skip_special_tokens=True)]
