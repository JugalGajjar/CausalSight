"""Qwen2.5-VL backend via transformers. Works on CUDA, and on Apple MPS for small smoke tests."""

from __future__ import annotations

from PIL import Image

from causalsight.models.base import VLMBackend


class QwenVLBackend(VLMBackend):
    def __init__(self, model_id: str, device: str | None = None, dtype: str = "bfloat16", max_pixels: int = 360 * 420, **_) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.name = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_pixels = max_pixels

    def generate(self, frames: list[Image.Image], prompt: str, max_new_tokens: int = 64) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "max_pixels": self.max_pixels},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
