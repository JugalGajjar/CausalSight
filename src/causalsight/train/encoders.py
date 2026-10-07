"""Model-specific input encoding for the trainers, behind one interface.

The trainers (cs-sft, cs-grpo) only need: load a model+processor, build generation inputs for
(frames, prompt), build teacher-forcing inputs for (frames, prompt, response) with the prompt length,
and repeat the vision inputs for a group of completions. Everything Qwen2.5-VL- or InternVL-specific
lives here. `make_encoder(model_id)` picks by config.model_type.
"""

from __future__ import annotations

from typing import Any, ClassVar

from PIL import Image


class Encoder:
    model_type: str = "base"
    lora_default_targets: ClassVar[list[str] | str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.processor = None
        self.tokenizer = None

    def load_model(self, model_id: str, dtype, attn: str):
        raise NotImplementedError

    def messages(self, frames: list[Image.Image], text: str, system: str | None = None) -> list[dict]:
        content = [{"type": "video", "video": frames}, {"type": "text", "text": text}]
        msgs = [{"role": "user", "content": content}]
        if system:
            msgs.insert(0, {"role": "system", "content": system})
        return msgs

    def prompt_text(self, msgs: list[dict]) -> str:
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def encode(self, texts: list[str], frames_list: list[list[Image.Image]], device) -> dict:
        """Processor inputs for already-templated texts and their videos (left-padded)."""
        raise NotImplementedError

    def prompt_inputs(self, frames: list[Image.Image], text: str, device, system: str | None = None) -> dict:
        return self.encode([self.prompt_text(self.messages(frames, text, system))], [frames], device)

    def full_inputs(self, frames: list[Image.Image], text: str, response: str, device, system: str | None = None) -> tuple[dict, int]:
        """Inputs for prompt+response+eos and the prompt length in tokens (prefix is identical)."""
        pt = self.prompt_text(self.messages(frames, text, system))
        full = self.encode([pt + response + self.tokenizer.eos_token], [frames], device)
        prompt = self.encode([pt], [frames], "cpu")
        return full, int(prompt["input_ids"].shape[1])

    def vision_kwargs(self, inputs: dict, g: int) -> dict[str, Any]:
        """The vision tensors of a single-prompt `inputs`, repeated for g sequences."""
        raise NotImplementedError


class QwenEncoder(Encoder):
    model_type = "qwen2_5_vl"

    def load_model(self, model_id: str, dtype, attn: str):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        return model

    def encode(self, texts, frames_list, device):
        from qwen_vl_utils import process_vision_info

        videos = []
        for f in frames_list:
            _imgs, vids = process_vision_info([{"role": "user", "content": [{"type": "video", "video": f}]}])
            videos.extend(vids)
        return self.processor(text=texts, videos=videos, padding=True, return_tensors="pt").to(device)

    def vision_kwargs(self, inputs, g):
        kw = {}
        if "pixel_values_videos" in inputs:
            kw["pixel_values_videos"] = inputs["pixel_values_videos"].repeat(g, 1)
            kw["video_grid_thw"] = inputs["video_grid_thw"].repeat(g, 1)
        if "second_per_grid_ts" in inputs:
            kw["second_per_grid_ts"] = inputs["second_per_grid_ts"].repeat(g)
        return kw


class InternVLEncoder(Encoder):
    """InternVL3 (HF-native). Video = list of frames, 256 tokens per 448x448 frame; the processor's
    `videos=` argument takes lists of PIL frames. No system-role support in the template is assumed, so a
    system string is prepended to the user text."""

    model_type = "internvl"
    lora_default_targets: ClassVar[str] = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

    def load_model(self, model_id: str, dtype, attn: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, attn_implementation=attn)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        return model

    def messages(self, frames, text, system=None):
        if system:
            text = system + "\n" + text
        return [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": text}]}]

    def encode(self, texts, frames_list, device):
        return self.processor(text=texts, videos=frames_list, padding=True, return_tensors="pt").to(device)

    def vision_kwargs(self, inputs, g):
        pv = inputs["pixel_values"]
        return {"pixel_values": pv.repeat(g, 1, 1, 1)}


ENCODERS = {QwenEncoder.model_type: QwenEncoder, InternVLEncoder.model_type: InternVLEncoder}


def make_encoder(model_id: str) -> Encoder:
    from transformers import AutoConfig

    mt = AutoConfig.from_pretrained(model_id).model_type
    if mt not in ENCODERS:
        raise ValueError(f"no encoder for model_type {mt!r} ({model_id}); known: {list(ENCODERS)}")
    return ENCODERS[mt](model_id)
