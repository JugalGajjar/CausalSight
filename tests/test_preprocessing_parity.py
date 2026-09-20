"""The evaluation backend must build video messages exactly like the trainers (no pixel cap by default)."""

from causalsight.models.qwen_vl import QwenVLBackend
from causalsight.train.grpo import build_messages


def test_eval_messages_match_trainer_messages():
    be = QwenVLBackend.__new__(QwenVLBackend)  # no weights needed
    be.max_pixels = None
    frames = ["f0", "f1"]
    ev = be._messages(frames, "PROMPT")[0]["content"][0]
    tr = build_messages({"problem": "PROMPT"}, frames, "plain")[0]["content"][0]
    assert ev == tr == {"type": "video", "video": frames}
    be.max_pixels = 151200
    assert be._messages(frames, "PROMPT")[0]["content"][0]["max_pixels"] == 151200
