import json
from pathlib import Path

import pytest

from causalsight.data.export_rl import export_rl, export_sft, question_record

ROOT = Path("data/raw/clevrer")
HAVE = (ROOT / "questions" / "train.json").exists() and Path("data/triplets/clevrer_train.jsonl").exists()


def test_question_record_shapes(tmp_path):
    q_desc = {"question_id": 0, "question": "How many spheres?", "question_type": "descriptive", "question_subtype": "count", "program": [], "answer": "3"}
    r = question_record(tmp_path, "train", 12, q_desc)
    assert r["problem_type"] == "numerical" and r["solution"] == "<answer>3</answer>" and r["problem"].endswith("Answer with a number.")
    q_mc = {"question_id": 1, "question": "Which?", "question_type": "counterfactual", "program": [],
            "choices": [{"choice_id": 0, "choice": "x", "program": [], "answer": "correct"}, {"choice_id": 1, "choice": "y", "program": [], "answer": "correct"}, {"choice_id": 2, "choice": "z", "program": [], "answer": "wrong"}]}
    r = question_record(tmp_path, "train", 12, q_mc)
    assert r["problem_type"] == "multiple choice" and r["gold"] == "AB" and r["options"] == ["A. x", "B. y", "C. z"]
    assert "video_00012.mp4" in r["path"]


@pytest.mark.skipif(not HAVE, reason="CLEVRER data absent")
def test_exports_on_real_data(tmp_path):
    stats = export_rl(ROOT, "train", tmp_path / "rl.jsonl", n_videos=3)
    assert sum(stats.values()) > 30
    stats = export_sft(ROOT, "train", Path("data/triplets/clevrer_train.jsonl"), tmp_path / "sft.jsonl", limit=200)
    assert sum(stats.values()) == 200
    rec = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    assert rec["response"].startswith("<think>") and rec["response"].endswith(f"<answer>{rec['gold']}</answer>")
