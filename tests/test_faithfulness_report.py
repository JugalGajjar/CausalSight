import json
from pathlib import Path

from causalsight.eval.faithfulness.report import report


def write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_report_blind_gap_and_es_gap(tmp_path):
    ids = [f"q{i}" for i in range(4)]
    mk = lambda correct: [{"id": i, "question_type": "descriptive", "correct": c} for i, c in zip(ids, correct)]
    write(tmp_path / "m_plain.jsonl", mk([1, 1, 1, 0]))
    write(tmp_path / "m_blind.jsonl", mk([1, 0, 0, 0]))
    write(tmp_path / "m_mask_evidence.jsonl", mk([0, 0, 1, 0]))
    write(tmp_path / "m_mask_random.jsonl", mk([1, 1, 0, 0]))
    r = report(tmp_path, "m")["ALL"]
    assert r["n"] == 4 and abs(r["blind_gap"] - 0.5) < 1e-9
    assert abs(r["flip_evidence"] - 2 / 3) < 1e-9 and abs(r["flip_random"] - 1 / 3) < 1e-9
    assert abs(r["es_gap"] - 1 / 3) < 1e-9
