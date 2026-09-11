import json
from pathlib import Path

from causalsight.eval.benchmarks.answers import normalize_short, parse_letters
from causalsight.eval.benchmarks.clevrer import ClevrerBenchmark
from causalsight.eval.harness import run


def test_normalize_short():
    assert normalize_short("The answer is: Three.") == "3"
    assert normalize_short("sphere") == "sphere"
    assert normalize_short("Yes, there are.") == "yes"


def test_parse_letters():
    assert parse_letters("A, C", 4) == "AC"
    assert parse_letters("Options B and D are correct.", 4) == "BD"
    assert parse_letters("AC", 4) == "AC"
    assert parse_letters("none", 3) == ""
    assert parse_letters("E", 4) == ""  # out of range


def fake_clevrer(tmp_path: Path) -> Path:
    root = tmp_path / "clevrer"
    (root / "questions").mkdir(parents=True)
    qs = [{"scene_index": 10000, "video_filename": "video_10000.mp4", "questions": [
        {"question_id": 0, "question": "How many spheres?", "question_type": "descriptive", "question_subtype": "count", "program": [], "answer": "2"},
        {"question_id": 1, "question": "Which will happen?", "question_type": "predictive", "program": [],
         "choices": [{"choice_id": 0, "choice": "x", "program": [], "answer": "correct"}, {"choice_id": 1, "choice": "y", "program": [], "answer": "wrong"}]},
    ]}]
    (root / "questions" / "validation.json").write_text(json.dumps(qs))
    return root


def test_clevrer_items_and_scoring(tmp_path):
    b = ClevrerBenchmark(fake_clevrer(tmp_path))
    items = list(b.items("validation"))
    assert [i.question_type for i in items] == ["descriptive", "predictive"]
    assert items[1].gold == "A" and "A. x" in items[1].prompt
    assert b.score(items[0], "two")["correct"] == 1
    s = b.score(items[1], "A and B")
    assert s == {"pred": "AB", "correct": 0, "option_correct": 1, "option_total": 2}
    assert b.score(items[1], "A")["correct"] == 1


def test_harness_end_to_end_with_dummy(tmp_path):
    root = fake_clevrer(tmp_path)
    out = tmp_path / "res.jsonl"
    summary = run("dummy", "clevrer", "validation", root, out, None, False, 16)
    assert summary["n"] == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[1]["correct"] == 1  # dummy answers "A", which is the correct option
    assert summary["by_type"]["predictive"]["per_option"] == 1.0
    assert out.with_suffix(".summary.json").exists()
