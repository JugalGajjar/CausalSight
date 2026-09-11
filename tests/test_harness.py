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


def test_exist_prompt_and_numeric_fallback(tmp_path):
    root = fake_clevrer(tmp_path)
    qs = json.loads((root / "questions" / "validation.json").read_text())
    qs[0]["questions"].append({"question_id": 2, "question": "Are there any cubes?", "question_type": "descriptive", "question_subtype": "exist", "program": [], "answer": "no"})
    (root / "questions" / "validation.json").write_text(json.dumps(qs))
    b = ClevrerBenchmark(root)
    it = list(b.items("validation"))[2]
    assert it.prompt.endswith("Answer with yes or no.")
    assert b.score(it, "0")["pred"] == "no" and b.score(it, "1")["pred"] == "yes"
    assert b.score(it, "No.")["correct"] == 1


def test_stratified_sample_is_deterministic():
    from causalsight.eval.benchmarks import Item
    from causalsight.eval.harness import stratified_sample

    items = [Item(f"{t}{i}", t, Path("x"), "", "") for t in ("a", "b") for i in range(10)]
    s1 = stratified_sample(items, 3, seed=1)
    s2 = stratified_sample(items, 3, seed=1)
    assert [i.id for i in s1] == [i.id for i in s2] and len(s1) == 6
    assert [i.id for i in s1] != [i.id for i in stratified_sample(items, 3, seed=2)]


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
