import csv
from pathlib import Path

from causalsight.eval.benchmarks.causalvqa import CausalVQABenchmark


def mk(tmp_path: Path):
    d = tmp_path / "CausalVQA" / "debug"
    (d / "videos").mkdir(parents=True)
    rows = [{"qid": "q1", "type": "Anticipation", "question": '"What next?"', "choices1": "1. a| 2. b| 3. c| 4. d| 5. e", "correct1": "2", "choices2": "1. e| 2. d| 3. c| 4. b| 5. a", "correct2": "4", "difficulty": "easy", "renamed_video": "v.mp4"}]
    with (d / "debug_metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return tmp_path


def test_multi_items_and_paired_summary(tmp_path):
    b = CausalVQABenchmark(mk(tmp_path))
    items = list(b.items("debug"))
    assert [i.gold for i in items] == ["B", "D"] and "Exactly one option" in items[0].prompt and len(items[0].options) == 5
    rows = [{"id": it.id, "question_type": it.question_type, **it.meta, **b.score(it, p)} for it, p in zip(items, ["B", "A"])]
    s = b.summarize(rows)["ALL"]
    assert s["n_questions"] == 1 and s["set_acc"] == 0.5 and s["paired_acc"] == 0.0


def test_option_mode(tmp_path):
    b = CausalVQABenchmark(mk(tmp_path), mc_mode="option")
    items = list(b.items("debug"))
    assert len(items) == 10 and sum(i.gold == "yes" for i in items) == 2
    rows = [{"id": it.id, "question_type": it.question_type, **it.meta, **b.score(it, it.gold)} for it in items]
    s = b.summarize(rows)["ALL"]
    assert s["per_option"] == 1.0 and s["paired_acc"] == 1.0
