from causalsight.eval.masking import decisive_triplets


def mk(q, role=None):
    return {"question": q, "role": role, "evidence": {"t_start": 0, "t_end": 0, "box": [0, 0, 0.1, 0.1]}}


def test_counterfactual_uses_observe_step():
    c = {"question_type": "counterfactual", "triplets": [mk("Which object is X?", "resolve"), mk("Which object is Y?", "resolve"), mk("Do X and Y collide during the video?", "observe"), mk("Which object is Z?", "resolve")]}
    assert [t["question"] for t in decisive_triplets(c)] == ["Do X and Y collide during the video?"]


def test_legacy_prefix_fallback_and_last_default():
    c = {"question_type": "predictive", "triplets": [mk("Which object is X?"), mk("Do X and Y collide during the video?")]}
    assert decisive_triplets(c)[0]["question"].startswith("Do ")
    d = {"question_type": "descriptive", "triplets": [mk("a"), mk("b")]}
    assert decisive_triplets(d)[0]["question"] == "b"
