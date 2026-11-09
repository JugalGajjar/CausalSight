"""Splits of the self-citation results: by gold answer, by citation hit, masked area, and what the masked run cites."""

import json

from causalsight.eval.cite_grounding import frame_of
from causalsight.eval.faithfulness.splits import cited_steps, hit_status, recite, splits

CH = lambda box, t="4-12", ent="the dog": f"<think>\n[1] Q: Where is {ent}? | A: {ent} | E: t={t} box=({box}) | deps=\n</think>\n<answer>yes</answer>"


def write(tmp_path, name, rows):
    (tmp_path / f"m_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_cited_steps_and_hit_status():
    raw = CH("0.100,0.200,0.400,0.600")
    assert cited_steps(raw) == [("the dog", 4, 12, (0.1, 0.2, 0.4, 0.6))]
    f = str(frame_of(4, 12))
    assert hit_status(raw, {"the dog": {f: [0.1, 0.2, 0.4, 0.6]}}) == "all"
    assert hit_status(raw, {"the dog": {f: [0.6, 0.6, 0.9, 0.9]}}) == "not_all"
    assert hit_status(raw, {}) == "not_all"  # never detected on the cited frame
    assert hit_status("<answer>yes</answer>", {}) is None


def test_recite_classifies_where_the_masked_run_cites():
    plain = {"a": {"raw": CH("0.100,0.200,0.400,0.600")}, "b": {"raw": CH("0.100,0.200,0.400,0.600")}, "c": {"raw": CH("0.100,0.200,0.400,0.600")}}
    masked = {
        "a": {"raw": CH("0.100,0.200,0.400,0.600")},  # the blacked-out box again
        "b": {"raw": CH("0.600,0.600,0.900,0.900")},  # same frame, another place
        "c": {"raw": CH("0.100,0.200,0.400,0.600", t="100-108", ent="the cat")},  # a frame the mask did not touch
    }
    r = recite(plain, masked, ["a", "b", "c"])
    assert r["steps"] == 3 and abs(r["same_box"] - 1 / 3) < 1e-9 and abs(r["other_frame"] - 1 / 3) < 1e-9
    assert abs(r["same_entities"] - 2 / 3) < 1e-9


def test_splits_by_gold_separates_affirmations_from_rejections(tmp_path):
    ids = [f"q{i}_{j}" for i in range(4) for j in range(2)]  # option 0 correct (gold yes), option 1 wrong
    mk = lambda fn: [{"id": i, "qid": i.split("_")[0], "question_type": "causal", "gold": "yes" if i.endswith("_0") else "no",
                      "correct": fn(i), "raw": CH("0.100,0.200,0.400,0.600"), "masked_fraction": 0.05} for i in ids]
    write(tmp_path, "plain", mk(lambda i: 1))
    write(tmp_path, "blind", mk(lambda i: 0 if i.endswith("_0") else 1))
    write(tmp_path, "mask_cite", mk(lambda i: 0 if i in ("q0_0", "q1_0") else 1))  # only affirmations flip
    write(tmp_path, "mask_cite_random", mk(lambda i: 1))
    s = splits(tmp_path, "m")
    assert s["by_gold"]["yes"]["cite"]["gap"] == 0.5 and s["by_gold"]["no"]["cite"]["gap"] == 0.0
    assert s["by_gold"]["yes"]["blind_flip"] == 1.0 and s["by_gold"]["no"]["blind_flip"] == 0.0
    assert abs(s["area"]["mask_cite"] - 0.05) < 1e-9 and s["recite"]["cite"]["same_box"] == 1.0
