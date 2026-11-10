"""Splits of the self-citation results: by gold answer, by citation hit, masked area, and what the masked run cites."""

import json

from causalsight.eval.cite_grounding import frame_of
from causalsight.eval.faithfulness.splits import cited_steps, control_coverage, entities_in_text, hit_status, pooled, recite, splits

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


def test_entities_in_text_and_control_replay(tmp_path):
    raw = CH("0.100,0.200,0.400,0.600")
    assert entities_in_text(raw, "what does the dog do?\nOption A: bark\nIs this option correct? Answer yes or no.") is True
    assert entities_in_text(raw, "what happens at the end?\nOption A: bark\nIs this option correct?") is False
    assert entities_in_text("<answer>yes</answer>", "what does the dog do?") is None
    rows = [{"id": f"q{i}_0", "raw": raw, "correct": 1} for i in range(50)]
    write(tmp_path, "plain", rows)
    cov = control_coverage(tmp_path / "m_plain.jsonl", {r["id"]: r for r in rows})
    assert len(cov) == 50 and all(0.0 <= v <= 1.0 for v in cov.values())
    assert cov == control_coverage(tmp_path / "m_plain.jsonl", {r["id"]: r for r in rows})  # seeded: replays exactly
    assert any(v == 0 for v in cov.values()) and any(v > 0 for v in cov.values())


def test_pooled_keeps_questions_as_clusters(tmp_path):
    for tag, flips in (("a", {"q0_0"}), ("b", {"q0_0", "q1_0"})):
        ids = [f"q{i}_0" for i in range(4)]
        mk = lambda fn: [{"id": i, "qid": i.split("_")[0], "question_type": "causal", "gold": "yes", "correct": fn(i), "raw": CH("0.100,0.200,0.400,0.600")} for i in ids]
        for name, fn in (("plain", lambda i: 1), ("mask_cite", lambda i: 0 if i in flips else 1), ("mask_cite_random", lambda i: 1)):
            (tmp_path / f"{tag}_{name}.jsonl").write_text("\n".join(json.dumps(r) for r in mk(fn)) + "\n")
    r = pooled(tmp_path, ["a", "b"])
    assert r["cite"]["all"]["n_correct"] == 8 and abs(r["cite"]["all"]["gap"] - 3 / 8) < 1e-9 and r["cite"]["yes"]["n_correct"] == 8


def test_entity_control_share_counts_other_entities_only(tmp_path):
    from causalsight.eval.faithfulness.splits import entity_control_share

    raw = CH("0.050,0.050,0.250,0.250")  # names "the dog", small box in the top-left corner
    rows = [{"id": f"v1_{i}_0", "frames_key": "v1", "raw": raw, "correct": 1} for i in range(40)]
    write(tmp_path, "plain", rows)
    dog = {str(f): [0.05, 0.05, 0.25, 0.25] for f in range(16)}
    cat = {str(f): [0.60, 0.60, 0.80, 0.80] for f in range(16)}       # another entity, elsewhere
    floor = {str(f): [0.0, 0.3, 1.0, 1.0] for f in range(16)}         # too large to count as an entity
    alias = {str(f): [0.06, 0.05, 0.26, 0.25] for f in range(16)}     # the same entity under another name
    (tmp_path / "tracks.json").write_text(json.dumps({"v1": {"the dog": dog, "the cat": cat, "the floor": floor, "the puppy": alias}}))
    share = entity_control_share(tmp_path / "tracks.json", tmp_path / "m_plain.jsonl", {r["id"]: r for r in rows})
    assert len(share) == 40 and all(0.0 <= v <= 1.0 for v in share.values())
    assert 0.0 < sum(share.values()) / len(share) < 0.5   # only boxes that land on the cat count
    assert share == entity_control_share(tmp_path / "tracks.json", tmp_path / "m_plain.jsonl", {r["id"]: r for r in rows})
