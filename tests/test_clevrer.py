"""Executor and chain-builder tests. Real-data tests skip when data/raw/clevrer is absent."""

import json
from pathlib import Path

import pytest

from causalsight.data.clevrer_executor import TracedExecutor
from causalsight.data.clevrer_sim import Event, Scene

ROOT = Path("data/raw/clevrer")
HAVE_DATA = (ROOT / "questions" / "train.json").exists()


# ---------------------------------------------------------------- synthetic scene


def synthetic_annotation(tmp_path: Path) -> Path:
    """3 objects, 20 frames. obj1 enters at frame 5, obj0 exits at frame 15, collisions 0-1 @8 and 1-2 @12."""
    props = [
        {"object_id": 0, "color": "red", "material": "metal", "shape": "cube"},
        {"object_id": 1, "color": "blue", "material": "rubber", "shape": "sphere"},
        {"object_id": 2, "color": "gray", "material": "metal", "shape": "cylinder"},
    ]
    traj = []
    for f in range(20):
        objs = []
        for i in range(3):
            vis = not (i == 1 and f < 5) and not (i == 0 and f >= 15)
            speed = 0.0 if (i == 2 and f < 12) else 1.0  # cylinder stationary until hit at 12
            objs.append({"object_id": i, "location": [f * 0.1, 0, 0], "orientation": [0, 0, 0],
                         "velocity": [speed, 0, 0], "angular_velocity": [0, 0, 0], "inside_camera_view": vis})
        traj.append({"frame_id": f, "objects": objs})
    ann = {"scene_index": 0, "video_filename": "v.mp4", "object_property": props, "motion_trajectory": traj,
           "collision": [{"object_ids": [0, 1], "frame_id": 8, "location": [0, 0, 0]},
                         {"object_ids": [1, 2], "frame_id": 12, "location": [0, 0, 0]}]}
    p = tmp_path / "ann.json"
    p.write_text(json.dumps(ann))
    return p


def test_scene_events_and_ancestors(tmp_path):
    s = Scene.from_annotation(synthetic_annotation(tmp_path))
    assert Event("in", 5, (1,)) in s.in_out
    assert Event("out", 15, (0,)) in s.in_out
    col12 = Event("collision", 12, (1, 2))
    anc = s.ancestors(col12)
    assert Event("collision", 8, (0, 1)) in anc and Event("in", 5, (1,)) in anc
    assert col12 not in anc
    assert s.is_moving(2, 5) is False and s.is_moving(2, 13) is True
    assert s.is_moving(2) is True  # moving at some point


def test_executor_descriptive_programs(tmp_path):
    s = Scene.from_annotation(synthetic_annotation(tmp_path))
    run = lambda pg: TracedExecutor.answer_str(TracedExecutor(s).run(pg))
    assert run(["objects", "metal", "filter_material", "count"]) == "2"
    assert run(["events", "objects", "filter_in", "count"]) == "1"
    assert run(["events", "objects", "blue", "filter_color", "unique", "filter_collision", "first", "filter_order",
                "objects", "blue", "filter_color", "unique", "get_col_partner", "query_color"]) == "red"
    assert run(["objects", "gray", "filter_color", "start", "get_frame", "filter_stationary", "exist"]) == "yes"
    assert run(["objects", "gray", "filter_color", "end", "get_frame", "filter_stationary", "exist"]) == "no"
    assert run(["events", "events", "objects", "sphere", "filter_shape", "unique", "filter_in", "unique",
                "filter_after", "objects", "filter_collision", "count"]) == "2"


def test_executor_explanatory_belong_to(tmp_path):
    s = Scene.from_annotation(synthetic_annotation(tmp_path))
    q = ["events", "events", "objects", "blue", "filter_color", "unique", "filter_collision", "objects", "gray",
         "filter_color", "unique", "filter_collision", "unique", "filter_ancestor", "belong_to"]
    choice_col = ["events", "objects", "red", "filter_color", "unique", "filter_collision", "objects", "blue",
                  "filter_color", "unique", "filter_collision", "unique"]
    choice_obj = ["objects", "gray", "filter_color", "unique"]
    assert TracedExecutor.answer_str(TracedExecutor(s).run(choice_col + q)) == "yes"
    assert TracedExecutor.answer_str(TracedExecutor(s).run(choice_obj + q)) == "no"


def test_executor_counterfactual_is_unknown(tmp_path):
    s = Scene.from_annotation(synthetic_annotation(tmp_path))
    pg = ["all_events", "objects", "red", "filter_color", "unique", "filter_collision", "objects", "blue", "filter_color",
          "unique", "filter_collision", "unique", "all_events", "objects", "gray", "filter_color", "unique",
          "get_counterfact", "belong_to"]
    assert TracedExecutor.answer_str(TracedExecutor(s).run(pg)) == "<unknown>"


# ---------------------------------------------------------------- real data


@pytest.mark.skipif(not HAVE_DATA, reason="CLEVRER not downloaded")
def test_real_data_executor_matches_gt():
    from causalsight.data.clevrer_validate import find_annotation

    videos = json.loads((ROOT / "questions" / "train.json").read_text())[:5]
    for v in videos:
        s = Scene.from_annotation(find_annotation(ROOT, "train", v["scene_index"]))
        for q in v["questions"]:
            if q["question_type"] == "descriptive":
                assert TracedExecutor.answer_str(TracedExecutor(s).run(q["program"])) == str(q["answer"]), q["question"]
            elif q["question_type"] == "explanatory":
                for c in q["choices"]:
                    out = TracedExecutor.answer_str(TracedExecutor(s).run(c["program"] + q["program"]))
                    assert {"yes": "correct", "no": "wrong"}[out] == c["answer"], (q["question"], c["choice"])


@pytest.mark.skipif(not HAVE_DATA, reason="CLEVRER not downloaded")
def test_real_data_chains_are_well_formed():
    from causalsight.data.clevrer_evidence import ProposalIndex
    from causalsight.data.clevrer_triplets import ChainBuilder, find_annotation

    videos = json.loads((ROOT / "questions" / "train.json").read_text())[:3]
    n_ok = 0
    for v in videos:
        s = Scene.from_annotation(find_annotation(ROOT, "train", v["scene_index"]))
        p = ProposalIndex.load(ROOT / "derender_proposals" / f"proposal_{v['scene_index']:05d}.json")
        b = ChainBuilder(s, p)
        for q in v["questions"]:
            for c in ([None] if q["question_type"] == "descriptive" else q["choices"]):
                chain, status = b.build(q, c)
                assert status in ("ok", "no_evidence"), status
                if chain is None:
                    continue
                n_ok += 1
                chain.validate()
                answers = [t.answer for t in chain.triplets if t.question.startswith("Which object is")]
                assert len(answers) == len(set(answers)), "duplicate object resolution"
                for t in chain.triplets:
                    assert 0 <= t.evidence.t_start <= t.evidence.t_end < 128
                if c is None:
                    assert chain.final_answer == str(q["answer"])
                else:
                    assert chain.final_answer == ("yes" if c["answer"] == "correct" else "no")
    assert n_ok > 50
