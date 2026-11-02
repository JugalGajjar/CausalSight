import random

from causalsight.eval.cite_grounding import entity_of, frame_of, score, steps_of


def test_entity_and_frame():
    assert entity_of("Where is 2: white dog: the white dog?") == "the white dog"
    assert entity_of("Where is the man in the beige vest?") == "the man in the beige vest"
    assert entity_of("Which object is the cube?") is None
    assert frame_of(13, 21) == 2 and frame_of(126, 126) == 15 and frame_of(0, 0) == 0 and frame_of(38, 46) == 5


RAW = (
    "<think>\n[1] Q: Where is the red cup? | A: the red cup: on the table | E: t=13-21 box=(0.200,0.400,0.400,0.800) | deps=\n"
    "[2] Q: Where is the man? | A: the man: standing | E: t=126-126 box=(0.000,0.000,1.000,1.000) | deps=\n"
    "[3] Q: Where is the dog? | A: the dog: lying | E: t=38-46 box=(0.600,0.100,0.800,0.500) | deps=\n</think>\n<answer>yes</answer>"
)


def test_steps_drop_whole_frame():
    s = steps_of(RAW)
    assert [(e, f) for e, f, _ in s] == [("the red cup", 2), ("the dog", 5)]


def test_score_cached_and_controlled():
    calls = []

    def detect(video, frame, phrase):
        calls.append((video, frame, phrase))
        if phrase == "the dog":
            return None  # undetected
        return (0.21, 0.40, 0.41, 0.80)  # nearly the cited cup box

    rows = [
        {"id": "v1_1_0", "frames_key": "v1", "question_type": "causal", "raw": RAW},
        {"id": "v1_1_1", "frames_key": "v1", "question_type": "causal", "raw": RAW},  # per-option repeat: no new calls
    ]
    rep = score(rows, detect, seed=0)
    assert len(calls) == 2 and rep["ALL"]["unique_detector_calls"] == 2
    a = rep["ALL"]
    assert a["n_items"] == 2 and a["n_steps"] == 4 and a["detected_rate"] == 0.5
    assert a["mean_iou"] > 0.9 and a["hit50"] == 1.0
    assert a["random_mean_iou"] < a["mean_iou"]
    assert rep["causal"]["n_steps"] == 4
