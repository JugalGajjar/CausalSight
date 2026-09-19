from causalsight.train.rewards.chain_rewards import grounding, process

GT = {"question_type": "descriptive", "final_answer": "blue", "choice_id": None, "triplets": [
    {"question": "Which object is the gray sphere?", "answer": "the gray rubber sphere", "depends_on": [], "role": "resolve", "evidence": {"t_start": 0, "t_end": 0, "box": [0.3, 0.6, 0.5, 0.8]}},
    {"question": "Which collision involving the gray sphere happens?", "answer": "the collision between the blue rubber sphere and the gray rubber sphere at frame 19", "depends_on": [0], "role": "resolve", "evidence": {"t_start": 17, "t_end": 21, "box": [0.0, 0.4, 0.2, 0.6]}},
]}
REC = {"problem_type": "free-form", "gt_chains": [GT]}
GOOD = "<think>\n[1] Q: Which object is the gray sphere? | A: the gray rubber sphere | E: t=0-0 box=(0.3,0.6,0.5,0.8) | deps=\n[2] Q: Which collision? | A: the collision between the blue rubber sphere and the gray rubber sphere at frame 19 | E: t=17-21 box=(0.0,0.4,0.2,0.6) | deps=1\n</think>\n<answer>blue</answer>"
REVERSED = "<think>\n[1] Q: Which collision? | A: the collision between the blue rubber sphere and the gray rubber sphere at frame 19 | E: t=17-21 box=(0.0,0.4,0.2,0.6) | deps=\n[2] Q: Which object is the gray sphere? | A: the gray rubber sphere | E: t=0-0 box=(0.3,0.6,0.5,0.8) | deps=\n</think>\n<answer>blue</answer>"
OFF = "<think>\n[1] Q: x | A: the red metal cube | E: t=100-100 box=(0.8,0.0,0.9,0.1) | deps=\n</think>\n<answer>blue</answer>"


def test_grounding_and_process_reward_ordering():
    assert abs(grounding(GOOD, REC) - 1.0) < 1e-9
    assert grounding(OFF, REC) == 0.0
    assert grounding("no chain", REC) == 0.0
    good, rev, off = process(GOOD, REC), process(REVERSED, REC), process(OFF, REC)
    assert good > rev > off  # reversed causal order is penalized but still matches content
    assert abs(good - 1.0) < 1e-9 and abs(rev - 0.5) < 1e-9


def test_grounding_obj_without_proposals_is_zero(monkeypatch):
    from causalsight.train.rewards.chain_rewards import grounding_obj

    monkeypatch.delenv("CS_PROPOSALS", raising=False)
    assert grounding_obj(GOOD, {"problem_id": "12_3"}) == 0.0
