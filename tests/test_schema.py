import pytest

from causalsight.data.schema import Evidence, Triplet, TripletChain
from causalsight.train.rewards import format_reward, necessity, process


def ev(t0=0, t1=5, box=(0.1, 0.1, 0.5, 0.5)):
    return Evidence(t_start=t0, t_end=t1, box=box)


def chain():
    # 0: red cube location; 1: blue sphere location (independent); 2: collision depends on 0,1
    return TripletChain(
        video_id="v0",
        question="What caused the blue sphere to move?",
        triplets=[
            Triplet("Where is the red cube?", "left", ev(0, 3)),
            Triplet("Where is the blue sphere?", "center", ev(0, 3, (0.4, 0.4, 0.8, 0.8))),
            Triplet("Do they collide?", "yes", ev(10, 14, (0.3, 0.3, 0.7, 0.7)), depends_on=(0, 1)),
        ],
        final_answer="the red cube hit it",
        question_type="explanatory",
    )


def test_validate_ok():
    chain().validate()


def test_box_must_be_normalized():
    with pytest.raises(ValueError):
        ev(box=(0, 0, 1.5, 1)).validate()


def test_dependency_must_point_backward():
    c = chain()
    c.triplets[0] = Triplet("q", "a", ev(), depends_on=(2,))
    with pytest.raises(ValueError):
        c.validate()


def test_iou():
    a, b = ev(0, 9, (0, 0, 0.5, 0.5)), ev(5, 14, (0, 0, 0.5, 0.5))
    assert a.spatial_iou(b) == pytest.approx(1.0)
    assert a.temporal_iou(b) == pytest.approx(5 / 15)
    assert a.st_iou(b) == pytest.approx(5 / 15)


def test_without_step_drops_dependency_closure():
    c = chain()
    c_minus_0 = c.without_step(0)
    assert [t.question for t in c_minus_0.triplets] == ["Where is the blue sphere?"]
    c_minus_2 = c.without_step(2)
    assert len(c_minus_2.triplets) == 2
    assert c_minus_2.meta["removed_step"] == 2


def test_json_roundtrip():
    c = chain()
    assert TripletChain.from_json(c.to_json()) == c


def test_format_reward():
    assert format_reward.score(chain(), num_frames=16) == 1.0
    assert format_reward.score(chain(), num_frames=12) == 0.0  # step 2 ends at frame 14
    assert format_reward.score(None, num_frames=16) == 0.0


def test_necessity_from_deltas():
    # two necessary steps, one padding step
    r = necessity.score_from_deltas([0.5, 0.4, 0.0], delta=0.1, eps=0.01, gamma=0.5)
    assert r == pytest.approx(2 / 3 - 0.5 * 1 / 3)


def test_order_violations_penalize_reversed_causal_chain():
    ref = chain()
    # prediction lists the collision first, then the two locations: causal order reversed
    pred = TripletChain(
        video_id="v0",
        question=ref.question,
        triplets=[
            Triplet("Do they collide?", "yes", ev(10, 14, (0.3, 0.3, 0.7, 0.7))),
            Triplet("Where is the red cube?", "left", ev(0, 3)),
            Triplet("Where is the blue sphere?", "center", ev(0, 3, (0.4, 0.4, 0.8, 0.8))),
        ],
        final_answer=ref.final_answer,
    )
    matching = {0: 1, 1: 2, 2: 0}
    v, total = process.order_violations(pred, ref, matching)
    assert (v, total) == (2, 2)


def test_order_violations_allow_swapping_independent_steps():
    ref = chain()
    pred = TripletChain(
        video_id="v0",
        question=ref.question,
        triplets=[
            Triplet("Where is the blue sphere?", "center", ev(0, 3, (0.4, 0.4, 0.8, 0.8))),
            Triplet("Where is the red cube?", "left", ev(0, 3)),
            Triplet("Do they collide?", "yes", ev(10, 14, (0.3, 0.3, 0.7, 0.7)), depends_on=(0, 1)),
        ],
        final_answer=ref.final_answer,
    )
    matching = {0: 1, 1: 0, 2: 2}
    assert process.order_violations(pred, ref, matching) == (0, 2)
