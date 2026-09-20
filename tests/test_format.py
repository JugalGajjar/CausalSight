from pathlib import Path

import pytest

from causalsight.data.schema import Evidence, Triplet, TripletChain
from causalsight.train.format import format_chain, parse_chain

CHAINS = Path("data/triplets/clevrer_validation.jsonl")


def sample_chain():
    return TripletChain(
        video_id="v",
        question="q?",
        triplets=[
            Triplet("Which object is the gray sphere?", "the gray rubber sphere", Evidence(0, 0, (0.36, 0.62, 0.47, 0.79))),
            Triplet("Which collision happens?", "the collision between A and B at frame 19", Evidence(17, 21, (0.06, 0.39, 0.23, 0.55)), depends_on=(0,)),
        ],
        final_answer="blue",
        question_type="descriptive",
    )


def test_round_trip():
    c = sample_chain()
    text = format_chain(c)
    assert text.startswith("<think>\n[1] Q: Which object is the gray sphere? | A: the gray rubber sphere | E: t=0-0 box=(0.360,0.620,0.470,0.790) | deps=")
    assert text.endswith("<answer>blue</answer>")
    p = parse_chain(text, "v", "q?", "descriptive")
    assert p.chain is not None and p.n_steps == 2 and p.answer == "blue"
    assert p.chain.triplets[1].depends_on == (0,)
    assert p.chain.triplets[0].evidence.box == pytest.approx(c.triplets[0].evidence.box, abs=1e-3)


def test_malformed_inputs():
    assert parse_chain("no tags at all").chain is None
    assert parse_chain("<think>\nfree text reasoning\n</think>\n<answer>3</answer>").chain is None
    p = parse_chain("<think>\n[1] Q: a | A: b | E: t=5-3 box=(0,0,1,1) | deps=\n</think>\n<answer>x</answer>")
    assert p.chain is None and p.has_tags and p.n_steps == 1  # bad span fails validation
    p = parse_chain("<think>\n[1] Q: a | A: b | E: t=0-0 box=(0,0,0.5,0.5) | deps=7\n</think>\n<answer>x</answer>")
    assert p.chain is not None and p.chain.triplets[0].depends_on == ()  # unknown dep dropped
    assert parse_chain("<think>\n[1] Q: a | A: b | E: t=0-0 box=(0,0,0.5,0.5) | deps=\n</think>").chain is None  # no answer


@pytest.mark.skipif(not CHAINS.exists(), reason="generated chains absent")
def test_round_trip_on_generated_chains():
    n = 0
    with CHAINS.open() as f:
        for line in f:
            c = TripletChain.from_json(line)
            p = parse_chain(format_chain(c), c.video_id, c.question, c.question_type)
            assert p.chain is not None, c.to_json()
            assert [t.depends_on for t in p.chain.triplets] == [t.depends_on for t in c.triplets]
            assert [t.answer for t in p.chain.triplets] == [t.answer for t in c.triplets]
            n += 1
            if n >= 2000:
                break
    assert n == 2000


def test_step_spans_align_with_triplets():
    from causalsight.train.format import step_spans

    text = format_chain(sample_chain())
    spans = step_spans(text)
    p = parse_chain(text)
    assert len(spans) == len(p.chain.triplets) == 2
    assert text[spans[0][0] : spans[0][1]].startswith("[1] Q: Which object is the gray sphere?")
    assert text[spans[1][0] : spans[1][1]].startswith("[2] Q:") and text[spans[1][0] : spans[1][1]].endswith("deps=1")
    assert step_spans("no think block") == []
