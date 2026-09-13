import random

from causalsight.data.schema import Evidence, Triplet, TripletChain
from causalsight.train.sft import corrupt_answer, corrupt_chain, on_path_indices


def test_corrupt_answer_variants():
    rng = random.Random(0)
    a = corrupt_answer("the gray rubber sphere", rng)
    assert a != "the gray rubber sphere" and a.startswith("the ")
    assert corrupt_answer("yes, at frame 45", rng) == "no"
    assert corrupt_answer("no", rng) == "yes"
    assert corrupt_answer("at frame 45", rng).startswith("at frame ") and corrupt_answer("at frame 45", rng) != "at frame 45"
    assert corrupt_answer("3", rng) is None


def test_corrupt_chain_targets_on_path_step():
    ev = Evidence(0, 0, (0.1, 0.1, 0.2, 0.2))
    chain = TripletChain("v", "q", [
        Triplet("Which object is the cube?", "the red metal cube", ev),
        Triplet("Which object is the sphere?", "the blue rubber sphere", ev),  # off-path: nothing depends on it
        Triplet("Which collision involving the cube happens?", "the collision between the red metal cube and the gray rubber cylinder at frame 40", ev, depends_on=(0,)),
    ], "cylinder", "descriptive")
    assert on_path_indices(chain) == [0, 2]
    c = corrupt_chain(chain, random.Random(1))
    i = c.meta["corrupted"]
    assert i in (0, 2) and c.triplets[i].answer != chain.triplets[i].answer
    assert c.triplets[1].answer == chain.triplets[1].answer
