from causalsight.train.rewards.registry import format_chain, format_tags, outcome, outcome_lenient


def test_outcome_descriptive_and_mc():
    rec = {"problem_type": "numerical", "subtype": "count", "gold": "3"}
    assert outcome("<think>x</think><answer>three</answer>", rec) == 1.0
    assert outcome("<answer>4</answer>", rec) == 0.0
    assert outcome("no tags", rec) == 0.0
    rec = {"problem_type": "multiple choice", "options": ["A. x", "B. y", "C. z"], "gold": "AC"}
    assert outcome("<think>..</think><answer>A, C</answer>", rec) == 1.0
    assert outcome("<answer>A</answer>", rec) == 0.0
    rec = {"problem_type": "free-form", "subtype": "exist", "gold": "no"}
    assert outcome("<answer>0</answer>", rec) == 1.0


def test_format_rewards():
    assert format_tags("<think>a</think>\n<answer>b</answer>", {}) == 1.0
    assert format_tags("<answer>b</answer>", {}) == 0.0
    good = "<think>\n[1] Q: a | A: b | E: t=0-0 box=(0,0,0.5,0.5) | deps=\n</think>\n<answer>x</answer>"
    assert format_chain(good, {}) == 1.0
    assert format_chain("<think>free text</think><answer>x</answer>", {}) == 0.2
    assert format_chain("nothing", {}) == 0.0


def test_outcome_lenient_without_tags():
    rec = {"problem_type": "multiple choice", "options": ["A. x", "B. y"], "gold": "B"}
    assert outcome_lenient("B", rec) == 1.0 and outcome("B", rec) == 0.0
    assert outcome_lenient("<think>..</think><answer>B</answer>", rec) == 1.0
    assert outcome_lenient("A", rec) == 0.0
