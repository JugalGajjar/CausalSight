"""Text format for triplet chains: what the model emits, and the parser that turns generations back
into `TripletChain` objects for the reward terms.

    <think>
    [1] Q: Which object is the gray sphere? | A: the gray rubber sphere | E: t=0-0 box=(0.36,0.62,0.47,0.79) | deps=
    [2] Q: Which collision involving the gray sphere happens? | A: ... | E: t=17-21 box=(0.06,0.39,0.23,0.55) | deps=1
    </think>
    <answer>blue</answer>

Step ids are 1-based in text and 0-based in `TripletChain.depends_on`. The <think>/<answer> tags keep the
format compatible with Video-R1-style format rewards; the step lines are what R_fmt, R_ground, R_nec and
R_proc consume. `parse_chain` is tolerant of whitespace and returns None on anything unrecoverable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from causalsight.data.schema import Evidence, Triplet, TripletChain

STEP_RE = re.compile(
    r"\[(?P<id>\d+)\]\s*Q:\s*(?P<q>.*?)\s*\|\s*A:\s*(?P<a>.*?)\s*\|\s*E:\s*t=(?P<t0>\d+)\s*-\s*(?P<t1>\d+)\s*"
    r"box=\(\s*(?P<x0>[\d.]+)\s*,\s*(?P<y0>[\d.]+)\s*,\s*(?P<x1>[\d.]+)\s*,\s*(?P<y1>[\d.]+)\s*\)\s*"
    r"\|\s*deps=(?P<deps>[\d,\s]*)\s*$"
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

SYSTEM_PROMPT = (
    "You answer questions about a video by reasoning in grounded steps. Inside <think></think>, write one "
    "step per line as [n] Q: <sub-question> | A: <intermediate answer> | E: t=<start>-<end> box=(x0,y0,x1,y1) "
    "| deps=<comma-separated earlier step numbers>. t is a frame range of the video and the box is the "
    "evidence region in normalized coordinates. Then give the final answer inside <answer></answer>."
)


def format_step(i: int, t: Triplet) -> str:
    e = t.evidence
    box = ",".join(f"{v:.3f}" for v in e.box)
    deps = ",".join(str(d + 1) for d in t.depends_on)
    return f"[{i + 1}] Q: {t.question} | A: {t.answer} | E: t={e.t_start}-{e.t_end} box=({box}) | deps={deps}"


def format_chain(chain: TripletChain) -> str:
    body = "\n".join(format_step(i, t) for i, t in enumerate(chain.triplets))
    return f"<think>\n{body}\n</think>\n<answer>{chain.final_answer}</answer>"


@dataclass
class Parsed:
    chain: TripletChain | None
    answer: str | None
    n_lines: int  # lines inside <think>
    n_steps: int  # lines that parsed as steps
    has_tags: bool


def parse_chain(text: str, video_id: str = "", question: str = "", question_type: str | None = None) -> Parsed:
    think = THINK_RE.search(text)
    ans = ANSWER_RE.search(text)
    answer = ans.group(1).strip() if ans else None
    if not think:
        return Parsed(None, answer, 0, 0, False)
    lines = [ln.strip() for ln in think.group(1).strip().splitlines() if ln.strip()]
    triplets: list[Triplet] = []
    ids: dict[int, int] = {}
    n_steps = 0
    for ln in lines:
        m = STEP_RE.match(ln)
        if not m:
            continue
        n_steps += 1
        sid = int(m.group("id"))
        try:
            ev = Evidence(
                t_start=int(m.group("t0")),
                t_end=int(m.group("t1")),
                box=(float(m.group("x0")), float(m.group("y0")), float(m.group("x1")), float(m.group("y1"))),
            )
            deps_txt = [d.strip() for d in m.group("deps").split(",") if d.strip()]
            deps = tuple(sorted(ids[int(d)] for d in deps_txt if int(d) in ids))
        except (ValueError, KeyError):
            return Parsed(None, answer, len(lines), n_steps, True)
        ids[sid] = len(triplets)
        triplets.append(Triplet(question=m.group("q").strip(), answer=m.group("a").strip(), evidence=ev, depends_on=deps))
    if not triplets or answer is None:
        return Parsed(None, answer, len(lines), n_steps, True)
    chain = TripletChain(video_id=video_id, question=question, triplets=triplets, final_answer=answer, question_type=question_type)
    try:
        chain.validate()
    except ValueError:
        return Parsed(None, answer, len(lines), n_steps, True)
    return Parsed(chain, answer, len(lines), n_steps, True)


def step_spans(text: str) -> list[tuple[int, int]]:
    """Character spans [start, end) of the step lines inside <think>, in order. When `parse_chain(text)`
    returns a chain, span k belongs to triplet k (both walk the same matching lines)."""
    think = THINK_RE.search(text)
    if not think:
        return []
    out = []
    pos = think.start(1)
    for raw in think.group(1).splitlines(keepends=True):
        ln = raw.strip()
        if ln and STEP_RE.match(ln):
            lead = len(raw) - len(raw.lstrip())
            out.append((pos + lead, pos + lead + len(ln)))
        pos += len(raw)
    return out
