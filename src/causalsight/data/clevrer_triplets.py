"""Programmatic CLEVRER triplet generator (proposal Section 5.1, source 1).

Runs each question program through the traced executor and turns the *entity-producing* steps into
triplets; scalar steps (count, exist, query_*, belong_to, negate) fold into the final answer so no
triplet merely restates the answer.

Landmark kinds
  resolve    unique/filter_order/get_col_partner/get_object -> one object or one event
  enumerate  a set consumed by count/exist, or filter_ancestor -> list of entities (or "none")
  observe    a possible collision from a choice program -> "do X and Y collide in the video?"

Predictive and counterfactual outcomes need the physics simulator, so their final answers come from
the choice labels; every observable intermediate step is still grounded and verifiable.

CLI:  cs-triplets --split train --n-videos 100 --out data/triplets/clevrer_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from causalsight.data.clevrer_evidence import (
    ProposalIndex,
    evidence_for_objects,
    whole_video_evidence,
)
from causalsight.data.clevrer_executor import UNKNOWN, Step, TracedExecutor, Val
from causalsight.data.clevrer_sim import Event, Scene
from causalsight.data.schema import Evidence, Triplet, TripletChain

SINGULAR = {
    "collisions": "collision",
    "possible collisions": "possible collision",
    "entries into the scene": "entry into the scene",
    "exits from the scene": "exit from the scene",
}


class ChainBuilder:
    def __init__(self, scene: Scene, proposals: ProposalIndex, tol: int = 3) -> None:
        self.s = scene
        self.p = proposals
        self.tol = tol

    # ---------------------------------------------------------------- public

    def build(self, question: dict, choice: dict | None = None) -> tuple[TripletChain | None, str]:
        """Returns (chain, status). status is 'ok' or a skip reason."""
        ex = TracedExecutor(self.s)
        program = (choice["program"] if choice else []) + question["program"]
        out = ex.run(program)
        exec_answer = TracedExecutor.answer_str(out)
        if out.kind == "error":
            return None, "executor_error"

        if choice is None:
            final = str(question["answer"])
            if exec_answer != final:
                return None, "executor_disagrees"
        else:
            final = "yes" if choice["answer"] == "correct" else "no"
            if exec_answer not in (UNKNOWN,) and {"yes": "yes", "no": "no"}.get(exec_answer) not in (final, None):
                return None, "executor_disagrees"

        landmarks = self._select_landmarks(ex.steps)
        if not landmarks:
            return None, "no_landmarks"

        triplets: list[Triplet] = []
        step_to_triplet: dict[int, int] = {}
        seen_qa: dict[tuple, int] = {}
        for si in landmarks:
            st = ex.steps[si]
            made = self._make_triplet(ex, st, landmarks[si])
            if made is None:
                return None, "no_evidence"
            q, a, ev = made
            # the program often re-derives the same entity ("the gray sphere" / "the gray object");
            # one triplet per resolved object or event, keyed by the entity, not the phrasing
            key = ("entity", st.out.kind, repr(st.out.data)) if landmarks[si] == "resolve" else ("qa", q, a)
            if key in seen_qa:
                step_to_triplet[si] = seen_qa[key]
                continue
            deps = tuple(sorted(self._landmark_deps(ex.steps, si, step_to_triplet)))
            step_to_triplet[si] = len(triplets)
            seen_qa[key] = len(triplets)
            triplets.append(Triplet(question=q, answer=a, evidence=ev, depends_on=deps, role=landmarks[si]))

        meta = {
            "scene_index": self.s.scene_index,
            "question_id": question["question_id"],
            "question_subtype": question.get("question_subtype"),
            "program": question["program"],
            "executor_answer": exec_answer,
        }
        if choice is not None:
            meta.update({"choice_id": choice["choice_id"], "choice": choice["choice"], "choice_program": choice["program"]})
        chain = TripletChain(
            video_id=self.s.video_filename,
            question=question["question"],
            triplets=triplets,
            final_answer=final,
            question_type=question["question_type"],
            meta=meta,
        )
        chain.validate()
        return chain, "ok"

    # ---------------------------------------------------------------- landmarks

    def _select_landmarks(self, steps: list[Step]) -> dict[int, str]:
        marks: dict[int, str] = {}
        for st in steps:
            if st.op in ("unique", "filter_order", "get_col_partner", "get_object") and st.out.kind in ("object", "event"):
                src = st.inputs[0]
                if st.out.kind == "event" and st.out.data.frame < 0:
                    marks[st.idx] = "observe"  # possible collision from all_events
                else:
                    marks[st.idx] = "resolve"
            elif st.op in ("count", "exist"):
                src = st.inputs[0]
                if src.step is not None and src.kind in ("objects", "events"):
                    marks[src.step] = "enumerate"
            elif st.op == "filter_ancestor" and st.out.kind == "events":
                marks[st.idx] = "enumerate"
        return dict(sorted(marks.items()))

    def _landmark_deps(self, steps: list[Step], si: int, step_to_triplet: dict[int, int]) -> set[int]:
        out: set[int] = set()
        seen: set[int] = set()
        stack = list(steps[si].input_steps)
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            if j in step_to_triplet:
                out.add(step_to_triplet[j])
            else:
                stack.extend(steps[j].input_steps)
        return out

    # ---------------------------------------------------------------- triplet construction

    def _make_triplet(self, ex: TracedExecutor, st: Step, kind: str) -> tuple[str, str, Evidence] | None:
        v = st.out
        if kind == "resolve" and v.kind == "object":
            return self._resolve_object(ex, st)
        if kind == "resolve" and v.kind == "event":
            return self._resolve_event(ex, st)
        if kind == "observe":
            return self._observe_possible_collision(ex, st)
        if kind == "enumerate" and v.kind == "objects":
            return self._enumerate_objects(ex, st)
        if kind == "enumerate" and v.kind == "events":
            return self._enumerate_events(ex, st)
        return None

    # -- objects

    def _resolve_object(self, ex: TracedExecutor, st: Step) -> tuple[str, str, Evidence] | None:
        obj = st.out.data
        src = st.inputs[0]
        if st.op == "unique":
            partial = self._partial_desc(src)
            q = f"Which object is {partial}?"
            frame = self._motion_frame(ex, src)
        elif st.op == "get_col_partner":
            ev: Event = st.inputs[0].data
            other = st.inputs[1].data
            q = f"Which object does {self.s.describe(other)} collide with at frame {ev.frame}?"
            frame = ev.frame
        elif st.op == "get_object":
            ev = st.inputs[0].data
            verb = "enters" if ev.type == "in" else "exits"
            q = f"Which object {verb} the scene at frame {ev.frame}?"
            frame = ev.frame if ev.type == "in" else ev.frame - 1
        else:
            return None
        if frame is None:
            frame = self._best_frame(obj)
        a = self.s.describe(obj)
        ev_box = self._objects_evidence([obj], frame)
        return (q, a, ev_box) if ev_box else None

    def _best_frame(self, obj: int, candidates: list[int] | None = None) -> int:
        """Frame with the largest detected box (objects sliding in at the edge give slivers)."""
        attrs = self.s.attrs[obj]
        cands = candidates if candidates is not None else [f for f in range(0, self.p.n_frames, 3) if self.s.is_visible(obj, f)]
        best_f, best_area = (cands[0] if cands else 0), -1.0
        for f in cands:
            hit = self.p.box(attrs, f, 0)
            if hit:
                b = hit[0]
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > best_area:
                    best_f, best_area = f, area
        return best_f

    def _partial_desc(self, v: Val) -> str:
        base = self.s.describe(-1, v.attrs) if v.attrs else "the object"
        if v.quals:
            return f"{base} that is {v.quals[0]}"
        return base

    def _motion_frame(self, ex: TracedExecutor, v: Val) -> int | None:
        """If the object set came through filter_moving/stationary at a frame, return that frame."""
        if v.step is None:
            return None
        st = ex.steps[v.step]
        if st.op in ("filter_moving", "filter_stationary"):
            fr = st.inputs[1]
            return None if fr.kind == "null" else int(fr.data)
        return None

    def _representative_frame(self, objs: list[int]) -> int:
        """Earliest frame where the most of `objs` are visible and detected."""
        best_f, best_n = 0, -1
        for f in range(0, self.p.n_frames, 4):
            n = sum(1 for o in objs if self.s.is_visible(o, f) and self.p.box(self.s.attrs[o], f, 0))
            if n > best_n:
                best_f, best_n = f, n
            if n == len(objs):
                break
        return best_f

    def _objects_evidence(self, objs: list[int], frame: int, span: tuple[int, int] | None = None) -> Evidence | None:
        attrs = [self.s.attrs[o] for o in objs]
        ev = evidence_for_objects(self.p, attrs, frame, span, self.tol)
        if ev is None and self.tol < 10:
            ev = evidence_for_objects(self.p, attrs, frame, span, 10)
        return ev

    # -- events

    def _event_evidence(self, e: Event) -> Evidence | None:
        n = self.p.n_frames
        if e.type == "collision":
            f = e.frame
            return self._objects_evidence(list(e.objects), f, (max(0, f - 2), min(n - 1, f + 2)))
        if e.type == "in":
            f0, f1 = e.frame, min(n - 1, e.frame + 4)
            f = self._best_frame(e.objects[0], list(range(f0, f1 + 1)))
            return self._objects_evidence(list(e.objects), f, (f0, f1))
        if e.type == "out":
            f0, f1 = max(0, e.frame - 5), max(0, e.frame - 1)
            f = self._best_frame(e.objects[0], list(range(f0, f1 + 1)))
            return self._objects_evidence(list(e.objects), f, (f0, f1))
        return None

    def _resolve_event(self, ex: TracedExecutor, st: Step) -> tuple[str, str, Evidence] | None:
        e: Event = st.out.data
        src = st.inputs[0]
        quals = src.quals if st.op == "unique" else st.out.quals
        phrase = self._event_phrase(quals, singular=True)
        q = f"What is the {phrase}?" if phrase.split(" ")[0] in ("first", "second", "last") else f"Which {phrase} happens in the video?"
        a = ex.describe_event(e)
        ev = self._event_evidence(e)
        return (q, a, ev) if ev else None

    def _event_phrase(self, quals: list[str], singular: bool) -> str:
        parts = [q for q in quals if q != "events"]
        if not parts:
            return "event"
        order = None
        if parts[0] in ("first", "second", "last"):
            order, parts = parts[0], parts[1:]
        head = parts[0] if parts else "event"
        if singular:
            for k, v in SINGULAR.items():
                if head.startswith(k):
                    head = v + head[len(k):]
                    break
        phrase = " ".join([head] + parts[1:])
        return f"{order} {phrase}" if order else phrase

    def _observe_possible_collision(self, ex: TracedExecutor, st: Step) -> tuple[str, str, Evidence] | None:
        e: Event = st.out.data
        a_id, b_id = e.objects
        d = self.s.describe
        q = f"Do {d(a_id)} and {d(b_id)} collide during the video?"
        real = next((c for c in self.s.collisions if set(c.objects) == {a_id, b_id}), None)
        if real:
            a = f"yes, at frame {real.frame}"
            ev = self._event_evidence(real)
        else:
            a = "no"
            frame = self._last_common_frame([a_id, b_id])
            ev = self._objects_evidence([a_id, b_id], frame, (max(0, frame - 4), frame))
        return (q, a, ev) if ev else None

    def _last_common_frame(self, objs: list[int]) -> int:
        for f in range(self.p.n_frames - 1, -1, -1):
            if all(self.s.is_visible(o, f) for o in objs):
                return f
        return self.p.n_frames - 1

    # -- sets

    def _enumerate_objects(self, ex: TracedExecutor, st: Step) -> tuple[str, str, Evidence] | None:
        v = st.out
        objs: list[int] = v.data
        a_ = v.attrs
        base = " ".join(p for p in (a_.get("color"), a_.get("material"), (a_.get("shape") + "s") if a_.get("shape") else "objects") if p)
        q = f"Which {base} are {v.quals[0]}?" if v.quals else f"Which {base} are there?"
        a = ", ".join(self.s.describe(o) for o in objs) if objs else "none"
        if not objs:
            return q, a, whole_video_evidence(self.p)
        frame = self._motion_frame(ex, v)
        if frame is None:
            frame = self._representative_frame(objs)
        span = (max(0, frame - 3), min(self.p.n_frames - 1, frame + 3)) if v.quals else None
        ev = self._objects_evidence(objs, frame, span)
        return (q, a, ev) if ev else None

    def _enumerate_events(self, ex: TracedExecutor, st: Step) -> tuple[str, str, Evidence] | None:
        v = st.out
        evs: list[Event] = [e for e in v.data if e.type not in ("start", "end")]
        phrase = self._event_phrase(v.quals, singular=False)
        q = f"Which {phrase} are there?"
        a = "; ".join(ex.describe_event(e) for e in evs) if evs else "none"
        if not evs:
            return q, a, whole_video_evidence(self.p)
        boxes = []
        for e in evs:
            ev = self._event_evidence(e)
            if ev is None:
                return None
            boxes.append(ev)
        t0 = min(b.t_start for b in boxes)
        t1 = max(b.t_end for b in boxes)
        box = (min(b.box[0] for b in boxes), min(b.box[1] for b in boxes), max(b.box[2] for b in boxes), max(b.box[3] for b in boxes))
        return q, a, Evidence(t_start=t0, t_end=t1, box=box)


# -------------------------------------------------------------------- CLI


def find_annotation(root: Path, split: str, scene_index: int) -> Path:
    lo = (scene_index // 1000) * 1000
    return root / "annotations" / split / f"annotation_{split}" / f"annotation_{lo:05d}-{lo + 1000:05d}" / f"annotation_{scene_index:05d}.json"


def generate(root: Path, split: str, n_videos: int | None, out: Path, types: set[str]) -> dict:
    videos = json.loads((root / "questions" / f"{split}.json").read_text())
    if n_videos:
        videos = videos[:n_videos]
    stats = Counter()
    lengths: dict[str, Counter] = defaultdict(Counter)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for v in videos:
            scene = Scene.from_annotation(find_annotation(root, split, v["scene_index"]))
            props = ProposalIndex.load(root / "derender_proposals" / f"proposal_{v['scene_index']:05d}.json")
            b = ChainBuilder(scene, props)
            for q in v["questions"]:
                t = q["question_type"]
                if t not in types:
                    continue
                units = [None] if t == "descriptive" else q["choices"]
                for c in units:
                    chain, status = b.build(q, c)
                    stats[(t, status)] += 1
                    if chain:
                        lengths[t][len(chain.triplets)] += 1
                        f.write(chain.to_json() + "\n")
    return {"stats": stats, "lengths": lengths}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate CLEVRER triplet chains")
    ap.add_argument("--root", type=Path, default=Path("data/raw/clevrer"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-videos", type=int, default=None)
    ap.add_argument("--types", default="descriptive,explanatory,predictive,counterfactual")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = generate(args.root, args.split, args.n_videos, args.out, set(args.types.split(",")))
    print("chains by (type, status):")
    for k, c in sorted(r["stats"].items()):
        print(f"  {k[0]:15s} {k[1]:20s} {c}")
    print("chain length histogram by type:")
    for t, h in r["lengths"].items():
        print(f"  {t:15s} {dict(sorted(h.items()))}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
