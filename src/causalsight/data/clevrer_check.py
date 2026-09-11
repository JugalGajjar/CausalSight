"""CLEVRER annotation format checker (notes/PLAN.md, Day 2).

Discovers the actual on-disk structure of the questions file, the per-video annotations, and the
derender proposals, then reports whether each field the triplet generator needs can be derived:

  q_i  sub-question        <- question `program` (postfix module list)
  a_i  intermediate answer <- executing the program prefix against the annotation
  e_i  evidence box        <- derender proposal masks (2D); annotations only have 3D world locations
  e_i  frame span          <- collision `frame_id`, motion_trajectory `frame_id`
  d_i  dependencies        <- program data flow

Usage: cs-clevrer-check --root data/raw/clevrer --split train --n-videos 20 --out results/clevrer_format_report.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from causalsight.data.rle import mask_to_box, rle_to_mask

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _type_tree(obj: Any, depth: int = 0, max_depth: int = 4, max_keys: int = 12) -> list[str]:
    """Render the shape of a JSON object: keys, types, list lengths, sample scalars."""
    pad = "  " * depth
    lines: list[str] = []
    if depth > max_depth:
        return [pad + "..."]
    if isinstance(obj, dict):
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_keys:
                lines.append(f"{pad}... ({len(obj) - max_keys} more keys)")
                break
            if isinstance(v, (dict, list)):
                desc = f"dict[{len(v)}]" if isinstance(v, dict) else f"list[{len(v)}]"
                lines.append(f"{pad}{k}: {desc}")
                lines.extend(_type_tree(v, depth + 1, max_depth, max_keys))
            else:
                lines.append(f"{pad}{k}: {type(v).__name__} = {_short(v)}")
    elif isinstance(obj, list) and obj:
        lines.append(f"{pad}[0]:")
        lines.extend(_type_tree(obj[0], depth + 1, max_depth, max_keys))
    return lines


def _short(v: Any, n: int = 60) -> str:
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


class Report:
    def __init__(self) -> None:
        self.sections: list[tuple[str, list[str]]] = []
        self.checks: list[tuple[str, str, str]] = []  # (status, name, detail)

    def section(self, title: str, lines: list[str]) -> None:
        self.sections.append((title, lines))

    def check(self, status: str, name: str, detail: str = "") -> None:
        self.checks.append((status, name, detail))

    def render(self) -> str:
        out = ["# CLEVRER format report", ""]
        out.append("## Checks")
        out.append("")
        out.append("| status | check | detail |")
        out.append("|---|---|---|")
        for s, n, d in self.checks:
            out.append(f"| {s} | {n} | {d} |")
        out.append("")
        for title, lines in self.sections:
            out.append(f"## {title}")
            out.append("")
            out.append("```")
            out.extend(lines)
            out.append("```")
            out.append("")
        return "\n".join(out)


# ---------------------------------------------------------------- questions


def check_questions(path: Path, rep: Report, sample_n: int) -> dict[str, Any]:
    if not path.exists():
        rep.check(FAIL, "questions file present", str(path))
        return {}
    data = json.loads(path.read_text())
    videos = data if isinstance(data, list) else data.get("videos", [])
    n_q = sum(len(v.get("questions", [])) for v in videos)
    rep.check(PASS, "questions file present", f"{len(videos)} videos, {n_q} questions")

    types = Counter()
    subtypes = Counter()
    prog_present = Counter()
    prog_missing = Counter()
    modules = Counter()
    n_choices = Counter()
    correct_per_q = Counter()
    choice_answers = Counter()
    desc_answers = Counter()
    samples: dict[str, dict] = {}
    for v in videos:
        for q in v.get("questions", []):
            t = q.get("question_type", "?")
            types[t] += 1
            subtypes[(t, q.get("question_subtype", "-"))] += 1
            if "program" in q:
                prog_present[t] += 1
                for m in q["program"]:
                    modules[m] += 1
            else:
                prog_missing[t] += 1
            if "choices" in q:
                n_choices[len(q["choices"])] += 1
                n_correct = 0
                for c in q["choices"]:
                    choice_answers[c.get("answer", "?")] += 1
                    if c.get("answer") == "correct":
                        n_correct += 1
                    for m in c.get("program", []):
                        modules[m] += 1
                correct_per_q[(t, n_correct)] += 1
            elif t == "descriptive":
                desc_answers[str(q.get("answer"))] += 1
            samples.setdefault(t, q)

    lines = [f"question types: {dict(types)}", ""]
    lines.append("subtypes:")
    for (t, st), c in sorted(subtypes.items()):
        lines.append(f"  {t:14s} {st:24s} {c}")
    lines.append("")
    lines.append(f"program present by type: {dict(prog_present)}")
    lines.append(f"program MISSING by type: {dict(prog_missing)}")
    lines.append(f"distinct program modules: {len(modules)}")
    lines.append("top modules:")
    for m, c in modules.most_common(40):
        lines.append(f"  {m:32s} {c}")
    lines.append("")
    lines.append(f"choices per MC question: {dict(n_choices)}")
    lines.append(f"choice answer values: {dict(choice_answers)}")
    lines.append(f"correct choices per question (type, n_correct): {dict(correct_per_q)}")
    lines.append(f"descriptive answer vocab ({len(desc_answers)}): {desc_answers.most_common(25)}")
    rep.section("Questions: statistics", lines)

    for t, q in samples.items():
        rep.section(f"Questions: sample {t}", _type_tree(q) + ["", json.dumps(q, indent=1)[:1500]])

    if prog_missing:
        rep.check(FAIL, "every train/val question has a program", f"missing: {dict(prog_missing)}")
    else:
        rep.check(PASS, "every train/val question has a program", f"{len(modules)} distinct modules")
    for t in ("descriptive", "explanatory", "predictive", "counterfactual"):
        if t not in types:
            rep.check(WARN, f"question type {t} present", "absent")
    return {"videos": videos, "modules": modules}


# ---------------------------------------------------------------- annotations


def _find_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def check_annotations(root: Path, rep: Report, n: int, rng: random.Random) -> dict[str, Any]:
    files = _find_files(root, "annotation_*.json")
    if not files:
        rep.check(FAIL, "annotation files found", f"no annotation_*.json under {root}")
        return {}
    rep.check(PASS, "annotation files found", f"{len(files)} files, e.g. {files[0].relative_to(root)}")
    picks = rng.sample(files, min(n, len(files)))

    keys = Counter()
    n_objects = Counter()
    n_frames = Counter()
    loc_dims = Counter()
    obj_fields = Counter()
    frame_fields = Counter()
    coll_fields = Counter()
    n_collisions = Counter()
    coll_frame_ids: list[int] = []
    frame_id_ranges: list[tuple[int, int]] = []
    inside_view_vals = Counter()
    first: dict | None = None
    for p in picks:
        a = json.loads(p.read_text())
        first = first or a
        keys.update(a.keys())
        n_objects[len(a.get("object_property", []))] += 1
        traj = a.get("motion_trajectory", [])
        n_frames[len(traj)] += 1
        if traj:
            frame_fields.update(traj[0].keys())
            fids = [f.get("frame_id", i) for i, f in enumerate(traj)]
            frame_id_ranges.append((min(fids), max(fids)))
            for o in traj[0].get("objects", []):
                obj_fields.update(o.keys())
                loc = o.get("location")
                if isinstance(loc, (list, tuple)):
                    loc_dims[len(loc)] += 1
            for f in traj:
                for o in f.get("objects", []):
                    inside_view_vals[o.get("inside_camera_view", "absent")] += 1
        colls = a.get("collision", [])
        n_collisions[len(colls)] += 1
        for c in colls:
            coll_fields.update(c.keys())
            if "frame_id" in c:
                coll_frame_ids.append(c["frame_id"])

    lines = [
        f"sampled {len(picks)} of {len(files)} annotation files",
        f"top-level keys: {dict(keys)}",
        f"objects per video: {dict(sorted(n_objects.items()))}",
        f"trajectory frames per video: {dict(sorted(n_frames.items()))}",
        f"frame_id ranges seen: {sorted(set(frame_id_ranges))}",
        f"per-frame fields: {dict(frame_fields)}",
        f"per-object fields: {dict(obj_fields)}",
        f"location dimensionality: {dict(loc_dims)}",
        f"inside_camera_view values: {dict(inside_view_vals)}",
        f"collisions per video: {dict(sorted(n_collisions.items()))}",
        f"collision fields: {dict(coll_fields)}",
        f"collision frame_id range: {(min(coll_frame_ids), max(coll_frame_ids)) if coll_frame_ids else 'none'}",
    ]
    rep.section("Annotations: statistics", lines)
    if first:
        rep.section("Annotations: structure of one file", _type_tree(first))

    if loc_dims and set(loc_dims) == {3}:
        rep.check(WARN, "annotation locations are 3D world coords", "no 2D boxes here; use derender proposals")
    elif loc_dims and 2 in loc_dims:
        rep.check(PASS, "annotation locations include 2D coords", str(dict(loc_dims)))
    else:
        rep.check(FAIL, "annotation object location field", f"unexpected: {dict(loc_dims)}")
    if coll_frame_ids:
        rep.check(PASS, "collision events carry frame_id", f"{len(coll_frame_ids)} events in sample")
    else:
        rep.check(FAIL, "collision events carry frame_id", "none found")
    if "object_property" in keys and {"color", "material", "shape"} <= (set(obj_fields) | _prop_fields(first)):
        rep.check(PASS, "object_property has color/material/shape", "")
    else:
        rep.check(FAIL, "object_property has color/material/shape", str(dict(obj_fields)))
    return {"n_frames": n_frames, "files": files}


def _prop_fields(a: dict | None) -> set[str]:
    if not a or not a.get("object_property"):
        return set()
    return set(a["object_property"][0].keys())


# ---------------------------------------------------------------- derender proposals


def check_proposals(root: Path, rep: Report, n: int, rng: random.Random, expected_frames: Counter | None) -> None:
    cands = [p for p in root.rglob("*.json") if "proposal" in str(p.parent).lower() or p.name.startswith("sim_")]
    cands = [p for p in cands if "annotation_" not in p.name and p.name not in ("train.json", "validation.json", "test.json")]
    if not cands:
        rep.check(FAIL, "derender proposals found", "download derender_proposals.zip; no proposal json found")
        return
    cands = sorted(cands)
    rep.check(PASS, "derender proposals found", f"{len(cands)} files, e.g. {cands[0].relative_to(root)}")
    picks = rng.sample(cands, min(n, len(cands)))

    top_keys = Counter()
    frame_keys = Counter()
    obj_keys = Counter()
    n_frames = Counter()
    objs_per_frame = Counter()
    mask_types = Counter()
    box_ok = 0
    box_fail = 0
    example_box = None
    mask_size = Counter()
    first: dict | None = None
    for p in picks:
        d = json.loads(p.read_text())
        first = first or d
        top_keys.update(d.keys() if isinstance(d, dict) else ["<list>"])
        frames = d.get("frames", d if isinstance(d, list) else [])
        n_frames[len(frames)] += 1
        for f in frames:
            frame_keys.update(f.keys())
            objs = f.get("objects", [])
            objs_per_frame[len(objs)] += 1
            for o in objs:
                obj_keys.update(o.keys())
                m = o.get("mask")
                if m is None:
                    mask_types["absent"] += 1
                    continue
                mask_types[type(m.get("counts", None)).__name__ if isinstance(m, dict) else type(m).__name__] += 1
                if isinstance(m, dict) and "size" in m:
                    mask_size[tuple(m["size"])] += 1
                    if example_box is None or box_ok < 50:
                        try:
                            box = mask_to_box(rle_to_mask(m))
                            if box is None:
                                box_fail += 1
                            else:
                                box_ok += 1
                                example_box = example_box or box
                        except Exception as e:  # noqa: BLE001
                            box_fail += 1
                            rep.section("Proposals: RLE decode error", [repr(e)])

    lines = [
        f"sampled {len(picks)} of {len(cands)} proposal files",
        f"top-level keys: {dict(top_keys)}",
        f"frames per file: {dict(sorted(n_frames.items()))}",
        f"per-frame keys: {dict(frame_keys)}",
        f"objects per frame: {dict(sorted(objs_per_frame.items()))}",
        f"per-object keys: {dict(obj_keys)}",
        f"mask counts type: {dict(mask_types)}",
        f"mask sizes (h, w): {dict(mask_size)}",
        f"RLE -> box decode: ok={box_ok} fail={box_fail}, example normalized box={example_box}",
    ]
    if expected_frames:
        lines.append(f"annotation trajectory frames per video (for comparison): {dict(sorted(expected_frames.items()))}")
    rep.section("Proposals: statistics", lines)
    if first:
        rep.section("Proposals: structure of one file", _type_tree(first, max_depth=5))

    if box_ok and not box_fail:
        rep.check(PASS, "proposal masks decode to 2D boxes", f"{box_ok} decoded, sizes {dict(mask_size)}")
    elif box_ok:
        rep.check(WARN, "proposal masks decode to 2D boxes", f"ok={box_ok} fail={box_fail}")
    else:
        rep.check(FAIL, "proposal masks decode to 2D boxes", f"mask types {dict(mask_types)}")
    if {"color", "material", "shape"} <= set(obj_keys):
        rep.check(PASS, "proposal objects carry attributes for matching to annotation objects", "")
    else:
        rep.check(WARN, "proposal objects carry attributes", f"keys {dict(obj_keys)}; will need trajectory-based matching")
    if expected_frames and n_frames and set(n_frames) != set(expected_frames):
        rep.check(WARN, "proposal frame count matches annotation frame count", f"proposals {dict(n_frames)} vs annotations {dict(expected_frames)}")
    elif expected_frames and n_frames:
        rep.check(PASS, "proposal frame count matches annotation frame count", str(dict(n_frames)))


# ---------------------------------------------------------------- verdict


def verdict(rep: Report) -> None:
    status = {n: s for s, n, _ in rep.checks}
    rows = [
        ("q_i sub-question from program", status.get("every train/val question has a program", FAIL), "program templates per module"),
        ("a_i intermediate answer", WARN, "needs a program executor over annotations (port from chuangg/CLEVRER executor)"),
        ("e_i box from proposals", status.get("proposal masks decode to 2D boxes", FAIL), "mask -> tight box, normalized"),
        ("e_i frame span", status.get("collision events carry frame_id", FAIL), "collision frame_id; trajectory frame_id for object steps"),
        ("d_i dependencies", status.get("every train/val question has a program", FAIL), "postfix program data flow"),
    ]
    rep.section("Verdict: triplet field derivability", [f"{s:5s} {n:36s} {d}" for n, s, d in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description="Check CLEVRER on-disk format against triplet generator needs")
    ap.add_argument("--root", type=Path, default=Path("data/raw/clevrer"))
    ap.add_argument("--split", choices=["train", "validation"], default="train")
    ap.add_argument("--n-videos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/clevrer_format_report.md"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rep = Report()
    check_questions(args.root / "questions" / f"{args.split}.json", rep, sample_n=1)
    ann_root = args.root / "annotations"
    ann = check_annotations(ann_root if ann_root.exists() else args.root, rep, args.n_videos, rng)
    check_proposals(args.root, rep, args.n_videos, rng, ann.get("n_frames"))
    verdict(rep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rep.render())
    print(f"\n{'status':6s} check")
    for s, n, d in rep.checks:
        print(f"{s:6s} {n}  {('- ' + d) if d else ''}")
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
