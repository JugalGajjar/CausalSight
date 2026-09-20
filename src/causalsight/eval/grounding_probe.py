"""Probe: frame-consistent grounding of a chain model on N items, greedy and sampled, on either split.

  cs-grounding-probe --model <adapter> --records <rl jsonl> --frames-root <train frames dir> --proposals <dir>
  cs-grounding-probe --model <adapter> --evalpack <dir>

Same scoring function as the training reward (`grounding_obj_steps`). Used to reconcile the grounding
measured during RL sampling on training videos with the one measured by greedy evaluation on validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from causalsight.models.qwen_vl import QwenVLBackend
from causalsight.train.data import load_frames
from causalsight.train.format import SYSTEM_PROMPT, parse_chain
from causalsight.train.rewards.chain_rewards import grounding_obj_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--frames-root", type=Path, default=None)
    ap.add_argument("--proposals", type=Path, default=None)
    ap.add_argument("--evalpack", type=Path, default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.evalpack:
        from causalsight.eval.benchmarks.clevrer import ClevrerBenchmark

        items = list(ClevrerBenchmark(a.evalpack, mc_mode="option").items("validation"))
        have = {p.name for p in (a.evalpack / "frames").iterdir()}
        items = [it for it in items if it.video_path.stem in have]
        recs = [{"problem_id": it.id, "problem": it.prompt, "frames": f"frames/{it.video_path.stem}", "qtype": it.question_type} for it in rng.sample(items, a.n)]
        root, props = a.evalpack, a.evalpack / "derender_proposals"
    else:
        allr = [json.loads(line) for line in a.records.open()]
        recs = [{**r, "frames": f"frames/{Path(r['path']).stem}", "qtype": r["question_type"]} for r in rng.sample(allr, a.n)]
        root, props = a.frames_root, a.proposals
    os.environ["CS_PROPOSALS"] = str(props)
    be = QwenVLBackend(a.model)
    for label, temp in (("greedy", 0.0), ("sampled T=1", 1.0)):
        ious, fmt, nsteps = [], 0, 0
        for s in range(0, len(recs), a.batch_size):
            b = recs[s : s + a.batch_size]
            outs = be.generate_batch([load_frames(root, r, 16) for r in b], [r["problem"] + "\n" + SYSTEM_PROMPT for r in b], 640, temperature=temp)
            for r, o in zip(b, outs):
                fmt += parse_chain(o).chain is not None
                st = grounding_obj_steps(o, r)
                ious.extend(st.values())
                nsteps += len(st)
        print(f"{label:12s} items {len(recs)} | format {fmt / len(recs):.3f} | scored steps {nsteps} | obj_iou {sum(ious) / max(1, len(ious)):.3f} | >=0.5 {sum(v >= 0.5 for v in ious) / max(1, len(ious)):.3f}")


if __name__ == "__main__":
    main()
