"""Pseudo-labeled grounded chains for real video (NExT-QA), proposal Section 5.1 source 2.

  cs-pseudolabel teach  --pool <pack>/nextqa_train_pool.jsonl --frames <pack> --out teach.jsonl
      Teacher (Qwen2.5-VL-7B) sees 16 frames, the question, the options and the correct option, and writes
      2-4 observation steps, each naming a visible entity and the frame index (0-15) where it is seen.
  cs-pseudolabel ground --teach teach.jsonl --frames <pack> --out-sft sft.jsonl --out-rl rl.jsonl
      Grounding DINO localizes each step's named entity at its frame; steps whose phrase is not detected
      (score < threshold) are dropped; chains with no surviving step are dropped. The detector box becomes
      the step's evidence. Writes per-option SFT records (correct option -> "yes", one sampled wrong
      option -> "no", same observation steps) and per-option RL records with the chain as `gt_chains`.

Frames are the 16 uniformly sampled frames of the pack; step spans are indices in that 16-frame grid, and
the harness/trainers treat them as such for real video (n_total = 16 for these records).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

LETTERS = "ABCDE"

TEACHER_PROMPT = (
    "You are given 16 frames of a video, numbered 0 to 15 in time order, a question and its answer options. "
    "The correct option is {letter}: {correct}.\n"
    "Write the visual evidence that justifies this answer as 2 to 4 numbered observation steps. Each step must "
    "name ONE visible person or object as a short noun phrase that a detector could find (for example 'the man in "
    "the blue sweater', 'the white dog', 'the red cup'), say what it is doing or where it is, and give the frame index "
    "where it is clearly visible. Use exactly this format per line:\n"
    "[n] entity: <noun phrase> | observation: <one sentence> | frame: <0-15>\n"
    "Question: {question}?\nOptions:\n{options}\nSteps:"
)
STEP_RE = re.compile(r"\[(\d+)\]\s*entity:\s*(.+?)\s*\|\s*observation:\s*(.+?)\s*\|\s*frame:\s*(\d+)", re.IGNORECASE)


def parse_teacher(text: str) -> list[dict]:
    steps = []
    for m in STEP_RE.finditer(text):
        f = int(m.group(4))
        if 0 <= f <= 15 and m.group(2).strip():
            steps.append({"entity": m.group(2).strip().rstrip("."), "observation": m.group(3).strip(), "frame": f})
    return steps[:4]


def teach(pool: Path, frames_root: Path, out: Path, model: str, batch_size: int, limit: int | None) -> None:
    from causalsight.models.qwen_vl import QwenVLBackend
    from causalsight.train.data import load_frames

    recs = [json.loads(line) for line in pool.open()]
    if limit:
        recs = recs[:limit]
    be = QwenVLBackend(model)
    done = set()
    if out.exists():
        done = {json.loads(line)["problem_id"] for line in out.open()}
    stats = Counter()
    with out.open("a") as f:
        todo = [r for r in recs if r["problem_id"] not in done]
        for s in range(0, len(todo), batch_size):
            b = todo[s : s + batch_size]
            frames = [load_frames(frames_root, {"frames": f"frames/{r['video']}"}, 16) for r in b]
            prompts = []
            for r in b:
                opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(r["options"]))
                prompts.append(TEACHER_PROMPT.format(letter=LETTERS[r["answer_idx"]], correct=r["options"][r["answer_idx"]], question=r["question"], options=opts))
            outs = be.generate_batch(frames, prompts, max_new_tokens=320)
            for r, o in zip(b, outs):
                steps = parse_teacher(o)
                stats["ok" if steps else "unparsed"] += 1
                f.write(json.dumps({**r, "teacher_raw": o, "steps": steps}) + "\n")
            f.flush()
            if (s // batch_size) % 20 == 0:
                print(f"  {s + len(b)}/{len(todo)} {dict(stats)}", flush=True)
    print("teach done:", dict(stats))


def ground(teach_path: Path, frames_root: Path, out_sft: Path, out_rl: Path, detector: str, threshold: float, seed: int) -> None:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from causalsight.data.schema import Evidence, Triplet, TripletChain
    from causalsight.train.data import load_frames
    from causalsight.train.format import format_chain

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(detector)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(detector).to(device).eval()
    rng = random.Random(seed)
    stats = Counter()
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with out_sft.open("w") as fs, out_rl.open("w") as fr:
        for line in teach_path.open():
            r = json.loads(line)
            if not r["steps"]:
                stats["no_steps"] += 1
                continue
            frames = load_frames(frames_root, {"frames": f"frames/{r['video']}"}, 16)
            triplets = []
            for st in r["steps"]:
                best = None
                for f in (st["frame"], max(0, st["frame"] - 1), min(15, st["frame"] + 1)):
                    img = frames[f]
                    inputs = proc(images=img, text=st["entity"].lower() + " .", return_tensors="pt").to(device)
                    with torch.no_grad():
                        o = det(**inputs)
                    res = proc.post_process_grounded_object_detection(o, inputs.input_ids, threshold=threshold, text_threshold=threshold, target_sizes=[img.size[::-1]])[0]
                    if len(res["scores"]):
                        i = int(res["scores"].argmax())
                        sc = float(res["scores"][i])
                        if best is None or sc > best[0]:
                            x0, y0, x1, y1 = [float(v) for v in res["boxes"][i]]
                            W, H = img.size
                            best = (sc, f, (max(0, x0 / W), max(0, y0 / H), min(1, x1 / W), min(1, y1 / H)))
                if best is None:
                    stats["step_undetected"] += 1
                    continue
                sc, f, box = best
                if box[2] - box[0] <= 0.01 or box[3] - box[1] <= 0.01:
                    stats["step_degenerate"] += 1
                    continue
                stats["step_kept"] += 1
                triplets.append(Triplet(question=f"Where is {st['entity']}?", answer=f"{st['entity']}: {st['observation']}", evidence=Evidence(f, f, box), role="observe"))
            if not triplets:
                stats["chain_dropped"] += 1
                continue
            stats["chain_kept"] += 1
            correct = r["answer_idx"]
            wrong = rng.choice([i for i in range(5) if i != correct])
            for idx, gold in ((correct, "yes"), (wrong, "no")):
                prompt = f"{r['question']}?\nOption {LETTERS[idx]}: {r['options'][idx]}\nIs this option correct? Answer yes or no."
                chain = TripletChain(video_id=r["video"], question=prompt, triplets=triplets, final_answer=gold, question_type=r["family"], meta={"source": "nextqa", "problem_id": r["problem_id"], "option": idx})
                pid = f"{r['problem_id']}_{idx}"
                fs.write(json.dumps({"problem_id": pid, "data_type": "video", "path": r["path"], "frames": f"frames/{r['video']}", "question_type": r["family"], "subtype": r["type_code"], "problem": prompt, "response": format_chain(chain), "gold": gold, "chain": json.loads(chain.to_json()), "n_total_frames": 16}) + "\n")
                fr.write(json.dumps({"problem_id": pid, "data_type": "video", "path": r["path"], "frames": f"frames/{r['video']}", "question_type": r["family"], "subtype": r["type_code"], "problem_type": "free-form", "problem": prompt, "options": [], "gold": gold, "solution": f"<answer>{gold}</answer>", "gt_chains": [{"question_type": r["family"], "final_answer": gold, "choice_id": idx, "triplets": json.loads(chain.to_json())["triplets"]}], "n_total_frames": 16}) + "\n")
            if stats["chain_kept"] % 200 == 0:
                print(f"  {dict(stats)}", flush=True)
    print("ground done:", dict(stats))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach")
    t.add_argument("--pool", type=Path, required=True)
    t.add_argument("--frames", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--limit", type=int, default=None)
    g = sub.add_parser("ground")
    g.add_argument("--teach", type=Path, required=True)
    g.add_argument("--frames", type=Path, required=True)
    g.add_argument("--out-sft", type=Path, required=True)
    g.add_argument("--out-rl", type=Path, required=True)
    g.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    g.add_argument("--threshold", type=float, default=0.35)
    g.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "teach":
        teach(a.pool, a.frames, a.out, a.model, a.batch_size, a.limit)
    else:
        ground(a.teach, a.frames, a.out_sft, a.out_rl, a.detector, a.threshold, a.seed)


if __name__ == "__main__":
    main()
