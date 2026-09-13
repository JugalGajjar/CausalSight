"""Stage 1: triplet-chain SFT with the bridge loss (proposal Section 5.1).

  cs-sft --config configs/stage1_sft.yaml --data data/frames/train --out runs/stage1 [--resume]

Per example (video frames, question, chain text): standard NLL over the response tokens, plus, with
probability `bridge_prob`, the bridge term  L_bridge = -log(1 - p(a* | v, q, c~))  where c~ is the chain
with one triplet on the dependency path to the answer corrupted (its intermediate answer replaced by a
distractor). The term says: if the chain is broken, the correct final answer should not remain likely.
LoRA, gradient accumulation, resumable checkpoints, same conventions as cs-grpo.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import yaml

from causalsight.data.clevrer_sim import COLORS, MATERIALS, SHAPES
from causalsight.data.schema import Triplet, TripletChain
from causalsight.train.data import load_frames, load_records
from causalsight.train.format import SYSTEM_PROMPT, format_chain

ATTR_VOCAB = {**{c: COLORS for c in COLORS}, **{m: MATERIALS for m in MATERIALS}, **{s: SHAPES for s in SHAPES}}


# ---------------------------------------------------------------- corruption


def corrupt_answer(answer: str, rng: random.Random) -> str | None:
    """Replace one attribute word with a different value, flip yes/no, or shift a frame number.
    Returns None if nothing corruptible was found."""
    words = answer.split()
    idx = [i for i, w in enumerate(words) if w.strip(",;.") in ATTR_VOCAB]
    if idx:
        i = rng.choice(idx)
        w = words[i].strip(",;.")
        words[i] = words[i].replace(w, rng.choice([v for v in ATTR_VOCAB[w] if v != w]))
        return " ".join(words)
    low = answer.strip().lower()
    if low.startswith("yes"):
        return "no"
    if low.startswith("no"):
        return "yes"
    m = re.search(r"frame (\d+)", answer)
    if m:
        f = int(m.group(1))
        return answer.replace(m.group(0), f"frame {max(0, f + rng.choice([-20, -12, 12, 20]))}")
    return None


def on_path_indices(chain: TripletChain) -> list[int]:
    """Indices of triplets on the dependency path to the final step (the last triplet and its ancestors)."""
    seen: set[int] = set()
    stack = [len(chain.triplets) - 1]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(chain.triplets[i].depends_on)
    return sorted(seen)


def corrupt_chain(chain: TripletChain, rng: random.Random) -> TripletChain | None:
    cands = on_path_indices(chain)
    rng.shuffle(cands)
    for i in cands:
        t = chain.triplets[i]
        new_a = corrupt_answer(t.answer, rng)
        if new_a and new_a != t.answer:
            ts = list(chain.triplets)
            ts[i] = Triplet(t.question, new_a, t.evidence, t.depends_on, t.role)
            return TripletChain(chain.video_id, chain.question, ts, chain.final_answer, chain.question_type, {**chain.meta, "corrupted": i})
    return None


# ---------------------------------------------------------------- trainer


class SFTTrainer:
    def __init__(self, cfg: dict, data_root: Path, out: Path, resume: bool) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg, self.out, self.data_root = cfg, out, data_root
        out.mkdir(parents=True, exist_ok=True)
        self.device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
        self.processor = AutoProcessor.from_pretrained(cfg["model"])
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["model"], torch_dtype=self.dtype, attn_implementation=cfg.get("attn", "sdpa"))
        if cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lora = cfg["lora"]
        self.model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0), target_modules=lora["targets"], task_type="CAUSAL_LM")).to(self.device)
        self.model.print_trainable_parameters()
        self.records = load_records(data_root / cfg["data_file"])
        self.rng = random.Random(cfg.get("seed", 0))
        self.order = list(range(len(self.records)))
        self.rng.shuffle(self.order)
        self.accum = int(cfg.get("grad_accum", 8))
        self.total_opt_steps = math.ceil(min(cfg.get("max_examples") or len(self.order), len(self.order)) * cfg.get("epochs", 1) / self.accum)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: min(1.0, (s + 1) / max(1, cfg.get("warmup", 20))) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, self.total_opt_steps)))))
        self.example = 0  # examples consumed
        self.opt_step = 0
        self.log = (out / "log.jsonl").open("a")
        if resume:
            self._load_state()

    def _save_state(self) -> None:
        import torch

        self.model.save_pretrained(self.out / "adapter")
        torch.save({"example": self.example, "opt_step": self.opt_step, "order": self.order, "rng": self.rng.getstate(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict()}, self.out / "trainer_state.pt")

    def _load_state(self) -> None:
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        p = self.out / "trainer_state.pt"
        if not p.exists():
            print("no checkpoint; starting fresh")
            return
        st = torch.load(p, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(self.model, load_file(self.out / "adapter" / "adapter_model.safetensors"))
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.example, self.opt_step, self.order = st["example"], st["opt_step"], st["order"]
        self.rng.setstate(st["rng"])
        print(f"resumed at example {self.example}, opt step {self.opt_step}")

    # ---- encoding

    def _encode(self, rec: dict, frames, response: str):
        """Returns processor inputs for prompt+response and the prompt length (tokens), so labels can mask the prompt."""
        from qwen_vl_utils import process_vision_info

        user = {"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": rec["problem"] + "\n" + SYSTEM_PROMPT}]}
        prompt_text = self.processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + response + self.processor.tokenizer.eos_token
        _imgs, videos = process_vision_info([user])
        full = self.processor(text=[full_text], videos=videos, padding=True, return_tensors="pt").to(self.device)
        prompt = self.processor(text=[prompt_text], videos=videos, padding=True, return_tensors="pt")
        return full, int(prompt["input_ids"].shape[1])

    def _token_logprobs(self, inputs, prompt_len: int):
        import torch

        logits = self.model(**inputs).logits[:, prompt_len - 1 : -1, :].float()
        targets = inputs["input_ids"][:, prompt_len:]
        lp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return lp, targets

    def _answer_span(self, response: str) -> tuple[int, int]:
        """Token offsets (within the response) of the text inside <answer>...</answer>."""
        tok = self.processor.tokenizer
        head = response[: response.index("<answer>") + len("<answer>")]
        body = response[: response.index("</answer>")]
        return len(tok(head, add_special_tokens=False)["input_ids"]), len(tok(body, add_special_tokens=False)["input_ids"])

    # ---- step

    def train_example(self, rec: dict) -> dict:
        import torch

        frames = load_frames(self.data_root, rec, self.cfg.get("n_frames"))
        chain = TripletChain.from_json(json.dumps(rec["chain"]))
        response = rec["response"]
        inputs, plen = self._encode(rec, frames, response)
        lp, _ = self._token_logprobs(inputs, plen)
        nll = -lp.mean()
        loss = nll
        stats = {"nll": nll.item()}
        if self.rng.random() < self.cfg.get("bridge_prob", 0.5) and self.cfg.get("bridge_weight", 0.0) > 0:
            corrupted = corrupt_chain(chain, self.rng)
            if corrupted is not None:
                c_resp = format_chain(corrupted)
                c_in, c_plen = self._encode(rec, frames, c_resp)
                c_lp, _ = self._token_logprobs(c_in, c_plen)
                a0, a1 = self._answer_span(c_resp)
                lp_ans = c_lp[0, a0:a1].sum()  # log p(a* | v, q, c~)
                p_ans = torch.exp(lp_ans.clamp(max=-1e-4))
                bridge = -torch.log1p(-p_ans)
                loss = loss + self.cfg["bridge_weight"] * bridge
                stats.update(bridge=bridge.item(), p_ans_corrupted=p_ans.item())
        (loss / self.accum).backward()
        stats["loss"] = loss.item()
        return stats

    def train(self) -> None:
        import torch

        n = min(self.cfg.get("max_examples") or len(self.order), len(self.order))
        t0 = time.time()
        self.model.train()
        acc: list[dict] = []
        while self.example < n:
            rec = self.records[self.order[self.example]]
            acc.append(self.train_example(rec))
            self.example += 1
            if self.example % self.accum == 0 or self.example == n:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("max_grad_norm", 1.0))
                self.opt.step()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
                self.opt_step += 1
                row = {"opt_step": self.opt_step, "example": self.example, "sec": round(time.time() - t0, 1), "lr": self.sched.get_last_lr()[0], "grad_norm": float(gn)}
                for k in ("nll", "loss", "bridge", "p_ans_corrupted"):
                    vals = [a[k] for a in acc if k in a]
                    if vals:
                        row[k] = sum(vals) / len(vals)
                acc = []
                self.log.write(json.dumps(row) + "\n")
                self.log.flush()
                if self.opt_step % self.cfg.get("log_every", 1) == 0:
                    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
                if self.opt_step % self.cfg.get("save_every", 100) == 0 or self.example == n:
                    self._save_state()
                    print(f"saved checkpoint at opt step {self.opt_step}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-examples", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.max_examples:
        cfg["max_examples"] = a.max_examples
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.yaml").write_text(yaml.safe_dump(cfg))
    SFTTrainer(cfg, a.data, a.out, a.resume).train()


if __name__ == "__main__":
    main()
