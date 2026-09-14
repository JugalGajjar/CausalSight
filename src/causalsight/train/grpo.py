"""Minimal single-GPU GRPO for Qwen2.5-VL with LoRA and pluggable rewards (proposal Sections 5.0 / 5.2).

  cs-grpo --config configs/stage0_grpo.yaml --data data/frames/train --out runs/stage0 [--resume]

Per step: one prompt -> G sampled completions -> rewards -> group-normalized advantages ->
one on-policy policy-gradient update with a KL penalty to the reference (the same weights with the
LoRA adapter disabled, so no second copy of the model is needed). Completions are scored in one
batched forward with the video inputs repeated per completion. Checkpoints (adapter + trainer
state) go to <out> every `save_every` steps and `--resume` continues from the last one; Colab can
disconnect at any time.

Rewards are named in the config; each maps to a function (completion, record) -> float in
causalsight.train.rewards.registry. Stage 0 uses outcome + format only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import yaml

from causalsight.data.schema import TripletChain
from causalsight.train.data import load_frames, load_records
from causalsight.train.format import SYSTEM_PROMPT, format_step, parse_chain
from causalsight.train.rewards.necessity import score_from_deltas
from causalsight.train.rewards.registry import REWARDS

PLAIN_SYSTEM = (
    "Answer the question about the video. First think step by step inside <think></think>, then give the "
    "final answer inside <answer></answer>. For multiple choice, the answer is the letters of all correct "
    "options; otherwise a single word or number."
)


def build_messages(rec: dict, frames, prompt_style: str, instr_in_user: bool = True) -> list[dict]:
    """`instr_in_user`: put the format instruction after the question in the user turn instead of a system
    prompt; small instruct models follow user-turn instructions far more reliably."""
    instr = SYSTEM_PROMPT if prompt_style == "chain" else PLAIN_SYSTEM
    if instr_in_user:
        return [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": rec["problem"] + "\n" + instr}]}]
    return [
        {"role": "system", "content": instr},
        {"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": rec["problem"]}]},
    ]


class GRPOTrainer:
    def __init__(self, cfg: dict, data_root: Path, out: Path, resume: bool) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = cfg
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.get("dtype", "bfloat16")]
        self.processor = AutoProcessor.from_pretrained(cfg["model"])
        self.processor.tokenizer.padding_side = "left"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["model"], torch_dtype=self.dtype, attn_implementation=cfg.get("attn", "sdpa"))
        if cfg.get("init_adapter"):
            # Stage 2 starts from the Stage 1 SFT adapter: merge it into the weights, so the reference
            # model (RL adapter disabled) *is* the SFT model, which also serves as the necessity verifier
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, cfg["init_adapter"]).merge_and_unload()
            print(f"merged init adapter {cfg['init_adapter']}")
        if cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        lora = cfg["lora"]
        self.model = get_peft_model(
            model,
            LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0), target_modules=lora["targets"], task_type="CAUSAL_LM"),
        ).to(self.device)
        self.model.print_trainable_parameters()
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)
        self.records = load_records(data_root / cfg["data_file"])
        self.data_root = data_root
        self.rewards = [(name, REWARDS.get(name), float(w)) for name, w in cfg["rewards"].items()]
        self.nec_cfg = cfg.get("necessity", {"delta": 0.1, "eps": 0.02, "gamma": 0.5})
        self.kl_clip = float(cfg.get("kl_log_ratio_clip", 5.0))
        self.rng = random.Random(cfg.get("seed", 0))
        self.order = list(range(len(self.records)))
        self.rng.shuffle(self.order)
        self.step = 0
        self.log = (out / "log.jsonl").open("a")
        if resume:
            self._load_state()

    # ---------------------------------------------------------------- checkpointing

    def _save_state(self) -> None:
        import torch

        self.model.save_pretrained(self.out / "adapter")
        torch.save({"step": self.step, "order": self.order, "rng": self.rng.getstate(), "opt": self.opt.state_dict()}, self.out / "trainer_state.pt")
        (self.out / "step.txt").write_text(str(self.step))

    def _load_state(self) -> None:
        import torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        st_path = self.out / "trainer_state.pt"
        if not st_path.exists():
            print("no checkpoint to resume from; starting fresh")
            return
        st = torch.load(st_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(self.model, load_file(self.out / "adapter" / "adapter_model.safetensors"))
        self.opt.load_state_dict(st["opt"])
        self.step, self.order = st["step"], st["order"]
        self.rng.setstate(st["rng"])
        print(f"resumed at step {self.step}")

    # ---------------------------------------------------------------- core

    def _inputs(self, rec: dict, frames):
        from qwen_vl_utils import process_vision_info

        messages = build_messages(rec, frames, self.cfg.get("prompt_style", "plain"), self.cfg.get("instr_in_user", True))
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _imgs, videos = process_vision_info(messages)
        return self.processor(text=[text], videos=videos, padding=True, return_tensors="pt").to(self.device)

    def _generate(self, inputs, g: int):
        import torch

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.cfg.get("temperature", 1.0),
                top_p=self.cfg.get("top_p", 1.0),
                max_new_tokens=self.cfg["max_new_tokens"],
                num_return_sequences=g,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return out[:, prompt_len:]

    def _completion_logprobs(self, inputs, completions, use_adapter: bool):
        """Per-token log-probs of `completions` (G x T, right-padded) given the prompt inputs."""
        import torch

        g = completions.shape[0]
        pad = self.processor.tokenizer.pad_token_id
        prompt_ids = inputs["input_ids"].repeat(g, 1)
        prompt_mask = inputs["attention_mask"].repeat(g, 1)
        comp_mask = (completions != pad).long()
        # keep everything up to the last non-pad token (an EOS may equal pad; handle via first-pad cut)
        ids = torch.cat([prompt_ids, completions], dim=1)
        mask = torch.cat([prompt_mask, comp_mask], dim=1)
        kw = {"input_ids": ids, "attention_mask": mask}
        if "pixel_values_videos" in inputs:
            kw["pixel_values_videos"] = inputs["pixel_values_videos"].repeat(g, 1)
            kw["video_grid_thw"] = inputs["video_grid_thw"].repeat(g, 1)
        if "second_per_grid_ts" in inputs:
            kw["second_per_grid_ts"] = inputs["second_per_grid_ts"].repeat(g)
        ctx = self.model.disable_adapter() if not use_adapter else _nullcontext()
        with ctx, (torch.no_grad() if not use_adapter else _nullcontext()):
            logits = self.model(**kw).logits
        logits = logits[:, prompt_ids.shape[1] - 1 : -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(-1, completions.unsqueeze(-1)).squeeze(-1)
        return tok_lp, comp_mask

    def _gold_logprob(self, inputs, chain_texts: list[str], gold: str) -> list[float]:
        """log p(gold answer | prompt, chain) under the reference model, for each chain text, in one
        batched teacher-forced forward (video inputs repeated per variant)."""
        import torch

        tok = self.processor.tokenizer
        tok.padding_side = "right"
        heads = [f"<think>\n{c}\n</think>\n<answer>" for c in chain_texts]
        tail = f"{gold}</answer>"
        seqs = [tok(h + tail, add_special_tokens=False)["input_ids"] for h in heads]
        head_lens = [len(tok(h, add_special_tokens=False)["input_ids"]) for h in heads]
        gold_len = len(tok(tail, add_special_tokens=False)["input_ids"])
        pad = tok.pad_token_id
        T = max(len(x) for x in seqs)
        comp = torch.full((len(seqs), T), pad, dtype=torch.long, device=self.device)
        for i, x in enumerate(seqs):
            comp[i, : len(x)] = torch.tensor(x, device=self.device)
        tok.padding_side = "left"
        lp, _mask = self._completion_logprobs(inputs, comp, use_adapter=False)
        out = []
        for i in range(len(seqs)):
            a0 = head_lens[i]
            out.append(float(lp[i, a0 : a0 + gold_len].sum()))
        return out

    def _necessity(self, inputs, rec: dict, text: str) -> float:
        """R_nec: fraction of steps whose removal (with their dependency closure) lowers p(gold) by more
        than delta, minus gamma x fraction of padding steps (drop <= eps). Verifier = reference model."""
        import math as _m

        p = parse_chain(text)
        if p.chain is None:
            return 0.0
        chain: TripletChain = p.chain
        variants = ["\n".join(format_step(i, t) for i, t in enumerate(chain.triplets))]
        for i in range(len(chain.triplets)):
            sub = chain.without_step(i)
            variants.append("\n".join(format_step(j, t) for j, t in enumerate(sub.triplets)) if sub.triplets else "")
        lps = self._gold_logprob(inputs, variants, rec["gold"])
        p_full = _m.exp(lps[0])
        deltas = [p_full - _m.exp(lp) for lp in lps[1:]]
        return score_from_deltas(deltas, self.nec_cfg["delta"], self.nec_cfg["eps"], self.nec_cfg["gamma"])

    def train_step(self, rec: dict) -> dict:
        import torch

        g = self.cfg["group_size"]
        frames = load_frames(self.data_root, rec, self.cfg.get("n_frames"))
        inputs = self._inputs(rec, frames)
        completions = self._generate(inputs, g)
        pad = self.processor.tokenizer.pad_token_id
        # cut each completion after its first EOS/pad so trailing tokens are masked out
        eos = self.processor.tokenizer.eos_token_id
        texts = self.processor.batch_decode(completions, skip_special_tokens=True)
        for i in range(g):
            row = completions[i]
            ends = ((row == eos) | (row == pad)).nonzero()
            if len(ends):
                completions[i, ends[0, 0] + 1 :] = pad
        rewards = torch.zeros(g)
        parts = {name: [] for name, _, _ in self.rewards}
        for i, t in enumerate(texts):
            for name, fn, w in self.rewards:
                v = self._necessity(inputs, rec, t) if name == "necessity" else float(fn(t, rec))
                parts[name].append(v)
                rewards[i] += w * v
        stats_common = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()) if g > 1 else 0.0,
            **{f"r_{k}": sum(v) / len(v) for k, v in parts.items()},
            "comp_len": float((completions != pad).sum(1).float().mean()),
            "sample": texts[0][:300],
        }
        if g > 1 and float(rewards.std()) < 1e-6:
            # no learning signal in this group; an update here would only be optimizer-scaled numerical
            # noise between the reference and policy passes, so skip it
            return {**stats_common, "loss": 0.0, "kl": 0.0, "grad_norm": 0.0, "skipped": 1}
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4) if g > 1 else rewards
        adv = adv.to(self.device)

        self.model.train()
        micro = self.cfg.get("micro_batch", g)
        beta = self.cfg.get("kl_beta", 0.04)
        total_loss, total_kl = 0.0, 0.0
        n_clipped, n_tokens = 0, 0
        self.opt.zero_grad(set_to_none=True)
        for s in range(0, g, micro):
            comp = completions[s : s + micro]
            ref_lp, mask = self._completion_logprobs(inputs, comp, use_adapter=False)
            pol_lp, _ = self._completion_logprobs(inputs, comp, use_adapter=True)
            # on-policy single update: ratio == 1 in value, gradient flows through pol_lp
            ratio = torch.exp(pol_lp - pol_lp.detach())
            # k3 KL estimator, >= 0. The log-ratio is clamped: with temperature-1 sampling an occasional
            # low-probability token that the reference rates highly gives exp(ref - pol) in the thousands
            # and a single token then dominates the step (observed: window-mean KL > 200 in Stage 0 run 1).
            log_ratio = (ref_lp - pol_lp).clamp(-self.kl_clip, self.kl_clip)
            n_clipped += int(((ref_lp - pol_lp).abs() > self.kl_clip).sum())
            n_tokens += int(mask.sum())
            kl = torch.exp(log_ratio) - log_ratio - 1
            per_tok = -(ratio * adv[s : s + micro].unsqueeze(1) - beta * kl)
            loss = ((per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)).sum() / g
            loss.backward()
            total_loss += loss.detach().item()
            total_kl += ((kl * mask).sum() / mask.sum().clamp(min=1)).detach().item()
        # the on-policy loss value is ~0 by construction (advantages are mean-centered); the gradient norm
        # is the informative number
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("max_grad_norm", 1.0))
        self.opt.step()
        self.model.eval()
        return {
            **stats_common,
            "loss": total_loss,
            "kl": total_kl / max(1, math.ceil(g / micro)),
            "kl_clip_frac": n_clipped / max(1, n_tokens),
            "grad_norm": float(grad_norm),
            "skipped": 0,
        }

    def train(self) -> None:
        total = min(self.cfg["steps"], len(self.order))
        t0 = time.time()
        while self.step < total:
            rec = self.records[self.order[self.step]]
            stats = self.train_step(rec)
            self.step += 1
            row = {"step": self.step, "problem_id": rec.get("problem_id"), "qtype": rec.get("question_type"), "sec": round(time.time() - t0, 1), **stats}
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()
            if self.step % self.cfg.get("log_every", 1) == 0:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "sample"}))
            if self.step % self.cfg.get("save_every", 50) == 0 or self.step == total:
                self._save_state()
                print(f"saved checkpoint at step {self.step}")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="directory produced by cs-frames")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override config steps (smoke tests)")
    ap.add_argument("--data-file", type=Path, default=None, help="override config data_file (absolute path or relative to --data)")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    if a.steps:
        cfg["steps"] = a.steps
    if a.data_file:
        cfg["data_file"] = str(a.data_file)
    (a.out).mkdir(parents=True, exist_ok=True)
    (a.out / "config.yaml").write_text(yaml.safe_dump(cfg))
    GRPOTrainer(cfg, a.data, a.out, a.resume).train()


if __name__ == "__main__":
    main()
