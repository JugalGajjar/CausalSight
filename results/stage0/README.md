# Stage 0 — zero-shot baseline, Qwen2.5-VL-3B-Instruct on CLEVRER validation

Pilot: 50 items per question type (200 total), 16 uniformly sampled frames, greedy decoding, run on
an M4 Pro (MPS, ~2.9 s/item). Item set restricted to questions with generated evidence chains and
identical across the four conditions (`--chains ... --per-type 50 --seed 0`).

Conditions: plain; blind (all frames black); evidence-masked (GT evidence boxes blacked out on the
sampled frames nearest each region's span); random-masked (same box sizes and spans, random positions).

Faithfulness table: `qwen3b_pilot_faithfulness.md` (paired bootstrap 95% CIs).

Reading (pilot, wide CIs):
* Blind gap is large on descriptive (0.62) and predictive (0.44) but small on explanatory (0.08):
  explanatory answers are close to what the model produces without seeing the video.
* Evidence masking flips 18.6% of correct answers vs 5.9% for random masks of equal size (ES gap 0.13),
  largest on counterfactual (0.20). The base model does use the annotated evidence, but weakly.
* Per-option accuracy on the MC types averages ~0.72, in the range Wu et al. (CVPR 2026) report for the
  same base model (73.9%), so their CLEVRER "inferential" number is most likely per-option.

Next: 250 per type (1000 items) for tight CIs, then the same four conditions after outcome-only GRPO.
