# Stage 0 — zero-shot baseline, Qwen2.5-VL-3B-Instruct on CLEVRER validation

Pilot: 50 items per question type (200 total), 16 uniformly sampled frames, greedy decoding, run on
an M4 Pro (MPS, ~2.9 s/item). Item set restricted to questions with generated evidence chains and
identical across the four conditions (`--chains ... --per-type 50 --seed 0`).

Conditions: plain; blind (all frames black); evidence-masked (GT evidence boxes blacked out on the
sampled frames nearest each region's span); random-masked (same box sizes and spans, random positions).

Faithfulness table: `qwen3b_pilot_faithfulness.md` (paired bootstrap 95% CIs).

Full run (`qwen3b_*`, 250 per type, 1000 items; ~48 min per condition on the M4 Pro):
`qwen3b_faithfulness.md`. Pilot (`qwen3b_pilot_*`, 50 per type) kept for reference.

Reading (1000 items):
* Per-question plain: descriptive 0.728, explanatory 0.288, predictive 0.660, counterfactual 0.320;
  inferential per-question 0.423, per-option mean 0.736. Wu et al. (CVPR 2026) report 73.9% for the
  same base model, so their CLEVRER "inferential" figure is per-option; Stage 0 reports both.
* Blind gap 0.30 [0.27, 0.33]; positive on every type (explanatory 0.17 [0.12, 0.22]). The pilot's
  near-zero explanatory gap was noise.
* Moment-masking ES gap 0.06 [0.03, 0.10] overall, but -0.01 [-0.05, 0.03] on descriptive: masking an
  object's box at one frame removes nothing when the object is visible in the other 15 frames. The
  moment-level metric only works for event evidence (explanatory 0.17, counterfactual 0.11).
  => added `--mask track` / `--mask track_random`: remove the *decisive* evidence objects (last triplet
  of each chain) for the whole video vs remove the same number of other objects from the same video.
  Items without a same-size control are skipped in both modes (kept: descriptive 172, explanatory 134,
  predictive 43, counterfactual 250 of 250 each; mean 1.4 evidence objects). This is the CLEVRER
  evidence-sensitivity metric going forward; the moment-level one stays for real video.

Next: track conditions on the same 1000 items, then the same six conditions after outcome-only GRPO.
