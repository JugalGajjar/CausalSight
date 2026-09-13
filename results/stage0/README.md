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

Track conditions (599 of the 1000 items have a same-size control):
* descriptive track gap 0.65 [0.55, 0.74] (flip 0.74 vs 0.09), explanatory 0.57 [0.40, 0.74],
  predictive 0.36 [0.14, 0.57]. Removing the decisive evidence object breaks the answer; removing
  another object mostly does not. This is the evidence-sensitivity metric for CLEVRER.
* counterfactual came out at -0.20 in the first track run because the "last triplet" of a counterfactual
  chain is the hypothetically *removed* object, not the pair whose collision is asked about. Fixed by
  recording triplet roles in the generator and using the observed-fact step (role=observe) as the
  decisive evidence for predictive and counterfactual chains.
* The "remove other objects" control needs spare objects, which pooled MC evidence rarely leaves
  (2/250 counterfactual items). Control changed to size-matched copies of the evidence tracks at random
  positions that avoid the evidence boxes: same masked area and timing, different place. Coverage
  931/1000. Caveat: MC decisive evidence is pooled over all options (mean ~4 objects), so the
  predictive/counterfactual track gaps are a weaker test than descriptive/explanatory (1-2 objects).
  Both track conditions must be re-run after regenerating chains with roles.

Final track results (chains with roles, size-matched random-position control, 931/1000 items):

| type | n_track | flip_track | flip_track_random | track_gap [95% CI] |
|---|---|---|---|---|
| ALL | 931 | 0.599 | 0.093 | 0.505 [0.45, 0.56] |
| descriptive | 207 | 0.753 | 0.093 | 0.660 [0.56, 0.74] |
| explanatory | 224 | 0.742 | 0.030 | 0.712 [0.60, 0.82] |
| predictive | 250 | 0.412 | 0.085 | 0.327 [0.25, 0.41] |
| counterfactual | 250 | 0.575 | 0.163 | 0.412 [0.28, 0.54] |

This is the pre-RL reference row for Stage 0. The same six conditions are run on every trained checkpoint.


## Stage 0 result: outcome-only GRPO, balanced mix (run 2, `grpo3b_bal_*`)

Training: 1,500 GRPO steps on 1,500 CLEVRER train questions (375 per type), G=8, LoRA r=64, 4.1 h on
an A100 80GB (`notes/2026-09-13-stage0-run1.md`). The checkpoint answers with bare letters/words (no tags).

| type | plain base -> GRPO | blind base -> GRPO | blind gap base -> GRPO | track gap base -> GRPO |
|---|---|---|---|---|
| descriptive | 0.728 -> 0.780 | 0.156 -> 0.192 | 0.57 -> 0.59 [0.52,0.66] | 0.66 -> 0.69 [0.61,0.78] |
| explanatory | 0.288 -> 0.808 | 0.116 -> 0.280 | 0.17 -> 0.53 [0.46,0.59] | 0.71 -> 0.69 [0.61,0.75] |
| predictive | 0.660 -> 0.900 | 0.340 -> 0.468 | 0.32 -> 0.43 [0.36,0.50] | 0.33 -> 0.42 [0.35,0.49] |
| counterfactual | 0.320 -> 0.500 | 0.176 -> 0.208 | 0.14 -> 0.29 [0.22,0.37] | 0.41 -> 0.58 [0.50,0.67] |

Per-option: explanatory 0.937, predictive 0.900, counterfactual 0.806 (base 0.692 / 0.798 / 0.719).

Reading: no shortcut collapse on the balanced mix. The accuracy gain is mostly video-dependent (blind
gaps grow), and evidence dependence is maintained or increased (track gaps). Wu et al.'s degradation
was observed with ~74% observational training questions; run 3 (`stage0_natural`, 72% descriptive)
tests that setting on this trainer.
