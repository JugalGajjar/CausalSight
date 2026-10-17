# CausalVQA debug split (real video, 99 questions x 2 choice sets, 5 options each), 16 frames, greedy

Bare-answer models: letters prompt. Chain models: one yes/no item per option (as trained); a choice set is
correct only if all five options are judged correctly. `paired` = both choice sets of a question correct
(the benchmark's metric). Blind = black frames.

| model | set acc | paired | blind set acc | blind gap | antic / cf / descr / hypo / plan |
|---|---|---|---|---|---|
| Qwen2.5-VL-3B base | 0.580 | 0.430 | 0.410 | 0.17 | 0.42 / 0.53 / 0.75 / 0.68 / 0.53 |
| GRPO balanced (ours) | 0.515 | 0.350 | 0.390 | 0.13 | 0.40 / 0.53 / 0.68 / 0.55 / 0.42 |
| VideoThinker-R1-3B (Wu et al.) | 0.475 | 0.310 | 0.340 | 0.14 | 0.45 / 0.45 / 0.60 / 0.50 / 0.38 |
| SFT chains (ours) | 0.020 | 0.010 | 0.020 | 0.00 | per-option 0.461; says "yes" to 54% of options (20% are correct) |
| chain GRPO (ours) | 0.050 | 0.020 | 0.010 | 0.04 | per-option 0.479; "yes" to 60% |
| CSR (ours) | 0.040 | 0.020 | 0.015 | 0.03 | per-option 0.426; "yes" to 63% |

Reading. (1) Every CLEVRER-trained model loses real-video accuracy relative to the base model: outcome-only
GRPO -6.5 points on set accuracy, Wu et al.'s released model -10.5. (2) The chain models transfer their
*format* (99-100% parseable chains, one grounded step naming a real object) but not their judgment: they
answer "yes" to most options and fall below the always-"no" rate (0.80 per option). Synthetic-only chain
training does not transfer to real video; this is reported as a limitation and as motivation for mixed
real/synthetic chain data in future work. (3) The base model's blind gap on real video is 0.17, so the
benchmark is only moderately video-dependent for a 3B model.
