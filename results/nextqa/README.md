# NExT-QA val subset (real video, 999 questions = 333 causal / 333 temporal / 333 descriptive, 5 options), 16 frames, greedy

"Before" row block: the models trained on CLEVRER only, evaluated before the mixed real/synthetic experiment.
Bare-answer models: letters prompt (`mc=multi`). Chain models: one yes/no item per option (as trained,
`mc=option`), 4,995 items; a question is correct only if all five options are judged correctly.
`selection acc` (derived from the same outputs) picks the single option judged "yes" and splits credit
evenly when several are, so it is comparable to the letters prompt. Blind = black frames.

## CLEVRER-only models ("before")

| model | prompt | acc | causal | temporal | descr | blind acc | blind gap |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B base | letters | 0.730 | 0.736 | 0.691 | 0.763 | 0.437 | 0.29 |
| GRPO balanced (ours) | letters | 0.750 | 0.760 | 0.706 | 0.784 | 0.437 | 0.31 |
| VideoThinker-R1-3B (Wu et al.) | letters | 0.736 | 0.754 | 0.682 | 0.772 | 0.385 | 0.35 |
| SFT chains (ours) | per option | 0.122 | 0.144 | 0.111 | 0.111 | 0.031 | 0.09 |
| chain GRPO (ours) | per option | 0.194 | 0.201 | 0.201 | 0.180 | 0.048 | 0.15 |

Chain models, per-option view (always-"no" baseline = 0.80):

| model | per-option acc | blind per-option | "yes" rate (20% correct) | recall yes / no | selection acc | blind selection | format rate | mean steps |
|---|---|---|---|---|---|---|---|---|
| SFT chains | 0.622 | 0.498 | 0.38 | 0.74 / 0.59 | 0.451 | 0.259 | 0.96 | 2.7 |
| chain GRPO | 0.682 | 0.559 | 0.41 | 0.80 / 0.65 | 0.470 | 0.272 | 0.98 | 2.7 |

Reading. (1) Outcome-only GRPO on CLEVRER transfers mildly to NExT-QA (+2.0 points over the base, every
type up), unlike CausalVQA where it lost 6.5; Wu et al.'s released model is +0.6. The blind gap of all
three bare-answer models is 0.29-0.35, so this subset is strongly video-dependent. (2) The chain models
keep their format (96-98% parseable, 2.7 steps) but not their judgment: they say "yes" to 38-41% of
options when 20% are correct, and their per-option accuracy sits below the always-"no" rate. Under the
selection view they reach 0.45-0.47, well below the base (0.73), and the video still matters to them
(selection drops to 0.26-0.27 blind). (3) Chain GRPO is ahead of chain SFT on every number, so RL on
synthetic outcomes helps the chain model a little on real video but does not close the gap. This block
is the baseline for the mixed real/synthetic arms (`sft_mix_*`, `grpo_mix_*`, `*_real_chains`).
