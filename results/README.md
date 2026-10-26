# Results index (all matched preprocessing; CLEVRER = 1,000 stratified validation questions; CausalVQA = debug split; NExT-QA = 999-question val subset)

* `stage0/` Qwen2.5-VL-3B base and outcome-only GRPO (balanced / natural mix); faithfulness suite.
* `stage1/`, `stage2/` grounded-chain SFT, chain GRPO (3 seeds), CSR (2 seeds), step-credit variants, no-necessity ablation.
* `baselines/` VideoRFT-3B, VideoThinker-R1-3B (Wu et al.), Video-R1-7B on our suite.
* `causalvqa/` real-video transfer (debug split).
* `nextqa/` real-video transfer on a 999-question NExT-QA val subset; "before" block for the mixed-data experiment.
* `colab/` per-condition summaries and faithfulness tables for every run (per-item jsonl not tracked).

## Second base: InternVL3-2B (language model Qwen2.5-1.5B, InternViT-300M), lr 3e-6 for RL (1e-5 diverged)

| model | descr | expl (opt) | pred (opt) | cf (opt) | box IoU | blind gap | track gap |
|---|---|---|---|---|---|---|---|
| InternVL base | 0.836 | 0.855 | 0.826 | 0.764 | - | 0.43 | 0.51 |
| + outcome-only GRPO | 0.856 | 0.927 | 0.896 | 0.770 | - | 0.50 | 0.56 |
| grounded-chain SFT | 0.824 | 0.694 | 0.794 | 0.773 | 0.750 | 0.28 | 0.55 |
| + chain GRPO | 0.840 | 0.762 | 0.806 | 0.776 | 0.744 | 0.30 | 0.57 |

CausalVQA (set acc / paired / blind gap): base 0.410 / 0.290 / +0.12; GRPO 0.415 / 0.250 / +0.13;
SFT 0.000 (per-option 0.324); chain GRPO 0.005 (per-option 0.338).

Reading. The two robust findings replicate on the second base: outcome-only GRPO does not collapse
(explanatory +7, predictive +7, blind gap grows) and the chain pipeline yields well-grounded, verifiable
steps (box IoU 0.75, track gap 0.55-0.57, format ~99%). What does not transfer is accuracy parity: on
InternVL the chain models stay below the base on explanatory (0.76 vs 0.86) and well below outcome-only
GRPO (0.93); on Qwen they matched it. Real-video transfer of synthetic-only chains fails on both bases.
The InternVL RL runs at the Qwen learning rate (1e-5) diverged (KL 11-58, reward falling) and are not
reported; `runs/internvl_stage0`, `runs/internvl_stage2_outcome_only` are kept for the record.
