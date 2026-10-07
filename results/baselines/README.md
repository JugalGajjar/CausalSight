# Published small-model video-RL checkpoints on our suite (matched preprocessing, 1,000 questions)

Same items, prompts (multi-option letters, <think>/<answer> instruction), frames, stock Qwen2.5-VL
processor, greedy decoding with pinned settings, 384 new tokens, as our Stage 0 GRPO rows.

| model | descr | expl (opt) | pred (opt) | cf (opt) | expl (q) | cf (q) | blind gap | track gap |
|---|---|---|---|---|---|---|---|---|
| Base Qwen2.5-VL-3B | 0.720 | 0.702 | 0.764 | 0.712 | 0.312 | 0.316 | 0.28 | 0.56 |
| VideoRFT-3B (768 tokens) | 0.752 | 0.542 | 0.752 | 0.626 | 0.084 | 0.260 | 0.27 | 0.49 |
| VideoThinker-R1-3B (Wu et al.) | 0.748 | 0.824 | 0.836 | 0.848 | 0.424 | 0.624 | 0.49 | 0.56 |
| Video-R1-7B | 0.780 | 0.823 | 0.686 | 0.718 | 0.556 | 0.368 | 0.35 | 0.58 |
| Ours: GRPO balanced (3B) | 0.776 | 0.939 | 0.900 | 0.820 | 0.820 | 0.520 | 0.47 | 0.59 |
| Ours: GRPO natural (3B) | 0.776 | 0.957 | 0.868 | 0.823 | 0.860 | 0.544 | 0.50 | 0.63 |
| Ours: SFT + CSR (3B) | 0.900 | 0.945 | 0.828 | 0.807 | 0.848 | 0.456 | 0.38 | 0.50 |

Notes. VideoThinker-R1-3B is the released model of the paper whose collapse claim we test; on our subset
it is the strongest published checkpoint on counterfactual (0.848 per option, above our 0.82) and well
below our GRPO rows on explanatory (0.824 vs 0.939-0.957). Its faithfulness gaps match ours. VideoRFT-3B
writes open-ended reasoning: 27.5% of its answers were cut at 384 tokens and 15.4% still at 768 (row above);
its accuracy barely moved between the two budgets, so the remaining truncation is the model's own behaviour.
Our rows are trained on 1,500 CLEVRER questions; the published models were not trained on CLEVRER
validation but Video-R1/VideoRFT training mixes include CLEVRER train questions.
