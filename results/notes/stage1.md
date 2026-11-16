# Stage 1 — triplet-chain SFT with bridge loss (`sft3b_*`), evaluated per option on the A100

> The numbers in this note come from the first evaluation pass. Every model was later re-evaluated with the
> preprocessing used in training, and those summaries are the ones in `results/evals/` and in the paper; they
> differ slightly (for example the base model's overall CLEVRER accuracy is 0.499 here and 0.490 there). The
> files of the first pass are no longer in the repository.

Same 1,000 stratified CLEVRER validation questions as Stage 0; MC questions become one yes/no item per
option (2,287 option items + 250 descriptive), chain prompt, 640 new tokens, greedy.

| type | per-option: base / GRPO-bal / SFT | per-question: base / GRPO-bal / SFT | SFT blind (per-option) |
|---|---|---|---|
| descriptive | - | 0.728 / 0.780 / 0.876 | 0.236 (per-question) |
| explanatory | 0.692 / 0.937 / 0.895 | 0.288 / 0.808 / 0.744 | 0.509 |
| predictive | 0.798 / 0.900 / 0.776 | 0.660 / 0.900 / 0.620 | 0.436 |
| counterfactual | 0.719 / 0.806 / 0.789 | 0.320 / 0.500 / 0.420 | 0.443 |

Notes: base and GRPO answer one multi-select prompt, SFT answers each option independently, so per-option
is the comparable column (yes/no chance = 0.5; SFT blind sits at chance on every MC type). Chains: format
rate 99.5%, 4.1 steps mean. Grounding of emitted boxes (first pass, pooled-option reference): spatial IoU
within +-8 frames 0.18, strict spatiotemporal IoU 0.09, temporal offset 5.5 frames. Recall in that pass was
diluted by pooling all options' evidence; rescored per option with `cs-rescore-chains`.
