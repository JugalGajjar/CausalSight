# Stage 2 results (matched preprocessing, per-option evaluation, 2,537 items)

Run A = `csr3b` (CSR rewards, sequence-level credit). Run B = `csr3b_sc` (step-level credit, weight 1.0,
frame-consistent grounding reward). Both 1,000 steps from the Stage 1 SFT checkpoint.

| model | descr | expl (opt) | pred (opt) | cf (opt) | obj_iou | blind gap | track gap |
|---|---|---|---|---|---|---|---|
| base | 0.720 | 0.702 | 0.764 | 0.712 | - | 0.28 | 0.56 |
| GRPO balanced | 0.776 | 0.939 | 0.900 | 0.820 | - | 0.47 | 0.59 |
| GRPO natural | 0.776 | 0.957 | 0.868 | 0.823 | - | 0.50 | 0.63 |
| SFT chains | 0.888 | 0.902 | 0.820 | 0.805 | 0.679 | 0.39 | 0.51 |
| CSR run A | 0.900 | 0.945 | 0.828 | 0.807 | 0.668 | 0.38 | 0.51 |
| CSR run B | 0.880 | 0.924 | 0.768 | 0.788 | 0.607 | 0.38 | 0.50 |

Paired bootstrap (same items) vs SFT: run A +0.019 [+0.010, +0.028] overall, driven by explanatory
+0.043 [+0.025, +0.060]; other types within noise. Run B -0.009 [-0.021, +0.004] overall, predictive
-0.052 [-0.084, -0.018]; format rate fell to 0.962 and obj_iou to 0.607.

Reading: sequence-credit CSR (run A) improves accuracy modestly and keeps grounding and faithfulness at
SFT levels; explanatory now matches outcome-only GRPO (0.945 vs 0.939) while emitting verifiable boxes.
Step-level credit at weight 1.0 (run B) harmed everything (training reward fell, KL 10x, grad norm at
clip): the step-credit term needs a smaller weight, not abandonment, but that is an open item.
Still needed: outcome+format-only chain GRPO from the same checkpoint (does any of run A's gain come
from the process rewards?), and a second seed of run A.
