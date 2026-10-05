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

## Ablation: outcome + format only chain GRPO (`chainrl3b`), 2026-09-28

| model | descr | expl | pred | cf | obj_iou | acc vs SFT (paired) | obj_iou vs SFT (paired) |
|---|---|---|---|---|---|---|---|
| SFT | 0.888 | 0.902 | 0.820 | 0.805 | 0.679 | - | - |
| chain GRPO o+f | 0.924 | 0.947 | 0.832 | 0.801 | 0.654 | +0.020 [+0.011, +0.030] | -0.027 [-0.031, -0.023] |
| CSR run A | 0.900 | 0.945 | 0.828 | 0.807 | 0.668 | +0.019 [+0.010, +0.028] | -0.010 [-0.014, -0.006] |

Run A vs chain GRPO, paired: accuracy -0.002 [-0.009, +0.006]; obj_iou +0.017 [+0.013, +0.021].
Reading: the accuracy gain from RL on chains comes entirely from outcome+format; the process rewards
(grounding, necessity, process) add no accuracy. Their measurable effect is on grounding: plain RL erodes
box IoU by 0.027 relative to SFT, and the process rewards recover 0.017 of that. Faithfulness gaps are
unchanged across the three chain models. Next: run C (step credit 0.3) to test whether step-level credit
protects grounding further without run B's degradation; then a second seed of the final method.

## Run C: step credit 0.3 (`csr3b_sc03`), 2026-10-02

Stable training (KL ~0.05, no degradation) but no benefit: accuracy equal to run A (-0.003 [-0.011, +0.006]),
grounding below run A (-0.011 [-0.015, -0.007]) and only marginally above plain chain GRPO (+0.006
[+0.001, +0.011]); track gap 0.46, the lowest of the chain models. Step-level credit is a negative result at
both weights tested (1.0 harms, 0.3 is neutral-to-worse). Final method = run A (sequence-level CSR).
Replication plan: second seeds of run A and of chain GRPO o+f, the pair behind the grounding-protection claim.

## CSR seed 1 (`csr3b_s1`), 2026-10-04: replicates seed 0
Accuracy seed1−seed0 −0.005 [−0.014, +0.004]; obj_iou −0.003 [−0.007, +0.002]. Against the same
comparators: seed1−SFT accuracy +0.014 [+0.003, +0.024]; seed1−chainGRPO(s0) obj_iou +0.014 [+0.010, +0.019].
The grounding-protection and accuracy claims hold under a second seed.
