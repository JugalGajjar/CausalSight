# Training configs

One YAML per training run. They were launched on Colab, so `init_adapter` and `data_file` point at a mounted
Drive (`/content/drive/MyDrive/causalsight/...`) or at the repository clone (`/content/CausalSight/...`); replace
them with your own paths. The tag is the prefix of the run's result files in `results/`.

| Config | Run | Result tag |
|---|---|---|
| `stage0_grpo.yaml` | outcome-only GRPO from the base, balanced mix; the natural mix uses `--data-file data/subsets/rl_natural_1500.jsonl` | `grpo3b_bal`, `grpo3b_nat` |
| `stage1_sft.yaml` | CLEVRER chain SFT with the bridge loss | `sft3b` |
| `stage2_outcome_only.yaml`, `_seed1`, `_seed2` | CLEVRER chain GRPO (outcome + format rewards), three seeds | `chainrl3b`, `chainrl3b_s1`, `chainrl3b_s2` |
| `stage2_csr.yaml`, `_seed1`, `_seed2` | chain GRPO with grounding, necessity and process rewards (CSR) | `csr3b`, `csr3b_s1` |
| `stage2_no_nec.yaml` | CSR without the necessity reward | `csr3b_nonec` |
| `stage2_stepcredit.yaml`, `stage2_stepcredit_soft.yaml` | CSR with step-level credit at weight 1.0 and 0.3 | `csr3b_sc`, `csr3b_sc03` |
| `stage2_csr_objground.yaml` | CSR with the frame-consistent grounding reward only | not reported |
| `sft_mix_chains.yaml`, `_s1` | mixed chain SFT, two seeds | `mixsft3b`, `mixsft3b_s1` |
| `sft_mix_plain.yaml`, `_s1` | plain (chain-free) counterpart of the mixed arm, two seeds | `mixplainsft3b`, `mixplainsft3b_s1` |
| `sft_real_chains.yaml` | NExT-QA chain SFT | `realsft3b` |
| `grpo_mix_chains.yaml` | mixed chain GRPO | `mixrl3b` |
| `grpo_mix_plain.yaml` | GRPO on the plain mixed arm | `mixplainrl3b` |
| `grpo_real_chains.yaml` | NExT-QA chain GRPO | `realrl3b` |
| `internvl_stage0_grpo.yaml`, `_lr3e6` | InternVL3-2B outcome-only GRPO at learning rate 1e-5 and 3e-6 | `ivl_grpo2b`, `ivl_grpo2b_lr` |
| `internvl_stage1_sft.yaml` | InternVL3-2B chain SFT | `ivl_sft2b` |
| `internvl_stage2_outcome_only.yaml`, `_lr3e6` | InternVL3-2B chain GRPO at learning rate 1e-5 and 3e-6 | `ivl_chainrl2b`, `ivl_chainrl2b_lr` |
| `smoke_*_mac.yaml` | toy-scale configs that only check the training loop runs | none |
