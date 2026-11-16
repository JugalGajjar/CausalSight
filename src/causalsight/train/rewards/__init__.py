"""Reward terms for GRPO on chains. `registry` maps reward names to functions of (completion, record).

R = R_out + w_f R_fmt + w (R_ground + R_nec + R_proc)

format_reward   R_fmt: the chain parses and its spans fit the clip
outcome         R_out: final-answer correctness
necessity       R_nec: step necessity from a frozen verifier
process         R_proc: order-aware matching against the reference chain
chain_rewards   R_ground and R_proc against the ground-truth chains of a record
"""

from causalsight.train.rewards import format_reward, necessity, outcome, process

__all__ = ["format_reward", "necessity", "outcome", "process"]
