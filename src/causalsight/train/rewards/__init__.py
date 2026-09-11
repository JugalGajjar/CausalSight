"""CSR reward terms (proposal Section 5.2). One module per term; each exposes `score(...)`.

R = w_o R_out + w_f R_fmt + w_g R_ground + w_n R_nec + w_p R_proc
"""

from causalsight.train.rewards import format_reward, grounding, necessity, outcome, process

__all__ = ["format_reward", "grounding", "necessity", "outcome", "process"]
