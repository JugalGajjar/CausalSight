"""Faithfulness metrics (proposal Section 6.3).

blind_gap          acc(video) - acc(blank clip)
evidence_masking   flip rate when predicted evidence is masked, minus the random-mask control
chain_consistency  judge predicts the answer from the chain alone; agreement with the model
necessity_rate     fraction of steps with Delta_i > delta under a held-out verifier
"""
