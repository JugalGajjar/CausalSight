"""Bridge term for triplet SFT (proposal Section 5.1).

L_bridge = -log(1 - p(a* | v, q, c_tilde)), where c_tilde corrupts one triplet on the
dependency path to the answer. Requires a distractor sampler per answer type.
"""

from __future__ import annotations


def corrupt_chain(chain, rng):
    """Return a copy of `chain` with one on-path intermediate answer replaced by a distractor."""
    raise NotImplementedError


def bridge_loss(logp_correct_given_corrupted):
    """-log(1 - p). Input is log p(a* | v, q, c_tilde) as a tensor."""
    raise NotImplementedError
