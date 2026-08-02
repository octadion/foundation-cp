"""Base nonconformity score functions (Tier-1 baselines, AGENTS.md §7).

Convention everywhere in the repo: a score matrix `S` has `S[i, k]` = the
nonconformity score of label k for point i, HIGHER = more nonconforming, and a
label is included in the set iff `S[i, k] <= threshold`.

THR/LAC is implemented (it is the trivial base score and is used by the
coverage-validity test's mental model). APS/RAPS/SAPS are Tier-1 baselines to be
filled during Phase 0 / baseline reproduction; their exact hyperparameters must
match the reference papers (AGENTS.md §7) and be reproduced against reported
numbers before use.
"""

from __future__ import annotations

import numpy as np


def thr_lac(probs: np.ndarray) -> np.ndarray:
    """THR / LAC (Sadinle et al., JASA 2019): score = 1 - softmax prob.

    `probs[i, k]` is the model's softmax probability of class k. Returns a score
    matrix with the repo convention (higher = worse).
    """
    return 1.0 - probs


def aps(probs: np.ndarray, *, randomize: bool = True, seed: int = 42) -> np.ndarray:
    """APS (Romano, Sesia, Candès, NeurIPS 2020). Cumulative sorted-prob score.

    TODO(phase0/baselines): implement to match the reference; reproduce reported
    numbers on >=1 setting before using as a baseline (§7 "Reproduksi").
    """
    raise NotImplementedError("APS: implement during baseline reproduction (§7)")


def raps(probs: np.ndarray, *, k_reg: int, lam: float, randomize: bool = True,
         seed: int = 42) -> np.ndarray:
    """RAPS (Angelopoulos et al., ICLR 2021). APS + regularization (k_reg, lam).

    TODO(phase0/baselines): implement + reproduce before use (§7).
    """
    raise NotImplementedError("RAPS: implement during baseline reproduction (§7)")


def saps(probs: np.ndarray, *, lam: float, randomize: bool = True,
         seed: int = 42) -> np.ndarray:
    """SAPS (Huang et al., ICML 2024, arXiv 2310.06430).

    TODO(phase0/baselines): implement + reproduce before use (§7).
    """
    raise NotImplementedError("SAPS: implement during baseline reproduction (§7)")
