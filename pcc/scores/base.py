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


def _sorted_parts(probs: np.ndarray):
    """Descending sort, cumulative mass at each label, and each label's 1-based rank.

    Written to mirror `get_APS_scores_all` in the long-tail-conformal repo exactly:
    sort descending, cumsum, then gather back to original label order via argsort of
    the permutation. `pi.argsort(1)` is the rank of each label, and adding one makes it
    1-based, which is what RAPS regularizes against.
    """
    p = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-p, axis=1, kind="stable")
    rank = np.argsort(order, axis=1)                 # 0-based rank of each label
    cum = np.take_along_axis(np.cumsum(np.take_along_axis(p, order, axis=1), axis=1),
                             rank, axis=1)
    return p, cum, rank


def _derandomize(p, scores, randomize, seed):
    """Subtract U * p_y (randomized) or p_y (not), matching the reference exactly."""
    if not randomize:
        return scores - p
    return scores - np.random.RandomState(seed).rand(*p.shape) * p


def aps(probs: np.ndarray, *, randomize: bool = True, seed: int = 42) -> np.ndarray:
    """APS (Romano, Sesia, Candès, NeurIPS 2020). Cumulative sorted-prob score.

    Verified elementwise against `utils.conformal_utils.get_APS_scores_all` from the
    long-tail-conformal release (§7 reproduction) in `pcc/tests/test_scores.py`.
    """
    p, cum, _ = _sorted_parts(probs)
    return _derandomize(p, cum, randomize, seed)


def raps(probs: np.ndarray, *, k_reg: int, lam: float, randomize: bool = True,
         seed: int = 42) -> np.ndarray:
    """RAPS (Angelopoulos et al., ICLR 2021). APS plus a rank penalty.

    The penalty is `max(0, lam * (rank - k_reg))` with a 1-BASED rank, applied to every
    label as if it were the true one. Verified against `get_RAPS_scores_all`.
    """
    p, cum, rank = _sorted_parts(probs)
    reg = np.maximum(lam * ((rank + 1) - k_reg), 0.0)
    return _derandomize(p, cum + reg, randomize, seed)


def saps(probs: np.ndarray, *, lam: float, randomize: bool = True,
         seed: int = 42) -> np.ndarray:
    """SAPS (Huang et al., ICML 2024, arXiv 2310.06430).

    SAPS keeps only the MAXIMUM probability and replaces every other sorted probability
    with the constant `lam`, on the argument that the tail of the softmax is noise that
    inflates set size without buying coverage. For a label of 1-based rank o:

        o = 1:  u * p_max
        o > 1:  p_max + (o - 2 + u) * lam

    No reference implementation ships with the long-tail-conformal release, so this one
    is written from the paper and pinned by property tests instead: the rank-1 branch
    must reduce to THR-like behaviour, scores must increase with rank, and the whole
    matrix must be non-negative.
    """
    p, _, rank = _sorted_parts(probs)
    p_max = p.max(axis=1, keepdims=True)
    u = (np.random.RandomState(seed).rand(*p.shape) if randomize
         else np.ones(p.shape, dtype=np.float64))
    o = rank + 1                                     # 1-based rank
    return np.where(o == 1, u * p_max, p_max + (o - 2 + u) * lam)


SCORE_FNS = {
    "thr": lambda p, **kw: thr_lac(p),
    "aps": lambda p, seed=42, **kw: aps(p, seed=seed),
    # k_reg=5, lam=0.01: the ImageNet setting of the RAPS paper, and the same values
    # unified_eval_corruption.py uses (RAPS_K_REG / RAPS_LAMBDA_REG).
    "raps": lambda p, seed=42, k_reg=5, lam=0.01, **kw: raps(p, k_reg=k_reg, lam=lam,
                                                             seed=seed),
    # lam=0.2: the ImageNet value reported by the SAPS paper.
    "saps": lambda p, seed=42, lam=0.2, **kw: saps(p, lam=lam, seed=seed),
}


def score_matrix(probs: np.ndarray, name: str = "thr", *, seed: int = 42) -> np.ndarray:
    """Dispatch by name, so a driver can sweep the score function as an axis."""
    if name not in SCORE_FNS:
        raise ValueError("unknown score {!r}; available: {}".format(
            name, sorted(SCORE_FNS)))
    return SCORE_FNS[name](np.asarray(probs), seed=seed)
