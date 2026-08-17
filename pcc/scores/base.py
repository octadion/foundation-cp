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


# Rows are INDEPENDENT in every one of these scores, so the whole matrix never has to be
# resident in transformed form at once. That matters at the real scale: at 250k x 1000 the
# unchunked version peaked at 10 GB for APS, 14 for RAPS and 16 for SAPS against Colab's
# ~12.7 GB, which is exactly how notebook 12 died while notebook 09 -- THR only, 1 GB --
# did not. Chunking bounds the temporaries; the input dtype is preserved rather than
# upcast to float64, which also matches what the reference implementation actually
# computes in when handed a float32 softmax dump.
CHUNK_ROWS = 20_000


def _sorted_parts(p: np.ndarray):
    """Descending sort, cumulative mass at each label, and each label's 0-based rank.

    Mirrors `get_APS_scores_all` in the long-tail-conformal repo: sort descending, cumsum,
    then gather back to original label order via argsort of the permutation. `argsort` of
    the permutation is the rank of each label; adding one makes it 1-based, which is what
    RAPS regularizes against. int32 ranks halve two full-size intermediates and cannot
    overflow -- a label index is bounded by the number of classes.
    """
    order = np.argsort(-p, axis=1, kind="stable").astype(np.int32, copy=False)
    rank = np.argsort(order, axis=1).astype(np.int32, copy=False)
    cum = np.take_along_axis(np.cumsum(np.take_along_axis(p, order, axis=1), axis=1),
                             rank, axis=1)
    return cum, rank


def _blockwise(probs, fn, randomize, seed):
    """Apply a row-independent score to row blocks, sharing ONE uniform stream.

    `RandomState.rand` fills C-order, so drawing block by block in row order yields the
    identical draws as one full-matrix call -- which is what keeps this bit-identical to
    the unchunked version and to the reference. Drawing per block with a fresh RandomState
    would silently reuse the same uniforms for every block.
    """
    p = np.asarray(probs)
    out = np.empty(p.shape, dtype=p.dtype)
    rs = np.random.RandomState(seed) if randomize else None
    for i in range(0, p.shape[0], CHUNK_ROWS):
        blk = p[i:i + CHUNK_ROWS]
        u = rs.rand(*blk.shape) if rs is not None else None
        out[i:i + CHUNK_ROWS] = fn(blk, u)
    return out


def aps(probs: np.ndarray, *, randomize: bool = True, seed: int = 42) -> np.ndarray:
    """APS (Romano, Sesia, Candès, NeurIPS 2020). Cumulative sorted-prob score.

    Verified elementwise against `utils.conformal_utils.get_APS_scores_all` from the
    long-tail-conformal release (§7 reproduction) in `pcc/tests/test_scores.py`.
    """
    def _f(p, u):
        cum, _ = _sorted_parts(p)
        return cum - (u * p if u is not None else p)
    return _blockwise(probs, _f, randomize, seed)


def raps(probs: np.ndarray, *, k_reg: int, lam: float, randomize: bool = True,
         seed: int = 42) -> np.ndarray:
    """RAPS (Angelopoulos et al., ICLR 2021). APS plus a rank penalty.

    The penalty is `max(0, lam * (rank - k_reg))` with a 1-BASED rank, applied to every
    label as if it were the true one. Verified against `get_RAPS_scores_all`.
    """
    def _f(p, u):
        cum, rank = _sorted_parts(p)
        cum += np.maximum(lam * ((rank + 1) - k_reg), 0.0)
        return cum - (u * p if u is not None else p)
    return _blockwise(probs, _f, randomize, seed)


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
    def _f(p, u):
        _, rank = _sorted_parts(p)
        p_max = p.max(axis=1, keepdims=True)
        uu = u if u is not None else np.ones(p.shape, dtype=p.dtype)
        o = rank + 1                                 # 1-based rank
        return np.where(o == 1, uu * p_max, p_max + (o - 2 + uu) * lam)
    return _blockwise(probs, _f, randomize, seed)


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
