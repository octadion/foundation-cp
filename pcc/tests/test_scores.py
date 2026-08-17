"""Score functions, checked against the reference implementations where they exist.

AGENTS.md §7 forbids a baseline entering a paper table before it has been reproduced
against the authors' code. APS and RAPS ship in the long-tail-conformal release, so
these tests compare elementwise against a transcription of `get_APS_scores_all` and
`get_RAPS_scores_all` rather than against numbers I chose. SAPS has no reference in that
release, so it is pinned by the properties the paper states instead.
"""
import numpy as np
import pytest

from pcc.scores.base import SCORE_FNS, aps, raps, saps, score_matrix, thr_lac


@pytest.fixture()
def probs():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 40)) * 2.0
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


# --- reference transcriptions -------------------------------------------------------
# Copied structurally from utils/conformal_utils.py of the long-tail-conformal release,
# with torch swapped for numpy. Kept separate from the implementation on purpose: if the
# two ever drift, that is the signal, and a shared helper would hide it.
def _ref_aps(sm, randomize=True, seed=0):
    order = np.argsort(-sm, axis=1, kind="stable")
    inv = np.argsort(order, axis=1)
    scores = np.take_along_axis(np.cumsum(np.take_along_axis(sm, order, axis=1), axis=1),
                                inv, axis=1)
    if not randomize:
        return scores - sm
    np.random.seed(seed)
    return scores - np.random.rand(*sm.shape) * sm


def _ref_raps(sm, lmbda, kreg, randomize=True, seed=0):
    order = np.argsort(-sm, axis=1, kind="stable")
    inv = np.argsort(order, axis=1)
    scores = np.take_along_axis(np.cumsum(np.take_along_axis(sm, order, axis=1), axis=1),
                                inv, axis=1)
    scores = scores + np.maximum(lmbda * ((inv + 1) - kreg), 0.0)
    if not randomize:
        return scores - sm
    np.random.seed(seed)
    return scores - np.random.rand(*sm.shape) * sm


def test_aps_matches_the_released_implementation(probs):
    for rnd in (False, True):
        got = aps(probs, randomize=rnd, seed=0)
        assert np.allclose(got, _ref_aps(probs, randomize=rnd, seed=0), atol=1e-10)


def test_raps_matches_the_released_implementation(probs):
    for k_reg, lam in ((5, 0.01), (1, 0.1), (10, 0.0)):
        got = raps(probs, k_reg=k_reg, lam=lam, randomize=False)
        assert np.allclose(got, _ref_raps(probs, lam, k_reg, randomize=False), atol=1e-10)


def test_raps_with_zero_penalty_is_exactly_aps(probs):
    """lam=0 removes the regularizer, so RAPS must collapse onto APS -- if it does not,
    the rank is off by one somewhere, which is the classic way to get RAPS subtly wrong."""
    assert np.allclose(raps(probs, k_reg=5, lam=0.0, randomize=False),
                       aps(probs, randomize=False), atol=1e-12)


def test_every_score_keeps_the_repo_convention(probs):
    """Higher = more nonconforming, and the true class of a confident row scores low."""
    top = probs.argmax(1)
    rows = np.arange(len(probs))
    for name in SCORE_FNS:
        S = score_matrix(probs, name, seed=0)
        assert S.shape == probs.shape, name
        assert np.isfinite(S).all(), name
        # the top-1 label must be among the lowest-scoring labels of its row
        r = (S <= S[rows, top][:, None]).sum(1)
        assert np.median(r) <= 2, (name, np.median(r))


def test_saps_ranks_beyond_the_top_grow_with_rank(probs):
    """SAPS discards the tail probabilities and pays `lam` per rank, so past rank 1 the
    score must be monotone in rank -- that IS the method, so it is worth pinning."""
    S = saps(probs, lam=0.2, randomize=False)
    order = np.argsort(-probs, axis=1, kind="stable")
    walk = np.take_along_axis(S, order, axis=1)
    assert (np.diff(walk[:, 1:], axis=1) > 0).all()
    assert (S >= 0).all()


def test_saps_lambda_controls_set_size_growth(probs):
    """A bigger lam penalises depth harder, so at any fixed threshold the sets shrink."""
    a = saps(probs, lam=0.05, randomize=False)
    b = saps(probs, lam=0.50, randomize=False)
    t = float(np.quantile(a, 0.5))
    assert (b <= t).sum() < (a <= t).sum()


def test_thr_is_still_the_plain_one_minus_probability(probs):
    assert np.allclose(score_matrix(probs, "thr"), thr_lac(probs))


def test_unknown_score_name_lists_what_is_available():
    with pytest.raises(ValueError) as ei:
        score_matrix(np.ones((2, 3)) / 3.0, "nope")
    assert "aps" in str(ei.value) and "saps" in str(ei.value)
