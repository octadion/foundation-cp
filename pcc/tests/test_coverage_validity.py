"""Coverage-validity invariant (AGENTS.md §8.7).

For ANY δ̂_y — including a random or deliberately terrible one — the modified
score s'(x,y) = s(x,y) − δ̂_y must still yield marginal coverage ≈ 1−α under
split conformal, because s' is a measurable function of (x,y). If this test
fails, there is a bug in the score path, not a modeling problem.

Run: pytest pcc/tests/test_coverage_validity.py
"""

from __future__ import annotations

import numpy as np
import pytest

from pcc.eval.conformal import build_sets, conformal_quantile, coverage


def _synthetic(n=20000, n_classes=50, seed=0):
    """Well-specified synthetic: scores are true-label nonconformity for a
    softmax-like model. Returns (score_matrix, labels)."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, n)
    logits = rng.normal(size=(n, n_classes))
    logits[np.arange(n), labels] += 2.0  # true label gets a signal boost
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    scores = 1.0 - probs  # THR/LAC-style nonconformity: higher = worse
    return scores, labels


def _split(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[: n // 2], idx[n // 2 :]


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
@pytest.mark.parametrize("delta_kind", ["zero", "random", "adversarial"])
def test_marginal_coverage_holds_for_any_delta(alpha, delta_kind):
    scores, labels = _synthetic()
    n, n_classes = scores.shape
    cal_idx, eval_idx = _split(n, seed=1)

    rng = np.random.default_rng(7)
    if delta_kind == "zero":
        delta = np.zeros(n_classes)
    elif delta_kind == "random":
        delta = rng.normal(scale=0.2, size=n_classes)
    else:  # adversarial: large per-class offsets uncorrelated with anything real
        delta = rng.normal(scale=2.0, size=n_classes)

    # modified score s' = s - delta_y (broadcast per candidate class column)
    s_prime = scores - delta[None, :]

    cal_true = s_prime[cal_idx, labels[cal_idx]]
    qhat = conformal_quantile(cal_true, alpha)
    sets = build_sets(s_prime[eval_idx], qhat)
    cov = coverage(sets, labels[eval_idx])

    # finite-sample slack; must not systematically undercover
    assert cov >= (1 - alpha) - 0.02, (
        f"marginal coverage {cov:.4f} < target {1-alpha:.4f} for "
        f"delta={delta_kind}, alpha={alpha}: BUG in score path"
    )
