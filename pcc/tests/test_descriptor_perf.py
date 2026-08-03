"""Guards the exactness of the descriptor fast paths (pcc/descriptors/phi.py).

Both optimizations must stay NUMERICALLY EQUIVALENT to the naive version — they
were introduced for speed (the d^3 eigendecomposition made descriptor stability
take hours at d=2048), not as approximations. If someone "simplifies" them back
or swaps in a truncated/randomized solver, these tests fail.

Runnable with pytest OR directly: `python pcc/tests/test_descriptor_perf.py`.
"""

from __future__ import annotations

import numpy as np

from pcc.descriptors.phi import (cosine_knn_matrix, within_class_cov_stats,
                                 class_means)


def test_gram_eigenvalues_match_full_covariance():
    """q x q Gram trick == d x d covariance, for q << d."""
    rng = np.random.default_rng(0)
    for q, d in [(10, 2048), (50, 512), (100, 2048)]:
        f = rng.normal(size=(q, d))
        got = within_class_cov_stats(f, top_k=3)

        cov = np.cov(f, rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]

        assert abs(got["cov_trace"] - np.trace(cov)) < 1e-8 * max(1.0, abs(np.trace(cov))), \
            f"trace mismatch at q={q}, d={d}"
        for i in range(3):
            assert abs(got[f"cov_eig_{i}"] - eig[i]) < 1e-6 * max(1.0, abs(eig[i])), \
                f"eig_{i} mismatch at q={q}, d={d}"


def test_cosine_knn_matrix_matches_naive_loop():
    """Vectorized [K,K] cosine == per-class normalize-and-sort loop."""
    rng = np.random.default_rng(1)
    K, d = 40, 64
    means = rng.normal(size=(K, d))
    means[7] = np.nan  # an absent class must stay NaN and be excluded from others
    ks = (1, 5, 10)
    got = cosine_knn_matrix(means, ks)

    valid = ~np.isnan(means).any(axis=1)
    for y in range(K):
        if not valid[y]:
            for k in ks:
                assert np.isnan(got[k][y]), f"absent class {y} must be NaN"
            continue
        a = means[y] / np.linalg.norm(means[y])
        others = valid.copy(); others[y] = False
        B = means[others]
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        cos = np.sort(Bn @ a)[::-1]
        for k in ks:
            assert abs(got[k][y] - cos[:k].mean()) < 1e-9, \
                f"cos_knn_{k} mismatch at class {y}"


def test_class_grouping_is_order_independent():
    """build_descriptors groups samples by class via argsort+searchsorted; the
    result must not depend on the order samples arrive in."""
    from pcc.descriptors.phi import build_descriptors
    rng = np.random.default_rng(2)
    K, d, per = 12, 32, 20
    F, C = [], []
    for y in range(K):
        F.append(rng.normal(y, 1.0, (per, d))); C.append(np.full(per, y))
    F = np.vstack(F); C = np.concatenate(C)
    L = rng.normal(size=(len(F), K))

    P1, names = build_descriptors(F, L, C, K, ks=(1, 5))
    perm = rng.permutation(len(F))
    P2, _ = build_descriptors(F[perm], L[perm], C[perm], K, ks=(1, 5))
    assert np.allclose(P1, P2, equal_nan=True), "descriptors depend on sample order"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print(f"all {len(fns)} descriptor fast-path tests passed")
