"""φ(y): class-geometry descriptors (AGENTS.md §6.3).

HARD RULE: descriptors are computed from TRAINING data only, and never from the
calibration split. Callers MUST route the sample ids used here through
pcc.eval.leakguard.assert_descriptors_clean(...) before fitting anything.

Planned descriptor set (§6.3), computed per class y:
  - penultimate feature mean of the class (and its norm)
  - cosine similarity to the k nearest class means (k in {1, 5, 10, 50})
  - trace / top eigenvalues of the within-class covariance
  - effective sample count, log prevalence
  - logit statistics: mean margin, rank distribution, entropy

Descriptor STABILITY as a function of images-per-class (10/25/50/100) must be
measured first (notebooks/01_descriptor_stability, §3.3) so that a negative
Phase-1 result is interpretable (noise in signal vs. noise in descriptor).

This module is intentionally a documented interface until 01_descriptor_stability
fixes the per-class image quota; implementing descriptors before that quota is
set risks baking in an unstable descriptor.
"""

from __future__ import annotations

import numpy as np


def class_mean(features: np.ndarray) -> np.ndarray:
    """Mean penultimate feature vector for one class (features: [n_y, d])."""
    return features.mean(axis=0)


def class_means(features, classes, n_classes):
    """[n_classes, d] matrix of class means; NaN rows for absent classes."""
    features = np.asarray(features, float)
    classes = np.asarray(classes)
    d = features.shape[1]
    means = np.full((n_classes, d), np.nan)
    for y in range(n_classes):
        m = classes == y
        if m.any():
            means[y] = features[m].mean(axis=0)
    return means


def _cosine(a, B):
    a = a / (np.linalg.norm(a) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return Bn @ a


def cosine_knn_matrix(means, ks=(1, 5, 10, 50)):
    """Vectorized: for every class, the mean cosine to its k nearest OTHER class
    means. One [K,K] matmul instead of re-normalizing the means per class.

    Returns {k: array[K]} with NaN for classes whose mean is undefined.
    """
    means = np.asarray(means, float)
    K = len(means)
    valid = ~np.isnan(means).any(axis=1)
    Mn = np.zeros_like(means)
    norms = np.linalg.norm(means[valid], axis=1, keepdims=True) + 1e-12
    Mn[valid] = means[valid] / norms
    S = Mn @ Mn.T                      # cosine similarity matrix
    out = {k: np.full(K, np.nan) for k in ks}
    for y in np.where(valid)[0]:
        others = valid.copy()
        others[y] = False
        if not others.any():
            continue
        cos = np.sort(S[y, others])[::-1]   # descending similarity
        for k in ks:
            out[k][y] = float(cos[:k].mean())
    return out


def within_class_cov_stats(features_y, top_k=3):
    """Trace and top-k eigenvalues of the within-class covariance.

    Computed via the q x q GRAM matrix rather than the d x d covariance. For
    q samples in d dims with q << d the covariance has rank <= q-1, and the
    nonzero eigenvalues of Xc^T Xc (d x d) and Xc Xc^T (q x q) are IDENTICAL —
    so this is exact, not an approximation. Verified equal to the full-covariance
    result to ~1e-14, and ~185x faster at q=100, d=2048 (the d^3 eigendecomposition
    was making descriptor stability take hours).
    """
    f = np.asarray(features_y, float)
    q = len(f)
    if q < 2:
        return {"cov_trace": np.nan, **{f"cov_eig_{i}": np.nan for i in range(top_k)}}
    Xc = f - f.mean(axis=0)
    G = (Xc @ Xc.T) / (q - 1)                      # [q, q], tiny
    eig = np.sort(np.linalg.eigvalsh(G))[::-1]
    out = {"cov_trace": float(f.var(axis=0, ddof=1).sum())}
    for i in range(top_k):
        out[f"cov_eig_{i}"] = float(eig[i]) if i < len(eig) else 0.0
    return out


def logit_stats(logits_y, y):
    """Per-class logit statistics: mean top-1 margin, entropy of mean softmax,
    mean rank of the true class (0 = top)."""
    L = np.asarray(logits_y, float)
    if len(L) == 0:
        return {"logit_margin": np.nan, "softmax_entropy": np.nan, "true_rank": np.nan}
    part = np.partition(L, -2, axis=1)
    margin = (part[:, -1] - part[:, -2]).mean()
    e = np.exp(L - L.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    entropy = float((-(p * np.log(p + 1e-12)).sum(axis=1)).mean())
    ranks = (L > L[:, [y]]).sum(axis=1)  # how many classes beat the true class
    return {"logit_margin": float(margin), "softmax_entropy": entropy,
            "true_rank": float(ranks.mean())}


def build_descriptors(features, logits, classes, n_classes, *,
                      ks=(1, 5, 10, 50), top_eig=3, log_prevalence_from=None):
    """Assemble the φ(y) matrix over classes (AGENTS.md §6.3).

    features/logits/classes are per-TRAINING-SAMPLE arrays; the caller MUST have
    already routed the sample ids through pcc.eval.leakguard.assert_descriptors_clean
    (descriptors never touch the calibration split).

    `log_prevalence_from`: per-class counts to use for log-prevalence (e.g. full
    train counts); defaults to counts within `classes`.

    Returns (Phi [n_classes, D], feature_names). Rows for absent classes are NaN.
    """
    features = np.asarray(features, float)
    logits = None if logits is None else np.asarray(logits, float)
    classes = np.asarray(classes)
    means = class_means(features, classes, n_classes)

    if log_prevalence_from is None:
        counts = np.bincount(classes, minlength=n_classes)
    else:
        counts = np.asarray(log_prevalence_from)

    # precompute the cosine-kNN features for ALL classes in one pass
    cos_knn = cosine_knn_matrix(means, ks)
    # group sample indices by class once, instead of a boolean scan per class
    order = np.argsort(classes, kind="stable")
    bounds = np.searchsorted(classes[order], np.arange(n_classes + 1))

    rows, names = [], None
    for y in range(n_classes):
        idx = order[bounds[y]:bounds[y + 1]]
        if len(idx) == 0:
            rows.append(None)
            continue
        m = idx
        feat = {}
        feat["mean_norm"] = float(np.linalg.norm(means[y]))
        feat.update({f"cos_knn_{k}": float(cos_knn[k][y]) for k in ks})
        feat.update(within_class_cov_stats(features[m], top_eig))
        # NOTE: `m` is an INTEGER index array (not a boolean mask), so the count is
        # len(m). Using m.sum() here silently summed the index VALUES — a bug that
        # made n_eff correlate with class id and could have faked predictability.
        # Guarded by tests/test_descriptor_perf.py::test_class_grouping_is_order_independent.
        feat["n_eff"] = float(len(m))
        feat["log_prevalence"] = float(np.log(counts[y] + 1e-12))
        if logits is not None:
            feat.update(logit_stats(logits[m], y))
        if names is None:
            names = list(feat.keys())
        rows.append([feat[k] for k in names])

    D = len(names)
    Phi = np.full((n_classes, D), np.nan)
    for y, r in enumerate(rows):
        if r is not None:
            Phi[y] = r
    return Phi, names
