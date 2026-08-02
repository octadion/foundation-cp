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


def cosine_to_knn(y, means, ks=(1, 5, 10, 50)):
    """For class y, mean cosine to its k nearest OTHER class means, for each k."""
    valid = ~np.isnan(means).any(axis=1)
    valid[y] = False
    others = means[valid]
    if len(others) == 0:
        return {f"cos_knn_{k}": np.nan for k in ks}
    cos = np.sort(_cosine(means[y], others))[::-1]  # descending similarity
    return {f"cos_knn_{k}": float(cos[:k].mean()) if len(cos) >= 1 else np.nan
            for k in ks}


def within_class_cov_stats(features_y, top_k=3):
    """Trace and top-k eigenvalues of the within-class covariance."""
    f = np.asarray(features_y, float)
    if len(f) < 2:
        return {"cov_trace": np.nan, **{f"cov_eig_{i}": np.nan for i in range(top_k)}}
    cov = np.cov(f, rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    out = {"cov_trace": float(np.trace(cov))}
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

    rows, names = [], None
    for y in range(n_classes):
        m = classes == y
        if not m.any():
            rows.append(None)
            continue
        feat = {}
        feat["mean_norm"] = float(np.linalg.norm(means[y]))
        feat.update(cosine_to_knn(y, means, ks))
        feat.update(within_class_cov_stats(features[m], top_eig))
        feat["n_eff"] = float(m.sum())
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
