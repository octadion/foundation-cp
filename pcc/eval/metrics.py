"""Metrics that must ALWAYS be reported together (AGENTS.md §9). Reporting set
size alone is the pattern that got prior work accused of selective reporting.

Every results table must carry, simultaneously: average set size, marginal
coverage, class-conditional coverage gap (CovGap), worst-class coverage,
worst-slab coverage, set size split by head/tail, and set size on held-out vs
seen classes (separately).
"""

from __future__ import annotations

import numpy as np


def marginal_coverage(sets, labels) -> float:
    return float(sets[np.arange(len(labels)), labels].mean())


def per_class_coverage(sets, labels, n_classes: int) -> np.ndarray:
    """Coverage within each class; NaN for classes absent from `labels`."""
    cov = np.full(n_classes, np.nan)
    hit = sets[np.arange(len(labels)), labels]
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            cov[c] = hit[mask].mean()
    return cov


def covgap(sets, labels, n_classes: int, alpha: float) -> float:
    """Class-conditional coverage gap: mean |cov_c - (1-α)| over present classes."""
    cov = per_class_coverage(sets, labels, n_classes)
    present = ~np.isnan(cov)
    return float(np.abs(cov[present] - (1 - alpha)).mean())


def worst_class_coverage(sets, labels, n_classes: int) -> float:
    cov = per_class_coverage(sets, labels, n_classes)
    return float(np.nanmin(cov))


def worst_slab_coverage(sets, labels, n_classes: int, *, n_slabs: int = 10) -> float:
    """Worst coverage over contiguous slabs of classes ordered by prevalence
    (approximate slab metric; slab ordering supplied by caller when needed)."""
    cov = per_class_coverage(sets, labels, n_classes)
    present_idx = np.where(~np.isnan(cov))[0]
    if len(present_idx) == 0:
        return float("nan")
    slabs = np.array_split(present_idx, min(n_slabs, len(present_idx)))
    return float(min(np.nanmean(cov[s]) for s in slabs))


def set_size_by_group(sets, labels, group_of_class) -> dict:
    """Average set size split by a per-class group label (e.g. 'head'/'tail',
    or 'seen'/'held_out'). `group_of_class` maps class id -> group name."""
    sizes = sets.sum(axis=1)
    groups = np.array([group_of_class[int(c)] for c in labels])
    return {g: float(sizes[groups == g].mean()) for g in sorted(set(groups))}


def summary(sets, labels, n_classes, alpha, *, group_of_class=None) -> dict:
    """The full §9 bundle. group_of_class enables the head/tail and
    seen/held-out breakdowns; pass it whenever those splits are defined."""
    out = {
        "avg_set_size": float(sets.sum(axis=1).mean()),
        "marginal_coverage": marginal_coverage(sets, labels),
        "covgap": covgap(sets, labels, n_classes, alpha),
        "worst_class_coverage": worst_class_coverage(sets, labels, n_classes),
        "worst_slab_coverage": worst_slab_coverage(sets, labels, n_classes),
    }
    if group_of_class is not None:
        out["set_size_by_group"] = set_size_by_group(sets, labels, group_of_class)
    return out
