"""Leakage guards enforced by ASSERTION, not convention (AGENTS.md §8.2, §8.3).

Two failure modes from prior work these guards close:

1. Sample-level splitting for an extrapolation claim -> misleadingly optimistic
   numbers. The claim is about held-out *classes*, so the fit/held-out split
   must be at the CLASS level (§8.2).
2. Class descriptors touching the calibration split -> the correction is no
   longer being *predicted* from geometry, it has seen the calibration labels
   of the target class (§8.3).

Import these and call them in every experiment. They raise loudly. Prefer a
hard failure over a silent wrong number.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class LeakageError(AssertionError):
    """Raised when a data-flow invariant that protects the claim is violated."""


def assert_disjoint(*groups: Iterable, names: tuple[str, ...] | None = None) -> None:
    """Assert that the given index/id groups are pairwise disjoint."""
    sets = [set(g) for g in groups]
    labels = names or tuple(f"group[{i}]" for i in range(len(sets)))
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise LeakageError(
                    f"{labels[i]} and {labels[j]} overlap on {len(overlap)} items "
                    f"(e.g. {sorted(overlap)[:5]}). Splits must be disjoint."
                )


def class_level_split(classes, fractions=(0.4, 0.3, 0.3), *, seed: int,
                      names=("fit", "cal", "eval")):
    """Partition the set of CLASSES (not samples) into disjoint groups.

    For the extrapolation claim, 'fit' classes are 𝒴_train (g_θ / descriptor
    fitting) and must be disjoint from the classes used for calibration/eval.
    Returns a dict name -> np.ndarray of class ids. Deterministic in `seed`.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if len(fractions) != len(names):
        raise ValueError("fractions and names length mismatch")
    classes = np.asarray(sorted(set(classes)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(classes))
    cuts = np.cumsum([int(round(f * len(classes))) for f in fractions[:-1]])
    parts = np.split(perm, cuts)
    out = {name: classes[idx] for name, idx in zip(names, parts)}
    assert_disjoint(*out.values(), names=tuple(out.keys()))
    return out


def assert_descriptors_clean(descriptor_source_sample_ids, calibration_sample_ids) -> None:
    """Enforce §8.3: class descriptors must be computed from training data only
    and must NEVER include any sample that is in the calibration split.
    """
    overlap = set(descriptor_source_sample_ids) & set(calibration_sample_ids)
    if overlap:
        raise LeakageError(
            f"Descriptors were computed from {len(overlap)} sample(s) that are in "
            f"the calibration split (e.g. {sorted(overlap)[:5]}). Descriptors must "
            f"use TRAINING data only (AGENTS.md §6.3, §8.3)."
        )


def assert_no_target_labels(target_classes, descriptor_classes_used) -> None:
    """Sanity check for the extrapolation regime: computing a descriptor for a
    held-out target class must not require having drawn on that class's own
    calibration labels. (The descriptor itself is from training data; this guards
    against an experiment silently using target-class calibration info.)
    """
    leaked = set(target_classes) & set(descriptor_classes_used)
    if leaked:
        raise LeakageError(
            f"{len(leaked)} held-out target class(es) appear in the set the "
            f"correction was estimated from -> this tests estimation, not "
            f"extrapolation (AGENTS.md §0, §8.2)."
        )
