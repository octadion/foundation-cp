"""Multi-split protocol + uncertainty (AGENTS.md §8). Phase-agnostic.

- >=100 random splits, report mean + CI (§8.4)
- paired bootstrap, 10_000 iterations, for method comparisons (§8.5)
- Holm-Bonferroni across all reported comparisons (§8.6)

No number leaves this repo without a CI/SE (§0.4).
"""

from __future__ import annotations

import numpy as np


def mean_ci(values, alpha: float = 0.05):
    """Mean and normal-approx CI (SE-based) over split repetitions.

    NaN-AWARE, and that matters: a single non-finite repetition used to turn the whole
    CI into NaN, which is how `gate A: nan` reached a report instead of a diagnosable
    number. Non-finite entries are dropped and counted, so the loss is visible:
    `n` is how many were supplied, `n_finite` how many were used.

    Uses `statistics.NormalDist` rather than `scipy.stats.norm` — one normal quantile is
    not worth a hard scipy dependency, and scipy is absent in some environments this
    repo is expected to run in.
    """
    from statistics import NormalDist

    v = np.asarray(values, dtype=float)
    n = len(v)
    f = v[np.isfinite(v)]
    if len(f) == 0:
        return {"mean": float("nan"), "se": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": n, "n_finite": 0}
    mean = float(f.mean())
    se = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else float("nan")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    lo = mean - z * se if np.isfinite(se) else float("nan")
    hi = mean + z * se if np.isfinite(se) else float("nan")
    return {"mean": mean, "se": se, "ci_low": lo, "ci_high": hi,
            "n": n, "n_finite": int(len(f))}


def paired_bootstrap(a, b, *, n_boot: int = 10_000, seed: int = 42, alpha: float = 0.05):
    """Paired bootstrap on the per-unit difference (a - b). Returns the mean
    difference, its bootstrap CI, and a two-sided p-value. `a` and `b` are paired
    per-unit metrics (e.g. per-class set size under method A vs B).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap requires equal-length paired arrays")
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = diff[idx].mean()
    obs = float(diff.mean())
    ci_low, ci_high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    # two-sided p-value from the bootstrap sign
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_diff": obs, "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p_value": float(min(p, 1.0)), "n_boot": n_boot}


def holm_bonferroni(pvalues, alpha: float = 0.05):
    """Holm-Bonferroni step-down. Returns list of dicts with adjusted threshold
    and reject flag, preserving input order (§8.6)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    rejected = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        if p[idx] <= thresh:
            rejected[idx] = True
        else:
            break  # step-down stops at first non-rejection
    return [
        {"p_value": float(p[i]), "threshold": float(alpha / (m - int(np.where(order == i)[0][0]))),
         "reject": bool(rejected[i])}
        for i in range(m)
    ]
