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

def sign_flip_test(diffs, *, n_perm=20000, seed=42, alternative="greater"):
    """Exact-in-the-limit randomization test for a PAIRED design (§8.6 companion).

    Replaces the normal approximation used in the Amendment-9 run, where a p-value was
    derived from a CI width as `z = mean/(ciW/3.92)`. That approximation could not
    separate p=0.0125 from p=0.0133 at the Holm boundary, and the boundary is exactly
    where it mattered — so the primary test now uses this instead.

    Under H0 the paired differences are symmetric about 0, so flipping their signs at
    random generates the null distribution of the mean. `alternative="greater"` tests
    "full beats the ablation".

    LIMITATION, stated because it is not removable: the per-split differences share a
    class pool, so they are NOT independent and the test is only approximately exact.
    It is strictly better conditioned than the normal approximation, not a substitute
    for a class-label permutation null (`class_permutation_p`), which is the stronger
    and more expensive option.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"p_value": float("nan"), "n": n, "observed": float("nan"),
                "n_perm": 0, "undefined_reason": "fewer than 3 paired observations"}
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, n))
    null = (signs * d).mean(axis=1)
    if alternative == "greater":
        hits = int(np.sum(null >= obs))
    elif alternative == "less":
        hits = int(np.sum(null <= obs))
    else:
        hits = int(np.sum(np.abs(null) >= abs(obs)))
    # +1 in numerator and denominator: the observed assignment is itself one of the
    # equally likely sign patterns, so a p of exactly 0 is not attainable.
    return {"p_value": (hits + 1) / (n_perm + 1), "n": n, "observed": obs,
            "n_perm": int(n_perm), "undefined_reason": None}
