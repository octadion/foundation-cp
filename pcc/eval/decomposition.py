"""Phase 0 (§5) — decompose the efficiency gap on REAL logits.

Measures how much of the average-set-size gap is closed, SEPARATELY, by:
  (1) a single global temperature,
  (2) per-sample energy reweighting (free energy of the pre-softmax logits),
  (3) a per-class offset fit on ABUNDANT data (not a realistic budget).

Pass criterion (§5): (3) must close a gap SUBSTANTIALLY larger than (1) and (2),
with non-overlapping CIs. If not, "structure lives at the class level" does not
hold on real data — STOP, do not enter Phase 1 (§5).

Why this exists: the motivating evidence came from synthetic injection-and-
recovery, which is circular. This must replicate on real logits (ImageNet,
iNaturalist) first.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.conformal import build_sets, conformal_quantile, coverage, set_sizes


def temperature_softmax(logits, T):
    z = np.asarray(logits, float) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def thr_scores_from_softmax(probs):
    """THR/LAC nonconformity (repo convention, higher = worse)."""
    return 1.0 - probs


def free_energy(logits):
    """Free energy of the pre-softmax logits: E(x) = -logsumexp(logits)."""
    L = np.asarray(logits, float)
    m = L.max(axis=1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(L - m).sum(axis=1)))


def group_quantile(scores, alpha, estimator="empirical"):
    """Threshold for a GROUP of calibration scores.

    CRITICAL for §5 (verified 2026-08-03, see reports/protocol_amendments.md):

    - `conformal`: the finite-sample-valid quantile at level ceil((n+1)(1-α))/n.
      Required for DEPLOYMENT (guarantees coverage), but the level DEPENDS ON n.
      A per-class group with n=50 targets the class's ~98th percentile while the
      pooled global group targets the ~95th — so the difference q̂_y − q̂_global is
      contaminated by an estimator artefact that grows as n shrinks. On real
      CIFAR-100 logits this made EVERY component of the decomposition negative
      (per-class offset −5.6), which is not a finding, it is a level mismatch.

    - `empirical`: the plain (1−α) quantile, the SAME target percentile for every
      group regardless of n. This is the right choice for MEASURING STRUCTURE
      (§5 asks how much gap class-level structure could close, fit on abundant
      data). Verified on synthetic at realistic accuracy: per-class gap
      −5.98 (conformal) → +1.98 (empirical) ≈ +1.84 (abundant-data oracle).

    Structure measurement and deployment validity are different questions; §8.7
    guards validity separately. Do not use `empirical` to build deployed sets.
    """
    scores = np.asarray(scores, float)
    if len(scores) == 0:
        return np.inf
    if estimator == "conformal":
        return conformal_quantile(scores, alpha)
    if estimator == "empirical":
        return float(np.quantile(scores, 1 - alpha))
    raise ValueError(f"unknown estimator {estimator!r}")


def efficiency(score_matrix, labels, alpha, cal_idx, eval_idx, threshold=None,
               estimator="empirical"):
    """Avg set size at target coverage. Calibrates the marginal threshold on
    cal_idx (unless `threshold` given) and reports size + coverage on eval_idx."""
    cal_true = score_matrix[cal_idx, labels[cal_idx]]
    q = group_quantile(cal_true, alpha, estimator) if threshold is None else threshold
    sets = build_sets(score_matrix[eval_idx], q)
    return {"avg_set_size": float(set_sizes(sets).mean()),
            "coverage": coverage(sets, labels[eval_idx]), "threshold": float(q)}


def gap_from_global_temperature(logits, labels, alpha, cal_idx, eval_idx,
                                Ts=None, estimator="empirical"):
    """Best global temperature by eval avg-set-size (component 1). Returns the
    baseline (T=1) size and the best-T size; the gap closed is baseline - best."""
    Ts = Ts if Ts is not None else np.linspace(0.5, 3.0, 26)
    base = efficiency(thr_scores_from_softmax(temperature_softmax(logits, 1.0)),
                      labels, alpha, cal_idx, eval_idx, estimator=estimator)
    best = base
    best_T = 1.0
    for T in Ts:
        e = efficiency(thr_scores_from_softmax(temperature_softmax(logits, T)),
                       labels, alpha, cal_idx, eval_idx, estimator=estimator)
        if e["avg_set_size"] < best["avg_set_size"]:
            best, best_T = e, T
    return {"baseline_size": base["avg_set_size"], "best_size": best["avg_set_size"],
            "gap_closed": base["avg_set_size"] - best["avg_set_size"], "best_T": best_T}


def gap_from_per_class_offset(score_matrix, labels, n_classes, alpha,
                              cal_idx, eval_idx, estimator="empirical"):
    """Per-class offset fit on ABUNDANT calibration data (component 3): apply
    classwise offsets estimated on cal_idx, measure eval avg set size. This is the
    'structure at the class level' upper bound (not a realistic budget).

    Uses `group_quantile(..., estimator)` for BOTH the per-class and the global
    threshold so both target the same percentile — see group_quantile's docstring
    for why the default is `empirical` here and why `conformal` silently destroys
    this measurement.
    """
    from pcc.eval.setsize import corrected_thresholds

    cal_true = score_matrix[cal_idx, labels[cal_idx]]
    cal_lab = labels[cal_idx]
    q_global = group_quantile(cal_true, alpha, estimator)
    delta = np.full(n_classes, np.nan)
    for y in range(n_classes):
        m = cal_lab == y
        if m.any():
            qy = group_quantile(cal_true[m], alpha, estimator)
            delta[y] = qy - q_global
    base = efficiency(score_matrix, labels, alpha, cal_idx, eval_idx,
                      threshold=q_global)
    sets = build_sets(score_matrix[eval_idx], corrected_thresholds(q_global, delta))
    size = float(set_sizes(sets).mean())
    return {"baseline_size": base["avg_set_size"], "offset_size": size,
            "gap_closed": base["avg_set_size"] - size,
            "coverage": coverage(sets, labels[eval_idx]),
            "n_free_params": int(np.isfinite(delta).sum()), "estimator": estimator}


def gap_from_energy_reweighting(logits, score_matrix, labels, alpha,
                                cal_idx, eval_idx, *, n_bins=10,
                                estimator="empirical"):
    """Component (2): per-sample correction indexed by free energy.

    DESIGN DECISION (recorded 2026-07-25). §5 specifies only "pembobotan
    per-sampel berbasis energi (free energy dari ruang pre-softmax)". The
    instantiation chosen makes all three components **structurally identical** —
    the same conformal offset estimator, fit on abundant data — differing ONLY in
    what indexes the correction:

        (1) temperature      : 1 global scalar
        (2) energy (this fn) : offset per ENERGY BIN, E(x) = -logsumexp(logits)
        (3) class offset     : offset per CLASS (δ_y)

    So (2) is "correction indexed by per-sample difficulty" and (3) is
    "correction indexed by class". If (3) closes a substantially larger gap with
    non-overlapping CIs, that is evidence the structure genuinely lives at the
    CLASS level rather than being a relabelling of per-sample difficulty — which
    is exactly the §5 hypothesis. Any other scheme is a fair alternative; this one
    is chosen because it makes the comparison apples-to-apples.

    Bin edges come from CALIBRATION energies only (no eval leakage). Each eval
    sample gets its own threshold `q̂_global + offset[bin(E(x))]` — a per-sample
    threshold, unlike the per-class threshold of component (3).

    NOTE on parameter count: n_bins vs K classes are not equal, so sweep n_bins
    and report the curve — a component with more free parameters can close more gap
    trivially. `phase0_energy_bin_sweep` does this.
    """
    E = free_energy(logits)
    cal_true = score_matrix[cal_idx, labels[cal_idx]]
    q_global = group_quantile(cal_true, alpha, estimator)

    edges = np.quantile(E[cal_idx], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_of = np.clip(np.digitize(E, edges) - 1, 0, n_bins - 1)

    offsets = np.zeros(n_bins)
    cal_bins = bin_of[cal_idx]
    for b in range(n_bins):
        m = cal_bins == b
        if m.any():
            qb = group_quantile(cal_true[m], alpha, estimator)
            offsets[b] = 0.0 if not np.isfinite(qb) else qb - q_global

    base = efficiency(score_matrix, labels, alpha, cal_idx, eval_idx,
                      threshold=q_global)
    thr = q_global + offsets[bin_of[eval_idx]]          # per-SAMPLE threshold
    sets = score_matrix[eval_idx] <= thr[:, None]
    size = float(sets.sum(axis=1).mean())
    return {"baseline_size": base["avg_set_size"], "energy_size": size,
            "gap_closed": base["avg_set_size"] - size,
            "coverage": coverage(sets, labels[eval_idx]),
            "n_bins": n_bins, "n_free_params": n_bins, "estimator": estimator}


def _sets_from_thresholds(score_matrix, thresholds, inflate=0.0):
    """thresholds: [K] (per candidate class) or [n] (per sample). Adds `inflate`."""
    t = np.asarray(thresholds, float) + inflate
    if t.ndim == 0:
        return score_matrix <= t
    if len(t) == score_matrix.shape[1]:
        return score_matrix <= t[None, :]
    if len(t) == score_matrix.shape[0]:
        return score_matrix <= t[:, None]
    raise ValueError(f"threshold length {len(t)} matches neither classes nor samples")


def min_size_at_worst_class_coverage(score_matrix, labels, n_classes, thresholds,
                                     target, *, tol=1e-4, max_inflate=10.0,
                                     allow_deflate=False):
    """Smallest average set size achieving worst-class coverage >= `target`, by
    uniformly inflating `thresholds`.

    This is the well-posed efficiency measure for a CLASS-level mechanism
    (Amendment 3, reports/protocol_amendments.md): average set size at nominal
    MARGINAL coverage cannot reward a class-indexed correction, because marginal
    split-CP is already optimal for marginal coverage. Here every mechanism is
    held to the SAME class-conditional requirement and we ask what it costs.

    Returns (avg_set_size, achieved_worst_coverage, inflation). If the target is
    unreachable within `max_inflate`, returns the max-inflation result.
    """
    from pcc.eval.metrics import per_class_coverage

    def worst(c):
        sets = _sets_from_thresholds(score_matrix, thresholds, c)
        cov = per_class_coverage(sets, labels, n_classes)
        return float(np.nanmin(cov)), sets

    lo = 0.0
    w0, _ = worst(0.0)
    if w0 >= target:
        if not allow_deflate:
            hi = 0.0
        else:
            # The vector already over-covers, so it is WASTEFUL: shrink it until the
            # target is only just met. Without this a threshold vector carrying a
            # large positive constant is punished for being generous rather than
            # judged neutral — a constant is a pure global shift and must be a
            # no-op (verified: constant goes from -16.27 to ~0 once deflation is
            # allowed). Required for a fair §6.4 comparison.
            hi, lo = 0.0, -max_inflate
            while lo < hi - tol:
                mid = (lo + hi) / 2
                w, _ = worst(mid)
                if w >= target:
                    hi = mid
                else:
                    lo = mid
    else:
        hi = 0.5
        while hi <= max_inflate:
            w, _ = worst(hi)
            if w >= target:
                break
            lo, hi = hi, hi * 2
        hi = min(hi, max_inflate)
        while hi - lo > tol:
            mid = (lo + hi) / 2
            w, _ = worst(mid)
            if w >= target:
                hi = mid
            else:
                lo = mid
    w, sets = worst(hi)
    return float(sets.sum(axis=1).mean()), w, float(hi)


def phase0_cc_decomposition(logits, labels, n_classes, alpha, cal_idx, eval_idx, *,
                            bin_grid=(2, 10, 50), estimator="empirical",
                            target=None):
    """ADOPTED §5 measurement (Amendment 3): average set size required to reach
    worst-class coverage >= 1-alpha, for each mechanism, with every per-group
    correction estimated OUT OF SAMPLE (on cal, evaluated on eval).

    Out-of-sample estimation is essential: in-sample per-class thresholds win even
    when no class structure exists (+8.77 measured), because per-class flexibility
    absorbs noise. Verified discrimination (abundant data): structure present
    +44.83, structure absent -0.32.

    Returns a dict mechanism -> {avg_set_size, worst_coverage, inflation}, plus
    `gap_vs_global` per mechanism (global_size - mechanism_size; POSITIVE = the
    mechanism is cheaper). §5 passes if `class` has the largest gap with
    non-overlapping CIs.
    """
    target = (1 - alpha) if target is None else target
    S = thr_scores_from_softmax(temperature_softmax(logits, 1.0))
    cal_true = S[cal_idx, labels[cal_idx]]
    cal_lab = labels[cal_idx]
    q_global = group_quantile(cal_true, alpha, estimator)
    Se, le = S[eval_idx], labels[eval_idx]

    out = {}

    def record(name, thresholds):
        sz, w, inf = min_size_at_worst_class_coverage(Se, le, n_classes,
                                                     thresholds, target)
        out[name] = {"avg_set_size": sz, "worst_coverage": w, "inflation": inf}

    record("global", np.full(n_classes, q_global))

    # (1) temperature: pick T on CAL by marginal efficiency, then hold to the same
    # class-conditional requirement on EVAL
    best_T, best_size = 1.0, np.inf
    for T in np.linspace(0.5, 3.0, 26):
        St = thr_scores_from_softmax(temperature_softmax(logits, T))
        qt = group_quantile(St[cal_idx, cal_lab], alpha, estimator)
        s = float((St[cal_idx] <= qt).sum(axis=1).mean())
        if s < best_size:
            best_T, best_size = T, s
    St = thr_scores_from_softmax(temperature_softmax(logits, best_T))
    qt = group_quantile(St[cal_idx, cal_lab], alpha, estimator)
    sz, w, inf = min_size_at_worst_class_coverage(St[eval_idx], le, n_classes,
                                                 np.full(n_classes, qt), target)
    out["temperature"] = {"avg_set_size": sz, "worst_coverage": w,
                          "inflation": inf, "best_T": best_T}

    # (2) energy: per-bin offsets estimated on CAL, applied per EVAL sample
    E = free_energy(logits)
    for b in bin_grid:
        edges = np.quantile(E[cal_idx], np.linspace(0, 1, b + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bin_of = np.clip(np.digitize(E, edges) - 1, 0, b - 1)
        thr_b = np.full(b, q_global)
        for j in range(b):
            m = bin_of[cal_idx] == j
            if m.any():
                qj = group_quantile(cal_true[m], alpha, estimator)
                if np.isfinite(qj):
                    thr_b[j] = qj
        record(f"energy_b{b}", thr_b[bin_of[eval_idx]])   # per-sample thresholds

    # (3) per-class offsets estimated on CAL (OUT OF SAMPLE)
    thr_c = np.full(n_classes, q_global)
    for y in range(n_classes):
        m = cal_lab == y
        if m.any():
            qy = group_quantile(cal_true[m], alpha, estimator)
            if np.isfinite(qy):
                thr_c[y] = qy
    record("class", thr_c)

    g = out["global"]["avg_set_size"]
    for k in out:
        out[k]["gap_vs_global"] = g - out[k]["avg_set_size"]
    return out


def phase0_energy_bin_sweep(logits, score_matrix, labels, alpha, cal_idx, eval_idx,
                            bin_grid=(2, 5, 10, 20, 50, 100), estimator="empirical"):
    """Gap closed by component (2) as a function of its free-parameter count.

    Needed for a fair §5 comparison: component (3) has K free parameters, so the
    energy component must be shown across a comparable range rather than at one
    arbitrary bin count.
    """
    return {b: gap_from_energy_reweighting(logits, score_matrix, labels, alpha,
                                           cal_idx, eval_idx, n_bins=b,
                                           estimator=estimator)
            for b in bin_grid}
