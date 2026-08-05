"""Phase 0 (§5) — where does the threshold structure live, on REAL logits?

Three candidate locations for the structure, all fit on CAL, all scored on EVAL:
  (1) a single global temperature      — global miscalibration,
  (2) per-sample energy bins           — per-sample difficulty (free energy),
  (3) a per-class offset               — the class level (the §5 hypothesis).

**The verdict comes from `phase0_explain_class_level` (Amendment 6).** It asks which
mechanism EXPLAINS the class-level conformal quantile q*_y, via out-of-sample R²
across classes. Pass requires all three: class-level structure exists above sampling
noise, it survives EVERY global temperature, and `class` beats every energy rival with
non-overlapping CIs.

`phase0_cc_decomposition` (the earlier set-size criterion, Amendment 3) is retained
ONLY as a deployment-cost report and **must not be used to rank mechanisms** — it
charges a mechanism for its threshold variance regardless of source, and named
`temperature` the winner in 4 of 4 planted worlds. See that function's docstring and
reports/protocol_amendments.md#amendment-6.

Why this module exists: the motivating evidence came from synthetic injection-and-
recovery, which is circular. It must replicate on real logits. For the same reason the
measurement itself is validated against planted worlds before any verdict is read —
pcc/tests/test_phase0_explain.py.
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
    """SECONDARY deployment-cost report. **This function CANNOT answer §5** and its
    ranking of mechanisms must never be used as the Phase-0 verdict (Amendment 6).

    It reports the average set size each mechanism needs to reach worst-class coverage
    >= 1-alpha, with every per-group correction estimated OUT OF SAMPLE. As a statement
    of "what does class-conditional coverage cost at this calibration budget" it is
    meaningful. As a comparison BETWEEN mechanisms it is invalid, because reaching the
    target by UNIFORM inflation charges a mechanism for its threshold VARIANCE
    regardless of whether that variance is real structure or estimation noise:

      - the classes still short after class-adaptation are those with the LOWEST
        thresholds, so a global additive repair over-inflates the entire vector;
      - with s = 1 - p and a weak model nearly every wrong label sits in [0.98, 1.0],
        so set size is near-vertical in the threshold (+0.0117 inflation took avg set
        size from 13.6 to 55.7);
      - consequently nested capacity degrades monotonically (energy_b2 -47.2,
        b10 -67.4, b50 -74.7 on real Pl@ntNet logits), and on synthetic worlds with a
        planted mechanism it named `temperature` the winner in 4 of 4 worlds --
        including the world whose only structure was per-class difficulty.

    Use `phase0_explain_class_level` for the §5 verdict.

    Returns a dict mechanism -> {avg_set_size, worst_coverage, inflation}, plus
    `gap_vs_global` per mechanism (global_size - mechanism_size).
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


def phase0_explain_class_level(logits, labels, n_classes, alpha, cal_idx, eval_idx, *,
                               bin_grid=(2, 10, 50), min_eval=20,
                               estimator="empirical", Ts=None, T_scan=None,
                               reliability_splits=30, T_scan_splits=10):
    """ADOPTED §5 measurement (Amendment 6): which mechanism EXPLAINS the class-level
    quantile? Replaces the set-size criterion, which was shown to be unable to answer
    §5 at all.

    WHY THE SET-SIZE CRITERION WAS ABANDONED (all three legs verified 2026-08-05):

      1. Nested capacity degrades monotonically on real Pl@ntNet logits:
         energy_b2 -47.2, energy_b10 -67.4, energy_b50 -74.7. b50 can represent
         everything b2 can, so a metric where extra capacity always loses is not
         measuring where structure lives.
      2. On synthetic worlds with a PLANTED mechanism it named `temperature` the
         winner in 4 of 4 worlds -- including the world where per-class difficulty
         was the only structure present. Zero discriminating power.
      3. Mechanically: reaching worst-class coverage by UNIFORM inflation punishes
         threshold VARIANCE regardless of its source. The classes still short after
         class-adaptation are the ones with the LOWEST thresholds, so repairing them
         with a global constant over-inflates the whole vector. Measured: a +0.0117
         inflation took avg set size from 13.6 to 55.7, because with s = 1 - p and a
         weak model nearly every wrong label sits in [0.98, 1.0].

    Set size confounds §5 with score saturation, the repair operator, and the
    conditional-vs-marginal efficiency tradeoff -- none of which §5 asks about.
    `phase0_cc_decomposition` is retained as a SECONDARY deployment-cost report.

    WHAT THIS MEASURES INSTEAD. Target: q*_y, the (1-alpha) quantile of class y's
    true-label scores on EVAL. Every mechanism is fit on CAL and predicts q*_y:

        global      -> a constant                (no class-level variance, by design)
        energy_bK   -> class-mean of the per-sample thresholds it would assign
        class       -> the per-class q_hat fit on CAL

    R2 is taken across classes, out of sample. A mechanism whose extra parameters are
    noise earns NEGATIVE R2 (measured: energy_b50 -> -33.3 in the energy world), so
    capacity is punished rather than rewarded. The rival is NOT crippled: when class
    difficulty is real it also shows up in per-sample confidence, so energy_b10
    reached +0.594 against class's +0.896 -- a genuine competitor.

    Temperature is judged by the question it can actually answer: does the best global
    temperature REMOVE the class-level structure? -- via split-half reliability of q̂_y
    on temperature-rescaled scores.

    Validated 4/4 on planted worlds {class, global, energy, none}.

    Returns r2 per mechanism, sd of the target (the §5 discriminant: 0.0618 when class
    structure was planted vs 0.0017 when only global miscalibration was),
    reliability on the identity and best-temperature scales, and `best_T_at_boundary`
    so a temperature search pinned at the grid edge is never read as a fitted value.
    """
    from pcc.targets.delta import class_quantile_reliability

    labels = np.asarray(labels)
    S = thr_scores_from_softmax(temperature_softmax(logits, 1.0))
    cal_true, cal_lab = S[cal_idx, labels[cal_idx]], labels[cal_idx]
    ev_true, ev_lab = S[eval_idx, labels[eval_idx]], labels[eval_idx]
    E = free_energy(logits)
    q_global = group_quantile(cal_true, alpha, estimator)

    scored, target = [], []
    for y in range(n_classes):
        m = ev_lab == y
        if m.sum() >= min_eval:
            scored.append(y)
            target.append(group_quantile(ev_true[m], alpha, estimator))
    scored = np.asarray(scored, int)
    target = np.asarray(target, float)
    if len(scored) < 10:
        raise ValueError(f"only {len(scored)} classes have >= {min_eval} eval samples; "
                         "cannot measure class-level explanation")

    preds = {"global": np.full(len(scored), q_global)}
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
        ev_bins = bin_of[eval_idx]
        preds[f"energy_b{b}"] = np.array(
            [thr_b[ev_bins[ev_lab == y]].mean() for y in scored])

    q_c = np.full(n_classes, q_global)
    for y in range(n_classes):
        m = cal_lab == y
        if m.any():
            v = group_quantile(cal_true[m], alpha, estimator)
            if np.isfinite(v):
                q_c[y] = v
    preds["class"] = q_c[scored]

    r2 = {k: _r2_across_classes(target, v) for k, v in preds.items()}

    Ts = np.linspace(0.25, 4.0, 31) if Ts is None else np.asarray(Ts, float)
    best_T, best_sz = 1.0, np.inf
    for T in Ts:
        St = thr_scores_from_softmax(temperature_softmax(logits, T))
        qt = group_quantile(St[cal_idx, cal_lab], alpha, estimator)
        sz = float((St[cal_idx] <= qt).sum(axis=1).mean())
        if sz < best_sz:
            best_T, best_sz = T, sz
    rel_id = class_quantile_reliability(cal_true, cal_lab, n_classes, alpha,
                                        n_splits=reliability_splits,
                                        estimator=estimator)

    # §5's alternative hypothesis is "this is only global miscalibration". Refuting it
    # properly requires that NO global temperature removes the class-level structure --
    # not merely that the efficiency-optimal one does not. So scan a coarse grid and
    # take the WORST (minimum) reliability; that is the number the verdict uses. Scanning
    # matters because the efficiency-optimal T is frequently pinned at a grid edge, where
    # it is not a fitted value at all.
    T_scan = np.geomspace(0.2, 5.0, 7) if T_scan is None else np.asarray(T_scan, float)
    rel_by_T = {}
    for T in T_scan:
        St = thr_scores_from_softmax(temperature_softmax(logits, T))
        # fewer split-half reps than the identity estimate: the scan only needs to
        # locate the WORST temperature, and reliability is nearly flat in T (measured
        # 0.973-0.985 across T in [0.2, 5.0]), so precision here is not the bottleneck.
        rel_by_T[float(T)] = class_quantile_reliability(
            St[cal_idx, cal_lab], cal_lab, n_classes, alpha,
            n_splits=T_scan_splits, estimator=estimator)["reliability_mean"]
    finite = {k: v for k, v in rel_by_T.items() if np.isfinite(v)}
    T_at_min = min(finite, key=finite.get) if finite else float("nan")
    rel_min_T = finite[T_at_min] if finite else float("nan")
    St = thr_scores_from_softmax(temperature_softmax(logits, best_T))
    rel_T = class_quantile_reliability(St[cal_idx, cal_lab], cal_lab, n_classes, alpha,
                                       n_splits=reliability_splits,
                                       estimator=estimator)

    rivals = {k: v for k, v in r2.items() if k not in ("class", "global")}
    return {"r2": r2, "n_classes_scored": int(len(scored)),
            "target_sd": float(target.std()), "q_global": float(q_global),
            "best_T": float(best_T),
            "best_T_at_boundary": bool(best_T in (float(Ts[0]), float(Ts[-1]))),
            "reliability_identity": rel_id["reliability_mean"],
            "reliability_after_T": rel_T["reliability_mean"],
            "reliability_min_over_T": rel_min_T,
            "T_at_min_reliability": float(T_at_min),
            "reliability_by_T": rel_by_T,
            "class_beats_rivals": bool(np.isfinite(r2["class"]) and rivals and
                                       r2["class"] > max(rivals.values())),
            "best_rival": (max(rivals, key=rivals.get) if rivals else None)}


def _r2_across_classes(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    if ss == 0:
        return np.nan
    return float(1 - np.sum((y - p) ** 2) / ss)


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
