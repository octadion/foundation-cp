"""§6.4 — translate predicted δ̂_y into prediction-set size.

╔══════════════════════════════════════════════════════════════════════════════╗
║ STATUS 2026-08-03: RESOLVED. Use `setsize_translation_heldout_space`.        ║
║ See Amendment 4 in reports/protocol_amendments.md.                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Getting here took four operationalizations; the first three are kept below for
reference and are documented as confounded. The difficulty was real: comparing set
sizes requires fixing a coverage objective, AND an apples-to-apples label space.

1. `setsize_translation_holdout` — sizes at nominal marginal coverage.
   CONFOUNDED: coverage drifts, so it compares two different coverage levels. An
   uncorrected +0.116 constant in δ̂ made it report sets growing by 12.7 labels.

2. `setsize_translation_holdout_matched` — sizes at MATCHED MARGINAL coverage.
   CONFOUNDED: marginal split-CP is already optimal for marginal coverage, so a
   class-indexed correction provably cannot win. Oracle δ̂ +1.58 (worse), pure
   constant −0.93 (better) — ranking meaningless.

3. Class-conditional coverage, but δ̂ applied only to held-out classes while sets
   still spanned ALL classes. CONFOUNDED: coverage constrained on held-out classes
   while size is paid across every class, so "raise exactly the measured classes'
   thresholds" wins for free. Pure constant +15.54 BEAT oracle −16.57.

4. `setsize_translation_heldout_space` — WORKS. Restrict the whole problem to the
   held-out LABEL SPACE (removing the seen/held-out asymmetry) and compare at a
   matched objective with the threshold vector free to deflate as well as inflate
   (so a constant is a genuine no-op). Controls: oracle +0.045, oracle+constant
   +0.046, pure constant ≈ −0.29 (neutral), shuffled oracle −15.39, random −12.54.

Objective choice matters and is now explicit. δ_y = q̂_y − q̂_global is a difference
of per-class QUANTILES, so it targets CLASS-CONDITIONAL coverage — that is the
primary objective. Under MACRO-coverage (the unweighted mean of per-class
coverages) even an oracle δ_y scores −0.64, because per-class quantiles give every
class exactly 1−α while the macro optimum deliberately trades over- and
under-coverage across classes. That trade-off is what Bhattacharyya, Ding & Barber
(arXiv 2606.28598) characterize; their reference implementation defines
`macro_cov = class_cov[valid].mean()` and also formalizes restricting the objective
to a subset of classes (`macro_cov_plus`), which is what design (4) does.

--- original module notes ---

δ_y is a PROXY; what is actually cared about is set-size reduction, and the
δ_y → set-size relation is not simply monotone across classes (§6.1). So gate B/C
predictability that does NOT translate into smaller held-out sets means the wrong
target was chosen — report it and propose an alternative (§6.4, §6.5).

Correction acts on the true-class score: s'(x,y) = s(x,y) − δ̂_y. Label k is in
the set iff s'(x,k) ≤ q̂, i.e. s(x,k) ≤ q̂ + δ̂_k. So the corrected rule is a
PER-CLASS threshold `q̂ + δ̂_k`. Marginal coverage validity is preserved for any
δ̂ (see tests/test_coverage_validity.py); the question here is efficiency.
"""

from __future__ import annotations

import numpy as np

from pcc.eval.conformal import build_sets
from pcc.eval.metrics import summary


EQUITY_STATS = {
    "worst": lambda c: float(np.nanmin(c)),
    "p05": lambda c: float(np.nanpercentile(c, 5)),
    "p10": lambda c: float(np.nanpercentile(c, 10)),
    "p25": lambda c: float(np.nanpercentile(c, 25)),
    "macro": lambda c: float(np.nanmean(c)),
    "neg_covgap": lambda c: -float(np.nanmean(np.abs(c - np.nanmean(c)))),
}


def avg_set_size_at_shift(score_matrix, thresholds, shift):
    t = np.asarray(thresholds, float) + shift
    return float((np.asarray(score_matrix) <= t[None, :]).sum(axis=1).mean())


def shift_to_size(score_matrix, thresholds, target_size, *, lo=-2.0, hi=2.0,
                  tol=1e-6, max_iter=200):
    """Scalar shift making the average set size equal `target_size`.

    Bisection is valid because average set size is monotone non-decreasing in the
    shift. This is the well-conditioned direction: average set size is smooth and
    monotone in the threshold, whereas the inverse problem (hit a coverage target,
    read the size) is near-vertical — with s = 1 - p on a weak model nearly every
    wrong label sits in [0.98, 1.0], so a +0.012 threshold change moved average set
    size from 13.6 to 55.7. Matching the RESOURCE and reading the BENEFIT is what
    Amendment 8 inverts.
    """
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if avg_set_size_at_shift(score_matrix, thresholds, mid) < target_size:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def equity_at_matched_size(score_matrix, labels, n_classes, thresholds, target_size):
    """Per-class coverage equity of a threshold vector, shifted to `target_size`.

    Returns every statistic in EQUITY_STATS plus the achieved size, so the caller can
    verify the match rather than trust it.
    """
    from pcc.eval.metrics import per_class_coverage

    sh = shift_to_size(score_matrix, thresholds, target_size)
    t = np.asarray(thresholds, float) + sh
    sets = np.asarray(score_matrix) <= t[None, :]
    cov = per_class_coverage(sets, labels, n_classes)
    out = {k: f(cov) for k, f in EQUITY_STATS.items()}
    out["avg_set_size"] = float(sets.sum(axis=1).mean())
    out["shift"] = float(sh)
    return out


def select_shrinkage(score_matrix, labels, n_classes, q_global, delta_hat,
                     target_size, *, lams=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
                     stat="worst"):
    """Choose the shrinkage λ in δ̃ = λ·δ̂ that maximizes `stat` at matched size.

    **MUST be called on the TRAIN label space only.** λ is a free parameter; selecting
    it on the held-out classes would let it manufacture a positive §6.4 result, which
    is precisely the failure mode Amendments 4 and 8 exist to prevent. `setsize_
    translation_shrunk` enforces this by construction.

    WHY SHRINKAGE IS NEEDED AT ALL (measured, Amendment 8): the raw δ̂ (λ=1) HURTS at
    realistic predictor quality, because a worst-class objective is governed by the
    LARGEST error in δ̂, not its variance — and R² controls mean squared error, so the
    requirement tightens as the number of classes grows. Shrinking trades that tail
    away. Optimal λ is far more aggressive than regression attenuation would suggest:

        predictor R² 0.30 -> λ≈0.10 -> Δworst +0.025   (raw λ=1 gives -0.403)
        predictor R² 0.56 -> λ≈0.10 -> Δworst +0.054
        predictor R² 0.90 -> λ≈0.20 -> Δworst +0.126
        oracle            -> λ≈0.70 -> Δworst +0.390
    """
    d = np.asarray(delta_hat, float)
    d = np.where(np.isfinite(d), d, 0.0)
    base = np.full(n_classes, float(q_global))
    best_lam, best_val, curve = 0.0, -np.inf, {}
    for lam in lams:
        e = equity_at_matched_size(score_matrix, labels, n_classes,
                                   base + lam * d, target_size)
        curve[float(lam)] = e[stat]
        if e[stat] > best_val:
            best_lam, best_val = float(lam), e[stat]
    return {"lambda": best_lam, "value": best_val, "curve": curve, "stat": stat}


def setsize_translation_shrunk(score_matrix, labels, alpha, cal_idx, eval_idx,
                               train_classes, heldout_classes, delta_hat, *,
                               stat="worst", lams=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
                               controls=True, q_global=None):
    """§6.4, ADOPTED design (Amendment 8): does the PREDICTED correction improve
    coverage EQUITY on held-out classes, at matched average set size?

    Two changes from design 4, both forced by measurement:

    1. **Inverted.** Design 4 matched coverage and read set size. Its own recorded
       controls show why that cannot work: `oracle +0.045` against
       `shuffled oracle -15.39` — a PERFECT δ̂ bought essentially nothing, so the
       ceiling was ~0 and the entire dynamic range sat on the negative side. Here the
       resource (average set size) is matched and the benefit (coverage equity) is
       read. Oracle headroom becomes **+0.34** on worst-class coverage.
    2. **Shrunk.** δ̃ = λ·δ̂ with λ chosen on the TRAIN label space only. Raw δ̂ hurts
       at realistic predictor quality; see `select_shrinkage` for the numbers.

    `stat="macro"` is deliberately NOT the default: macro coverage has **no headroom
    even for an oracle** (measured -0.0016), because a uniform threshold is already
    near-optimal for an unweighted mean — that is Jensen, not a property of δ_y. Macro
    remains the right statistic for the TAIL report (`pcc.eval.tail`), where the
    question is descriptive rather than comparative.

    λ is selected on 𝒴_train and applied unchanged to 𝒴_held-out. Selecting it on the
    held-out classes would let a free parameter manufacture a positive result.
    """
    from pcc.eval.conformal import restrict_to_classes

    S = np.asarray(score_matrix)
    labels = np.asarray(labels, int)
    d_all = np.asarray(delta_hat, float)
    d_all = np.where(np.isfinite(d_all), d_all, 0.0)

    # Marginal threshold from the pooled calibration split. Held-out classes have no
    # labelled calibration data of their own -- that is the premise of Sec 6.4 -- but
    # the marginal quantile is available regardless. `cal_idx=None` with an explicit
    # `q_global` covers the common case where cal and eval come from separate arrays.
    if cal_idx is None:
        if q_global is None:
            raise ValueError("pass cal_idx or q_global")
    else:
        q_global = float(np.quantile(S[cal_idx, labels[cal_idx]], 1 - alpha))
    q_global = float(q_global)
    if eval_idx is None:
        eval_idx = np.arange(len(labels))

    def _space(classes, idx):
        Ss, ys, Ks, ids = restrict_to_classes(S[idx], labels[idx], classes)
        return Ss, ys, Ks, ids

    # --- step 1: select lambda on the TRAIN label space -------------------------
    S_tr, y_tr, K_tr, ids_tr = _space(train_classes, eval_idx)
    base_tr = np.full(K_tr, q_global)
    target_tr = avg_set_size_at_shift(S_tr, base_tr, 0.0)
    sel = select_shrinkage(S_tr, y_tr, K_tr, q_global, d_all[ids_tr],
                           target_tr, lams=lams, stat=stat)
    lam = sel["lambda"]

    # --- step 2: apply it, unchanged, on the HELD-OUT label space ---------------
    S_ho, y_ho, K_ho, ids_ho = _space(heldout_classes, eval_idx)
    base_ho = np.full(K_ho, q_global)
    target_ho = avg_set_size_at_shift(S_ho, base_ho, 0.0)
    e_base = equity_at_matched_size(S_ho, y_ho, K_ho, base_ho, target_ho)
    e_corr = equity_at_matched_size(S_ho, y_ho, K_ho,
                                    base_ho + lam * d_all[ids_ho], target_ho)

    out = {"stat": stat, "lambda_selected_on_train": lam,
           "lambda_curve_train": sel["curve"],
           "n_classes_heldout": int(K_ho), "n_classes_train": int(K_tr),
           "target_avg_set_size": float(target_ho),
           "uncorrected": e_base, "corrected": e_corr,
           "delta": {k: e_corr[k] - e_base[k] for k in EQUITY_STATS},
           "size_matched": bool(abs(e_corr["avg_set_size"]
                                    - e_base["avg_set_size"]) < 1e-2),
           "pass": bool(e_corr[stat] - e_base[stat] > 0)}

    if controls:
        rng = np.random.default_rng(0)
        ctl = {}
        # A CEILING, not a method: the oracle uses held-out EVAL labels, so it is not
        # achievable -- it exists only to show the metric has headroom at all.
        s_true_ho = S_ho[np.arange(len(y_ho)), y_ho]
        oracle = np.zeros(K_ho)
        for k in range(K_ho):
            m = y_ho == k
            if m.sum() >= 5:
                oracle[k] = float(np.quantile(s_true_ho[m], 1 - alpha)) - q_global
        for name, vec in (("oracle_ceiling", oracle),
                          ("pure_constant", np.full(K_ho, 0.05)),
                          ("shuffled_delta", rng.permutation(d_all[ids_ho])),
                          ("raw_delta_lambda1", d_all[ids_ho])):
            e = equity_at_matched_size(S_ho, y_ho, K_ho, base_ho + vec, target_ho)
            ctl[name] = e[stat] - e_base[stat]
        out["controls"] = ctl
    return out


def corrected_thresholds(q_global: float, delta_hat: np.ndarray) -> np.ndarray:
    """Per-class threshold q̂ + δ̂_k. NaN δ̂ (no prediction) falls back to q̂."""
    d = np.array(delta_hat, float)
    d[~np.isfinite(d)] = 0.0
    return q_global + d


def compare_setsize(score_matrix, labels, n_classes, alpha, q_global, delta_hat,
                    *, group_of_class=None):
    """Uncorrected (global q̂) vs corrected (q̂ + δ̂_k) prediction sets.

    Returns the full §9 metric bundle for BOTH, so any set-size gain is shown
    together with its coverage cost (§9). `group_of_class` enables the
    seen/held-out and head/tail breakdowns that §6.4 requires (never merge them).
    """
    sets_unc = build_sets(score_matrix, q_global)
    sets_cor = build_sets(score_matrix, corrected_thresholds(q_global, delta_hat))
    m_unc = summary(sets_unc, labels, n_classes, alpha, group_of_class=group_of_class)
    m_cor = summary(sets_cor, labels, n_classes, alpha, group_of_class=group_of_class)
    return {
        "uncorrected": m_unc,
        "corrected": m_cor,
        "avg_set_size_delta": m_cor["avg_set_size"] - m_unc["avg_set_size"],
        "marginal_coverage_delta": m_cor["marginal_coverage"] - m_unc["marginal_coverage"],
    }


def macro_coverage(sets, labels, n_classes):
    """Macro-coverage: the UNWEIGHTED mean of per-class coverages.

    Matches the reference implementation of Bhattacharyya, Ding & Barber
    (arXiv 2606.28598), `conformal.py`: `macro_cov = class_cov[valid].mean()`.
    Marginal coverage weights classes by frequency; macro-coverage does not.
    """
    from pcc.eval.metrics import per_class_coverage
    cov = per_class_coverage(sets, labels, n_classes)
    return float(np.nanmean(cov))


def setsize_translation_heldout_space(score_matrix, labels, alpha, q_global,
                                      delta_hat, held_out_classes, *,
                                      objective="class_conditional"):
    """§6.4 — RESOLVED design (Amendment 4, reports/protocol_amendments.md).

    Does a PREDICTED δ̂ buy efficiency on held-out classes? Two design choices make
    the question answerable; both were forced by measured confounds.

    1. **Restrict the whole problem to the held-out LABEL SPACE.** Earlier versions
       applied δ̂ only to held-out classes while prediction sets still spanned all
       classes. Coverage was then constrained on held-out classes but size was paid
       across every class, so "raise exactly the measured classes' thresholds" won
       for reasons unrelated to class structure — a pure constant beat the oracle
       (+15.54 vs −16.57). Restricting to held-out columns removes that asymmetry.
       The reference implementation formalizes the same idea as `macro_cov_plus`
       ("MacroCov restricted to active classes").

    2. **Compare at a matched coverage objective, allowing the threshold vector to
       DEFLATE as well as inflate.** Without deflation a δ̂ carrying a positive
       constant is punished for over-covering instead of judged neutral.

    `objective`:
      - "class_conditional" (default): worst-class coverage ≥ 1−α. This MATCHES the
        target: δ_y = q̂_y − q̂_global is a difference of per-class QUANTILES, i.e. it
        aims at per-class coverage.
      - "macro": macro-coverage ≥ 1−α. Reported as a secondary view. NOTE δ_y is
        *not* the optimizer of this objective — per-class quantiles give every class
        exactly 1−α, whereas the macro optimum trades over- and under-coverage
        across classes (that is what arXiv 2606.28598 characterizes). Measured:
        even an ORACLE δ_y scores −0.64 under "macro", so do not read a negative
        here as a failure of δ̂.

    Validated controls (`objective="class_conditional"`): oracle +0.045,
    oracle+constant +0.046 (constant is a no-op, as it must be), pure constant
    ≈ −0.29 (neutral), shuffled oracle −15.39, random −12.54.

    Returns sizes for the uncorrected and corrected arms and `gap` = uncorrected −
    corrected (POSITIVE = δ̂ achieves the same objective with smaller sets).
    """
    from pcc.eval.decomposition import min_size_at_worst_class_coverage

    held = np.asarray(sorted(set(int(c) for c in held_out_classes)), dtype=int)
    Kh = len(held)
    if Kh == 0:
        raise ValueError("held_out_classes is empty")
    mask = np.isin(labels, held)
    if not mask.any():
        raise ValueError("no eval points fall in held_out_classes")

    S = score_matrix[mask][:, held]                 # held-out label space only
    remap = {int(c): i for i, c in enumerate(held)}
    lab = np.array([remap[int(y)] for y in labels[mask]], dtype=int)

    d = np.array(delta_hat, float)[held]
    d[~np.isfinite(d)] = 0.0
    target = 1 - alpha

    if objective == "class_conditional":
        unc, unc_w, _ = min_size_at_worst_class_coverage(
            S, lab, Kh, np.full(Kh, q_global), target, allow_deflate=True)
        cor, cor_w, _ = min_size_at_worst_class_coverage(
            S, lab, Kh, q_global + d, target, allow_deflate=True)
    elif objective == "macro":
        def _size_at_macro(base):
            lo, hi = -1.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                m = macro_coverage(build_sets(S, base + mid), lab, Kh)
                if m < target:
                    lo = mid
                else:
                    hi = mid
            sets = build_sets(S, base + hi)
            return float(sets.sum(axis=1).mean()), macro_coverage(sets, lab, Kh)
        unc, unc_w = _size_at_macro(np.full(Kh, q_global))
        cor, cor_w = _size_at_macro(q_global + d)
    else:
        raise ValueError(f"unknown objective {objective!r}")

    return {"objective": objective, "n_heldout_classes": int(Kh),
            "n_heldout_points": int(mask.sum()),
            "uncorrected_size": unc, "uncorrected_achieved": unc_w,
            "corrected_size": cor, "corrected_achieved": cor_w,
            "gap": unc - cor, "reduces_size": bool(cor < unc)}


def _coverage_of(score_matrix, labels, thresholds):
    sets = build_sets(score_matrix, thresholds)
    return float(sets[np.arange(len(labels)), labels].mean()), sets


def size_at_matched_coverage(score_matrix, labels, base_thresholds,
                            target_coverage, *, tol=1e-4, max_shift=1.0):
    """Average set size after shifting `base_thresholds` by a single global
    constant so that empirical coverage equals `target_coverage`.

    WHY THIS EXISTS (measured): a predicted δ̂ generally carries a CONSTANT
    component, and a constant offset is a GLOBAL recalibration — not class-level
    information. Comparing set sizes without absorbing it compares two different
    coverage levels, which is meaningless. On CIFAR-100 an uncorrected +0.116
    offset in δ̂ (see delta_y_matched_n's docstring) made §6.4 report sets GROWING
    by 12.7 labels — an artefact of coverage drift, not a prediction failure.

    Same principle as Amendment 3: efficiency claims are only meaningful at
    matched coverage.
    """
    lo, hi = -max_shift, max_shift
    for _ in range(60):
        mid = (lo + hi) / 2
        cov, _ = _coverage_of(score_matrix, labels, np.asarray(base_thresholds, float) + mid)
        if cov < target_coverage:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    cov, sets = _coverage_of(score_matrix, labels,
                             np.asarray(base_thresholds, float) + hi)
    return float(sets.sum(axis=1).mean()), cov, float(hi)


def setsize_translation_holdout_matched(score_matrix, labels, n_classes, alpha,
                                       q_global, delta_hat, held_out_classes):
    """§6.4 at matched MARGINAL coverage. CONFOUNDED - see the module header;
    do NOT report a §6.4 verdict from this. Kept because absorbing the constant
    component of δ̂ is a necessary ingredient of any correct version.

    Restricts to points whose true label is a held-out class, then compares:
      - uncorrected: a single global threshold, shifted to hit target coverage
      - corrected:   q̂_global + δ̂_y, shifted by a global constant to hit the SAME coverage

    It isolates the class-specific VARIATION in δ̂ (good), but matching MARGINAL
    coverage means a class-indexed correction provably cannot win (bad). Returns avg sizes for both, their difference (negative =
    corrected is smaller = δ̂ helps), and the achieved coverages.
    """
    held = set(int(c) for c in held_out_classes)
    mask = np.array([int(y) in held for y in labels])
    if not mask.any():
        raise ValueError("no eval points fall in held_out_classes")
    S, lab = score_matrix[mask], labels[mask]

    target = 1 - alpha
    unc_size, unc_cov, unc_shift = size_at_matched_coverage(
        S, lab, np.full(n_classes, q_global), target)
    d = np.array(delta_hat, float)
    d[~np.isfinite(d)] = 0.0
    cor_size, cor_cov, cor_shift = size_at_matched_coverage(
        S, lab, q_global + d, target)
    return {"n_holdout_points": int(mask.sum()),
            "uncorrected_size": unc_size, "uncorrected_coverage": unc_cov,
            "corrected_size": cor_size, "corrected_coverage": cor_cov,
            "size_delta": cor_size - unc_size,
            "shift_uncorrected": unc_shift, "shift_corrected": cor_shift,
            "reduces_size": bool(cor_size < unc_size)}


def setsize_translation_holdout(score_matrix, labels, n_classes, alpha, q_global,
                                delta_hat, held_out_classes):
    """§6.4 gate: does δ̂_y reduce set size ON HELD-OUT CLASSES?

    Restricts the size/coverage comparison to points whose true label is a
    held-out class. Returns corrected vs uncorrected avg set size + coverage on
    that subset, and a `reduces_size` flag (corrected < uncorrected).
    """
    held = set(int(c) for c in held_out_classes)
    mask = np.array([int(y) in held for y in labels])
    if not mask.any():
        raise ValueError("no eval points fall in held_out_classes")
    res = compare_setsize(score_matrix[mask], labels[mask], n_classes, alpha,
                          q_global, delta_hat)
    res["n_holdout_points"] = int(mask.sum())
    res["reduces_size"] = bool(res["avg_set_size_delta"] < 0)
    return res
