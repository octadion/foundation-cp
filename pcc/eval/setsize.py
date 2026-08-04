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
