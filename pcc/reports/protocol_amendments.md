# Protocol amendments — estimator artefacts in δ_y (2026-08-03)

Two amendments, both forced by **measured** artefacts, not by preference. Each was
verified numerically before being adopted. Both are recorded here because they
change how §5 and §6 are computed, and because a reviewer will ask.

Trigger: the first real Phase-0 run on CIFAR-100 returned `FAIL` with **every**
component negative or ~zero (temperature +0.014, all energy bins negative,
per-class offset −5.615). A result where every method makes prediction sets
*bigger* is a signal of a broken measurement, not a finding.

---

## Amendment 1 — §5 decomposition must use LEVEL-MATCHED quantiles

### The artefact

The split-conformal quantile uses level `ceil((n+1)(1−α))/n`, which **depends on
n**. So a per-class group with n=50 targets that class's ~98th percentile, while
the pooled global group (n=5000) targets the ~95th. The quantity
`q̂_y − q̂_global` therefore mixes real class structure with a pure estimator
offset, and the offset grows as n_y shrinks. On a high-accuracy model the score
distribution is heavy-tailed near zero, so p98 ≫ p95 and the inflation is large.

This explains the entire observed pattern coherently:

| component | per-group n | inflation | observed gap (α=0.05) |
|---|---|---|---|
| temperature | 1 global quantile only | none | +0.050 |
| energy, 2 bins | ~2500 | small | −0.675 |
| energy, 50 bins | ~100 | larger | −2.032 |
| energy, 100 bins | ~50 | largest | −6.712 |
| per-class offset | ~50 | large | −5.615 |

### Verification

Synthetic at realistic accuracy (0.752, 100 samples/class — matching CIFAR-100):

| estimator for the per-class offset | gap closed |
|---|---|
| `conformal` (level `ceil((n+1)(1−α))/n`) | **−5.98** (reproduces the real −5.615) |
| `empirical` (plain 1−α quantile) | **+1.98** |
| abundant-data oracle (true per-class 1−α percentile) | **+1.84** |

`empirical` ≈ oracle, so the problem was **not** finite-sample noise — it was
purely the level mismatch.

### Amendment

For the §5 **structure measurement**, all components use
`decomposition.group_quantile(..., estimator="empirical")` — the same target
percentile for every group regardless of n. §5 asks "how much gap could
class-level structure close, fit on abundant data"; it is not a deployment.

**Deployment validity is unaffected and separately guarded.** Deployed sets still
use the finite-sample `conformal` quantile, and `tests/test_coverage_validity.py`
(§8.7) enforces marginal coverage for any δ̂. `estimator="conformal"` remains
available and Phase-0 reports **both**, labelled.

Regression tests: `tests/test_phase0_decomposition.py::test_class_offset_positive_at_realistic_accuracy_and_sample_count`
and `::test_group_quantile_levels_differ_with_n`. The original Phase-0 tests used
400 samples/class with a weak model — which is exactly why they missed this.

---

## Amendment 2 — δ_y must be estimated at MATCHED n per class (gate C protection)

### The artefact — this one can kill the project for the wrong reason

Because the estimator bias depends on `n_y`, and `n_y ∝ prevalence`, **δ_y
correlates mechanically with prevalence even when no class structure exists.**

Control: 200 classes, long-tailed `n_y` (10–2000), **every class given an
identical score distribution**, so true δ_y = 0 for all. Any correlation is
artefact:

| estimator | corr(δ_y, log n_y) | mean δ_y (should be 0) |
|---|---|---|
| `conformal` | **−0.287** | +0.053 |
| `empirical` | **+0.227** | −0.020 |

Both are biased, in **opposite directions**. Also: 48/200 classes had an
**infinite** conformal quantile at α=0.05 (n_y too small).

Why this is dangerous: §6.3 **gate C** asks whether geometry beats log-prevalence
alone at predicting δ_y, and §6.5 says that if log-prevalence explains most of
δ_y then "Ding et al. dan TACP sudah punya semuanya dan tidak ada paper di sini"
— i.e. the project stops. A **spurious** prevalence↔δ_y link would therefore
terminate the project for an estimator artefact. It also compounds the
prevalence-linked descriptor-noise risk already recorded in
`descriptor_stability_findings.md`.

### Verification of the fix

Same control, δ_y estimated from a **common n_cal per class** (subsample every
class to the same count, so the bias is identical everywhere):

| estimation | corr(δ_y, log n_y) | classes retained |
|---|---|---|
| all available samples | **−0.385** | 200/200 |
| matched n_cal = 20 | **+0.016** | 150/200 |
| matched n_cal = 50 | +0.154 | 76/200 |

Matched-n at 20 removes the artefact (−0.385 → +0.016). The residual at n_cal=50
is a small-sample/selection effect: only 76 head classes survive, so the
correlation is estimated over a restricted range with fewer points.

### Amendment (to be applied when Phase 1 is built)

1. **PRIMARY:** estimate δ_y with a **matched n_cal per class**. Report n_cal and
   the number of classes retained. This is the analysis gate C is judged on.
2. **MANDATORY null control:** a permutation test that destroys true class
   structure while preserving `n_y` (shuffle labels within the calibration pool),
   then measures how much log-prevalence "predicts" δ_y under that null. Gate C
   conclusions must exceed this null, not merely exceed zero.
3. **SECONDARY:** the all-samples estimate, as a sensitivity analysis, always
   reported alongside — never instead.
4. Report the **fraction of classes with undefined (infinite) δ_y** per α; it
   interacts with `fallback_policy.md`.

### Unavoidable tension, stated explicitly

Matched-n discards data: head classes are subsampled and classes below n_cal are
dropped. That trades statistical power for freedom from the prevalence artefact —
and dropping tail classes is itself a prevalence-linked selection, which must be
reported (which classes, what fraction of the tail). On a long-tailed dataset the
two cannot both be maximised. Choosing n_cal is therefore a **human decision** on
Pl@ntNet, made before gate C runs and recorded here.

---

---

## Amendment 3 — ADOPTED 2026-08-03 (human-approved): the §5 gap metric was ill-posed

**Status: APPROVED and implemented.** The human reviewed the evidence below and chose to adopt
the corrected metric. This is written as evidence, not as a change.
§0.3 forbids changing a metric after seeing a negative result, and this proposal
follows a FAIL — so it must be judged explicitly rather than adopted quietly.

### What the second Phase-0 run showed

After Amendment 1 the per-class offset moved from −5.615 to **−0.066** (α=0.05)
and **−0.097** (α=0.1) — the level artefact was removed, but the gap is still
slightly negative, and **every** mechanism is ≤ ~0 (temperature +0.050/+0.014,
energy −0.38…−1.12). Verdict: FAIL at both α, on both estimators.

### Why the metric, not the hypothesis, is the likely problem

The metric as implemented is *avg set size at nominal marginal coverage*. But
marginal split-CP is already the most efficient way to obtain **marginal**
coverage; adding a class-conditional constraint can only spend size. So a
class-indexed mechanism **cannot** win on this metric regardless of whether class
structure exists. A test that cannot return a positive is not a test.

Verified — abundant data, strong real class structure, out-of-sample per-class
thresholds, avg set size at matched **class-conditional** coverage
(worst-class ≥ 1−α):

| cal samples/class | structure PRESENT | structure ABSENT |
|---|---|---|
| 50 (CIFAR-100's regime) | +3.17 | −60.41 |
| 200 | +2.79 | −22.03 |
| 1000 | **+40.89** | −2.55 |
| 4000 | **+44.83** | −0.32 |

This metric discriminates cleanly **and has a working negative control** (the gap
goes negative without structure, converging to 0 as data grows). It is also the
objective `AGENTS.md` §9 actually cares about (CovGap, worst-class coverage).

Two rejected alternatives, for the record:
- *avg size at matched class-conditional coverage, in-sample*: wins by +8.77 even
  with NO class structure (per-class flexibility absorbs noise). Biased.
- *threshold-shuffling null*: too destructive (null 97.77 / 69.71) — it models
  "actively wrong structure", not "no structure". Invalid as a null.

Only **out-of-sample** per-class estimation separates real structure from noise
absorption.

### The other conclusion this forces

At 50 calibration samples/class — CIFAR-100's regime — even strong real structure
yields only +3.17 versus +40.89 at 1000/class. **CIFAR-100 structurally cannot
answer Phase 0.** This is consistent with the two limits already recorded
(α=0.01 infeasible; gate A noise-limited at 25/class). It also entangles Phase 0
with Phase 1: distinguishing "no class structure" from "structure not estimable at
this n" *is* what gate A (split-half reliability of δ_y) measures. So §5 and §6.2
cannot be interpreted independently, and §5 should be run on classes that actually
have abundant calibration data — which §5 itself demands ("di-fit pada data
berlimpah, bukan pada budget realistis").

### The decision required

1. **Adopt the corrected metric** (avg set size at matched class-conditional
   coverage, out-of-sample per-class estimation), re-run Phase 0 on Pl@ntNet head
   classes where data is abundant; OR
2. **Keep the original metric** and record Phase 0 as FAIL, i.e. stop the project
   per §5.

**Decision: (1) adopted.** Implemented as
`decomposition.phase0_cc_decomposition` / `min_size_at_worst_class_coverage`.
Guarded by `tests/test_phase0_decomposition.py`, including a negative control and a
test asserting the metric CANNOT resolve at CIFAR-100's ~50 cal samples/class — so
if that ever starts passing, the "debug only" justification gets re-examined.

---

## Status

- Amendment 1: **implemented + tested** (`pcc/eval/decomposition.py:group_quantile`).
- Amendment 2: **implemented + tested** (`targets/delta.py:delta_y_matched_n`,
  `:prevalence_null`), wired into `notebooks/04_phase1_gate.ipynb`. `n_cal` still
  needs a **human decision for Pl@ntNet** (the head/tail trade-off).
- Amendment 3: **adopted + implemented + tested**
  (`decomposition.phase0_cc_decomposition`), wired into `notebooks/03`.
- `δ_y` as defined in §6.1 (conformal quantiles) is retained as the **deployment**
  target definition; these amendments concern how it is **measured** for the
  structure and predictability questions.
