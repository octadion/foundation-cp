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

---

## CIFAR-100 Phase-0 outcome under the adopted metric (2026-08-03) — out of regime

| α | global size | temperature | best energy | **class** |
|---|---|---|---|---|
| 0.05 | 9.10 | +2.07 [−0.18, +4.31] | −41.6 | **−87.9** [−90.4, −85.3] |
| 0.1 | 3.57 | +0.06 [+0.01, +0.11] | −47.0 | **−85.8** [−86.6, −85.1] |

Verdict FAIL at both α. The class mechanism needs ~97 of 100 labels to reach
worst-class coverage ≥ 1−α, versus 3.6–9.1 for a global threshold.

**Diagnosis (measured, not assumed).** The adopted metric behaves correctly where
it is applicable — validated at +40.89 (structure present) / −2.52 (absent) with
≥1000 calibration samples per class. CIFAR-100 sits outside that regime, and the
blow-up is the signature of it: a global threshold needs only a *small* inflation
to lift its worst class (final size 3.57 at α=0.1), which means **class-difficulty
heterogeneity in CIFAR-100 is small relative to the estimation noise at ~50
calibration samples per class**. Per-class thresholds estimated from 50 samples
are noisy, one badly-estimated class forces a large uniform inflation, and that
inflation is then paid by all 100 classes.

This is consistent with CIFAR-100 being balanced and curated — classes are roughly
equally hard, which is precisely why the class-conditional CP literature works on
long-tailed data. It **cannot** distinguish "no class structure" from "structure
not estimable at this n", and it is the fourth independent reason CIFAR-100 cannot
decide Phase 0 (after: α=0.01 infeasible, gate A noise-limited, prevalence
ablation undefined because balanced).

**No further metric redesign.** Two redesigns have already been made; a third,
tuned to make CIFAR-100 pass, would be exactly the behaviour §0.3 forbids. The
metric stands as validated; CIFAR-100 is recorded as out of regime. Phase 0 is
decided on Pl@ntNet. The notebook now prints each mechanism's achieved size and
the inflation it required, so this pathology is legible rather than mysterious.

---

---

## Amendment 4 — the §6.4 metric, RESOLVED (2026-08-03)

Phase 1 ran end to end on CIFAR-100 and reported §6.4 = **+12.66** (sets *growing*
by 12.7 labels, CI [10.77, 14.55]). That is an artefact, and chasing it exposed a
genuine methodological gap. Two bugs were fixed on the way, then a third problem
was found that should NOT be fixed by inventing another metric.

**Fixed bug 1 — δ_y target carried a constant positive bias.** `delta_y_matched_n`
defaulted to `estimator="conformal"`. Matching n removes the *n_y-dependence* of
the bias but not the *level mismatch*: at n_cal=25, α=0.1 the per-class group is
evaluated at level 0.96 while the pooled group sits at 0.9004. Control where the
true δ_y is 0 for every class:

| estimator | mean δ_y | fraction > 0 |
|---|---|---|
| conformal | **+0.1158** | 0.90 |
| empirical | −0.0093 | 0.39 |

On a [0,1] score scale that +0.116 inflates every threshold — which is exactly the
+12.66. **Default changed to `empirical`** for the prediction target; `conformal`
remains for deployment-valid corrections.

**Fixed bug 2 — the comparison drifted in coverage.** Applying δ̂ changes coverage,
so sizes were compared at two different coverage levels. `size_at_matched_coverage`
now absorbs the constant component before comparing.

**UNRESOLVED — which coverage objective?** Three operationalizations, three
structural confounds (all measured, not conjectured):

| operationalization | oracle per-class δ̂ | pure constant δ̂ | verdict |
|---|---|---|---|
| sizes at nominal marginal coverage | — | — | coverage drifts; invalid |
| sizes at MATCHED marginal coverage | +1.58 (worse) | −0.93 (better) | marginal CP is already optimal for marginal coverage, so a class-indexed correction *cannot* win |
| class-conditional, held-out classes only | −16.57 (worse) | **+15.54 (better)** | coverage is constrained only on held-out classes while the SET spans all classes, so "raise exactly the measured classes' thresholds" wins for reasons unrelated to class structure |

A pure constant beating the oracle is decisive evidence that the objective, not the
implementation, is wrong.

### RESOLVED by reading the literature, not by inventing a metric

Bhattacharyya, Ding & Barber (arXiv 2606.28598, "Conformal Prediction with
Macro-Coverage Guarantees"; code at github.com/tiffanyding/macro-guarantees) supply
the two missing pieces:

1. **The optimal set has the form `C(x) = {y : s(x,y) ≤ τ_y}`** — per-class
   thresholds. So the FORM of our correction (τ_y = q̂ + δ_y) was right all along;
   what was wrong was the objective it was scored against.
2. **`macro_cov = class_cov[valid].mean()`** — macro-coverage is the *unweighted*
   mean of per-class coverages (their `conformal.py`). Their code also defines
   **`macro_cov_plus` = "MacroCov restricted to active classes"**, i.e. restricting
   the objective to a subset of classes is a formalized concept, not a hack.

That second point is what fixed confound (3): restrict the whole problem to the
**held-out label space** — held-out columns of the score matrix, held-out samples,
labels remapped — so there is no seen/held-out asymmetry to exploit. Plus allow the
threshold vector to **deflate** as well as inflate, so a constant δ̂ is judged
neutral instead of punished for over-covering.

Implemented as `setsize.setsize_translation_heldout_space`. Controls, all measured:

| δ̂ | gap (positive = smaller sets at the same objective) |
|---|---|
| oracle δ_y | **+0.045** |
| oracle + constant | **+0.046** (constant is a no-op, as required) |
| pure constant | ≈ −0.29 (neutral) |
| shuffled oracle (null) | **−15.39** |
| random noise | −12.54 |

Regression tests: `tests/test_setsize.py::test_sec64_oracle_positive_and_null_negative`
and `::test_sec64_constant_delta_is_a_noop`.

### Objective choice, stated explicitly

`δ_y = q̂_y − q̂_global` is a difference of per-class QUANTILES, so it targets
**class-conditional** coverage — that is the PRIMARY objective. Under
**macro-coverage** even an ORACLE δ_y scores **−0.64**, because per-class quantiles
give every class exactly 1−α whereas the macro optimum deliberately trades over-
and under-coverage across classes (precisely what 2606.28598 characterizes). So a
negative under `objective="macro"` must NOT be read as δ̂ failing; it is reported as
a secondary view only. If macro-coverage is ever adopted as the deployment
objective, the *target* must change too — δ_y would no longer be the right thing to
predict, and their characterized optimum should be used instead.

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
