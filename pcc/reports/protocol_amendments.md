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

---

## EFFECT OF THE AMENDMENTS — gate B flipped FAIL -> PASS (2026-08-04)

Phase 1 was re-run on CIFAR-100 after Amendment 2's default changed from
`conformal` to `empirical`. **Nothing else about gate B changed** — same
descriptors, same splits, same ridge, same seed.

| run | δ_y estimator | held-out R² (PRIMARY `stable` set) | gate B |
|---|---|---|---|
| before | conformal (level-mismatched, +0.116 bias) | +0.025, CI [−0.376, +0.301] | **FAIL** |
| after | empirical (level-matched) | **+0.309, CI [+0.055, +0.484]** | **PASS** |

Normalized by the target ceiling: **+0.484, CI [+0.086, +0.758]**.

So the estimator artefact was **masking a real geometry → δ_y relationship**. This is
the fourth time an n-dependent quantile-level artefact distorted a result in this
repo (Phase-0 components, δ_y↔prevalence correlation, §6.4's +12.66, and now gate
B's null result). The pattern is worth naming: **whenever a per-group conformal
quantile is compared against a pooled one, check the level before believing the
number.**

**The pre-registered feature policy validated itself.** `stable` (9 features,
stability ≥0.90) PASSES; `full` (15 features) gives +0.226, CI [−0.103, +0.421] and
FAILS. The unstable features add noise exactly as
`descriptor_stability_findings.md` predicted — and because `stable` was named
PRIMARY *before* any of this was run, that is not selection after the fact.

### Gate C still fails, and CIFAR-100 cannot fix it

`full` beats `distance_only` by +0.147 but CI [−0.104, +0.278] includes 0 —
underpowered at 50 held-out classes. The prevalence ablation is **undefined**
(balanced dataset). Gate C is decided on Pl@ntNet.

### §6.4 beats its null but stays negative — and §9 shows why

| objective | observed | shuffled null | beats null |
|---|---|---|---|
| class_conditional (PRIMARY) | −37.56 [−40.55, −34.57] | −45.88 [−46.94, −44.83] | **yes** (+8.3) |
| macro (secondary) | −6.45 [−7.37, −5.54] | −12.49 [−13.31, −11.66] | **yes** (+6.0) |

So δ̂ carries genuine class-specific information (it decisively beats a null with
identical marginals), but is too weak to beat a global threshold. The §9 bundle
diagnoses the cost precisely:

| metric | uncorrected | corrected |
|---|---|---|
| avg set size | 1.30 | **4.18** |
| marginal coverage | 0.902 | 0.894 |
| CovGap | 0.0505 | 0.0497 |
| worst-class coverage | 0.717 | **0.717** |

The correction **triples set size while leaving worst-class coverage bit-identical
and CovGap essentially unchanged** — it is not helping the classes that need help.
At R² ≈ 0.31 the prediction noise costs more than the signal buys. Consistent with
every other CIFAR-100 limit; this is a debug reading, not a gate verdict.

---

---

## MEASURED: Pl@ntNet calibration is far thinner than assumed (2026-08-04)

From the released `cal` labels alone (21,783 samples, 1081 classes; zero images):

| statistic | value |
|---|---|
| classes with ≥1 cal sample | 976 |
| **classes with ZERO cal samples** | **105** (δ_y undefined; `fallback_policy.md` applies) |
| per-class count p25 / **median** / p75 / p90 / max | 1 / **2** / 10 / 43 / 616 |

**Half the classes have ≤2 calibration samples.**

### Multi-α (§8.8) on real data — deployment view

A classwise *conformal* quantile is finite only if `n ≥ ceil(1/α) − 1`:

| α | n needed | classes supported |
|---|---|---|
| 0.01 | ≥99 | **57 / 1081 (5.3%)** |
| 0.05 | ≥19 | **187 / 1081 (17.3%)** |
| 0.1 | ≥9 | **304 / 1081 (28.1%)** |

So **~72% of classes cannot receive a finite classwise threshold even at α=0.1** and
fall back to the global threshold. This is the *deployment* view; the *prediction
target* uses the level-matched `empirical` estimator (Amendment 2), which is defined
for any n ≥ 1 but is severely noisy at n = 1–2.

### Consequence 1 — Phase 1 must run on a prevalence-selected subset

Gate A needs to split each class's samples in half, so it needs n ≥ 2 at absolute
minimum and realistically ≥10. With a median of 2, gate A **cannot** run on the full
class set. Matched-n cost:

| n_cal | classes kept | fit / held-out | samples per gate-A half |
|---|---|---|---|
| 10 | 283 (26.2%) | 141 / 142 | 5 |
| 20 | 184 (17.0%) | 92 / 92 | 10 |
| 25 | 152 (14.1%) | 76 / 76 | 12–13 |
| 50 | 98 (9.1%) | 49 / 49 | 25 |

Every choice discards 74–91% of classes, and the discard is **prevalence-linked by
construction**. Gate A/B/C therefore measure extrapolation *among head-ish classes
only*. This must be stated in every table; it is not a detail.

### Consequence 2 — this VALIDATES the project's premise

If 72% of classes cannot get a classwise threshold, then every existing per-class
method is undefined there and falls back to global. A method that **predicts** δ_y
from geometry needs no calibration samples for the target class, so it addresses
exactly the regime where the incumbents have nothing. The thin tail is the problem
the project claims to solve, and the data confirms the problem is real.

### Consequence 3 — the tail can still be EVALUATED, just not used as a target

There is no ground-truth δ_y for a 2-sample class, so gate B (R² against δ_y) is
necessarily restricted to the estimable subset. But **efficiency can be evaluated on
the tail**: one tail class gives a useless coverage estimate (0/2, 1/2, 2/2), while
**macro-coverage aggregated over ~700 tail classes is estimable**. That is precisely
what macro-coverage is for (Bhattacharyya–Ding–Barber). So the split is:

- **gate A / B / C** → estimable subset (n_cal-restricted), reported with the
  selection made explicit;
- **§6.4 / efficiency** → ALL classes including the tail, aggregated via
  macro-coverage, which is where the claim actually pays off.

### PROPOSED `n_cal` — pre-register a PAIR, decided before any Phase-1 run

Two competing pressures: more classes → tighter gate-B/C CIs (the binding
constraint on CIFAR-100 at 50 held-out classes); more samples/class → higher gate-A
ceiling (the cap on achievable R²). No single value optimises both.

- **PRIMARY: `n_cal = 25`** — 152 classes, 12–13 samples per gate-A half. This
  matches the CIFAR-100 configuration that produced a usable ceiling
  (reliability 0.638), so the ceiling is known to be workable at this depth.
- **SENSITIVITY: `n_cal = 10`** — 283 classes, 5 per half. Nearly double the classes
  (more B/C power) at a lower gate-A ceiling.

**Both always reported.** Fixing the pair now, before any Pl@ntNet Phase-1 run,
is what keeps this from becoming a post-hoc choice. Awaiting human confirmation.

---

---

## Amendment 5 — ADOPTED 2026-08-05 (human-approved): how Phase 1 runs when §3.3 is not met

### The measurement that forced it

`notebooks/02_descriptor_stability.ipynb` on Pl@ntNet (45,756 train images, 1081
classes, bootstrap estimator):

| | q=5 | q=10 | q=25 | q=50 |
|---|---|---|---|---|
| overall | 0.701 | 0.759 | 0.792 | **0.815** |
| q0 tail (n=2–7) | 0.670 | 0.644 | 0.664 | **0.684** |
| q1 (n=7–21) | 0.665 | 0.774 | 0.814 | 0.810 |
| q2 (n=21–100) | 0.681 | 0.798 | 0.883 | 0.909 |
| q3 head (n=100) | 0.649 | 0.770 | 0.879 | **0.922** |
| **head − tail** | −0.021 | 0.126 | 0.215 | **+0.238** |

Images per class: p10=3, **p50=21**, p75=100. `recommended_quota = None` — **§3.3 is
NOT satisfied**, so a negative Phase-1 result is ambiguous by that section's own logic.

**The tail is flat and cannot be fixed by any quota.** q0 sits at ~0.67 for every q,
because those classes hold only 2–7 images: a bootstrap draw of 50 from 3 images just
resamples the same 3. Tail stability is capped by `n_y`, not by the quota. Raising the
quota would *widen* the spread (25% of classes were capped at our quota=100 and have
more available), so it makes the contamination worse, not better.

Only **4 features** reach ≥0.90 at q=50: logit_margin 0.945, cos_knn_5 0.916,
cos_knn_10 0.909, cos_knn_1 0.907 — versus 9 on CIFAR-100. Covariance features are
0.54–0.71, unsurprising when a 2048-dim covariance is estimated from a median of 21
images.

### Adopted protocol (three parts)

**1. Asymmetric reading of gate B.** Descriptor noise ATTENUATES R²; it cannot
manufacture predictability. So a gate-B **PASS remains meaningful** (the noise worked
against us), while a gate-B **FAIL is recorded as ambiguous** per §3.3 rather than as
evidence of no signal. This asymmetry must be stated wherever gate B is reported.

**2. Gate C is computed WITHIN prevalence strata (PRIMARY); pooled is secondary.**
Descriptor accuracy tracks prevalence, so pooled "geometry beats log-prevalence" is
confounded. Conditioning on the confound is the standard remedy: inside one quartile
prevalence barely varies and descriptor quality is roughly uniform.

Validated on planted causal stories (`tests/test_stratified_gatec.py`):

| δ_y driven by | pooled R² | strata where geometry beats prevalence |
|---|---|---|
| prevalence only | +0.935 | **0 / 4** ✓ correctly rejects |
| geometry only | +0.928 | **4 / 4** ✓ correctly accepts |

The pooled R² is nearly identical for the two stories (0.935 vs 0.928) — it carries
almost no information about which is true. That is the justification for making the
stratified form primary, and a test asserts it stays true.

**3. Feature sets.** PRIMARY `stable` = the ≥0.90 features measured above (4 on
Pl@ntNet); SECONDARY `full`. Both always reported, selection criterion never touches
δ_y.

**Reporting requirement.** The head−tail spread (**0.238**) must appear next to every
gate-C conclusion, and every Phase-1 report must state that §3.3 was not met and that
gate-B failures are consequently uninterpretable.

---

## Amendment 6 — the Phase-0 criterion cannot answer §5; replace it (2026-08-05)

**Status: ADOPTED, implemented, validated 4/4 on planted worlds.**

### What happened

Phase 0 on Pl@ntNet returned FAIL with every mechanism except `temperature` strongly
negative (`class` −54.8, `energy_b50` −74.7 at α=0.1). Before reading that as "the
structure does not live at the class level", the metric itself was audited. It fails.

### Three independent legs of evidence

1. **Nested capacity degrades monotonically** on the real logits:
   `energy_b2 −47.2 → energy_b10 −67.4 → energy_b50 −74.7`. The energy family is
   nested — b50 can represent everything b2 can — so a criterion in which extra
   capacity systematically loses is not measuring where structure lives.

2. **Zero discriminating power on planted worlds.** Four synthetic worlds, each with
   exactly one planted mechanism {class, global, energy, none}. The criterion named
   `temperature` the winner in **4 of 4**, including the world whose only structure
   was per-class difficulty. A macro-coverage variant with free deflation also scored
   2/4 and also failed the class world.

3. **The mechanism, arithmetically.** Reaching worst-class coverage by **uniform
   inflation** charges a mechanism for its threshold **variance**, whatever its source:
   - the classes still short after class-adaptation are exactly those with the
     *lowest* thresholds, so a global additive repair over-inflates the whole vector
     (`class` needed inflation 0.0492 vs `global` 0.0117);
   - with `s = 1 − p` and a weak model nearly every wrong label sits in [0.98, 1.0],
     so set size is near-vertical in the threshold: **+0.0117 inflation took average
     set size from 13.6 to 55.7**.

   So cost ∝ Var(δ̂) = Var(δ true) + Var(noise), and parameter count enters directly.
   `global` has zero opportunities to undershoot; `class` has K.

The `class` mechanism is *perfectly aligned with the criterion's own requirement*
(the criterion is per-class, the mechanism is indexed per-class) and still lost to
doing nothing. A criterion whose best-matched mechanism loses is measuring something
else.

### Why not patch it a fourth time

Amendments 1 and 3 and the four §6.4 designs were all patches to a set-size criterion.
Set size conflates §5's question with three things §5 never asked about: score
saturation, the choice of repair operator, and the conditional-vs-marginal efficiency
tradeoff (equalizing coverage across classes *always* costs average set size — that is
Jensen, not a finding). No patch removes those.

### The replacement

§5 asks **where the threshold structure is located**. That is a question about
explaining per-class quantiles. Target `q*_y` = the (1−α) quantile of class y's
true-label scores on EVAL; every mechanism is fit on CAL and predicts it:

| mechanism | class-level prediction |
|---|---|
| `global` | a constant (no class-level variance by construction) |
| `energy_bK` | class-mean of the per-sample thresholds it assigns |
| `class` | the per-class q̂ fit on CAL |

R² is taken **across classes, out of sample**, so a mechanism whose extra parameters
are noise earns **negative** R² (measured: `energy_b50 → −35.8` in the no-structure
world). Capacity is punished, not rewarded.

**Temperature is judged by the question it can actually answer.** A global temperature
cannot produce class-level variance, so scoring it by class-level R² would rig the
comparison. Instead: does the best global temperature **remove** the class-level
structure? — measured as split-half reliability of q̂_y on temperature-rescaled scores
(`targets/delta.py:class_quantile_reliability`).

**The rival is not crippled.** When class difficulty is real it also shows up in
per-sample confidence, so `energy_b10` reached **+0.523** against `class`'s **+0.969**
in the planted-class world. `class` must beat a genuine competitor.

### Pass criterion (pre-registered)

Class-level structure exists above noise (reliability > 0.30) **AND** survives the best
global temperature (reliability after T > 0.30) **AND** `class` R² exceeds every
`energy_bK` rival, with non-overlapping CIs across splits.

Reported alongside: `target_sd` = sd(q*_y), the §5 discriminant — **0.0839** when class
structure was planted vs **0.0020** when only global miscalibration was.

### Validation (mandatory before any Phase-0 verdict is read)

`pcc/tests/test_phase0_explain.py` — recovers the planted mechanism in **4/4** worlds,
confirms capacity is punished, and is the pre-registration guard: the criterion it
replaced scored 0/1 on the class world.

`phase0_cc_decomposition` is **retained as a SECONDARY deployment-cost report** — "what
does class-conditional coverage cost at this calibration budget" is a real question —
with a docstring stating that its ranking of mechanisms must never be the verdict.

---

## Amendment 7 — gate C was never run for the primary feature set (2026-08-05)

**Status: ADOPTED, implemented, tested.** A plain defect, not a judgement call.

Amendment 5 made the stability-screened set PRIMARY. On Pl@ntNet that set is
`['cos_knn_5', 'logit_margin']` — it contains **neither** `log_prevalence` **nor**
`cos_knn_1`. `predictability()` resolved its ablation columns from the Phi it was
handed, and notebook 04 handed it a **pre-sliced** Phi. So `ablations` came back empty,
`gate_C_pass` became `None`, and **the gate most likely to fail was silently never
run** for the primary feature set (`gate_C_primary_pass: null` in the report).

**Fix.** Ablations are BASELINES, not features of the full model. `predictability()`
now takes the complete `Phi` plus `feature_subset` naming the full model's columns, and
always resolves ablations against the complete matrix. `distance_col` became a tuple of
candidates (`cos_knn_5`, `cos_knn_1`, `cos_knn_10`), first present wins, so the distance
baseline survives a stability screen that drops `cos_knn_1`.

**Also added: an `underpowered` flag.** The stratified `full` result had **15 features
fit on 19 training classes** per stratum and scored negative held-out R² (−0.015,
+0.179, −0.105, −0.266) while the 2-feature stable set reached **+0.453** in the rarest
stratum. Comparing a 15-parameter model against 1–2-parameter baselines at n_train=19
is not a capacity-fair test, so "geometry beats prevalence in 0/4 strata" was
uninformative for `full`. `underpowered` (n_train < 3p) now marks that regime, and a
FAIL inside it may not be reported as evidence of no signal — the same asymmetry
Amendment 5 established for descriptor noise.

---

## Amendment 8 — §6.4 inverted: match the resource, read the benefit (2026-08-06)

**Status: ADOPTED, implemented, 5 tests.** Design 5. The previous four are in
`pcc/eval/setsize.py`'s module docstring.

### Design 4 had no headroom, and its own controls said so

Design 4's recorded controls: `oracle +0.045`, `oracle+constant +0.046`,
`pure constant ≈ -0.29`, `shuffled oracle -15.39`, `random -12.54`.

**A perfect δ̂ scored +0.045 while the noise floor was -15.** The ceiling was
essentially zero and the entire dynamic range sat on the negative side, so no real
predictor could ever return a positive. That is the same defect that retired design 2
("marginal split-CP is already optimal, so a class-indexed correction provably cannot
win") — and it was sitting in the file, unread, while Pl@ntNet reported **-60.58**.

Root cause is Amendment 6's: matching on COVERAGE and reading SET SIZE is the
ill-conditioned direction, because set size is near-vertical in the threshold.

### The inversion

Match the **resource** — average set size, which is smooth and monotone in a scalar
shift, so bisection is stable — and read the **benefit** — per-class coverage equity,
bounded in [0,1]. Oracle headroom becomes **+0.31** on worst-class coverage.

### Objective: worst-class coverage, NOT macro

| objective | oracle at matched size |
|---|---|
| worst-class | **+0.312** |
| macro | +0.012 |

Macro has no headroom **even for an oracle**, because a uniform threshold is already
near-optimal for an unweighted mean — Jensen, not a property of δ_y. Macro therefore
cannot be §6.4's objective, though it remains correct for the descriptive tail report
(`pcc.eval.tail`), where the question is not comparative.

### Shrinkage is mandatory, and λ must come from 𝒴_train

The raw δ̂ **hurts** at realistic predictor quality. A worst-class objective is governed
by the **largest** error in δ̂, not its variance — and R² controls *mean squared* error,
so the requirement tightens as K grows (the max of K Gaussian errors is ≈2.7σ at
K=60). Break-even in raw form is at ρ=1.0, i.e. a perfect predictor.

Shrinking δ̃ = λ·δ̂ fixes it, with λ far more aggressive than regression attenuation:

| predictor R² | best λ | Δ worst-class | raw λ=1 |
|---|---|---|---|
| 0.30 | 0.10 | **+0.025** | -0.403 |
| 0.56 | 0.10 | +0.054 | -0.352 |
| 0.90 | 0.20 | +0.126 | -0.116 |
| oracle | 0.70 | +0.390 | +0.340 |

**λ is a free parameter**, so selecting it on the held-out classes would manufacture a
positive result — exactly the failure mode Amendment 4 was written to prevent.
`setsize_translation_shrunk` selects λ on 𝒴_train and applies it unchanged to
𝒴_held-out, and a test asserts the train-selected λ need not equal the held-out
optimum (observed: 0.10 vs 0.05, giving a held-out **-0.050**).

### Honest limitation

At R²≈0.30 the expected gain (**+0.02**) is **smaller than the error in transferring λ
across class sets**. Synthetic end-to-end at that quality returns `observed -0.0017`
against `oracle +0.3533` and `raw λ=1 -0.3850`. So the correct reading of §6.4 at
current descriptor quality is **null with a quantified ceiling**, not "negative":
λ-selection successfully prevents the harm, and the distance from 0 to +0.35 is what
better descriptors would have to buy. Report the controls, never the verdict alone —
if `oracle_ceiling` ≤ 0.02 the metric cannot return a positive and the observed value
carries no information either way.

### Consequence for the tail report and §9

Both now apply the **shrunk** δ̂. Reporting them under the raw vector would describe a
correction the protocol does not endorse, and the earlier tail table (set sizes falling
2.31→2.03 etc.) was computed that way.

---

## Status

- Amendment 1: **implemented + tested** (`pcc/eval/decomposition.py:group_quantile`).
- Amendment 2: **implemented + tested** (`targets/delta.py:delta_y_matched_n`,
  `:prevalence_null`), wired into `notebooks/04_phase1_gate.ipynb`. `n_cal` still
  needs a **human decision for Pl@ntNet** (the head/tail trade-off).
- Amendment 3: **SUPERSEDED by Amendment 6.** `phase0_cc_decomposition` is retained
  only as a secondary deployment-cost report; it cannot rank mechanisms.
- Amendment 4: **SUPERSEDED by Amendment 8** (no oracle headroom; retained as a
  secondary, non-verdict view for continuity).
- Amendment 5: implemented (`predictability_by_stratum`), wired into `notebooks/04`.
- Amendment 6: **adopted + implemented + validated 4/4**
  (`decomposition.phase0_explain_class_level`, `targets/delta.py:class_quantile_reliability`,
  `pcc/tests/test_phase0_explain.py`), wired into `notebooks/03`.
- Amendment 7: **adopted + implemented + tested** (`predictability(feature_subset=...)`,
  ablations always resolved from the complete Phi; `underpowered` flag).
- Amendment 8: **adopted + implemented + tested**
  (`setsize.setsize_translation_shrunk`, `equity_at_matched_size`, `select_shrinkage`,
  `pcc/tests/test_sec64_shrunk.py`), wired into `notebooks/04`.
- `δ_y` as defined in §6.1 (conformal quantiles) is retained as the **deployment**
  target definition; these amendments concern how it is **measured** for the
  structure and predictability questions.
