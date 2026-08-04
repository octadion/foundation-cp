# Phase 0 prerequisite — checkpoint → score reproduction gate (PRE-REGISTERED)

**Written:** 2026-07-25, **before** any forward pass. Author: coding agent.
Promoted from "open item" to a **hard gate** by instruction (2026-07-25).

## Why this gate exists

The whole pipeline depends on our extracted embeddings coming from the *same*
model that produced the released softmax scores used for the baselines. If the
forward pass we build does not reproduce those scores — because of preprocessing
drift, a normalization mismatch, or the wrong checkpoint version — then:

- extraction still runs,
- descriptors still compute,
- Phase 1 still produces R² and reliability numbers,

and **every one of those numbers is meaningless**, with nothing visibly broken.
This reproduction check is the only alarm. It runs first and it can stop the project.

## Hypothesis under test

> A `resnet50` loaded from LTC's released `best-{dataset}-model.pth`, run with
> LTC's exact test transform, reproduces LTC's released softmax scores for that
> dataset's val/cal set (up to floating-point/hardware noise), and achieves the
> same top-1 accuracy.

## Method (permutation-invariant — see release_audit.md)

Released `*_softmax.npy` / `*_labels.npy` are in an **unrecoverable shuffled
order** (`shuffle=True` in LTC's loaders). So the check does not align rows to
source images. Instead:

1. Build the val (or cal) dataset in **deterministic order** (`shuffle=False`),
   exact LTC test transform: `Resize(256) → CenterCrop(224) → ToTensor →
   Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])`.
2. Forward pass → logits → cast to **float64** → `scipy.special.softmax(axis=1)`
   (matches LTC's `get_softmax_and_labels` exactly, incl. dtype).
3. Compare our score set `(S_mine, y_mine)` to the released set `(S_rel, y_rel)`
   using statistics invariant to row order:
   - **top-1 accuracy** (fully permutation-invariant),
   - **sorted true-class-probability curve** (per sample, prob of its own label,
     sorted ascending) — a distribution comparison,
   - **nearest-neighbor row match**: for K sampled rows of `S_mine`, the min
     L∞ distance to any row of `S_rel`; near-zero iff the score *multiset* matches.
   - a **label-multiset** check: `sorted(y_mine) == sorted(y_rel)` and per-class
     counts equal (confirms same underlying data & class indexing).

We may run on a **val subsample** (≥ ~3,000 images) — enough for accuracy and
NN matching to be decisive — since downloading every val image is not required
for the alarm to fire (the real failure modes are gross).

## Pass criteria (pre-registered — do NOT change after seeing results)

Let `acc_mine`, `acc_rel` be top-1 accuracies; NN distance = per-row min L∞
softmax distance from a sampled `S_mine` row to the released set.

**PASS** requires ALL of:

- **G1 accuracy:** `|acc_mine − acc_rel| ≤ 0.002` (0.2 percentage points).
- **G2 NN match:** ≥ **99%** of a ≥1,000-row `S_mine` subsample has an NN in
  `S_rel` with L∞ softmax distance ≤ **1e-4**; median NN distance ≤ **1e-5**.
- **G3 true-prob curve:** max abs difference between the sorted true-class-prob
  curves (interpolated to a common grid) ≤ **1e-3**.
- **G4 label multiset:** per-class sample counts of `y_mine` equal those of
  `y_rel` exactly.

**FAIL** if any of G1–G4 is violated. On FAIL → **STOP**, write the report with
conclusion `FAIL`, do **not** run extraction, and report to the human with the
observed numbers and the most likely cause (transform/normalize/checkpoint/
class-indexing).

Tolerance rationale: the guarded failures (preprocessing drift, normalization
mismatch, wrong checkpoint) move accuracy by whole points and softmax rows by
≫1e-2. GPU/cuDNN/torch-version float noise stays well under 1e-4 on softmax
probabilities. The thresholds sit in that gap. If, on real hardware, benign
noise turns out to exceed 1e-4, that is itself reportable — loosen only with a
recorded, timestamped justification and re-report both versions (same discipline
as `fallback_policy.md`).

## On PASS — record (goes into the report + Drive manifest)

- SHA256 of the `.pth` checkpoint and of each released `.npy` used.
- torch / torchvision / numpy / scipy versions, GPU name, CUDA/cuDNN versions.
- `acc_mine`, `acc_rel`, NN-distance quantiles, curve max-diff, subsample size.
- A `GATE_PASSED` marker file on Drive that `01_setup_extract` checks before
  running.

## CORRECTIONS to the criteria, made during the first Pl@ntNet run (2026-08-04)

Three specification errors were found when the gate first ran on real Pl@ntNet
data. All three are recorded here because §12 requires criteria changes to be
explicit, and because two of them would have produced a FALSE FAILURE on a
correct checkpoint.

**C1 — G1's tolerance was mis-derived (would false-alarm).** The pre-registered
`|acc_mine − acc_rel| ≤ 0.002` assumed a like-for-like comparison. But the gate
compares a **subsample** accuracy (3,000 images) against the **full** released set
(21,783), so binomial sampling noise alone is `sqrt(p(1−p)/n)` = **0.0073** at
p≈0.8, n=3000 — larger than the entire tolerance. A correct checkpoint would fail
most of the time.

Corrected rule: `tol_effective = max(0.002, 3 · SE)` with SE computed from the
actual subsample size. Observed on the real run: acc_mine 0.8040 vs acc_rel 0.7945,
diff 0.0095 = **1.31 σ**, tol_effective 0.0217 → **PASS**.

This is a specification fix, not a loosening to pass. Verified that the widened
tolerance still catches real breakage: a deliberately corrupted comparison failed
at **43.7 σ**. The failure modes G1 guards (preprocessing drift, normalization
mismatch, wrong checkpoint) move accuracy by whole percentage points, far beyond 3 σ.
`diff_in_sigmas` is now reported so the margin is always visible.

**C2 — G4 was applied to the wrong set (would always fail).** `label_multiset_equal`
was called on the forward-passed **subsample** (3,000) versus the released array
(21,783). Per-class counts can never match, so G4 was a guaranteed failure
regardless of the checkpoint. Fixed: G4 now takes `full_mine_labels`, the labels of
the **entire reconstructed cal subset**. Those need **no forward pass at all** —
they come straight from `ltc_cal_val_indices` — so G4 becomes what it was meant to
be: a real test of whether LTC's cal membership was reproduced. The result dict
records `scope` so a subsample-scoped G4 can never be misread as meaningful.

**C3 — G2 exhausted RAM (crashed the session).** `nn_match_distances` built
`abs(reference[None] - chunk[:, None])`, i.e. a `[256, 21783, 1081]` float64 array =
**48.2 GB**. The blocking was on the wrong axis. Replaced with an exact pruning
scheme: the row max is 1-Lipschitz in L∞, so `|max(a) − max(b)| ≤ ‖a−b‖_∞`, and
sorting the reference by row max lets a binary search discard everything outside
`[q_max ± tol]` with **no false negatives**. Guarantee, stated precisely: the
returned distance is exact whenever it is ≤ tol; beyond tol it is only guaranteed
to be > tol. That is all G2 needs (it asks what fraction lie within tol). Verified
against brute force: identical within/outside classification, bit-exact values for
within-tol rows, 40/40 planted duplicates found, 0 false positives.

## GATE RESULT — Pl@ntNet `cal`, FULL SET: **FAIL** (2026-08-04). NOT RESOLVED.

Run on all 21,783 released rows (no subsampling), `LOSS=cross_entropy`,
`best-plantnet-model.pth`, sha256 `4b82e4aa1a97d281…`:

| criterion | value | verdict |
|---|---|---|
| **G1** accuracy | ours 0.7950236 vs released 0.7945187 — diff **5.05e-4** (0.18 σ) | **PASS** |
| **G4** label multiset | 21,783 = 21,783, per-class counts exact | **PASS** |
| **G3** true-prob curve | max abs diff **0.0040** (tol 1e-3) | **FAIL** |
| **G2** NN row match | **19.5%** within 1e-4; median L∞ **0.554** (tol 1e-5) | **FAIL** |

**Extraction is BLOCKED. The gate did its job and the project does not proceed
past it until this is explained.**

### Why this pattern is confusing, and what it rules out

Going from a 3,000 subsample to the full set moved G1 from 0.0098 (1.36 σ) to
**5.05e-4** and G3 from 0.0202 to 0.0040 — so most of the earlier discrepancy was
sampling noise. What did NOT move is G2: median L∞ stayed ~0.55.

- Not a wrong-architecture or badly-loaded checkpoint: accuracy agrees to 11
  samples in 21,783.
- Not wrong images, labels or class indexing: G4 is exact, and an accidental
  agreement of per-class counts across 1,081 long-tailed classes is not credible.
- Not the focal-loss variant mix-up first suspected: that hypothesis was checked
  and **not confirmed** (the recursive glob resolves consistently in the observed
  layout). A `LOSS` knob and a same-variant assertion were added anyway as
  defensive code, and the notebook now lists every candidate path.

G1 and G3 are **order-invariant distribution statistics**; G2 is the only criterion
that requires **row-level correspondence**. So the live explanations are:

1. the two score sets are **different multisets of rows** (an equally accurate,
   similarly calibrated but genuinely different model, or different inputs), or
2. **G2 itself is at fault** — the matcher was rewritten this same day to fix a
   48 GB allocation, so it is a legitimate suspect.

### Diagnostic added to separate them (cell 7c)

If the two sets contain the same rows up to permutation, then sorting each set by
row-max must give **identical** vectors — a test that involves no matching and so
cannot be confounded by a bug in the matcher. Verified to discriminate: identical
rows permuted give max|diff| = **0.00e+00**; a different-but-equally-accurate model
gives **8.6e-02**.

Cell 7c also reports the mean max-probability of matched vs unmatched rows. If the
19.5% that matched are the saturated ones, that is the signature of two different
accurate models (confident rows coincide, uncertain rows diverge).

### What will NOT be done

The tolerances stay. G2's median of 0.554 is three orders of magnitude above float
noise; loosening it to obtain a PASS would reinstate exactly the silent failure this
gate exists to catch — extraction would run, descriptors would compute, Phase 1
would emit numbers, and none of it would mean anything.

## RESOLVED — the G2 failure was MY BUG, not the checkpoint (2026-08-04, later)

Cell 7c settled it, and it indicted the matcher rather than the model. Two numbers
read together:

- `sorted row-max: median|diff| = **2.90e-04**` — the two score sets agree closely
- matched rows mean max-prob **0.9957** vs unmatched **0.7965**

The G2 pruning window was `[q_max ± tol]` with **tol = 1e-4**, while the actual
per-row row-max difference has median **2.9e-4** — **three times wider than the
window**. So each row's true twin was *systematically excluded from the search*,
and a large distance was reported instead. Saturated rows (row-max ≈ 1.0 on both
sides, difference ≈ 0) stayed inside the window and matched; everything less
confident was pruned away. That is exactly the observed "19.5% matched, all of them
confident" signature. **The median of 0.554 was an artefact of my pruning radius.**

### Fix: progressive widening with a correctness certificate

Search radius `r` is widened over `(tol, 10·tol, 100·tol, 1e-2, 1e-1, ∞)` and
stops as soon as the best distance found satisfies `d ≤ r`. That condition is a
proof of optimality: any row outside the window differs in row-max by more than
`r ≥ d`, and since `L∞ ≥ |row-max difference|`, its distance exceeds `d`. So the
returned value is the TRUE nearest-neighbour distance, at bounded cost.

Verified exact against brute force (`rtol=1e-9`) at every noise level — 3e-7,
9e-4, 9e-3, and a genuinely different model. Regression test:
`tests/test_manifest_and_tail.py::test_nn_match_is_exact_across_noise_regimes`.

### Also retracted: the "DIFFERENT MODEL" claim in cell 7b

Cell 7b declared **DIFFERENT MODEL → Sec 2.3.4 BLOCKER** because the full-set
accuracy gap (5.05e-4 ≈ 11 argmax flips in 21,783) exceeded a 1e-4 threshold I had
picked. That was an over-claim: a handful of argmax flips is equally consistent with
numerical nondeterminism (PIL/torchvision resize behaviour, cuDNN kernel choice) as
with a different checkpoint. Accuracy alone cannot distinguish them. Cell 7b now
reports the flip count and defers the verdict to G2's — now exact — distances:

| G2 median distance | interpretation |
|---|---|
| ~1e-7 | same model, same inputs |
| ~1e-4 | same model, small numerical/preprocessing drift |
| >1e-2 | genuinely different model → §2.3.4 blocker |

### Status: gate must be RE-RUN

The verdict recorded above (FAIL) was produced by a broken matcher and is void.
Tolerances were NOT changed — G2's tolerance is still 1e-4 with a 1e-5 median
requirement. What changed is that G2 now computes the quantity it always claimed to
compute. Re-run required before any conclusion about the checkpoint.

## Scope (per the revised cheap-first order — see release_audit.md)

- **CIFAR-100 — reproduction gate N/A.** No released checkpoint exists, so we
  self-train a ResNet-50 and take scores + embeddings from our *own* model. There
  is nothing external to reconcile. Replace this gate with a **self-consistency
  check**: (i) recompute softmax from saved logits and confirm it matches the
  extracted softmax exactly, (ii) confirm the penultimate-embedding → `fc` →
  logits path reproduces the stored logits (hook correctness), (iii) sane val
  accuracy. Used to debug the pipeline before spending on Pl@ntNet.
- **Pl@ntNet-300K — the definitive feasible reproduction gate.** Released
  checkpoint + scores; run this gate in full (G1/G3/G4; G2 only if the class-index
  convention is confirmed). Phase 0 + Phase 1 run here.
- **iNaturalist-2018 (8142) — DEFERRED.** Its gate is attempted **only if**
  Pl@ntNet's gate passes, Phase 1 passes on Pl@ntNet (A/B/C), and §6.4 is
  positive. Val-image acquisition logistics are recorded in `release_audit.md`
  but must not be actioned before then.

Each `(dataset × backbone)` has its own gate/consistency report; extraction for a
pair is blocked until *its* report says PASS.

### CIFAR-100 gate result (run 2026-07-25) — PASS

`reports/00a_cifar100_selfconsistency.json`. Self-trained ResNet-50, 15 epochs,
Adam lr 1e-4. `softmax_recompute_maxdiff = 0.0`; `penultimate_to_logits_maxdiff =
3.62e-06` (float32 embedding → float64 reconstruction rounding, well inside the
1e-3 tolerance) → **the penultimate hook captures the correct tensor**. Test-set
accuracy 0.826.

**Defect found and fixed (recorded, not hidden):** in the first run, the
validation loader subset `train_full`, which carries `RandomResizedCrop(224)` +
flip, so per-epoch val accuracy was measured **under training augmentation**. It
read ~0.688 while true (clean-transform) test accuracy was 0.826, and best-epoch
selection ran on that augmented, noisy metric. Fixed in `00a` cell 4 (val now
uses a clean-transform dataset instance). **Not retrained**: 0.826 is more than
adequate for a debug-only backbone, and CIFAR-100 is never the gate verdict.

### α feasibility limit on CIFAR-100 (PRE-REGISTERED before Phase 1)

Split-conformal classwise quantiles need `ceil((n+1)(1−α))/n ≤ 1`. With
CIFAR-100's 100 test images per class, measured directly:

| samples/class | α=0.01 | α=0.05 | α=0.1 |
|---|---|---|---|
| 100 (full test) | degenerate (level=1.000) | ok | ok |
| 50 (50/50 cal/eval split) | **undefined (∞)** | ok | ok |
| 25 (split-half within cal, gate A) | **undefined (∞)** | degenerate (level=1.000) | ok |

Consequence, fixed in advance: on CIFAR-100 the §8.8 multi-α requirement
**cannot** be met — **gate A (split-half reliability) is reported at α=0.1 only**,
and α=0.01 classwise δ_y is undefined. This is a *sample-count* limit, not a
result, and it is a further reason CIFAR-100 is debug-only. The full multi-α
sweep {0.01, 0.05, 0.1} runs on Pl@ntNet, where head classes have enough
calibration samples — and the same arithmetic will make α=0.01 undefined for
Pl@ntNet's **tail** classes, which is exactly why §6.2 requires head vs. tail to
be reported separately rather than pooled.
