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
