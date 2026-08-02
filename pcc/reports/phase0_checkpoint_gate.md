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
