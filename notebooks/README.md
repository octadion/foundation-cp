# `notebooks/` — Colab runners only

Notebooks are **runners, not algorithms** (`AGENTS.md` §3.4). Heavy logic lives
in `pcc/*.py` so results reproduce off Colab and diffs are readable. Structure
mirrors `c:/jagr/heat` notebooks.

## Every notebook
- **Cell 0 (markdown):** title + what it does + its pre-registered pass criteria.
- **Cell 1:** GPU check (`nvidia-smi`).
- **Cell 2 (`# === EDIT ME ===`):** config — Drive paths, backbone id, seed, α.
- **Cell 3 (setup):** mount Drive; clone/pull repo; pip install **pinned**
  versions; set seed; **print GPU + library versions to log**.
- **Body:** call `pcc` functions; every cell **idempotent** (safe to re-run).
- **Last cell:** write `pcc/reports/<name>.json` via `pcc.utils.io.write_report`
  (config, versions, seed, runtime, results, explicit pass/fail).

## Execution order (cheap-first — see reports/release_audit.md)
CIFAR-100 (pipeline debug, self-trained model) → Pl@ntNet (checkpoint gate +
Phase 0 + Phase 1 in full) → iNaturalist **only if** Pl@ntNet's gate and Phase 1
(A/B/C + §6.4) pass. ImageNet with Phase 2+/3.

## Numbering (create as phases need them)
- **`00a_cifar100_train_extract` — CIFAR-100 self-train + one-pass extract**
  (no released checkpoint exists): fine-tune ResNet-50 from IMAGENET1K_V2
  (Ding/CFCP procedure), then extract scores + penultimate embeddings from that
  same model. Reproduction gate is N/A here → run the self-consistency check
  (`reports/phase0_checkpoint_gate.md`, CIFAR-100 scope) instead.
- **`00_verify_checkpoint` — HARD Phase-0 gate** (`reports/phase0_checkpoint_gate.md`):
  a forward pass on LTC's released checkpoint must reproduce the released softmax
  scores (permutation-invariant, since LTC loaders shuffle). On FAIL → STOP, do
  not extract. Writes a `GATE_PASSED_{dataset}_{split}.json` marker to Drive.
- `01_setup_extract` — extract logits + penultimate embeddings (frozen backbone).
  **Refuses to run without the gate marker** for its (dataset, split).
- **`02_descriptor_stability` — MANDATORY before Phase 1** (`AGENTS.md` §3.3):
  descriptor stability vs. images-per-class (10/25/50/100); fixes the per-class
  quota. Without it, a negative Phase-1 gate is uninterpretable.
- `03_phase0_decomposition` — §5 decomposition on real logits (GATE 0).
- `04_phase1_reliability` — §6 gate A/B/C + §6.4 (GATE 1).
- Phase 2+ notebooks: only after the gate passes.

A notebook that only runs if the previous one is still live in memory is a bug
(§3.1): each reads inputs from Drive and writes outputs to Drive.
