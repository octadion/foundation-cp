# `pcc/experiments/` — one numbered, deterministic script per experiment

Each script (`AGENTS.md` §4, §12):
- accepts a `--seed`,
- writes its **full config + pre-registered pass criteria** to `pcc/reports/`
  before/at run (via `pcc.utils.io.write_report`, which refuses a report with no
  pre-registered criteria),
- is reproducible **bit-for-bit** on the same seed,
- is the primary execution path (notebooks are thin Colab runners over these).

Numbering tracks phases: `phase0_*`, `phase1_*` (gate), then Phase 2+ only after
the gate passes. No script under here may import `pcc.method` before the gate.
