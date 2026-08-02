# `pcc/method/` — g_θ — DO NOT BUILD BEFORE THE GATE PASSES

This directory holds the predictive model `g_θ` that maps class descriptors
`φ(y)` to the correction `δ̂_y`. It is the Phase 2 method.

**Per `AGENTS.md` §0.2, §6.5, and §10: nothing here may be implemented until
Phase 1's gate passes** — that means criteria A (reliability ≥ 0.3), B (held-out
normalized R² clearly > 0), C (beats log-prevalence / Fargion-distance / fuzzy
baselines), **and** §6.4 (predicted δ̂_y actually shrinks set size on held-out
classes) are all satisfied, each with CIs that don't include the failing value.

> "Kapasitas model bukan masalahnya; keberadaan sinyal yang jadi masalah."
> (Model capacity is not the problem; the existence of signal is.)

An agent may **not** decide on its own to skip a failed gate. If the gate fails,
the deliverable is a clear report of that failure (`reports/`), not a workaround.

When the gate passes, the pre-registered pass criteria and the gate report must
be linked here before the first line of `g_θ` is written.
