# `pcc/data/` — loaders, manifests, checksums

**Never store images. Store embeddings** (`AGENTS.md` §3.2). The loader's job is
to *stream* images through a frozen backbone; images are discarded after the
forward pass. Everything persisted goes to Google Drive with a checksummed
manifest (§2.3: "Ekstrak dan simpan logit DAN embedding penultimate dalam satu
lintasan, dengan checksum manifest").

## Data acquisition priority (`AGENTS.md` §3.2)
1. **Released scores/labels** from the Ding repos (MB–GB, bit-level comparable)
   — for **all** Tier-1/Tier-2 baselines. Do not recompute. See
   [`reports/release_audit.md`](../reports/release_audit.md).
2. **Val-split embeddings** — full download feasible; calibration + evaluation.
3. **Per-class train-subset embeddings** — streamed, fixed per-class quota
   (10/25/50/100), the descriptor source. The quota is fixed by
   `notebooks/01_descriptor_stability` (§3.3) — **not before**.

## Manifest
Every extraction writes a manifest: per-shard SHA256, backbone id + checkpoint
hash, dataset + split + taxonomic level, per-class sample count, and the exact
preprocessing transform. A run resumes by reading the manifest first (§3.1).

`manifest.py` (TODO) provides `write_manifest` / `verify_manifest` /
`resume_point`.
