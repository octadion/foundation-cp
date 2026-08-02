# `pcc/extract/` — frozen forward pass → logits + penultimate embeddings

Backbones are **frozen** (`AGENTS.md` §1): no retraining, no fine-tuning. This
module runs one forward pass and captures, in a **single pass** (§2.3):
- **logits** (sanity vs. released scores; not a substitute for released scores)
- **penultimate embeddings** (the descriptor input — the thing no release contains)

## Hard requirements (Colab, `AGENTS.md` §3.1)
- **Resumable**: checkpoint every N batches to Drive; on start, read the
  checkpoint/manifest before computing anything.
- **Ephemeral local disk**: never leave results only in `/content`.
- **No A100 assumption**: use `pcc.utils.device.autobatch` for OOM backoff.
- **Stream → forward → write embedding → discard image** (§3.2).

## Backbone provenance — use the SAME checkpoint that made the released scores
Per the release audit: iNat-2018 + Pl@ntNet use **LTC's released ResNet-50**;
ImageNet uses the **public SimCLRv2 `r152_3x_sk1`** encoder. A forward pass must
reproduce the released softmax scores bit-for-bit on a sample before embeddings
from that checkpoint are trusted (verify on Colab; see release_audit open items).

`forward.py` (TODO) provides a `PenultimateExtractor` that registers a forward
hook on the penultimate layer and writes `(logits, embeddings)` shards + manifest.
