"""Sharded, resumable extraction of logits + penultimate embeddings.

Colab kills sessions, so a long extraction must survive a kill and continue
(AGENTS.md §3.1). Design:

- write EMBEDDINGS, never images (§3.2): stream -> forward -> write -> discard
- one `.npz` per shard, appended to a checksummed manifest immediately
- `resume_point()` counts only the unbroken, checksum-verified prefix, so a
  half-written shard is redone rather than skipped (a skipped shard would silently
  desynchronise embeddings from labels)

The core loop takes a `make_batch_iter(start_index)` callable rather than a model,
so the resume/sharding logic is testable without torch or a GPU. The torch
convenience wrapper is `extract_dataset`.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterator

import numpy as np

from pcc.data.manifest import (add_shard, init_manifest, load_manifest,
                               resume_point, truncate_to_valid_prefix,
                               verify_manifest)

SHARD_FMT = "shard_{:05d}.npz"


def extract_sharded(make_batch_iter: Callable[[int], Iterator[dict]],
                    n_total: int, out_dir: str, *, provenance: dict[str, Any],
                    shard_size: int = 2000, resume: bool = True,
                    verbose: bool = True) -> dict[str, Any]:
    """Run extraction into `out_dir` as verified shards.

    `make_batch_iter(start)` must yield dicts of equal-length arrays (e.g.
    {'softmax':…, 'labels':…, 'embeddings':…, 'logits':…}) covering samples
    `start … n_total-1` IN A DETERMINISTIC ORDER. Determinism is what makes the
    resume correct — build it from a `shuffle=False` loader over a fixed index list.

    Returns a summary dict; the manifest on disk is the source of truth.
    """
    os.makedirs(out_dir, exist_ok=True)
    if resume:
        truncate_to_valid_prefix(out_dir)
        start = resume_point(out_dir)
    else:
        start = 0
    m = load_manifest(out_dir) if resume else None
    if m is None:
        m = init_manifest(out_dir, provenance=provenance)
        start = 0
    else:
        # Provenance must not change mid-extraction, otherwise the shards on disk
        # came from a different model than the ones we are about to append.
        old = m.get("provenance", {})
        drift = {k: (old.get(k), v) for k, v in provenance.items()
                 if k in old and old[k] != v}
        if drift:
            raise ValueError(
                f"provenance drift vs existing shards in {out_dir}: {drift}. "
                f"These embeddings came from a different model/config — start a "
                f"fresh out_dir instead of mixing them.")

    if start >= n_total:
        if verbose:
            print(f"[extract] already complete: {start}/{n_total} samples")
        return {"n_written": 0, "n_total_on_disk": start, "complete": True,
                "shards": len(m["shards"])}

    if verbose:
        print(f"[extract] resuming at sample {start}/{n_total} "
              f"({len(m['shards'])} verified shards on disk)")

    buf: dict[str, list] = {}
    buffered = 0
    written = 0
    shard_i = len(m["shards"])

    def flush():
        nonlocal buf, buffered, shard_i, written, m
        if buffered == 0:
            return
        arrays = {k: np.concatenate(v, axis=0) for k, v in buf.items()}
        fname = SHARD_FMT.format(shard_i)
        tmp = os.path.join(out_dir, fname + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, os.path.join(out_dir, fname))
        m = add_shard(out_dir, m, fname, n_samples=buffered,
                      extra={"index_start": start + written})
        written += buffered
        shard_i += 1
        buf, buffered = {}, 0
        if verbose:
            print(f"[extract]   shard {fname} ({written + start}/{n_total})")

    for batch in make_batch_iter(start):
        n = len(next(iter(batch.values())))
        for k, v in batch.items():
            buf.setdefault(k, []).append(np.asarray(v))
        buffered += n
        if buffered >= shard_size:
            flush()
    flush()

    v = verify_manifest(out_dir)
    return {"n_written": written, "n_total_on_disk": v["n_verified_samples"],
            "complete": v["n_verified_samples"] >= n_total,
            "shards": len(m["shards"]), "verified": v["ok"]}


def load_extracted(out_dir: str, keys=None, *, verify: bool = True) -> dict[str, np.ndarray]:
    """Concatenate all shards back in manifest order.

    `verify=True` (default) refuses to return data if any shard fails its checksum —
    silently loading a corrupt shard is exactly the failure this module exists to
    prevent.
    """
    m = load_manifest(out_dir)
    if m is None:
        raise FileNotFoundError(f"no manifest in {out_dir}")
    if verify:
        v = verify_manifest(out_dir, m)
        if not v["ok"]:
            raise RuntimeError(
                f"shard verification failed in {out_dir}: missing={v['missing']} "
                f"corrupt={v['corrupt']}. Re-run extraction (it will resume from "
                f"the last good shard).")
    out: dict[str, list] = {}
    for sh in m["shards"]:
        with np.load(os.path.join(out_dir, sh["file"])) as z:
            names = keys if keys is not None else list(z.keys())
            for k in names:
                out.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# ---------------------------------------------------------------------------
# torch convenience wrapper
# ---------------------------------------------------------------------------

def per_class_quota_indices(labels, quota: int, *, seed: int = 42) -> np.ndarray:
    """Deterministic index list with at most `quota` samples per class, sorted so
    the extraction order is stable across sessions (required for resume)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    picks = []
    for y in np.unique(labels):
        idx = np.where(labels == y)[0]
        k = min(quota, len(idx))
        picks.append(rng.choice(idx, k, replace=False))
    return np.sort(np.concatenate(picks))


def extract_dataset(dataset, model, device, out_dir, *, provenance,
                    indices=None, shard_size=2000, batch_size=64, num_workers=2,
                    resume=True, capture_embeddings=True, verbose=True):
    """Extract from a torch dataset. Order is fixed by `indices` (or 0..N-1) and the
    loader uses shuffle=False, so resume is exact.

    Batch size backs off automatically on CUDA OOM (§3.1: no A100 assumption).
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from scipy.special import softmax as scipy_softmax

    from pcc.extract.backbones import _FcInputCapture

    idx = np.arange(len(dataset)) if indices is None else np.asarray(indices)
    n_total = len(idx)

    def make_batch_iter(start: int):
        sub = Subset(dataset, idx[start:].tolist())
        bs = batch_size
        while True:
            try:
                loader = DataLoader(sub, batch_size=bs, shuffle=False,
                                    num_workers=num_workers)
                cap = _FcInputCapture(model) if capture_embeddings else None
                model.eval()
                with torch.no_grad():
                    for images, labels in loader:
                        out = model(images.to(device))
                        logits = out.detach().cpu().numpy().astype(np.float64)
                        batch = {"logits": logits,
                                 "softmax": scipy_softmax(logits, axis=1),
                                 "labels": np.asarray(labels).astype(int)}
                        if cap is not None:
                            batch["embeddings"] = cap.buffer.cpu().numpy().astype(np.float32)
                        yield batch
                if cap is not None:
                    cap.remove()
                return
            except RuntimeError as e:  # pragma: no cover - needs a real GPU
                if "out of memory" in str(e).lower() and bs > 8:
                    torch.cuda.empty_cache()
                    bs //= 2
                    if verbose:
                        print(f"[extract] CUDA OOM -> retrying at batch_size={bs}")
                    continue
                raise

    return extract_sharded(make_batch_iter, n_total, out_dir,
                           provenance=provenance, shard_size=shard_size,
                           resume=resume, verbose=verbose)
