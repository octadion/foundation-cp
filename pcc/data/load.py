"""One loader for both storage layouts, so the analysis notebooks do not care.

Two layouts exist for historical reasons:
- **single `.npz`** — CIFAR-100 (notebook 00a wrote `test.npz` / `train_subset.npz`)
- **sharded dir + manifest** — Pl@ntNet (notebook 01, resumable extraction)

`load_split` accepts either and always verifies what it can: sharded loads refuse to
return data if any shard checksum fails, so a corrupt shard can never be silently
folded into an analysis.
"""

from __future__ import annotations

import os

import numpy as np


def _npz_candidates(root: str, split: str):
    return [os.path.join(root, f"{split}.npz"),
            os.path.join(root, split, f"{split}.npz")]


def load_split(root: str, split: str, *, keys=None, verify: bool = True) -> dict:
    """Load one split from `{root}/{split}/` (sharded) or `{root}/{split}.npz`.

    Returns a dict of arrays. `keys` restricts what is read (e.g. skip `embeddings`
    when only scores are needed — on Pl@ntNet the train embeddings are ~375 MB).
    """
    from pcc.extract.forward import load_extracted
    from pcc.data.manifest import load_manifest

    shard_dir = os.path.join(root, split)
    if load_manifest(shard_dir) is not None:
        return load_extracted(shard_dir, keys=keys, verify=verify)

    for p in _npz_candidates(root, split):
        if os.path.exists(p):
            with np.load(p) as z:
                names = keys if keys is not None else list(z.keys())
                return {k: z[k] for k in names if k in z}

    raise FileNotFoundError(
        f"no data for split {split!r} under {root}: neither a sharded manifest at "
        f"{shard_dir} nor an .npz at {_npz_candidates(root, split)}")


def split_provenance(root: str, split: str) -> dict:
    """Manifest provenance for a sharded split ({} for a plain .npz).

    Use this to surface `under_gate_exception` and `scores_source` in reports — the
    Pl@ntNet gate did not pass, and that fact must travel with the data rather than
    living only in a document (reports/phase0_checkpoint_gate.md).
    """
    from pcc.data.manifest import load_manifest
    m = load_manifest(os.path.join(root, split))
    return (m or {}).get("provenance", {})


def softmax_from_logits(logits) -> np.ndarray:
    """float64 softmax, matching how the released scores were produced.

    Implemented in numpy rather than `scipy.special.softmax` so analysis code has one
    fewer hard dependency. Numerically equivalent: both subtract the row max before
    exponentiating, and the arithmetic is done in float64 either way.
    """
    z = np.asarray(logits, np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_scores(root: str, split: str, *, verify: bool = True):
    """(softmax, labels) for a split, computing softmax from logits if absent."""
    d = load_split(root, split, verify=verify)
    labels = np.asarray(d["labels"]).astype(int)
    if "softmax" in d:
        return np.asarray(d["softmax"], np.float64), labels
    return softmax_from_logits(d["logits"]), labels


def per_class_counts(labels, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(labels, int), minlength=n_classes)
