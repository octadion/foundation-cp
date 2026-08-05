"""One-call extraction per Pl@ntNet split, so the notebook stays a thin runner.

Which splits and what is stored, following the 2026-08-04 decision to use OUR
scores everywhere (see reports/phase0_checkpoint_gate.md):

| split         | source directory | stored            | used for |
|---------------|------------------|-------------------|----------|
| `cal`         | `images/val` (70% subset, LTC's seeded split) | logits + labels | calibration, δ_y |
| `proper_val`  | `images/val` (the other 30%) | logits + labels | model-selection-free holdout |
| `test`        | `images/test`    | logits + labels   | evaluation |
| `train_quota` | `images/train` (quota per class) | logits + labels + **embeddings** | descriptors φ(y) |

Embeddings are stored ONLY for `train_quota`: §6.3 requires descriptors to come from
training data, and cal/test embeddings would waste Drive for nothing. Everything
goes through `pcc.extract.forward.extract_dataset`, so every split is sharded,
checksummed and resumable.
"""

from __future__ import annotations

import os

import numpy as np

from pcc.data.ltc_datasets import (NUM_CLASSES, plantnet_imagefolder,
                                   plantnet_scored_subset, test_transform)
from pcc.extract.forward import extract_dataset, per_class_quota_indices

# splits whose embeddings we keep (see the table above)
EMBEDDING_SPLITS = ("train_quota",)


def build_split(data_root: str, split: str, *, quota: int | None = None,
                seed: int = 42):
    """Return (dataset, indices_or_None, n_classes) for a split name."""
    tfm = test_transform()
    if split in ("cal", "proper_val"):
        score_split = "cal" if split == "cal" else "val"
        ds, labels, _ = plantnet_scored_subset(data_root, score_split, transform=tfm)
        return ds, None, len(labels)
    if split == "test":
        ds = plantnet_imagefolder(data_root, "test", transform=tfm)
        return ds, None, len(ds)
    if split == "train_quota":
        if quota is None:
            raise ValueError("train_quota needs a quota")
        ds = plantnet_imagefolder(data_root, "train", transform=tfm)
        idx = per_class_quota_indices(np.asarray(ds.targets), quota, seed=seed)
        return ds, idx, len(idx)
    raise ValueError(f"unknown split {split!r}")


def extract_split(split: str, *, data_root: str, out_root: str, model, device,
                  ckpt_sha: str, quota: int | None = None, seed: int = 42,
                  shard_size: int = 2000, batch_size: int = 64, num_workers: int = 2,
                  under_gate_exception: bool = False, verbose: bool = True):
    """Extract one split into `{out_root}/{split}` with a checksummed manifest.

    `under_gate_exception` is recorded in the manifest provenance so that a
    downstream reader can never lose track of the fact that the checkpoint gate did
    not pass (reports/phase0_checkpoint_gate.md).
    """
    ds, idx, n = build_split(data_root, split, quota=quota, seed=seed)
    out_dir = os.path.join(out_root, split)
    keep_emb = split in EMBEDDING_SPLITS
    provenance = {
        "dataset": "plantnet", "split": split, "backbone": "resnet50_ltc",
        "ckpt_sha256": ckpt_sha, "n_classes": NUM_CLASSES["plantnet"],
        "transform": "Resize(256)+CenterCrop(224)+ToTensor+Normalize(imagenet)",
        "embeddings_stored": keep_emb, "quota": quota, "seed": seed,
        "scores_source": "OURS (not LTC released) - see reports/phase0_checkpoint_gate.md",
        "under_gate_exception": bool(under_gate_exception),
    }
    if verbose:
        print(f"[{split}] {n} images -> {out_dir}  (embeddings={keep_emb})")
    res = extract_dataset(ds, model, device, out_dir, provenance=provenance,
                          indices=idx, shard_size=shard_size,
                          batch_size=batch_size, num_workers=num_workers,
                          resume=True, capture_embeddings=keep_emb, verbose=verbose)
    res["split"] = split
    res["out_dir"] = out_dir
    res["n_expected"] = n
    return res
