"""Selective extraction from the Pl@ntNet-300K zip.

Why this exists: `train` is ~28 GB of the 31.67 GB archive, but descriptors only
need a QUOTA of images per class (e.g. 100 x 1081 = 108k of 306k images). Extracting
all of train would blow `/content` (32 GB zip + 28 GB train + val). A zip stores a
central directory, so we can list members cheaply and extract only the ones we want.

Everything here is deterministic (seeded) and idempotent (existing files are
skipped), because Colab sessions die and the extraction must be resumable.
"""

from __future__ import annotations

import os
import zipfile

import numpy as np

ARCHIVE_PREFIX = "plantnet_300K/images"


def list_split_members(zip_path: str, split: str) -> dict[str, list[str]]:
    """Class folder name -> list of member paths, read from the central directory.

    Cheap: no decompression, just the archive index.
    """
    want = f"{ARCHIVE_PREFIX}/{split}/"
    by_class: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.startswith(want) or name.endswith("/"):
                continue
            rest = name[len(want):]
            if "/" not in rest:
                continue
            cls = rest.split("/", 1)[0]
            by_class.setdefault(cls, []).append(name)
    return by_class


def quota_members(zip_path: str, split: str, quota: int, *, seed: int = 42):
    """Deterministically pick at most `quota` members per class.

    Sorted output so the extraction (and therefore the later forward pass) has a
    stable order across sessions — required for the resume in
    `pcc.extract.forward` to be exact.
    """
    by_class = list_split_members(zip_path, split)
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    per_class: dict[str, int] = {}
    for cls in sorted(by_class):
        files = sorted(by_class[cls])
        k = min(quota, len(files))
        idx = rng.choice(len(files), k, replace=False)
        chosen = [files[i] for i in sorted(idx)]
        picked.extend(chosen)
        per_class[cls] = k
    return sorted(picked), per_class


def extract_members(zip_path: str, members, dest: str, *, verbose_every: int = 5000):
    """Extract `members` under `dest`, skipping any that already exist.

    Idempotent: re-running after a killed session completes the extraction instead
    of restarting it. Returns (n_extracted, n_skipped).
    """
    os.makedirs(dest, exist_ok=True)
    n_new = n_skip = 0
    with zipfile.ZipFile(zip_path) as z:
        for i, m in enumerate(members):
            target = os.path.join(dest, m)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                n_skip += 1
            else:
                z.extract(m, path=dest)
                n_new += 1
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"  {i+1}/{len(members)} (new={n_new} skipped={n_skip})")
    return n_new, n_skip


def ensure_train_quota(zip_path: str, dest_root: str, quota: int, *, seed: int = 42,
                       verbose: bool = True):
    """Make sure `quota` train images per class exist under `dest_root`.

    `dest_root` is the directory that will contain `plantnet_300K/images/train/...`
    (i.e. pass `/content` if DATA_ROOT is `/content/plantnet_300K`).
    Returns a dict summary including the per-class counts actually available.
    """
    members, per_class = quota_members(zip_path, "train", quota, seed=seed)
    if verbose:
        short = sum(1 for v in per_class.values() if v < quota)
        print(f"train quota={quota}: {len(members)} members across {len(per_class)} classes"
              f" ({short} classes have fewer than {quota} images)")
    n_new, n_skip = extract_members(zip_path, members, dest_root,
                                    verbose_every=5000 if verbose else 0)
    if verbose:
        print(f"extracted {n_new} new, {n_skip} already present")
    return {"quota": quota, "n_members": len(members), "n_classes": len(per_class),
            "n_extracted": n_new, "n_skipped": n_skip,
            "classes_below_quota": int(sum(1 for v in per_class.values() if v < quota))}
