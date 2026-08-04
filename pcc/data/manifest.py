"""Checksummed extraction manifests (AGENTS.md §2.3, §3.1).

Colab sessions die mid-run, so an extraction must be able to answer two questions
after a restart: *what did I already write*, and *is it intact*. A manifest records
one entry per shard with its SHA256, plus the provenance needed to make the
embeddings interpretable later (backbone id, checkpoint hash, dataset/split,
preprocessing, per-class counts).

The manifest is the resume point: `resume_point()` returns the number of verified
samples already on disk, and a re-run continues from there.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import numpy as np


MANIFEST_NAME = "manifest.json"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, MANIFEST_NAME)


def load_manifest(out_dir: str) -> dict[str, Any] | None:
    p = manifest_path(out_dir)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A manifest truncated by a killed session is worthless but must not crash
        # the resume path — treat it as absent and re-verify shards from scratch.
        return None


def init_manifest(out_dir: str, *, provenance: dict[str, Any]) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    m = {"provenance": provenance, "shards": []}
    write_manifest(out_dir, m)
    return m


def write_manifest(out_dir: str, manifest: dict[str, Any]) -> str:
    """Atomic-ish write: temp file then replace, so a kill mid-write cannot leave a
    half-written manifest in place of a good one."""
    p = manifest_path(out_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, p)
    return p


def add_shard(out_dir: str, manifest: dict[str, Any], shard_file: str, *,
              n_samples: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record a completed shard and persist the manifest immediately."""
    full = os.path.join(out_dir, shard_file)
    entry = {"file": shard_file, "n_samples": int(n_samples),
             "sha256": sha256_file(full), "bytes": os.path.getsize(full)}
    if extra:
        entry.update(extra)
    manifest["shards"].append(entry)
    write_manifest(out_dir, manifest)
    return manifest


def verify_manifest(out_dir: str, manifest: dict[str, Any] | None = None,
                    *, quick: bool = False) -> dict[str, Any]:
    """Check every recorded shard exists and matches its checksum.

    `quick=True` compares byte size only (cheap sanity for a long run); the default
    recomputes SHA256, which is what you want before trusting cached embeddings.
    Returns {ok, n_verified_samples, missing, corrupt, first_bad_index}.
    """
    manifest = manifest if manifest is not None else load_manifest(out_dir)
    if manifest is None:
        return {"ok": False, "n_verified_samples": 0, "missing": [], "corrupt": [],
                "first_bad_index": 0, "reason": "no manifest"}
    missing, corrupt, verified, first_bad = [], [], 0, None
    for i, sh in enumerate(manifest.get("shards", [])):
        full = os.path.join(out_dir, sh["file"])
        if not os.path.exists(full):
            missing.append(sh["file"])
            first_bad = i if first_bad is None else first_bad
            continue
        bad = (os.path.getsize(full) != sh.get("bytes") if quick
               else sha256_file(full) != sh["sha256"])
        if bad:
            corrupt.append(sh["file"])
            first_bad = i if first_bad is None else first_bad
            continue
        if first_bad is None:
            verified += int(sh["n_samples"])
    return {"ok": not missing and not corrupt, "n_verified_samples": verified,
            "missing": missing, "corrupt": corrupt,
            "first_bad_index": len(manifest.get("shards", [])) if first_bad is None
            else first_bad}


def resume_point(out_dir: str) -> int:
    """Number of samples already written AND verified, counting only the unbroken
    prefix of shards. A corrupt shard truncates the prefix, so the re-run redoes it
    rather than silently skipping a hole.
    """
    v = verify_manifest(out_dir)
    return int(v["n_verified_samples"])


def truncate_to_valid_prefix(out_dir: str) -> dict[str, Any]:
    """Drop manifest entries at/after the first bad shard so the resume is clean."""
    m = load_manifest(out_dir)
    if m is None:
        return {"dropped": 0}
    v = verify_manifest(out_dir, m)
    keep = v["first_bad_index"]
    dropped = len(m["shards"]) - keep
    if dropped > 0:
        m["shards"] = m["shards"][:keep]
        write_manifest(out_dir, m)
    return {"dropped": dropped, "kept": keep}


def per_class_counts(labels, n_classes: int) -> list[int]:
    return np.bincount(np.asarray(labels, int), minlength=n_classes).tolist()
