"""Idempotent, resumable download of the Pl@ntNet-300K archive.

Shared by notebooks 00 and 01 so the logic cannot drift between them. It exists as a
module because the first version lived only in notebook 00, and notebook 01 then
assumed `/content/plantnet_300K.zip` was present — which it is not after a Colab
session restart, since `/content` is ephemeral by design (AGENTS.md §3.1).

Correctness notes:
- A bare `os.path.exists` check is WRONG: an interrupted aria2c leaves a PARTIAL zip
  plus a `.aria2` control file, and "the file exists" would then skip the resume and
  hand a truncated archive to the extractor.
- aria2c verifies the MD5 during the download; the byte size is checked again after.
"""

from __future__ import annotations

import os
import shutil
import subprocess

ZENODO_URL = ("https://zenodo.org/records/5645731/files/"
              "plantnet_300K.zip?download=1")
ZENODO_MD5 = "db27d149f2a6c304b887353c07021687"
ZENODO_BYTES = 31670505069


def zip_status(zip_path: str) -> dict:
    ctrl = zip_path + ".aria2"
    exists = os.path.exists(zip_path)
    size = os.path.getsize(zip_path) if exists else 0
    has_ctrl = os.path.exists(ctrl)
    complete = exists and not has_ctrl and size == ZENODO_BYTES
    return {"exists": exists, "size": size, "control_file": has_ctrl,
            "complete": complete, "expected_bytes": ZENODO_BYTES}


def ensure_zip(zip_path: str, *, url: str = ZENODO_URL, md5: str = ZENODO_MD5,
               expected_bytes: int = ZENODO_BYTES, connections: int = 16,
               verbose: bool = True) -> dict:
    """Guarantee a complete archive at `zip_path`, downloading/resuming if needed."""
    st = zip_status(zip_path)
    if st["complete"]:
        if verbose:
            print(f"zip complete: {zip_path} ({st['size']/1e9:.2f} GB)")
        return {**st, "downloaded": False}

    free = shutil.disk_usage(os.path.dirname(zip_path) or "/").free
    if free < expected_bytes * 1.05:
        raise RuntimeError(
            f"not enough space for the archive: {free/1e9:.1f} GB free, "
            f"{expected_bytes/1e9:.1f} GB needed at {zip_path}")

    if verbose:
        if st["exists"]:
            print(f"zip INCOMPLETE ({st['size']/1e9:.2f} of "
                  f"{expected_bytes/1e9:.2f} GB"
                  f"{', control file present' if st['control_file'] else ''}) -> resuming")
        else:
            print(f"downloading {expected_bytes/1e9:.2f} GB with aria2c "
                  f"({connections} connections) ...")

    subprocess.run(["apt-get", "-qq", "install", "-y", "aria2"], check=False)
    cmd = ["aria2c", f"-x{connections}", f"-s{connections}", "-k1M", "-c",
           "--auto-file-renaming=false", "--allow-overwrite=false",
           "--console-log-level=warn", "--summary-interval=30",
           f"--checksum=md5={md5}",
           "-d", os.path.dirname(zip_path) or ".",
           "-o", os.path.basename(zip_path), url]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(
            f"aria2c exit {r.returncode}. Re-run — it RESUMES from where it stopped. "
            f"Do NOT delete {zip_path}.aria2.")

    st = zip_status(zip_path)
    if st["size"] != expected_bytes:
        raise RuntimeError(f"size mismatch after download: {st['size']} != "
                           f"{expected_bytes}. Re-run to resume.")
    if verbose:
        print(f"download complete and MD5-verified ({st['size']/1e9:.2f} GB)")
    return {**st, "downloaded": True}
