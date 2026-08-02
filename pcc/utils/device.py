"""Device + memory helpers. GPU is not guaranteed to be an A100 (AGENTS.md §3.1):
code must run on L4/V100/T4 with automatic batch-size backoff. Do not hardcode
memory assumptions.
"""

from __future__ import annotations


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gpu_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def autobatch(fn, batch_sizes=(256, 128, 64, 32, 16, 8)):
    """Call fn(batch_size) trying successively smaller sizes on CUDA OOM.

    Use for forward passes so extraction survives on smaller GPUs without a code
    change. Returns fn's result at the first size that fits.
    """
    import torch

    last_err = None
    for bs in batch_sizes:
        try:
            return fn(bs)
        except RuntimeError as e:  # pragma: no cover - depends on GPU
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                last_err = e
                continue
            raise
    raise RuntimeError(f"autobatch exhausted {batch_sizes}") from last_err
