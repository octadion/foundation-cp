"""Frozen backbone loading + single-pass logits/penultimate extraction.

For the LTC path (AGENTS.md decisions 2026-07-25): the released `.pth` is a full
`resnet50` state-dict with `fc` replaced by an `nn.Linear(2048, num_classes)`
head. We rebuild that architecture and load the state-dict. No retraining, no
fine-tuning (§1). The penultimate embedding is the 2048-d input to `fc`.

Scores are reproduced EXACTLY as LTC does (data/ltc_datasets faithfulness note):
logits -> float64 -> scipy.special.softmax(axis=1).
"""

from __future__ import annotations

import numpy as np


def load_ltc_resnet50(ckpt_path: str, num_classes: int, device=None):
    """Rebuild LTC's ResNet-50 and load released weights. Returns eval-mode model.

    Built with weights=None (the released state-dict fully overwrites them, so no
    torchvision-pretrained download is needed).
    """
    import torch
    from torch import nn
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # tolerate checkpoints saved as {"state_dict": ...} or with a prefix
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # strict=False so we can SEE mismatches instead of crashing; a real
        # mismatch here is itself a gate signal (wrong checkpoint / arch).
        print(f"[load_ltc_resnet50] missing={list(missing)[:5]} "
              f"unexpected={list(unexpected)[:5]} "
              f"(counts: {len(missing)}/{len(unexpected)})")
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


class _FcInputCapture:
    """Forward-pre-hook on model.fc: captures the (already flattened) 2048-d
    penultimate features that are fed into the classifier head."""

    def __init__(self, model):
        self.buffer = None
        self._handle = model.fc.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.buffer = inputs[0].detach()

    def remove(self):
        self._handle.remove()


def forward_logits_and_embeddings(model, dataloader, device, *,
                                  capture_embeddings=True, return_logits=False):
    """Single pass over `dataloader` (should be shuffle=False for reproducibility).

    Returns (softmax_f64 [N,C], labels [N], embeddings [N,2048] or None), or, if
    return_logits=True, (softmax, labels, embeddings, logits_f64) — logits are
    needed for the CIFAR-100 self-consistency check (penultimate @ fc -> logits).
    softmax matches LTC exactly: float64 logits -> scipy softmax.
    """
    import torch
    from scipy.special import softmax as scipy_softmax
    from tqdm.auto import tqdm

    cap = _FcInputCapture(model) if capture_embeddings else None
    logits_chunks, label_chunks, emb_chunks = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            out = model(inputs)
            logits_chunks.append(out.detach().cpu().numpy().astype(np.float64))
            label_chunks.append(np.asarray(labels))
            if cap is not None:
                emb_chunks.append(cap.buffer.cpu().numpy().astype(np.float32))
    if cap is not None:
        cap.remove()

    logits = np.concatenate(logits_chunks, axis=0)          # float64, LTC dtype
    softmax = scipy_softmax(logits, axis=1)                 # == LTC get_softmax
    labels = np.concatenate(label_chunks, axis=0).astype(int)
    embeddings = np.concatenate(emb_chunks, axis=0) if emb_chunks else None
    if return_logits:
        return softmax, labels, embeddings, logits
    return softmax, labels, embeddings
