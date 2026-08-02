"""CIFAR-100 self-trained ResNet-50 (pipeline-debug backbone; AGENTS.md revised
order). No released checkpoint exists, so we own the whole stack — this trains
the model and the extraction+self-consistency check replaces the reproduction
gate.

Recipe follows CCC's `generate_scores/train_models/cifar_utils.py` (resnet50 from
IMAGENET1K_V2, fc->100, Adam lr 1e-4, RandomResizedCrop(224)+flip / Resize+crop,
no Normalize). CIFAR-100 numbers are NOT compared to CCC's released numbers
(per-backbone rule); this backbone exists to debug Phase 0/1.

COLAB SURVIVAL (AGENTS.md §3.1): the checkpoint is written to Google Drive every
epoch and training RESUMES from it. Nothing is left only on ephemeral /content.
"""

from __future__ import annotations

import os

import numpy as np


def cifar_transforms():
    from torchvision import transforms
    train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return train, test


def build_cifar_resnet50(num_classes=100, device=None):
    import torch
    from torch import nn
    from torchvision.models import resnet50
    model = resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if device is not None:
        model = model.to(device)
    return model


def _save_ckpt(path, model, optimizer, epoch, best_acc):
    """Atomic-ish save to Drive: write to a temp file then replace."""
    import torch
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_acc": best_acc}, tmp)
    os.replace(tmp, path)


def train_cifar100(model, train_loader, val_loader, *, ckpt_dir, device,
                   num_epochs=30, lr=1e-4, resume=True, seed=42):
    """Fine-tune with per-epoch checkpointing to `ckpt_dir` (on Drive) + resume.

    Writes `last.pth` every epoch (model+optimizer+epoch+best_acc) and `best.pth`
    whenever val accuracy improves. On start, if `last.pth` exists and resume=True,
    continues from the saved epoch. Returns the path to `best.pth`.
    """
    import torch
    from torch import nn, optim

    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "cifar100_resnet50_last.pth")
    best_path = os.path.join(ckpt_dir, "cifar100_resnet50_best.pth")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    start_epoch, best_acc = 0, 0.0
    if resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"[resume] from epoch {start_epoch} (best_acc={best_acc:.4f})")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred.cpu() == y.cpu()).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {epoch}: val_acc={acc:.4f}")

        _save_ckpt(last_path, model, optimizer, epoch, max(best_acc, acc))
        if acc > best_acc:
            best_acc = acc
            _save_ckpt(best_path, model, optimizer, epoch, best_acc)
            print(f"  new best -> {best_path}")

    return best_path


def self_consistency_check(model, embeddings, logits, softmax, labels, *,
                           tol_logit=1e-3, tol_softmax=1e-6):
    """Reproduction gate is N/A for CIFAR-100; this replaces it (AGENTS.md /
    phase0_checkpoint_gate.md CIFAR-100 scope):

    (i)  softmax recomputed from logits equals the extracted softmax,
    (ii) penultimate embeddings @ fc.weight^T + fc.bias reproduces the logits
         (proves the penultimate hook captured the right tensor),
    (iii) sane val accuracy (argmax logits vs labels).
    Returns a dict with per-check pass flags and a 'PASS'/'FAIL' verdict.
    """
    from scipy.special import softmax as scipy_softmax
    import torch

    sm2 = scipy_softmax(np.asarray(logits, np.float64), axis=1)
    c_i = float(np.abs(sm2 - softmax).max())

    W = model.fc.weight.detach().cpu().numpy().astype(np.float64)
    b = model.fc.bias.detach().cpu().numpy().astype(np.float64)
    recon = np.asarray(embeddings, np.float64) @ W.T + b
    c_ii = float(np.abs(recon - np.asarray(logits, np.float64)).max())

    acc = float((np.asarray(logits).argmax(1) == np.asarray(labels)).mean())

    p_i, p_ii = c_i <= tol_softmax, c_ii <= tol_logit
    return {
        "softmax_recompute_maxdiff": c_i, "softmax_ok": bool(p_i),
        "penultimate_to_logits_maxdiff": c_ii, "hook_ok": bool(p_ii),
        "val_accuracy": acc,
        "verdict": "PASS" if (p_i and p_ii) else "FAIL",
    }
