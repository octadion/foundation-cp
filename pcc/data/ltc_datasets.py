"""LTC dataset loaders + released-score loader (rewritten per AGENTS.md §0.1 —
faithful to `tiffanyding/long-tail-conformal`, NOT imported from it).

Only what the Phase-0 checkpoint gate and extraction need: the exact TEST
transform, deterministic (shuffle=False) val datasets, and a loader for the
released softmax/label `.npy` files.

Faithfulness anchors (from LTC `train_models/train.py`):
- test transform: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(IN stats)
- iNaturalist-2018 val order = order of `val2018.json["images"]`; label =
  `annotations[i]["category_id"]` in 0..8141 (species). NO ambiguity in the
  class-index convention here — the model was trained on these ids directly.
- released files: `{model}-{dataset}-model_{split}_{softmax,labels}.npy`,
  `{dataset}_train_labels.npy`, `{model}` in {best, last-epoch, double-dip}.
"""

from __future__ import annotations

import json
import os

import numpy as np

# ImageNet normalization — LTC uses these for BOTH plantnet and inaturalist test.
IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 256
CROP_SIZE = 224

NUM_CLASSES = {"plantnet": 1081, "plantnet-trunc": 330,
               "inaturalist": 8142, "inaturalist-trunc": 857}


def test_transform():
    """The exact LTC val/test transform. Imported lazily so numpy-only analysis
    steps don't require torchvision."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE),
        transforms.CenterCrop(size=CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IN_MEAN, std=IN_STD),
    ])


class INaturalist2018Val:
    """Deterministic iNaturalist-2018 val dataset (json order). `root` is the dir
    containing the image tree; `ann_file` is val2018.json. Yields (image, label)
    with label = category_id (0..8141). Matches LTC's `iNaturalist.__getitem__`
    (path = root + file_name).
    """

    def __init__(self, root: str, ann_file: str, transform=None,
                 subset_indices=None):
        from torchvision.datasets.folder import default_loader

        with open(ann_file) as f:
            ann = json.load(f)
        self.imgs = [a["file_name"] for a in ann["images"]]
        if "annotations" in ann:
            self.labels = [a["category_id"] for a in ann["annotations"]]
        else:
            self.labels = [0] * len(self.imgs)
        self.root = root if root.endswith("/") else root + "/"
        self.transform = transform
        self._loader = default_loader
        # allow evaluating on a deterministic subset without touching all images
        self.index = list(range(len(self.imgs))) if subset_indices is None \
            else list(subset_indices)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        idx = self.index[i]
        img = self._loader(self.root + self.imgs[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def plantnet_val(root: str, split: str = "val", transform=None):
    """Pl@ntNet-300K split as a torchvision ImageFolder over
    `{root}/images/{split}`.

    CAUTION (see reports/phase0_checkpoint_gate.md, G2): ImageFolder assigns
    class indices by sorted folder name. If LTC's `PlantNet` class used a
    different species-id -> index convention, per-column comparison (NN matching,
    G2) will be misaligned even for the correct checkpoint, while accuracy (G1),
    true-prob curve (G3) and label-multiset (G4) stay valid because they are
    invariant to a consistent relabeling. Print `class_to_idx` and sanity-check
    accuracy before trusting G2 on Pl@ntNet.
    """
    from torchvision.datasets import ImageFolder

    path = os.path.join(root, "images", split)
    return ImageFolder(path, transform=transform)


def load_released_scores(folder: str, dataset: str, split: str = "val",
                         model_type: str = "best"):
    """Load released softmax + labels. Returns (softmax [N,C] float, labels [N])."""
    base = f"{folder}/{model_type}-{dataset}-model_{split}"
    softmax = np.load(base + "_softmax.npy")
    labels = np.load(base + "_labels.npy")
    return softmax, labels


def load_released_train_labels(folder: str, dataset: str):
    return np.load(f"{folder}/{dataset}_train_labels.npy")
