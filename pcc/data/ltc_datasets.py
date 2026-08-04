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


# LTC's score-split names are NOT directory names. `PlantNet.split_folder` is
# `os.path.join(root, split)` with split in {train, val, test} — there is no `cal`
# directory. The released `cal` and `val` score arrays are BOTH derived from the
# `val` directory (see ltc_cal_val_indices).
SCORE_SPLIT_TO_DIR = {"cal": "val", "val": "val", "test": "test", "train": "train"}


def ltc_cal_val_indices(n_val: int, frac_val: float = 0.3):
    """Reproduce LTC's 4-way split of the `val` directory into proper-val / cal.

    From `train_models/train.py:get_dataloaders` (proper_cal branch):

        np.random.seed(0)
        num_val_samples = int(np.floor(frac_val * len(val_dataset)))
        indices = np.arange(len(val_dataset)); np.random.shuffle(indices)
        proper_val_indices = indices[:num_val_samples]   # -> released `val` scores
        proper_cal_indices = indices[num_val_samples:]   # -> released `cal` scores

    The seed is fixed, so MEMBERSHIP is exactly reconstructable (row ORDER is not —
    the loaders use shuffle=True, which is why the gate stays permutation-invariant).
    Reconstructing membership is what makes N and the label multiset (G4) a real
    check rather than a mismatch.

    Uses the legacy global RNG deliberately: `np.random.seed(0)` +
    `np.random.shuffle` must be byte-identical to theirs, so do NOT switch to
    `default_rng` here.
    """
    state = np.random.get_state()
    try:
        np.random.seed(0)
        num_val = int(np.floor(frac_val * n_val))
        idx = np.arange(n_val)
        np.random.shuffle(idx)
        return idx[:num_val], idx[num_val:]
    finally:
        np.random.set_state(state)


def plantnet_imagefolder(root: str, score_split: str = "cal", transform=None):
    """The full Pl@ntNet-300K ImageFolder backing a given SCORE split.

    LTC's `PlantNet` subclasses `torchvision.datasets.ImageFolder`, so the class
    index convention IS ImageFolder's (sorted class-folder names). That resolves
    the earlier G2 uncertainty: per-column comparison is valid on Pl@ntNet.

    `root` must be the directory that CONTAINS the split folders. Both layouts are
    accepted: `{root}/{split}` and `{root}/images/{split}`.
    """
    from torchvision.datasets import ImageFolder

    d = SCORE_SPLIT_TO_DIR.get(score_split, score_split)
    for cand in (os.path.join(root, "images", d), os.path.join(root, d)):
        if os.path.isdir(cand) and any(
            e.is_dir() for e in os.scandir(cand)
        ):
            return ImageFolder(cand, transform=transform)
    raise FileNotFoundError(
        f"No populated Pl@ntNet split directory for score split {score_split!r} "
        f"(looked for '{d}' under {root}/images and {root}). Download the dataset "
        f"from Zenodo record 5645731 first — creating an empty folder does not help."
    )


def plantnet_scored_subset(root: str, score_split: str = "cal", transform=None):
    """The exact image subset whose scores LTC released for `score_split`.

    For `cal` / `val` this applies `ltc_cal_val_indices` to the `val` directory, so
    the returned subset matches the released array's size and label multiset. For
    `test` the whole `test` directory is used.

    Returns (dataset_or_subset, labels_array, class_to_idx).
    """
    from torch.utils.data import Subset

    ds = plantnet_imagefolder(root, score_split, transform=transform)
    targets = np.asarray(ds.targets)
    if score_split in ("cal", "val"):
        proper_val, proper_cal = ltc_cal_val_indices(len(ds))
        sel = proper_cal if score_split == "cal" else proper_val
        return Subset(ds, sel.tolist()), targets[sel], ds.class_to_idx
    return ds, targets, ds.class_to_idx


def load_released_scores(folder: str, dataset: str, split: str = "val",
                         model_type: str = "best"):
    """Load released softmax + labels. Returns (softmax [N,C] float, labels [N])."""
    base = f"{folder}/{model_type}-{dataset}-model_{split}"
    softmax = np.load(base + "_softmax.npy")
    labels = np.load(base + "_labels.npy")
    return softmax, labels


def load_released_train_labels(folder: str, dataset: str):
    return np.load(f"{folder}/{dataset}_train_labels.npy")
