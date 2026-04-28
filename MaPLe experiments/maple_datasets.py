"""
MaPLe Dataset Loader with Base-to-New Generalization Support
=============================================================

Extends the CoOp dataset loader with support for:
1. Base-to-New class splits for generalization evaluation
2. Dataset-specific split ratios (following MaPLe paper)

Base-to-New Evaluation Protocol:
- Split dataset classes in half: Base (seen) and New (unseen)
- Train on Base classes only
- Evaluate on both Base and New classes separately
- Report: Base Acc, New Acc, and Harmonic Mean (H-Mean)
"""

import json
import os
import random
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset


# Dataset Registry (same as CoOp)
DATASET_INFO = OrderedDict(
    caltech101=dict(
        dir_name="caltech-101",
        split_json="split_zhou_Caltech101.json",
        image_subdir="101_ObjectCategories",
        n_classes=100,  # Actually 101, but we use CoOp's setup
    ),
    oxfordpets=dict(
        dir_name="oxford_pets",
        split_json="split_zhou_OxfordPets.json",
        image_subdir="images",
        n_classes=37,
    ),
    stanfordcars=dict(
        dir_name="stanford_cars",
        split_json="split_zhou_StanfordCars.json",
        image_subdir="",
        n_classes=196,
    ),
    flowers102=dict(
        dir_name="oxford_flowers",
        split_json="split_zhou_OxfordFlowers.json",
        image_subdir="jpg",
        n_classes=102,
    ),
    food101=dict(
        dir_name="food-101",
        split_json="split_zhou_Food101.json",
        image_subdir="images",
        n_classes=101,
    ),
    fgvcaircraft=dict(
        dir_name="fgvc_aircraft",
        split_json="split_zhou_FGVCAircraft.json",
        image_subdir="images",
        n_classes=100,
    ),
    sun397=dict(
        dir_name="sun397",
        split_json="split_zhou_SUN397.json",
        image_subdir="SUN397",
        n_classes=397,
    ),
    dtd=dict(
        dir_name="dtd",
        split_json="split_zhou_DescribableTextures.json",
        image_subdir="images",
        n_classes=47,
    ),
    eurosat=dict(
        dir_name="eurosat",
        split_json="split_zhou_EuroSAT.json",
        image_subdir="2750",
        n_classes=10,
    ),
    ucf101=dict(
        dir_name="ucf101",
        split_json="split_zhou_UCF101.json",
        image_subdir="UCF-101-midframes",
        n_classes=101,
    ),
)


# Base-to-new class splits from the MaPLe protocol.
# These splits are deterministic for reproducibility.
def get_base_new_split(dataset_name: str, seed: int = 1) -> Tuple[List[int], List[int]]:
    """
    Get deterministic base-new class split for a dataset.

    Args:
        dataset_name: Name of the dataset
        seed: Random seed for split

    Returns:
        base_indices: List of class indices for base (training) set
        new_indices: List of class indices for new (evaluation) set
    """
    name = dataset_name.lower().replace("-", "").replace("_", "")
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_classes = DATASET_INFO[name]["n_classes"]
    n_base = n_classes // 2

    # Use a fixed seed for reproducible splits.
    rng = random.Random(seed)
    indices = list(range(n_classes))
    rng.shuffle(indices)

    base_indices = sorted(indices[:n_base])
    new_indices = sorted(indices[n_base:])

    return base_indices, new_indices


# Dataset classes.
class MaPLeSplitDataset(Dataset):
    """
    Dataset class that supports filtering by class indices for base-to-new evaluation.
    """
    def __init__(
        self,
        items: List[Tuple[str, int, str]],
        image_dir: str,
        transform=None,
        class_filter: Optional[List[int]] = None,
        remap_labels: bool = True,
    ):
        """
        Args:
            items: List of (relative_path, label, classname) tuples
            image_dir: Root directory for images
            transform: Image transform (e.g., CLIP preprocess)
            class_filter: Optional list of class indices to include
            remap_labels: If True, remap labels to 0...N-1 after filtering
        """
        self.image_dir = image_dir
        self.transform = transform
        self.remap_labels = remap_labels

        # Filter items by class when requested.
        if class_filter is not None:
            class_set = set(class_filter)
            items = [item for item in items if int(item[1]) in class_set]

        self.items = items

        # Build a class mapping.
        label_to_name = {}
        for _, lbl, cname in items:
            label_to_name[int(lbl)] = cname

        # Sort by original label.
        sorted_labels = sorted(label_to_name.keys())
        self.original_classes = [label_to_name[k] for k in sorted_labels]

        # Create label remapping when needed.
        if remap_labels and class_filter is not None:
            self.label_map = {old: new for new, old in enumerate(sorted_labels)}
        else:
            self.label_map = {k: k for k in sorted_labels}

        self.classes = self.original_classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        relpath, label, _classname = self.items[idx]
        img_path = os.path.join(self.image_dir, relpath)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Remap labels when needed.
        label = self.label_map[int(label)]
        return img, label

    def get_classnames(self) -> List[str]:
        return self.classes


# Few-shot sampling.
def sample_few_shot(items: List, num_shots: int, seed: int = 1) -> List:
    """
    Sample num_shots examples per class.

    Args:
        items: List of (path, label, classname) tuples
        num_shots: Number of examples per class
        seed: Random seed for sampling

    Returns:
        Sampled items list
    """
    if num_shots <= 0:
        return items

    rng = random.Random(seed)
    class_to_items = {}
    for item in items:
        _, lbl, _ = item
        lbl = int(lbl)
        class_to_items.setdefault(lbl, []).append(item)

    sampled = []
    for lbl in sorted(class_to_items.keys()):
        pool = class_to_items[lbl].copy()
        rng.shuffle(pool)
        sampled.extend(pool[:num_shots])

    return sampled


# Public API.
def read_split_json(json_path: str) -> Tuple[List, List, List]:
    """Read CoOp-style split JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]


def build_maple_loader(
    dataset_name: str,
    data_root: str,
    split: str,
    preprocess,
    batch_size: int = 32,
    num_workers: int = 2,
    num_shots: int = -1,
    seed: int = 1,
    class_filter: Optional[List[int]] = None,
    remap_labels: bool = True,
) -> Tuple[DataLoader, List[str], int]:
    """
    Build DataLoader for MaPLe experiments.

    Args:
        dataset_name: Dataset name (e.g., "eurosat", "caltech101")
        data_root: Path to dataset root directory
        split: "train", "val", or "test"
        preprocess: CLIP preprocess transform
        batch_size: Batch size
        num_workers: Number of data loader workers
        num_shots: K-shot per class (-1 for all)
        seed: Random seed
        class_filter: List of class indices to include (for base/new split)
        remap_labels: Remap labels to 0...N-1 after filtering

    Returns:
        (DataLoader, classnames, n_classes)
    """
    name = dataset_name.lower().replace("-", "").replace("_", "")
    if name not in DATASET_INFO:
        supported = ", ".join(DATASET_INFO.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {supported}")

    info = DATASET_INFO[name]
    ds_dir = os.path.join(data_root, info["dir_name"])
    json_path = os.path.join(ds_dir, info["split_json"])
    image_dir = os.path.join(ds_dir, info["image_subdir"]) if info["image_subdir"] else ds_dir

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Split file not found: {json_path}")

    train_items, val_items, test_items = read_split_json(json_path)

    if split == "train":
        items = train_items
    elif split == "val":
        items = val_items
    elif split == "test":
        items = test_items
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    # Apply few-shot sampling only on the train split.
    if split == "train" and num_shots > 0:
        items = sample_few_shot(items, num_shots, seed=seed)

    # Create the dataset with optional class filtering.
    dataset = MaPLeSplitDataset(
        items,
        image_dir=image_dir,
        transform=preprocess,
        class_filter=class_filter,
        remap_labels=remap_labels,
    )

    is_train = split == "train"
    dataset_size = len(dataset)
    drop = is_train and (dataset_size > batch_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop,
    )

    classnames = dataset.get_classnames()
    return loader, classnames, len(classnames)


def build_base_new_loaders(
    dataset_name: str,
    data_root: str,
    preprocess,
    batch_size: int = 32,
    num_workers: int = 2,
    num_shots: int = 16,
    seed: int = 1,
) -> Dict:
    """
    Build all loaders needed for base-to-new generalization evaluation.

    Returns a dict with:
    - train_loader: Training data (base classes only)
    - base_test_loader: Test data for base classes
    - new_test_loader: Test data for new classes
    - base_classes: List of base class names
    - new_classes: List of new class names
    - all_classes: Full class list
    """
    # Get the base/new split.
    base_indices, new_indices = get_base_new_split(dataset_name, seed=seed)

    # Training loader (base classes only, few-shot)
    train_loader, base_classes, n_base = build_maple_loader(
        dataset_name, data_root, "train", preprocess,
        batch_size=batch_size, num_workers=num_workers,
        num_shots=num_shots, seed=seed,
        class_filter=base_indices, remap_labels=True,
    )

    # Base test loader (base classes, full test set)
    base_test_loader, _, _ = build_maple_loader(
        dataset_name, data_root, "test", preprocess,
        batch_size=batch_size, num_workers=num_workers,
        num_shots=-1, seed=seed,
        class_filter=base_indices, remap_labels=True,
    )

    # New test loader (new classes, full test set)
    new_test_loader, new_classes, n_new = build_maple_loader(
        dataset_name, data_root, "test", preprocess,
        batch_size=batch_size, num_workers=num_workers,
        num_shots=-1, seed=seed,
        class_filter=new_indices, remap_labels=True,
    )

    # Also load all classes for reference.
    _, all_classes, _ = build_maple_loader(
        dataset_name, data_root, "test", preprocess,
        batch_size=1, num_workers=0, num_shots=-1, seed=seed,
    )

    return {
        "train_loader": train_loader,
        "base_test_loader": base_test_loader,
        "new_test_loader": new_test_loader,
        "base_classes": base_classes,
        "new_classes": new_classes,
        "all_classes": all_classes,
        "n_base": n_base,
        "n_new": n_new,
        "base_indices": base_indices,
        "new_indices": new_indices,
    }


def get_all_dataset_names() -> List[str]:
    """Return list of supported dataset names."""
    return list(DATASET_INFO.keys())


# Quick test.
if __name__ == "__main__":
    print("Supported datasets and their class counts:")
    for name, info in DATASET_INFO.items():
        n_classes = info["n_classes"]
        base_idx, new_idx = get_base_new_split(name)
        print(f"  {name:16s}  classes={n_classes:3d}  base={len(base_idx):3d}  new={len(new_idx):3d}")
