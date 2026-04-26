"""
Combined medical-image dataset loader.

Supports three data sources:
  1. breast_images.zip  – folder-organised images (class name == folder name)
  2. lungs_images.zip   – folder-organised images
  3. ISIC_2024_Training_Input.zip + ISIC_2024_Training_GroundTruth.csv

All three sources are merged into a single binary-classification dataset:
  label 0 → benign / normal
  label 1 → malignant / cancerous

Usage
-----
    from src.data import CombinedMedicalDataset, build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders()
"""

import os
import sys
import zipfile
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as cfg

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise ImportError("PyTorch is required. Run: pip install torch") from e

from .transforms import get_train_transforms, get_eval_transforms

# ─── Valid image extensions ───────────────────────────────────────────────────
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _infer_label_from_folder(folder_name: str) -> Optional[int]:
    """
    Map a folder name to a binary label using keyword lists from config.

    Returns 0 (benign), 1 (malignant), or None if the folder name is
    unrecognised.
    """
    name_lower = folder_name.lower().strip()
    if name_lower in cfg.BENIGN_KEYWORDS:
        return 0
    if name_lower in cfg.MALIGNANT_KEYWORDS:
        return 1
    # Partial-match fallback
    for kw in cfg.BENIGN_KEYWORDS:
        if kw in name_lower:
            return 0
    for kw in cfg.MALIGNANT_KEYWORDS:
        if kw in name_lower:
            return 1
    return None


def _extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip archive only if the target directory does not exist."""
    if os.path.isdir(extract_to) and os.listdir(extract_to):
        return  # already extracted
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _collect_from_folder_dataset(
    root: str,
    source_tag: str,
) -> List[Tuple[str, int, str]]:
    """
    Walk *root* and collect (image_path, label, source_tag) triples.

    Expects at least one sub-directory per class.  Sub-directories whose
    names cannot be mapped to a binary label are skipped with a warning.
    """
    samples: List[Tuple[str, int, str]] = []
    root_path = Path(root)

    # Gather all subdirectories that look like class folders
    class_dirs = [d for d in sorted(root_path.iterdir()) if d.is_dir()]

    if not class_dirs:
        # Flat directory – all images treated as unlabelled; skip
        warnings.warn(
            f"No sub-directories found in '{root}'. "
            "Expected class-named sub-folders. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return samples

    for class_dir in class_dirs:
        label = _infer_label_from_folder(class_dir.name)
        if label is None:
            # Try one level deeper (some zips have an extra wrapper folder)
            sub_dirs = [d for d in sorted(class_dir.iterdir()) if d.is_dir()]
            if sub_dirs:
                for sub in sub_dirs:
                    lbl = _infer_label_from_folder(sub.name)
                    if lbl is None:
                        warnings.warn(
                            f"Cannot infer label from folder '{sub}'. Skipping.",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    for img_path in sub.rglob("*"):
                        if img_path.suffix.lower() in _IMG_EXTS:
                            samples.append((str(img_path), lbl, source_tag))
            else:
                warnings.warn(
                    f"Cannot infer label from folder '{class_dir.name}'. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
            continue

        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in _IMG_EXTS:
                samples.append((str(img_path), label, source_tag))

    return samples


def _collect_isic_samples(
    image_dir: str,
    csv_path: str,
    source_tag: str = "isic",
) -> List[Tuple[str, int, str]]:
    """
    Load ISIC 2024 samples using the ground-truth CSV.

    The CSV must have at least:
      - a column containing the image id  (isic_id  or  image_name)
      - a column containing the binary label (target  or  label or  MEL)
    """
    df = pd.read_csv(csv_path)

    # ── Identify image-id column ──────────────────────────────────────────
    id_candidates = ["isic_id", "image_name", "image_id", "filename", "id"]
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if id_col is None:
        raise ValueError(
            f"Cannot find an image-ID column in '{csv_path}'. "
            f"Expected one of {id_candidates}. Found: {list(df.columns)}"
        )

    # ── Identify label column ─────────────────────────────────────────────
    label_candidates = ["target", "label", "MEL", "melanoma", "class", "diagnosis"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            f"Cannot find a label column in '{csv_path}'. "
            f"Expected one of {label_candidates}. Found: {list(df.columns)}"
        )

    image_dir_path = Path(image_dir)
    samples: List[Tuple[str, int, str]] = []

    for _, row in df.iterrows():
        img_id = str(row[id_col])
        label  = int(row[label_col])

        # Try common extensions
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = image_dir_path / (img_id + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # Search recursively (images may be in a sub-folder)
            matches = list(image_dir_path.rglob(img_id + ".*"))
            if matches:
                img_path = matches[0]

        if img_path is None:
            continue  # image file not found; skip

        samples.append((str(img_path), label, source_tag))

    return samples


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class CombinedMedicalDataset(Dataset):
    """
    A unified PyTorch Dataset that combines breast, lung, and ISIC skin-lesion
    images for binary classification (0 = benign, 1 = malignant).

    Parameters
    ----------
    samples : list of (image_path, label, source_tag)
    transform : torchvision transform (or None)
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        transform=None,
    ) -> None:
        self.samples   = samples
        self.transform = transform

    # ── Class helpers ─────────────────────────────────────────────────────
    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]

    @property
    def class_counts(self) -> dict:
        counts = {0: 0, 1: 0}
        for _, lbl, _ in self.samples:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    @property
    def source_counts(self) -> dict:
        counts: dict = {}
        for _, _, src in self.samples:
            counts[src] = counts.get(src, 0) + 1
        return counts

    # ── PyTorch interface ─────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, _ = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Return a black image on corrupt files instead of crashing
            image = Image.new("RGB", (cfg.IMG_SIZE, cfg.IMG_SIZE), color=0)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ─── Dataset builder ──────────────────────────────────────────────────────────

def _load_all_samples(
    breast_zip:  str = cfg.BREAST_IMAGES_ZIP,
    lung_zip:    str = cfg.LUNG_IMAGES_ZIP,
    isic_zip:    str = cfg.ISIC_IMAGES_ZIP,
    isic_csv:    str = cfg.ISIC_CSV,
    extract_dir: str = cfg.EXTRACTED_DIR,
) -> List[Tuple[str, int, str]]:
    """Extract archives (if needed) and collect all sample triples."""

    all_samples: List[Tuple[str, int, str]] = []

    # ── Breast ────────────────────────────────────────────────────────────
    if os.path.isfile(breast_zip):
        _extract_zip(breast_zip, cfg.BREAST_EXTRACT_DIR)
        samples = _collect_from_folder_dataset(cfg.BREAST_EXTRACT_DIR, "breast")
        print(f"[Dataset] Breast: {len(samples)} images loaded.")
        all_samples.extend(samples)
    else:
        warnings.warn(
            f"Breast images zip not found at '{breast_zip}'. Skipping.",
            UserWarning,
            stacklevel=2,
        )

    # ── Lung ──────────────────────────────────────────────────────────────
    if os.path.isfile(lung_zip):
        _extract_zip(lung_zip, cfg.LUNG_EXTRACT_DIR)
        samples = _collect_from_folder_dataset(cfg.LUNG_EXTRACT_DIR, "lung")
        print(f"[Dataset] Lung:   {len(samples)} images loaded.")
        all_samples.extend(samples)
    else:
        warnings.warn(
            f"Lung images zip not found at '{lung_zip}'. Skipping.",
            UserWarning,
            stacklevel=2,
        )

    # ── ISIC ──────────────────────────────────────────────────────────────
    if os.path.isfile(isic_zip) and os.path.isfile(isic_csv):
        _extract_zip(isic_zip, cfg.ISIC_EXTRACT_DIR)
        samples = _collect_isic_samples(cfg.ISIC_EXTRACT_DIR, isic_csv, "isic")
        print(f"[Dataset] ISIC:   {len(samples)} images loaded.")
        all_samples.extend(samples)
    else:
        missing = []
        if not os.path.isfile(isic_zip):
            missing.append(isic_zip)
        if not os.path.isfile(isic_csv):
            missing.append(isic_csv)
        warnings.warn(
            f"ISIC files not found: {missing}. Skipping.",
            UserWarning,
            stacklevel=2,
        )

    if not all_samples:
        raise RuntimeError(
            "No samples were loaded from any dataset. "
            "Please check the paths in config.py and ensure the data files exist."
        )

    return all_samples


def build_dataloaders(
    breast_zip:  str = cfg.BREAST_IMAGES_ZIP,
    lung_zip:    str = cfg.LUNG_IMAGES_ZIP,
    isic_zip:    str = cfg.ISIC_IMAGES_ZIP,
    isic_csv:    str = cfg.ISIC_CSV,
    batch_size:  int = cfg.BATCH_SIZE,
    num_workers: int = 4,
    train_ratio: float = cfg.TRAIN_RATIO,
    val_ratio:   float = cfg.VAL_RATIO,
    seed:        int   = cfg.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders from all three datasets.

    The combined dataset is stratified-split into:
      - 80 % train
      - 10 % validation
      - 10 % test

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    all_samples = _load_all_samples(breast_zip, lung_zip, isic_zip, isic_csv)

    labels = [s[1] for s in all_samples]

    # ── Stratified 80 / 10 / 10 split ─────────────────────────────────────
    # Step 1: hold out (val_ratio + test_ratio) = 20 % of data
    # Step 2: split the 20 % equally → 10 % val / 10 % test
    holdout_ratio = 1.0 - train_ratio  # 0.20

    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        all_samples,
        labels,
        test_size=holdout_ratio,
        stratify=labels,
        random_state=seed,
    )

    val_samples, test_samples, _, _ = train_test_split(
        temp_samples,
        temp_labels,
        test_size=val_ratio / holdout_ratio,  # 0.10 / 0.20 = 0.5
        stratify=temp_labels,
        random_state=seed,
    )

    print(
        f"\n[Split] Train: {len(train_samples)}  "
        f"Val: {len(val_samples)}  "
        f"Test: {len(test_samples)}"
    )

    # ── Build PyTorch datasets ─────────────────────────────────────────────
    train_dataset = CombinedMedicalDataset(train_samples, transform=get_train_transforms())
    val_dataset   = CombinedMedicalDataset(val_samples,   transform=get_eval_transforms())
    test_dataset  = CombinedMedicalDataset(test_samples,  transform=get_eval_transforms())

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
