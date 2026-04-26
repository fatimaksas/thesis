#!/usr/bin/env python3
import argparse
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_FILE_HINTS = ["label", "ground", "meta", "train", "annotation"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_zip(zip_path: Path, destination: Path) -> None:
    ensure_dir(destination)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination)


def find_zip(data_root: Path, contains: List[str]) -> Optional[Path]:
    zips = sorted(data_root.glob("*.zip"))
    for z in zips:
        name = z.name.lower()
        if all(token in name for token in contains):
            return z
    return None


def discover_image_files(root: Path) -> Dict[str, Path]:
    mapping = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            mapping[p.stem] = p
    return mapping


def normalize_label(raw_label: str) -> Optional[int]:
    value = str(raw_label).strip().lower()
    if value in {"", "nan", "none", "null"}:
        return None

    positive_tokens = {
        "1",
        "true",
        "yes",
        "positive",
        "malignant",
        "cancer",
        "tumor",
        "pneumonia",
        "abnormal",
    }
    negative_tokens = {
        "0",
        "false",
        "no",
        "negative",
        "benign",
        "normal",
        "healthy",
    }

    if value in positive_tokens:
        return 1
    if value in negative_tokens:
        return 0

    # Best effort token matching for longer labels.
    if any(token in value for token in ["malig", "cancer", "tumor", "pneumonia", "positive", "abnormal"]):
        return 1
    if any(token in value for token in ["benign", "normal", "healthy", "negative"]):
        return 0

    # Numeric fallback.
    try:
        num = float(value)
        return 1 if num > 0 else 0
    except ValueError:
        return None


def guess_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["isic_id", "image_id", "image", "image_name", "id", "filename", "file_name", "path"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    # heuristic: object-like column likely id/path
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


def guess_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["target", "label", "diagnosis", "class", "y", "malignant", "cancer"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    # fallback: binary numeric col if present
    for c in df.columns:
        values = pd.Series(df[c]).dropna().unique()
        if 0 < len(values) <= 5:
            return c
    return None


def read_label_table(csv_path: Path) -> pd.DataFrame:
    if csv_path.suffix.lower() == ".tsv":
        return pd.read_csv(csv_path, sep="\t")
    return pd.read_csv(csv_path)


def isic_metadata(data_root: Path, image_root: Path) -> pd.DataFrame:
    gt = data_root / "ISIC_2024_Training_GroundTruth.csv"
    if not gt.exists():
        raise FileNotFoundError(f"Missing ISIC ground truth CSV: {gt}")

    df = pd.read_csv(gt)
    id_col = guess_id_column(df)
    label_col = guess_label_column(df)
    if not id_col or not label_col:
        raise ValueError(f"Unable to find id/label columns in {gt}")

    image_map = discover_image_files(image_root)
    rows = []
    for _, row in df.iterrows():
        image_id = str(row[id_col])
        img_path = image_map.get(Path(image_id).stem)
        if not img_path:
            continue
        raw = row[label_col]
        label = normalize_label(raw)
        if label is None:
            continue
        rows.append(
            {
                "image_path": str(img_path.resolve()),
                "label": int(label),
                "source_dataset": "isic",
                "original_label": str(raw),
            }
        )
    return pd.DataFrame(rows)


def infer_label_from_path(path: Path) -> Optional[int]:
    parts = [p.lower() for p in path.parts]
    joined = " ".join(parts)
    if any(t in joined for t in ["malignant", "cancer", "tumor", "pneumonia", "positive"]):
        return 1
    if any(t in joined for t in ["benign", "normal", "healthy", "negative"]):
        return 0
    return None


def generic_dataset_metadata(dataset_root: Path, source_name: str) -> pd.DataFrame:
    # Try to discover label table first.
    label_files = []
    for ext in ("*.csv", "*.tsv"):
        label_files.extend(dataset_root.rglob(ext))
    label_files = [
        p
        for p in label_files
        if any(hint in p.name.lower() for hint in LABEL_FILE_HINTS)
    ]

    image_map = discover_image_files(dataset_root)
    if label_files:
        for label_file in sorted(label_files):
            try:
                df = read_label_table(label_file)
                id_col = guess_id_column(df)
                label_col = guess_label_column(df)
                if not id_col or not label_col:
                    continue
                rows = []
                for _, row in df.iterrows():
                    image_id = str(row[id_col])
                    img_path = image_map.get(Path(image_id).stem)
                    if not img_path:
                        continue
                    raw = row[label_col]
                    label = normalize_label(raw)
                    if label is None:
                        continue
                    rows.append(
                        {
                            "image_path": str(img_path.resolve()),
                            "label": int(label),
                            "source_dataset": source_name,
                            "original_label": str(raw),
                        }
                    )
                if rows:
                    return pd.DataFrame(rows)
            except Exception:
                continue

    # Fallback to folder-name inferred labels.
    rows = []
    for img in dataset_root.rglob("*"):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label_from_path(img)
        if label is None:
            continue
        rows.append(
            {
                "image_path": str(img.resolve()),
                "label": int(label),
                "source_dataset": source_name,
                "original_label": "inferred_from_folder",
            }
        )
    return pd.DataFrame(rows)


def validate_label_compatibility(meta: pd.DataFrame) -> None:
    if meta.empty:
        raise ValueError("Unified metadata is empty after parsing labels.")
    unique_labels = sorted(meta["label"].unique().tolist())
    if len(unique_labels) < 2:
        raise ValueError(f"Expected at least two classes for training, found labels: {unique_labels}")

    per_source = meta.groupby("source_dataset")["label"].nunique().to_dict()
    incompatible = [source for source, n in per_source.items() if n < 2]
    if incompatible:
        raise ValueError(
            f"Incompatible sources found (need at least 2 classes per source): {incompatible}"
        )


def can_stratify(keys: pd.Series) -> bool:
    counts = keys.value_counts()
    return bool(len(counts) > 1 and counts.min() >= 2)


def stratified_split(meta: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = meta.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    strat_key = meta["label"].astype(str) + "__" + meta["source_dataset"].astype(str)

    strat1 = None
    if can_stratify(strat_key):
        strat1 = strat_key
    elif can_stratify(meta["label"]):
        strat1 = meta["label"]

    train_val, test = train_test_split(
        meta,
        test_size=0.10,
        random_state=seed,
        stratify=strat1,
    )

    train_val_key = train_val["label"].astype(str) + "__" + train_val["source_dataset"].astype(str)
    strat2 = None
    if can_stratify(train_val_key):
        strat2 = train_val_key
    elif can_stratify(train_val["label"]):
        strat2 = train_val["label"]

    train, val = train_test_split(
        train_val,
        test_size=(0.10 / 0.90),
        random_state=seed,
        stratify=strat2,
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        return self.transform(img), int(row["label"]), row["source_dataset"]


class HybridCNNViT(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if pretrained:
            cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            cnn = models.resnet18(weights=None)
            vit = models.vit_b_16(weights=None)

        cnn_out = cnn.fc.in_features
        cnn.fc = nn.Identity()

        vit_out = vit.heads.head.in_features
        vit.heads = nn.Identity()

        self.cnn = cnn
        self.vit = vit
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out + vit_out, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(combined)


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    batch_size: int
    patience: int
    device: str


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def compute_class_weights(labels: pd.Series) -> torch.Tensor:
    unique = sorted(pd.Series(labels).dropna().astype(int).unique().tolist())
    expected = list(range(len(unique)))
    if unique != expected:
        raise ValueError(
            f"Labels must be contiguous integers starting at 0, got {unique}."
        )
    counts = labels.value_counts().sort_index()
    total = counts.sum()
    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return torch.tensor([weights[i] for i in expected], dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    y_true, y_pred, sources = [], [], []
    with torch.no_grad():
        for x, y, src in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())
            sources.extend(list(src))

    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    per_dataset = {}
    df = pd.DataFrame({"source": sources, "y_true": y_true, "y_pred": y_pred})
    for source, grp in df.groupby("source"):
        per_dataset[source] = {
            "accuracy": accuracy_score(grp["y_true"], grp["y_pred"]),
            "f1_macro": f1_score(grp["y_true"], grp["y_pred"], average="macro"),
        }

    return {"overall": overall, "per_dataset": per_dataset}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    cfg: TrainConfig,
) -> nn.Module:
    device = torch.device(cfg.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= max(len(val_loader.dataset), 1)

        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def build_unified_metadata(data_root: Path, extracted_root: Path) -> pd.DataFrame:
    breast_zip = find_zip(data_root, ["breast"])
    lungs_zip = find_zip(data_root, ["lung"])
    isic_zip = find_zip(data_root, ["isic", "input"])

    if not all([breast_zip, lungs_zip, isic_zip]):
        raise FileNotFoundError(
            "Could not find required zip files in data root. Expected names containing: breast, lung, and isic+input."
        )

    breast_dir = extracted_root / "breast_images"
    lungs_dir = extracted_root / "lungs_images"
    isic_dir = extracted_root / "isic_images"

    extract_zip(breast_zip, breast_dir)
    extract_zip(lungs_zip, lungs_dir)
    extract_zip(isic_zip, isic_dir)

    isic_meta = isic_metadata(data_root, isic_dir)
    breast_meta = generic_dataset_metadata(breast_dir, "breast")
    lungs_meta = generic_dataset_metadata(lungs_dir, "lungs")

    meta = pd.concat([isic_meta, breast_meta, lungs_meta], ignore_index=True)
    meta = meta.drop_duplicates(subset=["image_path", "source_dataset"]).reset_index(drop=True)
    validate_label_compatibility(meta)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid CNN+ViT model on combined breast/lungs/ISIC datasets.")
    parser.add_argument("--data-root", required=True, help="Absolute path to thesis_project/data")
    parser.add_argument("--output-dir", required=True, help="Absolute path to output artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained backbones")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root_input = Path(args.data_root).expanduser()
    if not data_root_input.is_absolute():
        raise ValueError("--data-root must be an absolute path")
    data_root = data_root_input.resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    extracted_root = ensure_dir(output_dir / "extracted")

    set_seed(args.seed)

    metadata = build_unified_metadata(data_root, extracted_root)
    metadata.to_csv(output_dir / "metadata_unified.csv", index=False)

    train_df, val_df, test_df = stratified_split(metadata, seed=args.seed)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    train_tf, eval_tf = make_transforms(args.image_size)

    train_ds = ImageDataset(train_df, train_tf)
    val_ds = ImageDataset(val_df, eval_tf)
    test_ds = ImageDataset(test_df, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = compute_class_weights(train_df["label"])

    model = HybridCNNViT(num_classes=2, pretrained=not args.no_pretrained)
    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
    )

    model = train_model(model, train_loader, val_loader, class_weights, cfg)

    device = torch.device(args.device)
    model.to(device)
    metrics = evaluate(model, test_loader, device)

    torch.save(model.state_dict(), output_dir / "hybrid_cnn_vit.pt")
    with open(output_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved outputs:")
    print(f"- {output_dir / 'metadata_unified.csv'}")
    print(f"- {output_dir / 'train.csv'}")
    print(f"- {output_dir / 'val.csv'}")
    print(f"- {output_dir / 'test.csv'}")
    print(f"- {output_dir / 'hybrid_cnn_vit.pt'}")
    print(f"- {output_dir / 'metrics_test.json'}")


if __name__ == "__main__":
    main()
