# Thesis – CNN-ViT Hybrid for Combined Medical Image Classification

A hybrid **CNN + Vision Transformer (ViT)** model that combines three medical-imaging datasets — breast cancer, lung cancer, and ISIC 2024 skin-lesion images — for binary classification (benign vs. malignant).

---

## Project Structure

```
thesis/
├── config.py               ← all hyperparameters and file paths
├── main.py                 ← training / evaluation entry point
├── requirements.txt
├── data/                   ← place the raw zip/csv files here
│   ├── breast_images.zip
│   ├── lungs_images.zip
│   ├── ISIC_2024_Training_Input.zip
│   └── ISIC_2024_Training_GroundTruth.csv
├── src/
│   ├── data/
│   │   ├── dataset.py      ← combined dataset loader + 80/10/10 split
│   │   └── transforms.py   ← augmentations for train/eval
│   ├── models/
│   │   └── hybrid_model.py ← CNN-ViT hybrid architecture
│   ├── training/
│   │   ├── trainer.py      ← training loop, early stopping, checkpointing
│   │   └── metrics.py      ← accuracy, F1, AUC, confusion matrix
│   └── utils/
│       └── helpers.py      ← seeding, checkpointing, plotting
├── checkpoints/            ← best model checkpoint saved here
└── results/                ← training curves and test metrics
```

---

## Model Architecture

```
Input image (B × 3 × 224 × 224)
       │
       ▼
┌─────────────────────────────────┐
│  CNN Backbone (EfficientNet-B0) │  ← pretrained on ImageNet
│  Output: (B, C, H', W')        │
└───────────────┬─────────────────┘
                │  flatten + linear projection
                ▼
        Token sequence (B, N, D)
        + [CLS] token prepended
        + Positional embeddings
                │
                ▼
┌────────────────────────────────┐
│  Transformer Encoder           │  ← 6 blocks
│  (Multi-Head Self-Attention    │
│   + Feed-Forward Network)      │
└───────────────┬────────────────┘
                │  [CLS] token
                ▼
       Classification Head (MLP)
                │
                ▼
         Logits (B × 2)
```

---

## Setup

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Place data files

Copy the four data files into the `data/` directory:

```
data/breast_images.zip
data/lungs_images.zip
data/ISIC_2024_Training_Input.zip
data/ISIC_2024_Training_GroundTruth.csv
```

The archives are extracted automatically on first run.

> **Breast / Lung zip format:** The images must be organised in class-named
> sub-folders inside the zip (e.g. `benign/img1.jpg`, `malignant/img2.jpg`).
> Common folder names like `normal`, `cancer`, `tumor`, `healthy`, etc. are
> recognised automatically.

### 3 – (Optional) Edit `config.py`

Adjust `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `CNN_BACKBONE`, etc. as
needed.

---

## Training

```bash
python main.py
```

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 50 | Number of training epochs |
| `--batch-size N` | 32 | Mini-batch size |
| `--lr F` | 1e-4 | Initial learning rate |
| `--device cpu\|cuda` | auto | Device selection |
| `--workers N` | 4 | DataLoader worker processes |
| `--freeze-backbone` | off | Freeze CNN backbone (train ViT head only) |
| `--no-pretrained` | off | Train backbone from scratch |

The best model checkpoint is saved to `checkpoints/best_model.pth`.
Training curves are saved to `results/training_history.png`.

## Evaluation

```bash
python main.py --eval-only
```

Loads `checkpoints/best_model.pth` and reports accuracy, F1, precision,
recall, AUC, and the confusion matrix on the held-out test split.

---

## Dataset Split

All three datasets are combined and stratified-split into:

| Split | Ratio |
|-------|-------|
| Train | 80 % |
| Val   | 10 % |
| Test  | 10 % |

Labels are unified to binary: **0 = benign / normal**, **1 = malignant**.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- timm ≥ 0.9
- scikit-learn, pandas, numpy, matplotlib, tqdm, einops, Pillow