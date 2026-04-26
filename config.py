"""
Configuration file for the CNN-ViT Hybrid model project.
All hyperparameters and path settings are defined here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

BREAST_IMAGES_ZIP  = os.path.join(DATA_DIR, "breast_images.zip")
LUNG_IMAGES_ZIP    = os.path.join(DATA_DIR, "lungs_images.zip")
ISIC_IMAGES_ZIP    = os.path.join(DATA_DIR, "ISIC_2024_Training_Input.zip")
ISIC_CSV           = os.path.join(DATA_DIR, "ISIC_2024_Training_GroundTruth.csv")

EXTRACTED_DIR      = os.path.join(DATA_DIR, "extracted")
BREAST_EXTRACT_DIR = os.path.join(EXTRACTED_DIR, "breast")
LUNG_EXTRACT_DIR   = os.path.join(EXTRACTED_DIR, "lung")
ISIC_EXTRACT_DIR   = os.path.join(EXTRACTED_DIR, "isic")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")

# ─── Image settings ───────────────────────────────────────────────────────────
IMG_SIZE    = 224   # input image size (height = width)
NUM_CHANNELS = 3    # RGB

# ─── Dataset split ratios ─────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ─── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE      = 32
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
LR_SCHEDULER    = "cosine"   # "cosine" | "step"
LR_STEP_SIZE    = 10
LR_GAMMA        = 0.1
EARLY_STOP_PATIENCE = 10
GRAD_CLIP_MAX_NORM  = 1.0   # max norm for gradient clipping

# ─── Model architecture ───────────────────────────────────────────────────────
CNN_BACKBONE      = "efficientnet_b0"  # backbone name from timm
PRETRAINED        = True               # use ImageNet pretrained weights
FREEZE_BACKBONE   = False              # fine-tune the backbone

# ViT / Transformer settings
EMBED_DIM              = 512   # transformer embedding dimension
NUM_HEADS              = 8     # multi-head attention heads
NUM_TRANSFORMER_LAYERS = 6     # number of transformer encoder blocks
MLP_RATIO              = 4.0   # ratio for the FFN hidden dim
DROPOUT                = 0.1
ATTENTION_DROPOUT      = 0.1

# ─── Classification ───────────────────────────────────────────────────────────
NUM_CLASSES    = 2                          # 0 = benign/normal, 1 = malignant
CLASS_NAMES    = ["benign", "malignant"]

# Keywords used to infer labels from folder names
BENIGN_KEYWORDS    = {"benign", "normal", "negative", "0", "non", "healthy",
                       "no_cancer", "no-cancer", "noncancer"}
MALIGNANT_KEYWORDS = {"malignant", "cancer", "positive", "1", "tumor",
                       "cancerous", "melanoma", "carcinoma"}

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = "cuda"   # overridden at runtime if CUDA is unavailable

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_INTERVAL = 10   # print metrics every N batches
SAVE_BEST_ONLY = True
