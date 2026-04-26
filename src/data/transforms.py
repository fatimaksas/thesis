"""
Data augmentation and preprocessing transforms.

Training transforms apply standard medical-image augmentations.
Eval (validation / test) transforms apply only deterministic preprocessing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config as cfg

try:
    import torchvision.transforms as T
except ImportError as e:
    raise ImportError("torchvision is required. Run: pip install torchvision") from e


# ImageNet normalisation stats work well even for medical images fine-tuned
# from pretrained weights.
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = cfg.IMG_SIZE) -> T.Compose:
    """Augmented transforms used during training."""
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


def get_eval_transforms(img_size: int = cfg.IMG_SIZE) -> T.Compose:
    """Deterministic transforms used during validation and testing."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])
