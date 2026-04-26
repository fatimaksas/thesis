"""
CNN-ViT Hybrid Model
====================

Architecture
------------
1. **CNN Backbone** (EfficientNet-B0 by default, pretrained on ImageNet)
   Extracts rich spatial feature maps of shape (B, C_cnn, H', W').

2. **Token Projection**
   The spatial feature map is flattened into a sequence of tokens:
       (B, C_cnn, H', W')  →  (B, H'×W', embed_dim)
   A [CLS] token is prepended and learnable position embeddings are added.

3. **Transformer Encoder**
   N stacked Transformer blocks (multi-head self-attention + feed-forward)
   capture long-range dependencies between spatial tokens.

4. **Classification Head**
   The [CLS] token representation is fed through a two-layer MLP to produce
   class logits.

Usage
-----
    from src.models import CNNViTHybrid
    model = CNNViTHybrid()
    logits = model(images)   # images: (B, 3, 224, 224)
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm is required for the CNN backbone. Run: pip install timm"
    ) from e


# ─── Multi-Head Self-Attention ────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.qkv     = nn.Linear(embed_dim, embed_dim * 3)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)      # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)               # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        return self.proj(out)


# ─── Transformer Encoder Block ────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (LayerNorm before attention and FFN)."""

    def __init__(
        self,
        embed_dim:  int,
        num_heads:  int,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
        attn_drop:  float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─── CNN-ViT Hybrid ───────────────────────────────────────────────────────────

class CNNViTHybrid(nn.Module):
    """
    CNN + Vision Transformer (ViT) hybrid classifier.

    Parameters
    ----------
    num_classes     : number of output classes (default from config)
    backbone_name   : timm model name for the CNN backbone
    pretrained      : load ImageNet pretrained backbone weights
    freeze_backbone : if True, backbone weights are not updated during training
    embed_dim       : transformer embedding dimension
    num_heads       : number of attention heads
    num_layers      : number of Transformer blocks
    mlp_ratio       : FFN hidden-dim expansion ratio
    dropout         : dropout rate
    attn_dropout    : attention dropout rate
    """

    def __init__(
        self,
        num_classes:     int   = cfg.NUM_CLASSES,
        backbone_name:   str   = cfg.CNN_BACKBONE,
        pretrained:      bool  = cfg.PRETRAINED,
        freeze_backbone: bool  = cfg.FREEZE_BACKBONE,
        embed_dim:       int   = cfg.EMBED_DIM,
        num_heads:       int   = cfg.NUM_HEADS,
        num_layers:      int   = cfg.NUM_TRANSFORMER_LAYERS,
        mlp_ratio:       float = cfg.MLP_RATIO,
        dropout:         float = cfg.DROPOUT,
        attn_dropout:    float = cfg.ATTENTION_DROPOUT,
    ) -> None:
        super().__init__()

        # ── CNN Backbone ──────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,   # return feature maps, not logits
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Determine the number of channels in the last feature map
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE)
            feats = self.backbone(dummy)
            last_feat = feats[-1]               # (1, C_cnn, H', W')
            cnn_channels = last_feat.shape[1]
            num_patches  = last_feat.shape[2] * last_feat.shape[3]

        # ── Token Projection ──────────────────────────────────────────────
        # Project CNN feature vectors to transformer embedding dimension
        self.token_proj = nn.Linear(cnn_channels, embed_dim)

        # [CLS] token and positional embeddings
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # ── Transformer Encoder ───────────────────────────────────────────
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim  = embed_dim,
                    num_heads  = num_heads,
                    mlp_ratio  = mlp_ratio,
                    dropout    = dropout,
                    attn_drop  = attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification Head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) image tensor

        Returns
        -------
        logits : (B, num_classes)
        """
        B = x.shape[0]

        # 1. CNN feature extraction
        feature_maps = self.backbone(x)
        feat = feature_maps[-1]             # (B, C_cnn, H', W')

        # 2. Flatten spatial dimensions → token sequence
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B, H*W, C)
        tokens = self.token_proj(tokens)            # (B, H*W, embed_dim)

        # 3. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, N+1, embed_dim)

        # 4. Add positional embedding
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # 5. Transformer encoder
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # 6. Classify from [CLS] token
        cls_out = tokens[:, 0]              # (B, embed_dim)
        logits  = self.head(cls_out)        # (B, num_classes)

        return logits

    # ── Utility ───────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
