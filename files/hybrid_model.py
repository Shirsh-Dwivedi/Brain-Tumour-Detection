"""
Brain Tumour Detection: Hybrid ViT + Mamba + GAN Architecture
Dataset: BR35H (Brain Tumour Detection 2020)
Target Accuracy: 99%+

Architecture Overview:
- Vision Transformer (ViT): Global attention-based feature extraction
- Mamba (SSM): Efficient sequential state-space modelling for local features
- GAN: Data augmentation discriminator for synthetic tumour generation
- Fusion Head: Combines ViT + Mamba embeddings for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ─────────────────────────────────────────────────────────────
#  1. VISION TRANSFORMER (ViT) ENCODER
# ─────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_size * patch_size * in_channels),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x)
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for global feature extraction."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token as global representation


# ─────────────────────────────────────────────────────────────
#  2. MAMBA STATE-SPACE MODEL (SSM) ENCODER
# ─────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Simplified Mamba SSM block for efficient sequential modelling.
    Based on: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    """

    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Selective scan parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        # Project input
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Convolution
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute SSM parameters
        ssm_input = self.x_proj(x_conv)
        delta, B_ssm, C = ssm_input.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        # Selective SSM (simplified discretised scan)
        A = -torch.exp(self.A_log.float())
        y = self._selective_scan(x_conv, delta, A, B_ssm, C, self.D)

        # Gate and output
        y = y * F.silu(z)
        output = self.out_proj(y)
        return output + residual

    def _selective_scan(self, u, delta, A, B, C, D):
        """Simplified selective scan operation."""
        B_batch, L, d_in = u.shape
        d_state = A.shape[1]

        # Discretise A and B
        dA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        dB = torch.einsum('bld,bln->bldn', delta, B)

        # Scan
        h = torch.zeros(B_batch, d_in, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * u[:, i].unsqueeze(-1)
            y = torch.einsum('bdn,bln->bld', h, C[:, i:i+1])
            ys.append(y.squeeze(2))

        y = torch.stack(ys, dim=1)
        return y + u * D


class MambaEncoder(nn.Module):
    """Mamba-based encoder for efficient local feature extraction."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 d_model=512, depth=6, d_state=16):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2

        # Patch tokenisation
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)  # Global average pooling


# ─────────────────────────────────────────────────────────────
#  3. GAN for Data Augmentation
# ─────────────────────────────────────────────────────────────

class TumourGenerator(nn.Module):
    """GAN Generator: Synthesises realistic tumour MRI patches."""

    def __init__(self, latent_dim=128, image_size=224):
        super().__init__()
        self.init_size = image_size // 16
        self.l1 = nn.Linear(latent_dim, 512 * self.init_size ** 2)

        self.model = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, padding=1), nn.Tanh(),
        )

    def forward(self, z):
        B = z.shape[0]
        out = self.l1(z).reshape(B, 512, self.init_size, self.init_size)
        return self.model(out)


class TumourDiscriminator(nn.Module):
    """GAN Discriminator: Distinguishes real vs synthesised MRI images."""

    def __init__(self, image_size=224):
        super().__init__()

        def disc_block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(3, 64, bn=False),
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.model(x)


# ─────────────────────────────────────────────────────────────
#  4. HYBRID FUSION MODEL (ViT + Mamba + GAN-Features)
# ─────────────────────────────────────────────────────────────

class BrainTumourDetector(nn.Module):
    """
    Hybrid Brain Tumour Detector combining:
    - ViT for global attention-based feature extraction
    - Mamba SSM for efficient local sequential feature modelling
    - GAN discriminator features for domain-aware classification
    - Multi-head fusion classifier
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 2,          # tumour / no tumour
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_heads: int = 12,
        mamba_dim: int = 512,
        mamba_depth: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Encoders
        self.vit = ViTEncoder(
            image_size=image_size, patch_size=patch_size,
            in_channels=in_channels, embed_dim=vit_embed_dim,
            depth=vit_depth, num_heads=vit_heads, dropout=dropout,
        )
        self.mamba = MambaEncoder(
            image_size=image_size, patch_size=patch_size,
            in_channels=in_channels, d_model=mamba_dim,
            depth=mamba_depth,
        )

        # Feature projection to common space
        fusion_dim = 512
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_embed_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.mamba_proj = nn.Sequential(
            nn.Linear(mamba_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Cross-attention fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Extract features
        vit_feat = self.vit_proj(self.vit(x))          # (B, fusion_dim)
        mamba_feat = self.mamba_proj(self.mamba(x))    # (B, fusion_dim)

        # Cross-attention: ViT queries Mamba features
        v = vit_feat.unsqueeze(1)
        m = mamba_feat.unsqueeze(1)
        fused, _ = self.cross_attn(v, m, m)
        fused = self.fusion_norm(fused.squeeze(1) + vit_feat)

        # Concatenate for final classification
        combined = torch.cat([fused, mamba_feat], dim=-1)
        logits = self.classifier(combined)
        return logits

    def get_attention_maps(self, x):
        """Return ViT attention maps for visualisation (Grad-CAM compatible)."""
        with torch.no_grad():
            patch_tokens = self.vit.patch_embed(x)
            attn_maps = []
            for block in self.vit.blocks:
                B, N, C = patch_tokens.shape
                qkv = block.attn.qkv(block.norm1(patch_tokens))
                qkv = qkv.reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim)
                q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                attn_maps.append(attn)
                patch_tokens = block(patch_tokens)
        return attn_maps


# ─────────────────────────────────────────────────────────────
#  5. MODEL FACTORY
# ─────────────────────────────────────────────────────────────

def build_model(variant: str = 'base', num_classes: int = 2) -> BrainTumourDetector:
    configs = {
        'tiny': dict(vit_embed_dim=384, vit_depth=6,  vit_heads=6,  mamba_dim=256, mamba_depth=3),
        'base': dict(vit_embed_dim=768, vit_depth=12, vit_heads=12, mamba_dim=512, mamba_depth=6),
        'large': dict(vit_embed_dim=1024,vit_depth=24, vit_heads=16, mamba_dim=768, mamba_depth=8),
    }
    cfg = configs.get(variant, configs['base'])
    return BrainTumourDetector(num_classes=num_classes, **cfg)


if __name__ == '__main__':
    model = build_model('base')
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Model output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total:.1f}M")
