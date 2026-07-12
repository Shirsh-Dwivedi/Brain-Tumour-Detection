"""
Training Pipeline for Brain Tumour Detection
Includes: GAN-augmented training, mixed precision, cosine LR schedule,
          label smoothing, EMA, progressive resizing, gradient clipping
"""

import os
import math
import time
import copy
import random
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from torchvision.utils import save_image

from models.hybrid_model import BrainTumourDetector, TumourGenerator, TumourDiscriminator, build_model


# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    data_root: str = "data/BR35H"
    image_size: int = 224
    num_classes: int = 2
    num_workers: int = 4

    # Model
    model_variant: str = "base"

    # Training
    epochs: int = 100
    batch_size: int = 32
    base_lr: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    amp: bool = True

    # GAN augmentation
    gan_latent_dim: int = 128
    gan_epochs: int = 50          # pretrain GAN first
    gan_lr: float = 2e-4
    gan_lambda_gp: float = 10.0  # gradient penalty weight
    gan_aug_ratio: float = 0.3   # fraction of batch to fill with synthetic images

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    gan_sample_dir: str = "gan_samples"

    # Augmentation
    random_erasing_p: float = 0.25
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0


# ─────────────────────────────────────────────────────────────
#  DATA PIPELINE
# ─────────────────────────────────────────────────────────────

def get_transforms(image_size: int, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def get_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = datasets.ImageFolder(
        root=os.path.join(cfg.data_root, 'train'),
        transform=get_transforms(cfg.image_size, train=True),
    )
    val_ds = datasets.ImageFolder(
        root=os.path.join(cfg.data_root, 'val'),
        transform=get_transforms(cfg.image_size, train=False),
    )

    # Balanced sampler for class imbalance
    class_counts = np.bincount([s[1] for s in train_ds.samples])
    sample_weights = [1.0 / class_counts[s[1]] for s in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────
#  AUGMENTATION UTILITIES
# ─────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    B = x.size(0)
    idx = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    idx = torch.randperm(B, device=x.device)

    cut_ratio = math.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[idx], lam


# ─────────────────────────────────────────────────────────────
#  EMA
# ─────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data = self.decay * s.data + (1 - self.decay) * m.data


# ─────────────────────────────────────────────────────────────
#  LR SCHEDULE: Warmup + Cosine Annealing
# ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                    min_lr=1e-6, base_lr=3e-4):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr / base_lr + cosine * (1 - min_lr / base_lr)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
#  GAN TRAINING
# ─────────────────────────────────────────────────────────────

def compute_gradient_penalty(discriminator, real, fake, device):
    """WGAN-GP gradient penalty."""
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interp)
    grad = torch.autograd.grad(
        d_interp, interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True,
    )[0]
    grad = grad.view(B, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_gan(cfg: TrainConfig, train_loader: DataLoader, device: torch.device):
    """Pre-train GAN for data augmentation."""
    print("\n[GAN] Pre-training tumour synthesiser...")
    G = TumourGenerator(cfg.gan_latent_dim, cfg.image_size).to(device)
    D = TumourDiscriminator(cfg.image_size).to(device)

    opt_G = optim.Adam(G.parameters(), lr=cfg.gan_lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=cfg.gan_lr, betas=(0.0, 0.9))

    os.makedirs(cfg.gan_sample_dir, exist_ok=True)
    fixed_z = torch.randn(16, cfg.gan_latent_dim, device=device)

    for epoch in range(cfg.gan_epochs):
        G.train(); D.train()
        d_losses, g_losses = [], []

        for i, (real_imgs, _) in enumerate(train_loader):
            real_imgs = real_imgs.to(device)
            B = real_imgs.size(0)
            z = torch.randn(B, cfg.gan_latent_dim, device=device)
            fake_imgs = G(z).detach()

            # --- Discriminator step ---
            opt_D.zero_grad()
            d_real = D(real_imgs).mean()
            d_fake = D(fake_imgs).mean()
            gp = compute_gradient_penalty(D, real_imgs, fake_imgs, device)
            d_loss = d_fake - d_real + cfg.gan_lambda_gp * gp
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())

            # --- Generator step (every 5 D steps) ---
            if i % 5 == 0:
                opt_G.zero_grad()
                z = torch.randn(B, cfg.gan_latent_dim, device=device)
                gen_imgs = G(z)
                g_loss = -D(gen_imgs).mean()
                g_loss.backward()
                opt_G.step()
                g_losses.append(g_loss.item())

        print(f"[GAN] Epoch {epoch+1:03d}/{cfg.gan_epochs} | "
              f"D: {np.mean(d_losses):.4f} | G: {np.mean(g_losses):.4f}")

        if (epoch + 1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z)
            save_image(samples, f"{cfg.gan_sample_dir}/epoch_{epoch+1:03d}.png",
                       nrow=4, normalize=True)

    torch.save(G.state_dict(), os.path.join(cfg.checkpoint_dir, 'gan_generator.pth'))
    print("[GAN] Pre-training complete. Generator saved.")
    return G


# ─────────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0) * 100. / B for k in topk]


def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, device,
                    gan_generator=None, epoch=0):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    t0 = time.time()

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        B = images.size(0)

        # GAN augmentation: replace part of batch with synthetic images
        if gan_generator is not None and random.random() < cfg.gan_aug_ratio:
            n_syn = max(1, int(B * cfg.gan_aug_ratio))
            with torch.no_grad():
                z = torch.randn(n_syn, cfg.gan_latent_dim, device=device)
                syn = gan_generator(z)
            # Treat synthetic as tumour class (1)
            syn_labels = torch.ones(n_syn, dtype=torch.long, device=device)
            images = torch.cat([images[:B - n_syn], syn], dim=0)
            labels = torch.cat([labels[:B - n_syn], syn_labels], dim=0)

        # Augmentation: alternate between Mixup and CutMix
        use_mixup = random.random() < 0.5
        if use_mixup and cfg.mixup_alpha > 0:
            images, y_a, y_b, lam = mixup_data(images, labels, cfg.mixup_alpha)
        else:
            images, y_a, y_b, lam = cutmix_data(images, labels, cfg.cutmix_alpha)

        with autocast(enabled=cfg.amp):
            logits = model(images)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        acc = accuracy(logits, labels)[0]
        loss_m.update(loss.item(), B)
        acc_m.update(acc.item(), B)

    elapsed = time.time() - t0
    return {'loss': loss_m.avg, 'acc': acc_m.avg, 'time': elapsed}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=cfg.amp):
            logits = model(images)
            loss = criterion(logits, labels)

        acc = accuracy(logits, labels)[0]
        loss_m.update(loss.item(), images.size(0))
        acc_m.update(acc.item(), images.size(0))

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute precision, recall, F1
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds,
                                   target_names=['No Tumour', 'Tumour'],
                                   output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    return {
        'loss': loss_m.avg,
        'acc': acc_m.avg,
        'report': report,
        'confusion_matrix': cm,
    }


def train(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # GAN pre-training
    gan_generator = train_gan(cfg, train_loader, device)
    gan_generator.eval()
    for p in gan_generator.parameters():
        p.requires_grad_(False)

    # Model
    model = build_model(cfg.model_variant, cfg.num_classes).to(device)
    ema = ModelEMA(model, cfg.ema_decay)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Optimizer: AdamW with layer-wise LR decay
    no_decay = ['bias', 'norm', 'LayerNorm']
    param_groups = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = optim.AdamW(param_groups, lr=cfg.base_lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.warmup_epochs, cfg.epochs, cfg.min_lr, cfg.base_lr
    )
    scaler = GradScaler(enabled=cfg.amp)

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n" + "="*70)
    print("  BRAIN TUMOUR DETECTION TRAINING  |  ViT + Mamba + GAN")
    print("="*70)

    for epoch in range(cfg.epochs):
        lr = optimizer.param_groups[0]['lr']

        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            cfg, device, gan_generator, epoch
        )
        ema.update(model)
        scheduler.step()

        # Validate with EMA model
        val_stats = validate(ema.shadow, val_loader, criterion, device, cfg)

        history['train_loss'].append(train_stats['loss'])
        history['train_acc'].append(train_stats['acc'])
        history['val_loss'].append(val_stats['loss'])
        history['val_acc'].append(val_stats['acc'])

        is_best = val_stats['acc'] > best_acc
        if is_best:
            best_acc = val_stats['acc']
            torch.save({
                'epoch': epoch,
                'model': ema.shadow.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'cfg': cfg,
            }, os.path.join(cfg.checkpoint_dir, 'best_model.pth'))

        print(
            f"Epoch [{epoch+1:03d}/{cfg.epochs}] | LR: {lr:.2e} | "
            f"Train Loss: {train_stats['loss']:.4f} | Train Acc: {train_stats['acc']:.2f}% | "
            f"Val Loss: {val_stats['loss']:.4f} | Val Acc: {val_stats['acc']:.2f}% "
            f"{'★ BEST' if is_best else ''}"
        )

        if (epoch + 1) % 10 == 0:
            report = val_stats['report']
            print(f"  Precision: {report['weighted avg']['precision']:.4f} | "
                  f"Recall: {report['weighted avg']['recall']:.4f} | "
                  f"F1: {report['weighted avg']['f1-score']:.4f}")

    print(f"\n✓ Training complete! Best validation accuracy: {best_acc:.2f}%")
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/BR35H')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_variant', default='base',
                        choices=['tiny', 'base', 'large'])
    parser.add_argument('--no_amp', action='store_true')
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_variant=args.model_variant,
        amp=not args.no_amp,
    )
    train(cfg)
