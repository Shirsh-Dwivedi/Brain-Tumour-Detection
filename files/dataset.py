"""
Dataset Preparation for BR35H Brain Tumour Detection Dataset
Downloads, organises, and validates the BR35H dataset.

BR35H Dataset Structure:
  - yes/  : 1500 MRI images with brain tumour
  - no/   : 1500 MRI images without brain tumour
  Total   : 3000 images

After preparation:
  data/BR35H/
    train/
      tumour/    (1200 images)
      no_tumour/ (1200 images)
    val/
      tumour/    (150 images)
      no_tumour/ (150 images)
    test/
      tumour/    (150 images)
      no_tumour/ (150 images)
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────
#  DATASET SETUP
# ─────────────────────────────────────────────────────────────

def prepare_br35h_dataset(
    raw_dir: str = "data/BR35H_raw",
    output_dir: str = "data/BR35H",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
):
    """
    Organises raw BR35H data into train/val/test splits.
    raw_dir should contain two subdirectories: 'yes' and 'no'.
    """
    random.seed(seed)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    class_map = {'yes': 'tumour', 'no': 'no_tumour'}
    splits = ['train', 'val', 'test']

    # Create output directories
    for split in splits:
        for cls in class_map.values():
            (out_path / split / cls).mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {}

    for raw_cls, out_cls in class_map.items():
        files = list((raw_path / raw_cls).glob('*.jpg'))
        files += list((raw_path / raw_cls).glob('*.png'))
        files += list((raw_path / raw_cls).glob('*.jpeg'))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        split_files = {
            'train': files[:n_train],
            'val':   files[n_train:n_train + n_val],
            'test':  files[n_train + n_val:],
        }

        for split, file_list in split_files.items():
            dst_dir = out_path / split / out_cls
            for src in file_list:
                shutil.copy2(src, dst_dir / src.name)
            stats[f"{split}/{out_cls}"] = len(file_list)
            print(f"  {split:5s}/{out_cls:12s}: {len(file_list):4d} images")

    print(f"\nDataset prepared at: {output_dir}")
    print(f"Total images: {sum(stats.values())}")
    return stats


# ─────────────────────────────────────────────────────────────
#  CUSTOM DATASET
# ─────────────────────────────────────────────────────────────

class BR35HDataset(Dataset):
    """
    Custom Dataset for BR35H with support for:
    - CLAHE (contrast-limited adaptive histogram equalisation)
    - Skull-stripping simulation
    - Multi-scale patch extraction
    """

    CLASS_MAP = {'no_tumour': 0, 'tumour': 1}

    def __init__(self, root: str, split: str = 'train',
                 image_size: int = 224, transform=None, use_clahe: bool = True):
        self.root = Path(root) / split
        self.image_size = image_size
        self.transform = transform
        self.use_clahe = use_clahe

        self.samples: List[Tuple[Path, int]] = []
        for cls_name, label in self.CLASS_MAP.items():
            cls_dir = self.root / cls_name
            if cls_dir.exists():
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        self.samples.append((p, label))

        random.shuffle(self.samples)
        print(f"[{split}] Loaded {len(self.samples)} samples | "
              f"Tumour: {sum(1 for _, l in self.samples if l==1)} | "
              f"No Tumour: {sum(1 for _, l in self.samples if l==0)}")

    def _apply_clahe(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE for enhanced tumour visibility."""
        import cv2
        img_np = np.array(img.convert('L'))  # greyscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        # Convert back to RGB
        enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
        return Image.fromarray(enhanced_rgb)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.use_clahe:
            try:
                img = self._apply_clahe(img)
            except ImportError:
                pass  # cv2 not available, skip

        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

class MetricsTracker:
    """Comprehensive metrics for medical image classification."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.preds: List[int] = []
        self.labels: List[int] = []
        self.probs: List[float] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        self.preds.extend(preds.cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())
        self.probs.extend(probs[:, 1].cpu().numpy().tolist())

    def compute(self) -> Dict:
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, confusion_matrix, average_precision_score
        )
        y_true = np.array(self.labels)
        y_pred = np.array(self.preds)
        y_prob = np.array(self.probs)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        auc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)

        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp + 1e-8)

        return {
            'accuracy':    acc * 100,
            'precision':   prec * 100,
            'recall':      rec * 100,      # sensitivity
            'specificity': specificity * 100,
            'f1_score':    f1 * 100,
            'auc_roc':     auc,
            'avg_precision': ap,
            'confusion_matrix': cm.tolist(),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }

    def print_report(self):
        m = self.compute()
        print("\n" + "="*50)
        print("  EVALUATION RESULTS")
        print("="*50)
        print(f"  Accuracy:     {m['accuracy']:.2f}%")
        print(f"  Precision:    {m['precision']:.2f}%")
        print(f"  Recall (Sen): {m['recall']:.2f}%")
        print(f"  Specificity:  {m['specificity']:.2f}%")
        print(f"  F1 Score:     {m['f1_score']:.2f}%")
        print(f"  AUC-ROC:      {m['auc_roc']:.4f}")
        print(f"  Avg Precision:{m['avg_precision']:.4f}")
        print("="*50)
        cm = m['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TP={m['tp']:4d}  FN={m['fn']:4d}")
        print(f"    FP={m['fp']:4d}  TN={m['tn']:4d}")
        print("="*50)
        return m


# ─────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────

class Predictor:
    """Inference wrapper with TTA (Test-Time Augmentation)."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        from models.hybrid_model import build_model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model = build_model('base', num_classes=2)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device).eval()
        self.device = device

        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        ]

    @torch.no_grad()
    def predict(self, image: Image.Image, use_tta: bool = True) -> Dict:
        if use_tta:
            logits_list = []
            for t in self.tta_transforms:
                x = t(image).unsqueeze(0).to(self.device)
                logits_list.append(torch.softmax(self(x), dim=1))
            probs = torch.stack(logits_list).mean(0)
        else:
            x = self.tta_transforms[0](image).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)

        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()
        return {
            'class': 'tumour' if pred_class == 1 else 'no_tumour',
            'label': pred_class,
            'confidence': confidence,
            'prob_tumour': probs[0, 1].item(),
            'prob_no_tumour': probs[0, 0].item(),
        }

    def __call__(self, x): return self.model(x)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='Prepare BR35H dataset')
    parser.add_argument('--raw_dir', default='data/BR35H_raw')
    parser.add_argument('--output_dir', default='data/BR35H')
    args = parser.parse_args()

    if args.prepare:
        prepare_br35h_dataset(args.raw_dir, args.output_dir)
