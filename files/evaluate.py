"""
Evaluation, Grad-CAM Visualisation, and Full Test Report
for Brain Tumour Detection Model
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.hybrid_model import build_model
from dataset import MetricsTracker, get_transforms


# ─────────────────────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping for ViT."""

    def __init__(self, model, target_layer_name='vit.blocks'):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name):
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None:
            # Fallback: hook last transformer block
            blocks = list(self.model.vit.blocks)
            layer = blocks[-1]

        def fwd_hook(_, __, output):
            self.activations = output

        def bwd_hook(_, __, grad_output):
            self.gradients = grad_output[0]

        layer.register_forward_hook(fwd_hook)
        layer.register_full_backward_hook(bwd_hook)

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        image_tensor.requires_grad_(True)

        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Pool gradients across patch tokens
        grads = self.gradients  # (B, N, D) or (B, D, N)
        acts = self.activations  # (B, N, D)

        if grads is None or acts is None:
            return None

        weights = grads.mean(dim=1, keepdim=True)  # (B, 1, D)
        cam = (weights * acts).sum(dim=-1)          # (B, N)
        cam = F.relu(cam)

        # Reshape to spatial grid
        n_patches = cam.shape[1] - 1  # remove CLS token
        grid_size = int(n_patches ** 0.5)
        cam = cam[:, 1:].reshape(1, 1, grid_size, grid_size)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ─────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumour', 'Tumour'],
                yticklabels=['No Tumour', 'Tumour'],
                ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(labels, probs, save_path: str = None):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#E74C3C', lw=2,
            label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history['train_acc']) + 1)
    ax1.plot(epochs, history['train_acc'], label='Train', color='#3498DB')
    ax1.plot(epochs, history['val_acc'], label='Validation', color='#E74C3C')
    ax1.axhline(99, color='#2ECC71', linestyle='--', alpha=0.7, label='99% threshold')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')

    ax2.plot(epochs, history['train_loss'], label='Train', color='#3498DB')
    ax2.plot(epochs, history['val_loss'], label='Validation', color='#E74C3C')
    ax2.set_title('Loss'); ax2.legend(); ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    fig.suptitle('Training History — Brain Tumour Detection (ViT+Mamba+GAN)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualise_predictions(model, dataset, n_samples=8, save_path=None):
    """Show model predictions with confidence scores."""
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    tf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    model.eval()
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs.max().item()

        # Denormalise for display
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        disp = img * std[:, None, None] + mean[:, None, None]
        disp = disp.permute(1, 2, 0).numpy().clip(0, 1)

        axes[i].imshow(disp)
        cls_names = ['No Tumour', 'Tumour']
        correct = pred == label
        colour = '#2ECC71' if correct else '#E74C3C'
        axes[i].set_title(
            f"GT: {cls_names[label]}\nPred: {cls_names[pred]} ({conf:.1%})",
            color=colour, fontsize=9
        )
        axes[i].axis('off')

    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────────
#  FULL EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model('base', num_classes=2)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    # Test dataset
    from torchvision import datasets
    from torch.utils.data import DataLoader
    test_ds = datasets.ImageFolder(
        root=os.path.join(args.data_root, 'test'),
        transform=get_transforms(224, train=False),
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    metrics = MetricsTracker()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            metrics.update(logits, labels)

    results = metrics.print_report()

    # Plots
    os.makedirs('results', exist_ok=True)
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, save_path='results/confusion_matrix.png')
    plot_roc_curve(
        np.array(metrics.labels), np.array(metrics.probs),
        save_path='results/roc_curve.png'
    )

    print(f"\n✓ Evaluation complete. Results saved to results/")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--data_root', default='data/BR35H')
    args = parser.parse_args()
    evaluate(args)
