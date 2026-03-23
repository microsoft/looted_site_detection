# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
CNN Evaluation Script for Looted Site Detection
Evaluates trained CNN models on test set using checkpoints produced by train.py.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __package__ is None or __package__ == '':
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_parent not in sys.path:
        sys.path.append(pkg_parent)
    from looted_site_detection.cnn_models import get_model
    from looted_site_detection.cnn_dataset import ImageDataset
else:
    from .cnn_models import get_model
    from .cnn_dataset import ImageDataset


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset split."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_site_ids = []

    with torch.no_grad():
        for images, labels, site_ids in dataloader:
            images = images.to(device)
            if len(images.shape) == 5:
                images = images[:, 0, :, :, :]
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_site_ids.extend(site_ids)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    auroc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': acc * 100,
        'f1': f1 * 100,
        'auroc': auroc * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'site_ids': all_site_ids,
    }


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Preserved', 'Looted'],
                yticklabels=['Preserved', 'Looted'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {output_path}')


def plot_roc_curve(labels, probs, output_path):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'ROC curve saved to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='CNN Evaluation for Looted Site Detection')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (model.pt from train.py)')
    parser.add_argument('--use_best_auroc', action='store_true',
                        help='Look for best_auroc.pth in the same directory as --checkpoint')
    parser.add_argument('--use_best_acc', action='store_true',
                        help='Look for best_acc.pth in the same directory as --checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as checkpoint dir)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if args.use_best_auroc:
        checkpoint_path = checkpoint_path.parent / 'best_auroc.pth'
    elif args.use_best_acc:
        checkpoint_path = checkpoint_path.parent / 'best_acc.pth'

    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    checkpoint_args = argparse.Namespace(**checkpoint['args'])

    if args.output_dir is None:
        args.output_dir = str(checkpoint_path.parent)
    os.makedirs(args.output_dir, exist_ok=True)

    splits_path = checkpoint_path.parent / 'splits.json'
    if not splits_path.exists():
        raise FileNotFoundError(
            f'splits.json not found at {splits_path}. '
            'This file is written by train.py when using --dynamic_split.'
        )
    with open(splits_path) as f:
        site_ids_dict = json.load(f)

    site_ids = site_ids_dict[args.split]
    print(f'\nCreating {args.split} dataset ({len(site_ids)} sites)...')

    dataset = ImageDataset(
        data_root=checkpoint_args.data_root,
        mask_root=getattr(checkpoint_args, 'mask_root', None),
        labels_csv=getattr(checkpoint_args, 'labels_csv', None),
        site_ids=site_ids,
        image_size=checkpoint_args.image_size,
        num_temporal_steps=checkpoint_args.num_temporal_steps,
        mask_mode=checkpoint_args.mask_mode,
        augment=False,
        normalize=True,
        year_filter=getattr(checkpoint_args, 'year', None),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'Model: {checkpoint_args.model}')

    model = get_model(
        checkpoint_args.model,
        pretrained=False,
        num_classes=2,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    print(f'\nEvaluating on {args.split} set...')
    metrics = evaluate(model, dataloader, args.device)

    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Split:      {args.split}')
    print(f'Model:      {checkpoint_args.model}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Epoch:      {checkpoint.get("epoch", "?")}')
    print('-'*60)
    print(f'Accuracy:  {metrics["accuracy"]:.2f}%')
    print(f'F1 Score:  {metrics["f1"]:.2f}%')
    print(f'AUROC:     {metrics["auroc"]:.2f}%')
    print(f'Precision: {metrics["precision"]:.2f}%')
    print(f'Recall:    {metrics["recall"]:.2f}%')
    print('-'*60)
    print('Confusion Matrix:')
    print(np.array(metrics['confusion_matrix']))
    print('='*60)

    metrics_to_save = {
        'split': args.split,
        'model': checkpoint_args.model,
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint.get('epoch'),
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'auroc': metrics['auroc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'confusion_matrix': metrics['confusion_matrix'],
    }

    metrics_path = os.path.join(args.output_dir, f'{args.split}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f'\nMetrics saved to {metrics_path}')

    predictions_df = pd.DataFrame({
        'site_id': metrics['site_ids'],
        'label': metrics['labels'],
        'prediction': metrics['predictions'],
        'probability': metrics['probabilities'],
    })
    predictions_path = os.path.join(args.output_dir, f'{args.split}_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f'Predictions saved to {predictions_path}')

    cm_path = os.path.join(args.output_dir, f'{args.split}_confusion_matrix.png')
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), cm_path)

    roc_path = os.path.join(args.output_dir, f'{args.split}_roc_curve.png')
    plot_roc_curve(metrics['labels'], metrics['probabilities'], roc_path)

    print(f'\nEvaluation completed! Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
