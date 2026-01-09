# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
CNN Evaluation Script for Looted Site Detection
Evaluates trained CNN models on test set
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cnn_models import get_model
from looting_image_dataset import LootingImageDataset


def evaluate(model, dataloader, device, args):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_site_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            if args.single:
                images, temporal_idx, masks, labels = batch
                images = images.to(device)
                masks = masks.to(device)
            else:
                images, temporal_idx, masks, labels = batch
                B, T, C, H, W = images.shape
                
                if 'temporal' in args.model_name:
                    images = images.to(device)
                else:
                    # Average over temporal dimension for non-temporal models
                    images = images.mean(dim=1).to(device)
                
                masks = masks.to(device)
            
            labels = labels.to(device).squeeze()
            
            # Forward pass
            if args.mask_mode == 'none':
                outputs = model(images)
            else:
                outputs = model(images, masks)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    auroc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'accuracy': acc * 100,
        'f1': f1 * 100,
        'auroc': auroc * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics


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
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--use_best_auroc', action='store_true',
                        help='Use best_auroc.pth instead of specified checkpoint')
    parser.add_argument('--use_best_acc', action='store_true',
                        help='Use best_acc.pth instead of specified checkpoint')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                        default=os.environ.get('CHANGE_DETECTION_DIR', 'change_detection'),
                        help='Root directory containing planet_mosaics_final_4bands/')
    parser.add_argument('--metadata_path', type=str,
                        default=os.environ.get('LOOTED_METADATA_PATH', 'data/metadata.csv'),
                        help='Path to metadata CSV file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    
    # Override parameters (if not using checkpoint args)
    parser.add_argument('--model_name', type=str, default=None,
                        help='Override model name from checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as checkpoint dir)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if args.use_best_auroc:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, 'best_auroc.pth')
    elif args.use_best_acc:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, 'best_acc.pth')
    
    print(f'Loading checkpoint: {checkpoint_path}')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    checkpoint_args = argparse.Namespace(**checkpoint['args'])
    
    # Override model name if specified
    if args.model_name is not None:
        checkpoint_args.model_name = args.model_name
    
    print(f'Model: {checkpoint_args.model_name}')
    print(f'Checkpoint epoch: {checkpoint["epoch"]}')
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset
    print(f'\nCreating {args.split} dataset...')
    dataset = LootingImageDataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata_path,
        split=args.split,
        image_size=(checkpoint_args.image_size, checkpoint_args.image_size),
        norm_stats=checkpoint_args.norm_stats,
        augment=False,
        single=checkpoint_args.single,
        mask_mode=checkpoint_args.mask_mode,
        use_buffered_masks=checkpoint_args.use_buffered_masks
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f'{args.split} dataset: {len(dataset)} samples')
    
    # Create model
    print(f'\nCreating model...')
    model_kwargs = {}
    if 'temporal' in checkpoint_args.model_name:
        model_kwargs['sequence_length'] = checkpoint_args.temporal_sampling
        model_kwargs['in_channels'] = 3
    
    model = get_model(
        checkpoint_args.model_name,
        pretrained=False,  # Don't load pretrained weights, we'll load from checkpoint
        num_classes=2,
        **model_kwargs
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    
    print(f'Model loaded from checkpoint')
    
    # Evaluate
    print(f'\nEvaluating on {args.split} set...')
    metrics = evaluate(model, dataloader, args.device, checkpoint_args)
    
    # Print results
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print(f'Split: {args.split}')
    print(f'Model: {checkpoint_args.model_name}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Epoch: {checkpoint["epoch"]}')
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
    
    # Save metrics
    metrics_to_save = {
        'split': args.split,
        'model_name': checkpoint_args.model_name,
        'checkpoint_path': checkpoint_path,
        'epoch': checkpoint['epoch'],
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'auroc': metrics['auroc'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    metrics_path = os.path.join(args.output_dir, f'{args.split}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f'\nMetrics saved to {metrics_path}')
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'label': metrics['labels'],
        'prediction': metrics['predictions'],
        'probability': metrics['probabilities']
    })
    predictions_path = os.path.join(args.output_dir, f'{args.split}_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f'Predictions saved to {predictions_path}')
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, f'{args.split}_confusion_matrix.png')
    plot_confusion_matrix(np.array(metrics['confusion_matrix']), cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(args.output_dir, f'{args.split}_roc_curve.png')
    plot_roc_curve(metrics['labels'], metrics['probabilities'], roc_path)
    
    print(f'\nEvaluation completed! Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()
