#!/usr/bin/env python
"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import sys, os
import logging

# Support running this file both as a module (python -m looted_site_detection.train)
# and as a script directly inside the package directory (python train.py) where
# relative imports normally fail. We append parent path and use absolute imports
# if __package__ is empty.
if __package__ is None or __package__ == '':
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_parent not in sys.path:
        sys.path.append(pkg_parent)
    try:
        from looted_site_detection.data import SiteFeatureDataset
        from looted_site_detection.models import build_model, is_cnn_model
        from looted_site_detection.metrics import compute_metrics
    except ImportError as e:
        raise ImportError(f"Failed absolute imports after path adjustment: {e}")
else:
    from .data import SiteFeatureDataset
    from .models import build_model, is_cnn_model
    from .metrics import compute_metrics

# Import CNN-specific modules
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    # Flexible import for ImageDataset (not always needed directly, but keep parity)
    try:
        from .cnn_dataset import ImageDataset  # package context
    except ImportError:
        from looted_site_detection.cnn_dataset import ImageDataset  # script context
    TORCH_AVAILABLE = True
except Exception as e:
    logging.warning(f"[train.py] PyTorch import failed: {e}")
    torch = None
    nn = None
    DataLoader = None
    ImageDataset = None


def parse_args():
    p = argparse.ArgumentParser(description='Train looted site classifier using pre-extracted features or raw images.')
    
    # Model selection (automatically determines feature-based vs CNN-based training)
    p.add_argument('--model', type=str, default='rf', 
                   help='Model: rf|logreg|gb|xgb|mlp|gru (feature-based) OR resnet20|resnet50|resnet18|resnet34|efficientnet_b0|efficientnet_b1 (CNN-based)')
    
    # Feature-based model arguments (used when model is rf/logreg/gb/xgb/mlp/gru)
    p.add_argument('--feature_type', type=str, default='handcrafted', 
                   help='For feature-based models: handcrafted|dinov2|satclip|satmae|prithvi|georsclip|satlas')
    p.add_argument('--aggregation', type=str, default='mean', 
                   help='For feature-based models: mean|median|max|min|std|concat|pca|none')
    
    # CNN-specific arguments (used when model is resnet*/efficientnet*/unet*)
    p.add_argument('--data_root', type=str, default='data/datasets', 
                   help='CNN: Root directory with monthly image folders (default "data/datasets").')
    p.add_argument('--mask_root', type=str, default=None, 
                   help='CNN: Directory with site masks.')
    p.add_argument('--labels_csv', type=str, default=None, 
                   help='CNN: CSV with site_id and label columns (optional, extracts from site names if not provided).')
    p.add_argument('--image_size', type=int, default=224, 
                   help='CNN: Input image size (default 224).')
    p.add_argument('--num_temporal_steps', type=int, default=1, 
                   help='CNN: Number of temporal images per site (1=single, >1=temporal).')
    p.add_argument('--mask_mode', type=str, default='multiply', 
                   help='CNN: Mask mode: multiply|concat|none.')
    p.add_argument('--pretrained', action='store_true', default=True,
                   help='CNN: Use ImageNet pretrained weights (default: True).')
    p.add_argument('--no_pretrained', dest='pretrained', action='store_false',
                   help='CNN: Disable pretrained weights (train from scratch).')
    p.add_argument('--test_use_buffered_masks', action='store_true',
                   help='CNN: At test time, use mask_buffered.png when available.')
    p.add_argument('--enforce_test_min_area', action='store_true',
                   help='CNN: Enforce minimum mask area at test time to the training median via dilation.')
    p.add_argument('--flat_runs_root', action='store_true',
                   help='CNN: Save outputs directly under <model_runs_root>/fold_<index> (no model subdir).')
    
    # Common arguments
    p.add_argument('--fold', type=int, default=1, help='Legacy validation fold (ignored if dynamic_split).')
    p.add_argument('--subset', type=int, default=None, help='Optional subset of sites.')
    p.add_argument('--year', type=int, default=None, help='Optional year filter (2016-2023).')
    p.add_argument('--output_dir', type=str, default='looted_site_detection/outputs')
    p.add_argument('--dynamic_split', action='store_true', help='Dynamic stratified train/val/test split.')
    p.add_argument('--test_size', type=float, default=0.2, help='Test fraction.')
    p.add_argument('--val_size', type=float, default=0.1, help='Validation fraction.')
    p.add_argument('--fold_index', type=int, default=0, help='Random seed offset.')
    p.add_argument('--model_runs_root', type=str, default='looted_site_detection/model_runs', 
                   help='Root directory for outputs.')
    
    # Training hyperparameters
    p.add_argument('--epochs', type=int, default=50, help='Training epochs (CNN: 50, GRU: 30).')
    p.add_argument('--batch_size', type=int, default=16, help='Batch size (CNN only).')
    p.add_argument('--lr', type=float, default=3e-4, help='Learning rate (CNN: 3e-4, GRU: 1e-3).')
    p.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (CNN only).')
    p.add_argument('--scheduler_step', type=int, default=20, help='LR scheduler step size (CNN only).')
    p.add_argument('--scheduler_gamma', type=float, default=0.1, help='LR scheduler gamma (CNN only).')
    p.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (CNN only, reduce if low shared memory).')
    p.add_argument('--no_pin_memory', action='store_true', help='Disable pin_memory in DataLoader to reduce host memory pressure.')
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience: stop if no val improvement for N epochs (CNN only, 0=disabled).')
    
    # Output options
    p.add_argument('--save_probs', action='store_true')
    p.add_argument('--normalize', action='store_true', help='Feature normalization (feature-based models only).')
    p.add_argument('--norm_method', type=str, default='standard', help='standard|robust scaling.')
    p.add_argument('--save_norm_stats', action='store_true', help='Save normalization stats.')
    
    return p.parse_args()


def train_cnn_model(args, site_ids_dict, out_dir):
    """Train a CNN model on raw images."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install: pip install torch torchvision")
    # Flexible import for script vs module execution
    try:
        from .cnn_dataset import create_image_datasets  # module context
    except ImportError:
        from looted_site_detection.cnn_dataset import create_image_datasets  # script context
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create image datasets
    train_ds, val_ds, test_ds = create_image_datasets(
        data_root=args.data_root,
        mask_root=args.mask_root,
        labels_csv=args.labels_csv,
        site_ids_dict=site_ids_dict,
        image_size=args.image_size,
        num_temporal_steps=args.num_temporal_steps,
        mask_mode=args.mask_mode,
        year_filter=args.year,  # Filter to specific year (e.g., 2023 for final state)
        test_use_buffered_masks=args.test_use_buffered_masks,
        enforce_test_min_area=args.enforce_test_min_area,
    )
    
    # Create data loaders
    # DataLoader resource configuration
    dl_workers = max(0, args.num_workers)
    pin = not args.no_pin_memory
    if dl_workers == 0:
        print("[DataLoader] Using single-process loading (num_workers=0) to avoid shared memory exhaustion.")
    else:
        print(f"[DataLoader] Using num_workers={dl_workers}, pin_memory={pin}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=dl_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=dl_workers, pin_memory=pin)
    
    # Build CNN model
    in_channels = 3
    if args.mask_mode == 'concat':
        in_channels = 4  # RGB + mask channel
    
    model = build_model(
        args.model,
        aggregation='none',  # Not used for CNNs
        input_dim=None,  # Not used for CNNs
        num_classes=2,
        pretrained=args.pretrained,
        mask_mode=args.mask_mode,
        in_channels=in_channels,
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = out_dir / 'model.pt'
    
    # Early stopping
    patience = args.patience if args.patience > 0 else None
    patience_counter = 0
    
    print(f"\nTraining CNN model: {args.model}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    if patience:
        print(f"Early stopping enabled: patience={patience} epochs")
    else:
        print("Early stopping disabled")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Handle temporal dimension if present
            if len(images.shape) == 5:  # [B, T, C, H, W]
                # For now, take first timestep (can extend to 3D CNN later)
                images = images[:, 0, :, :, :]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                if len(images.shape) == 5:
                    images = images[:, 0, :, :, :]
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        
        # LR scheduling
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0  # Reset counter on improvement
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience and patience_counter >= patience:
                print(f"\nEarly stopping triggered: no improvement for {patience} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
                break
    
    if not (patience and patience_counter >= patience):
        print(f"\nCompleted all {args.epochs} epochs")
        print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Load best model for testing
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []
    test_ids = []
    
    with torch.no_grad():
        for images, labels, site_ids in test_loader:
            images = images.to(device)
            
            if len(images.shape) == 5:
                images = images[:, 0, :, :, :]
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())
            test_ids.extend(site_ids)
    
    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    
    # Compute metrics (same format as feature-based models)
    accuracy = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='binary')
    auroc = roc_auc_score(test_labels, test_probs[:, 1])
    precision = precision_score(test_labels, test_preds, average='binary', zero_division=0)
    recall = recall_score(test_labels, test_preds, average='binary', zero_division=0)
    conf_mat = confusion_matrix(test_labels, test_preds).tolist()
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'auroc': float(auroc),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': conf_mat,
    }
    
    return test_preds, test_probs, test_labels, test_ids, metrics


def train_feature_model(args, site_ids_dict, out_dir):
    """Train a traditional feature-based model."""
    train_ds = SiteFeatureDataset(
        args.feature_type, args.aggregation, 'train', fold=args.fold, subset=args.subset, year=args.year, 
        site_ids_dict=site_ids_dict, normalize=args.normalize, norm_method=args.norm_method
    )
    norm_stats = train_ds.get_norm_stats() if args.normalize else None
    pca_model = train_ds.get_pca_model() if args.aggregation == 'pca' else None
    
    val_ds = SiteFeatureDataset(
        args.feature_type, args.aggregation, 'val', fold=args.fold, subset=args.subset, year=args.year, 
        site_ids_dict=site_ids_dict, normalize=args.normalize, norm_method=args.norm_method, 
        norm_stats=norm_stats, pca_model=pca_model
    )
    test_ds = SiteFeatureDataset(
        args.feature_type, args.aggregation, 'test', fold=args.fold, subset=args.subset, year=args.year, 
        site_ids_dict=site_ids_dict, normalize=args.normalize, norm_method=args.norm_method, 
        norm_stats=norm_stats, pca_model=pca_model
    )

    X_train, y_train, _ = train_ds.get_data()
    X_val, y_val, _ = val_ds.get_data()
    X_test, y_test, test_ids = test_ds.get_data()

    if args.model == 'gru':
        assert args.aggregation == 'none', 'GRU requires aggregation=none (raw sequences).'
        input_dim = X_train.shape[-1]
    else:
        input_dim = X_train.shape[-1]
    
    model = build_model(args.model, args.aggregation if args.model != 'gru' else 'none', input_dim)

    if args.model == 'gru':
        model.epochs = args.epochs
        model.lr = args.lr
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = np.stack([1-y_pred, y_pred], axis=1)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Save normalization stats if requested
    if args.normalize and args.save_norm_stats and norm_stats is not None:
        with open(out_dir / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)
    
    return y_pred, y_proba, y_test, test_ids, metrics, len(y_train), len(y_val)


def main():
    args = parse_args()

    # --- Robust data_root resolution for CNN workflows ---
    # When launching from parent directory or different working directories, the intended
    # dataset path may actually live under "looted_site_detection/datasets" rather than a
    # bare "datasets" sibling of the current CWD. We normalize here so downstream modules
    # (dynamic_split_images, cnn_dataset) receive a concrete, existing path that contains
    # both required class subdirectories: looted/ and preserved/.
    def _resolve_data_root(path_str: str) -> str:
        p = Path(path_str)
        candidates = []
        # If absolute, test directly
        if p.is_absolute():
            candidates.append(p)
        else:
            # CWD relative
            candidates.append(Path.cwd() / p)
            # Under package root (this file is inside looted_site_detection)
            pkg_root = Path(__file__).resolve().parent
            candidates.append(pkg_root / p)
            # Under project root with explicit looted_site_detection prefix
            candidates.append(Path.cwd() / 'looted_site_detection' / p)
            candidates.append(pkg_root.parent / 'looted_site_detection' / p)
        # If user already passed something like 'looted_site_detection/datasets'
        if 'looted_site_detection' in path_str:
            candidates.append(Path(path_str))
        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                unique_candidates.append(c)
                seen.add(c)
        # Validation: must contain both class subdirectories
        for c in unique_candidates:
            looted_dir = c / 'looted'
            preserved_dir = c / 'preserved'
            if looted_dir.exists() and preserved_dir.exists():
                print(f"[data_root] Resolved dataset root: {c}")
                return str(c)
        # Last resort: return original string; downstream will raise if unusable
        print(f"[data_root] Warning: Could not positively resolve '{path_str}'. Using as-is.")
        return path_str

    # Only attempt resolution for CNN models (feature-based uses its own data loaders)
    # We'll update args.data_root early so all subsequent code uses normalized path.
    if is_cnn_model(args.model):
        original_root = args.data_root
        args.data_root = _resolve_data_root(args.data_root)
        if args.data_root != original_root:
            print(f"[data_root] Normalized '{original_root}' -> '{args.data_root}'")
    
    # Determine if CNN or feature-based model
    use_cnn = is_cnn_model(args.model)
    
    # Setup output directory
    if args.dynamic_split:
        # Flexible import for dynamic split helper
        try:
            from .dynamic_split import generate_stratified_site_splits
        except ImportError:
            from looted_site_detection.dynamic_split import generate_stratified_site_splits
        seed = 42 + args.fold_index
        
        if use_cnn:
            # CNN models: organize by model type (no feature_type/aggregation)
            # Always use image-based splitting for CNN models
            try:
                from .dynamic_split_images import create_image_based_splits
            except ImportError:
                from looted_site_detection.dynamic_split_images import create_image_based_splits
            
            if args.year is not None:
                print(f"\n{'='*60}")
                print(f"Using image-based dynamic splitting for year {args.year}")
                print(f"This ensures labels match image content (final state)")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"Using image-based dynamic splitting for ALL YEARS (2017-2023)")
                print(f"WARNING: This creates temporal label noise!")
                print(f"Sites looted mid-period will have mislabeled pre-looting images")
                print(f"{'='*60}\n")
            
            site_ids_dict = create_image_based_splits(
                data_root=args.data_root,
                labels_csv=args.labels_csv,
                test_size=args.test_size,
                val_size=args.val_size,
                year=args.year,  # None = all years
                seed=seed
            )
            # Allow overriding base output root for CNN runs via --model_runs_root
            base_root = Path(args.model_runs_root) if args.model_runs_root else Path('results/model_runs_cnn')
            run_dir = (base_root / f'fold_{args.fold_index}') if args.flat_runs_root else (base_root / args.model / f'fold_{args.fold_index}')
        else:
            # Feature-based models: organize by feature_type/aggregation/model
            site_ids_dict = generate_stratified_site_splits(
                args.feature_type, test_size=args.test_size, val_size=args.val_size, seed=seed
            )
            run_dir = Path(args.model_runs_root) / args.feature_type / args.aggregation / args.model / f'fold_{args.fold_index}'
        
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'splits.json', 'w') as f:
            json.dump(site_ids_dict, f, indent=2)
        out_dir = run_dir
    else:
        site_ids_dict = None
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Train model (CNN or feature-based)
    if use_cnn:
        print(f"Training CNN model: {args.model}")
        y_pred, y_proba, y_test, test_ids, metrics = train_cnn_model(args, site_ids_dict, out_dir)
        num_train = len(site_ids_dict['train']) if site_ids_dict else 0
        num_val = len(site_ids_dict['val']) if site_ids_dict else 0
        num_test = len(y_test)
    else:
        print(f"Training feature-based model: {args.model}")
        y_pred, y_proba, y_test, test_ids, metrics, num_train, num_val = train_feature_model(args, site_ids_dict, out_dir)

    # Save results (same format for both CNN and feature-based)
    # Feature-based models use roc_auc, f1 (not f1_score) naming convention
    # Align CNN metrics to match for consistency
    if use_cnn and 'f1_score' in metrics:
        metrics['f1'] = metrics.pop('f1_score')
    if use_cnn and 'auroc' in metrics:
        metrics['roc_auc'] = metrics.pop('auroc')
    
    result = {
        'feature_type': args.feature_type if not use_cnn else 'N/A',
        'aggregation': args.aggregation if not use_cnn else 'N/A',
        'model': args.model,
        'fold': args.fold,
        'subset': args.subset,
        'year': args.year,
        'metrics': metrics,
        'num_train_val': int(num_train + num_val) if not use_cnn else None,
        'num_train': int(num_train) if use_cnn else None,
        'num_val': int(num_val) if use_cnn else None,
        'num_test': int(len(y_test)),
        'dynamic_split': args.dynamic_split,
        'test_size': args.test_size if args.dynamic_split else None,
        'val_size': args.val_size if args.dynamic_split else None,
        'fold_index': args.fold_index if args.dynamic_split else None,
        'normalized': args.normalize if not use_cnn else True,  # CNNs always normalize
        'norm_method': args.norm_method if args.normalize and not use_cnn else 'dataset_stats',
    }
    
    # Add CNN-specific info (optional, not in feature-based schema)
    if use_cnn:
        result['cnn_config'] = {
            'image_size': args.image_size,
            'num_temporal_steps': args.num_temporal_steps,
            'mask_mode': args.mask_mode,
            'pretrained': args.pretrained,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }
    
    # Add feature-based specific info (optional, not in CNN workflow)
    if not use_cnn:
        result['pca_aggregation'] = (args.aggregation == 'pca')
    
    # Persist unified results file (use eval_results.json to match feature-based naming)
    results_filename = 'eval_results.json'
    with open(out_dir / results_filename, 'w') as f:
        json.dump(result, f, indent=2)

    
    if args.save_probs:
        out = {
            'site_id': test_ids,
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'probs': y_proba.tolist()
        }
        with open(out_dir / 'test_predictions.json', 'w') as f:
            json.dump(out, f, indent=2)
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
