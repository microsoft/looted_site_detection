#!/usr/bin/env python
"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import SiteFeatureDataset
from .models import build_model
from .metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate looted site classifier (retrain on train+val, evaluate on test).')
    p.add_argument('--feature_type', type=str, default='handcrafted')
    p.add_argument('--aggregation', type=str, default='mean', help='mean|median|max|min|std|concat|pca|none (pca reduces concat via PCA; none for GRU)')
    p.add_argument('--model', type=str, default='rf', help='rf|logreg|gb|xgb|gru')
    p.add_argument('--fold', type=int, default=1, help='Legacy validation fold (ignored if dynamic_split).')
    p.add_argument('--subset', type=int, default=None, help='Optional subset of sites for quick experiments.')
    p.add_argument('--year', type=int, default=None, help='Optional year (2016-2023) to restrict monthly features.')
    p.add_argument('--output_dir', type=str, default='looted_site_detection/outputs')
    p.add_argument('--dynamic_split', action='store_true', help='On-the-fly stratified split.')
    p.add_argument('--test_size', type=float, default=0.2, help='Test fraction in dynamic mode.')
    p.add_argument('--val_size', type=float, default=0.1, help='Validation fraction (of total) in dynamic mode.')
    p.add_argument('--fold_index', type=int, default=0, help='Random seed offset used in dynamic split (seed=42+fold_index).')
    p.add_argument('--model_runs_root', type=str, default='looted_site_detection/model_runs', help='Root directory for dynamic runs.')
    p.add_argument('--epochs', type=int, default=30, help='Epochs for GRU model only.')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate for GRU model only.')
    p.add_argument('--save_probs', action='store_true')
    p.add_argument('--normalize', action='store_true', help='Apply per-month feature normalization (fit on train split, reused for val/test).')
    p.add_argument('--norm_method', type=str, default='standard', help='standard|robust scaling.')
    p.add_argument('--save_norm_stats', action='store_true', help='Persist normalization statistics JSON when using --normalize.')
    return p.parse_args()


def main():
    args = parse_args()
    if args.dynamic_split:
        from .dynamic_split import generate_stratified_site_splits
        seed = 42 + args.fold_index
        site_ids_dict = generate_stratified_site_splits(args.feature_type, test_size=args.test_size, val_size=args.val_size, seed=seed)
        # Per-aggregation/model directory to organize results by aggregation method
        run_dir = Path(args.model_runs_root) / args.feature_type / args.aggregation / args.model / f'fold_{args.fold_index}'
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / 'splits.json', 'w') as f:
            json.dump(site_ids_dict, f, indent=2)
        out_dir = run_dir
    else:
        site_ids_dict = None
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SiteFeatureDataset(
        args.feature_type, args.aggregation, 'train', fold=args.fold, subset=args.subset, year=args.year, site_ids_dict=site_ids_dict,
        normalize=args.normalize, norm_method=args.norm_method
    )
    norm_stats = train_ds.get_norm_stats() if args.normalize else None
    pca_model = train_ds.get_pca_model() if args.aggregation == 'pca' else None
    val_ds = SiteFeatureDataset(
        args.feature_type, args.aggregation, 'val', fold=args.fold, subset=args.subset, year=args.year, site_ids_dict=site_ids_dict,
        normalize=args.normalize, norm_method=args.norm_method, norm_stats=norm_stats, pca_model=pca_model
    )
    test_ds = SiteFeatureDataset(
        args.feature_type, args.aggregation, 'test', fold=args.fold, subset=args.subset, year=args.year, site_ids_dict=site_ids_dict,
        normalize=args.normalize, norm_method=args.norm_method, norm_stats=norm_stats, pca_model=pca_model
    )

    X_train, y_train, _ = train_ds.get_data()
    X_val, y_val, _ = val_ds.get_data()
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    X_test, y_test, test_ids = test_ds.get_data()

    if args.model == 'gru':
        assert args.aggregation == 'none', 'GRU requires aggregation=none.'
        input_dim = X_combined.shape[-1]
    else:
        input_dim = X_combined.shape[-1]
    model = build_model(args.model, args.aggregation if args.model != 'gru' else 'none', input_dim)

    if args.model == 'gru':
        model.epochs = args.epochs
        model.lr = args.lr
        model.fit(X_combined, y_combined)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
    else:
        model.fit(X_combined, y_combined)
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = np.stack([1-y_pred, y_pred], axis=1)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    result = {
        'feature_type': args.feature_type,
        'aggregation': args.aggregation,
        'model': args.model,
        'fold': args.fold,
        'subset': args.subset,
        'year': args.year,
        'metrics': metrics,
        'num_train_val': int(len(y_combined)),
        'num_test': int(len(y_test)),
        'dynamic_split': args.dynamic_split,
        'test_size': args.test_size if args.dynamic_split else None,
        'val_size': args.val_size if args.dynamic_split else None,
        'fold_index': args.fold_index if args.dynamic_split else None,
        'normalized': args.normalize,
        'norm_method': args.norm_method if args.normalize else None,
        'pca_aggregation': True if args.aggregation == 'pca' else False,
    }
    with open(out_dir / 'eval_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    if args.normalize and args.save_norm_stats and norm_stats is not None:
        with open(out_dir / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats, f)
    if args.save_probs:
        out = {
            'site_id': test_ids,
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'probs': y_proba.tolist()
        }
        with open(out_dir / 'test_predictions_eval.json', 'w') as f:
            json.dump(out, f, indent=2)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
