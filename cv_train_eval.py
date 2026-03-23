#!/usr/bin/env python
"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

K-Fold Cross-Validation plus final test evaluation for looted site detection.
Relocated from change_detection.classification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .utils import load_features, build_temporal_matrix, aggregate_features, extract_class, compute_monthly_stats, apply_monthly_normalization
from .splits import load_fold_dict
from .models import build_model
from .metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description='K-fold cross-validation + final test evaluation.')
    p.add_argument('--feature_type', type=str, default='handcrafted')
    p.add_argument('--aggregation', type=str, default='mean', help='mean|median|max|min|std|concat|pca|none')
    p.add_argument('--model', type=str, default='rf', help='rf|logreg|gb|xgb|gru')
    p.add_argument('--k_folds', type=int, default=5)
    p.add_argument('--year', type=int, default=None, help='Optional calendar year filter (2016-2023).')
    p.add_argument('--subset', type=int, default=None, help='Optional subset of CV pool for quick smoke tests.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30, help='GRU only')
    p.add_argument('--lr', type=float, default=1e-3, help='GRU only')
    p.add_argument('--save_probs', action='store_true')
    p.add_argument('--output_root', type=str, default='looted_site_detection/model_runs')
    p.add_argument('--normalize', action='store_true', help='Per-month normalization inside each CV fold (fit on that fold\'s train split).')
    p.add_argument('--norm_method', type=str, default='standard', help='standard|robust scaling method.')
    p.add_argument('--save_norm_stats', action='store_true', help='Persist normalization statistics for final test training.')
    return p.parse_args()


def build_raw_matrix(feature_df, year=None):
    X, site_ids, months, feature_cols, mask = build_temporal_matrix(feature_df, year=year)
    site_names = sorted(feature_df['site_name'].unique())
    labels = [extract_class(s) for s in site_names]
    y = np.array(labels, dtype=np.int64)
    return X, y, site_names, months, mask


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    full_df = load_features(args.feature_type)
    full_df['site_id'] = full_df['site_name'].apply(lambda s: int(s.split('_')[-1]) if '_' in s else -1)
    fold_dict = load_fold_dict()
    test_ids = set(fold_dict['test'])
    cv_pool_ids = set(fold_dict['train'])
    for k in range(1,6):
        cv_pool_ids.update(fold_dict.get(f'val_{k}', []))
    cv_pool_ids = cv_pool_ids - test_ids
    cv_df = full_df[full_df['site_id'].isin(cv_pool_ids)].copy()
    test_df = full_df[full_df['site_id'].isin(test_ids)].copy()
    if args.subset:
        site_names_all = sorted(cv_df['site_name'].unique())
        if args.subset < len(site_names_all):
            keep_subset = set(rng.choice(site_names_all, size=args.subset, replace=False))
            cv_df = cv_df[cv_df['site_name'].isin(keep_subset)].copy()
    X_pool_raw, y_pool, pool_site_names, months, mask_pool = build_raw_matrix(cv_df, year=args.year)
    X_test_raw, y_test, test_site_names, months_test, mask_test = build_raw_matrix(test_df, year=args.year)
    assert months == months_test, 'Month ordering mismatch between CV pool and test set.'
    year_tag = f'y{args.year}' if args.year else 'yall'
    out_dir = Path(args.output_root) / args.feature_type / 'cv' / f'{args.model}_{args.aggregation}_{year_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_pool_raw, y_pool)):
        X_tr_raw, y_tr = X_pool_raw[train_idx], y_pool[train_idx]
        X_val_raw, y_val = X_pool_raw[val_idx], y_pool[val_idx]
        mask_tr = mask_pool[train_idx]
        mask_val = mask_pool[val_idx]
        if args.normalize:
            norm_stats_fold = compute_monthly_stats(X_tr_raw, mask_tr, months, method=args.norm_method)
            X_tr_raw = apply_monthly_normalization(X_tr_raw, mask_tr, norm_stats_fold)
            X_val_raw = apply_monthly_normalization(X_val_raw, mask_val, norm_stats_fold)
        if args.aggregation == 'none':
            X_tr = X_tr_raw
            X_val = X_val_raw
        else:
            X_tr = aggregate_features(X_tr_raw, args.aggregation, mask_tr)
            X_val = aggregate_features(X_val_raw, args.aggregation, mask_val)
        if args.model == 'gru':
            input_dim = X_tr.shape[-1]
            model = build_model('gru', 'none', input_dim)
            model.epochs = args.epochs
            model.lr = args.lr
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
        else:
            input_dim = X_tr.shape[-1]
            model = build_model(args.model, args.aggregation if args.model != 'gru' else 'none', input_dim)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
            else:
                y_proba = np.stack([1-y_pred, y_pred], axis=1)
        metrics = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append({'fold': fold_idx, 'metrics': metrics, 'num_train': int(len(y_tr)), 'num_val': int(len(y_val))})
    metric_keys = list(fold_metrics[0]['metrics'].keys())
    agg = {}
    for k in metric_keys:
        if k == 'confusion_matrix':
            continue
        vals = [fm['metrics'][k] for fm in fold_metrics]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    with open(out_dir / 'cv_fold_metrics.json', 'w') as f:
        json.dump(fold_metrics, f, indent=2)
    with open(out_dir / 'cv_summary.json', 'w') as f:
        json.dump({'feature_type': args.feature_type, 'aggregation': args.aggregation, 'model': args.model, 'year': args.year, 'k_folds': args.k_folds, 'summary': agg}, f, indent=2)
    # Final train on entire CV pool (optionally normalized + PCA with stats from full pool)
    if args.normalize:
        norm_stats_final = compute_monthly_stats(X_pool_raw, mask_pool, months, method=args.norm_method)
        X_pool_raw_final = apply_monthly_normalization(X_pool_raw, mask_pool, norm_stats_final)
        X_test_raw_final = apply_monthly_normalization(X_test_raw, mask_test, norm_stats_final)
    else:
        norm_stats_final = None
        X_pool_raw_final = X_pool_raw
        X_test_raw_final = X_test_raw
    if args.aggregation == 'pca':
        # Fit PCA on full pool then transform pool and test
        from .utils import fit_pca_concat, transform_pca_concat
        n_components = X_pool_raw_final.shape[2]
        pca_full = fit_pca_concat(X_pool_raw_final, n_components)
        X_pool_final = transform_pca_concat(X_pool_raw_final, pca_full)
        X_test_final = transform_pca_concat(X_test_raw_final, pca_full)
    elif args.aggregation == 'none':
        X_pool_final = X_pool_raw_final
        X_test_final = X_test_raw_final
    else:
        X_pool_final = aggregate_features(X_pool_raw_final, args.aggregation, mask_pool)
        X_test_final = aggregate_features(X_test_raw_final, args.aggregation, mask_test)
    if args.model == 'gru':
        input_dim = X_pool_final.shape[-1]
        final_model = build_model('gru', 'none', input_dim)
        final_model.epochs = args.epochs
        final_model.lr = args.lr
        final_model.fit(X_pool_final, y_pool)
        y_pred_test = final_model.predict(X_test_final)
        y_proba_test = final_model.predict_proba(X_test_final)
    else:
        input_dim = X_pool_final.shape[-1]
        final_model = build_model(args.model, args.aggregation if args.model != 'gru' else 'none', input_dim)
        final_model.fit(X_pool_final, y_pool)
        y_pred_test = final_model.predict(X_test_final)
        if hasattr(final_model, 'predict_proba'):
            y_proba_test = final_model.predict_proba(X_test_final)
        else:
            y_proba_test = np.stack([1-y_pred_test, y_pred_test], axis=1)
    test_metrics = compute_metrics(y_test, y_pred_test, y_proba_test)
    test_result = {
        'feature_type': args.feature_type,
        'aggregation': args.aggregation,
        'model': args.model,
        'year': args.year,
        'k_folds': args.k_folds,
        'num_cv_pool': int(len(y_pool)),
        'num_test': int(len(y_test)),
        'metrics': test_metrics,
        'normalized': args.normalize,
        'norm_method': args.norm_method if args.normalize else None
    }
    with open(out_dir / 'test_results.json', 'w') as f:
        json.dump(test_result, f, indent=2)
    if args.normalize and args.save_norm_stats and norm_stats_final is not None:
        with open(out_dir / 'norm_stats.json', 'w') as f:
            json.dump(norm_stats_final, f)
    if args.save_probs:
        with open(out_dir / 'test_predictions.json', 'w') as f:
            json.dump({'site_name': test_site_names,'y_true': y_test.tolist(),'y_pred': y_pred_test.tolist(),'probs': y_proba_test.tolist()}, f, indent=2)
    print(json.dumps({'cv_summary': agg, 'test_metrics': test_metrics}, indent=2))

if __name__ == '__main__':
    main()
