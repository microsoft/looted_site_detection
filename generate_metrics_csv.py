#!/usr/bin/env python
"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Aggregate evaluation metrics across folds for each feature type and model.
Scans new layout: looted_site_detection/model_runs/<feature_type>/<model>/fold_*/eval_results.json
Output: looted_site_detection/model_runs/<feature_type>/metrics_summary.csv with columns model, fold, OA, F1, AUROC, FPR.
FPR = FP / (FP + TN) derived from confusion_matrix [[TN, FP],[FN, TP]].
If eval_results.json missing or malformed, row skipped.
Backward compatibility: if per-model subdirs absent, fall back to old pattern feature_type/fold_*/eval_results.json (model column = 'unknown').
"""
import argparse
import json
from pathlib import Path
import csv

def parse_args():
    p = argparse.ArgumentParser(description='Generate per-feature-type metrics summary CSVs.')
    p.add_argument('--feature_types', nargs='*', default=None, help='Optional list of feature types to process; if omitted, infer from model_runs directory.')
    p.add_argument('--model_runs_root', type=str, default='looted_site_detection/model_runs', help='Root directory for model runs.')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing CSVs if present.')
    return p.parse_args()


def extract_metrics(eval_path: Path, model_name: str):
    try:
        with open(eval_path, 'r') as f:
            data = json.load(f)
        metrics = data.get('metrics', {})
        cm = metrics.get('confusion_matrix')
        if not cm or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
            return None
        tn, fp = cm[0]
        fn, tp = cm[1]
        denom = fp + tn
        fpr = fp / denom if denom else float('nan')
        row = {
            'model': model_name,
            'fold': data.get('fold_index', data.get('fold', None)),
            'OA': metrics.get('accuracy'),
            'F1': metrics.get('f1'),
            'AUROC': metrics.get('roc_auc'),
            'FPR': fpr
        }
        return row
    except Exception:
        return None


def process_feature_type(feature_type: str, model_runs_root: Path):
    feature_dir = model_runs_root / feature_type
    if not feature_dir.exists():
        return None
    rows = []
    # Detect per-aggregation/model structure: feature_type/aggregation/model/fold_*
    agg_dirs = [d for d in feature_dir.iterdir() if d.is_dir()]
    found_agg_structure = False
    for agg_dir in agg_dirs:
        model_dirs = [d for d in agg_dir.iterdir() if d.is_dir() and d.name in {'rf','logreg','gb','xgb','gru','mlp'}]
        if model_dirs:
            found_agg_structure = True
            for mdir in model_dirs:
                model_name = mdir.name
                for fold_dir in sorted(mdir.glob('fold_*')):
                    eval_file = fold_dir / 'eval_results.json'
                    if eval_file.exists():
                        row = extract_metrics(eval_file, model_name)
                        if row is not None:
                            if row['fold'] is None:
                                try:
                                    row['fold'] = int(fold_dir.name.split('_')[-1])
                                except Exception:
                                    row['fold'] = fold_dir.name
                            rows.append(row)
    # Fallback for old per-model structure (no aggregation subdir)
    if not found_agg_structure:
        model_dirs = [d for d in feature_dir.iterdir() if d.is_dir() and d.name in {'rf','logreg','gb','xgb','gru','mlp'}]
        if model_dirs:
            for mdir in model_dirs:
                model_name = mdir.name
                for fold_dir in sorted(mdir.glob('fold_*')):
                    eval_file = fold_dir / 'eval_results.json'
                    if eval_file.exists():
                        row = extract_metrics(eval_file, model_name)
                        if row is not None:
                            if row['fold'] is None:
                                try:
                                    row['fold'] = int(fold_dir.name.split('_')[-1])
                                except Exception:
                                    row['fold'] = fold_dir.name
                            rows.append(row)
        else:
            # Fallback very old layout (no per-model, just fold_* at top level)
            for fold_dir in sorted(feature_dir.glob('fold_*')):
                eval_file = fold_dir / 'eval_results.json'
                if eval_file.exists():
                    row = extract_metrics(eval_file, 'unknown')
                    if row is not None:
                        if row['fold'] is None:
                            try:
                                row['fold'] = int(fold_dir.name.split('_')[-1])
                            except Exception:
                                row['fold'] = fold_dir.name
                        rows.append(row)
    if not rows:
        return None
    out_csv = feature_dir / 'metrics_summary.csv'
    return out_csv, rows


def write_csv(out_csv: Path, rows, overwrite: bool):
    if out_csv.exists() and not overwrite:
        return
    # Round metrics to 3 decimal places
    rounded = []
    for r in rows:
        rr = r.copy()
        for k in ['OA','F1','AUROC','FPR']:
            v = rr.get(k)
            if isinstance(v, (int,float)) and v == v:  # not NaN
                rr[k] = f"{v:.3f}"
            else:
                rr[k] = ''
        rounded.append(rr)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','fold','OA','F1','AUROC','FPR'])
        writer.writeheader()
        for r in rounded:
            writer.writerow(r)


def main():
    args = parse_args()
    model_runs_root = Path(args.model_runs_root)
    
    if args.feature_types is None:
        # Infer feature types as directories under model_runs (excluding special dirs like cv)
        args.feature_types = [d.name for d in model_runs_root.iterdir() if d.is_dir() and d.name != 'cv']
    for ft in args.feature_types:
        result = process_feature_type(ft, model_runs_root)
        if result is None:
            continue
        out_csv, rows = result
        write_csv(out_csv, rows, args.overwrite)
        print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == '__main__':
    main()
