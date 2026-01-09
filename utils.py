"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from .config import FEATURE_ROOT, FEATURE_FILE_MAP, MONTHS_UP_TO_CUTOFF, CLASS_MAP, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def load_features(feature_type: str) -> pd.DataFrame:
    assert feature_type in FEATURE_FILE_MAP, f"Unknown feature_type {feature_type}. Available: {list(FEATURE_FILE_MAP)}"
    path = FEATURE_ROOT / FEATURE_FILE_MAP[feature_type]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    assert 'site_name' in df.columns and 'month' in df.columns, 'CSV must contain site_name and month columns.'
    return df

def extract_site_id(site_name: str) -> int:
    try:
        return int(site_name.split('_')[-1])
    except Exception:
        return -1

def extract_class(site_name: str) -> int:
    prefix = site_name.split('_')[0]
    return CLASS_MAP.get(prefix, -1)

def build_temporal_matrix(df: pd.DataFrame, year: int | None = None) -> Tuple[np.ndarray, List[int], List[str], List[str], np.ndarray]:
    feature_cols = [c for c in df.columns if c not in {'site_name','month','site_id'}]
    if year is not None:
        assert 2016 <= year <= 2023, f"Year must be between 2016 and 2023 inclusive, got {year}."
        months_selected = [m for m in MONTHS_UP_TO_CUTOFF if m.startswith(f'{year}_')]
    else:
        months_selected = MONTHS_UP_TO_CUTOFF
    df_filt = df[df['month'].isin(months_selected)].copy()
    sites = sorted(df_filt['site_name'].unique())
    months = months_selected
    full_index = pd.MultiIndex.from_product([sites, months], names=['site_name','month'])
    df_filt = df_filt.set_index(['site_name','month'])[feature_cols]
    df_reindexed = df_filt.reindex(full_index)
    mask_present = (~df_reindexed.isna().any(axis=1)).to_numpy().reshape(len(sites), len(months)).astype(np.int8)
    df_reindexed = df_reindexed.fillna(0.0)
    values = df_reindexed.to_numpy(dtype=np.float32).reshape(len(sites), len(months), len(feature_cols))
    site_ids = [extract_site_id(s) for s in sites]
    return values, site_ids, months, feature_cols, mask_present

def compute_monthly_stats(X: np.ndarray, mask: np.ndarray, months: List[str], method: str = 'standard') -> Dict:
    """
    Compute per-month scaling statistics using only rows where mask==1 for that month.
    X: (N_sites, N_months, N_features)
    mask: (N_sites, N_months) with 1 if month present, 0 if missing.
    method: 'standard' (mean/std) or 'robust' (median/IQR).
    Returns a dict containing statistics for JSON serialization.
    """
    assert method in {'standard','robust'}, f"Unknown norm method {method}"
    stats = {
        'method': method,
        'months': months,
        'per_month': []
    }
    for mi, m in enumerate(months):
        valid_rows = mask[:, mi] == 1
        vals = X[valid_rows, mi, :]
        if vals.size == 0:
            # Fallback zeros if no data (should rarely happen)
            if method == 'standard':
                mean = np.zeros(X.shape[-1], dtype=np.float32)
                std = np.ones(X.shape[-1], dtype=np.float32)
                stats['per_month'].append({'mean': mean.tolist(), 'std': std.tolist()})
            else:
                median = np.zeros(X.shape[-1], dtype=np.float32)
                iqr = np.ones(X.shape[-1], dtype=np.float32)
                stats['per_month'].append({'median': median.tolist(), 'iqr': iqr.tolist()})
            continue
        if method == 'standard':
            mean = vals.mean(axis=0)
            std = vals.std(axis=0)
            std[std == 0] = 1.0  # avoid divide-by-zero
            stats['per_month'].append({'mean': mean.tolist(), 'std': std.tolist()})
        else:  # robust
            median = np.median(vals, axis=0)
            q75 = np.percentile(vals, 75, axis=0)
            q25 = np.percentile(vals, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1.0
            stats['per_month'].append({'median': median.tolist(), 'iqr': iqr.tolist()})
    return stats

def apply_monthly_normalization(X: np.ndarray, mask: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Apply previously computed monthly normalization statistics to X.
    For missing months (mask==0) we set normalized features to 0 to preserve the notion of absence.
    """
    method = stats['method']
    per_month = stats['per_month']
    Xn = np.empty_like(X, dtype=np.float32)
    for mi in range(X.shape[1]):
        month_stats = per_month[mi]
        if method == 'standard':
            mean = np.array(month_stats['mean'], dtype=np.float32)
            std = np.array(month_stats['std'], dtype=np.float32)
            Xn[:, mi, :] = (X[:, mi, :] - mean) / std
        else:
            median = np.array(month_stats['median'], dtype=np.float32)
            iqr = np.array(month_stats['iqr'], dtype=np.float32)
            Xn[:, mi, :] = (X[:, mi, :] - median) / iqr
    # Re-zero missing months so aggregations using mask skip them or concat retains clear marker
    missing = mask == 0
    Xn[missing] = 0.0
    return Xn

def fit_pca_concat(X: np.ndarray, n_components: int) -> PCA:
    """Fit PCA on concatenated temporal features.
    X shape: (N_sites, N_months, N_features). We flatten months then fit PCA to reduce
    (N_months * N_features) -> n_components (typically = N_features).
    
    Automatically adjusts n_components if it exceeds min(n_samples, n_features).
    """
    N_sites, N_months, N_feat = X.shape
    X_flat = X.reshape(N_sites, N_months * N_feat)
    
    # PCA requires n_components <= min(n_samples, n_features)
    max_components = min(X_flat.shape[0], X_flat.shape[1])
    
    if n_components > max_components:
        # Use 1024 as fallback when requested components exceed limit
        adjusted_components = min(1024, max_components)
        print(f"[PCA] Requested n_components={n_components} exceeds limit (max={max_components})")
        print(f"[PCA] Adjusting to n_components={adjusted_components}")
        n_components = adjusted_components
    
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(X_flat)
    return pca

def transform_pca_concat(X: np.ndarray, pca: PCA) -> np.ndarray:
    N_sites, N_months, N_feat = X.shape
    X_flat = X.reshape(N_sites, N_months * N_feat)
    X_proj = pca.transform(X_flat)  # shape (N_sites, n_components)
    return X_proj.astype(np.float32)

def aggregate_features(X: np.ndarray, method: str, mask: np.ndarray = None) -> np.ndarray:
    if mask is None:
        mask = np.ones(X.shape[:2], dtype=bool)
    valid = mask.astype(bool)
    if method == 'concat':
        return X.reshape(X.shape[0], -1)
    def masked_stat(func):
        out = []
        for i in range(X.shape[0]):
            rows = X[i][valid[i]]
            if rows.size == 0:
                rows = np.zeros((1, X.shape[2]), dtype=np.float32)
            out.append(func(rows, axis=0))
        return np.stack(out, axis=0)
    if method == 'mean':
        return masked_stat(np.mean)
    if method == 'median':
        return masked_stat(np.median)
    if method == 'max':
        return masked_stat(np.max)
    if method == 'min':
        return masked_stat(np.min)
    if method == 'std':
        return masked_stat(np.std)
    raise ValueError(f'Unknown aggregation method {method}')

def subset_sites(X: np.ndarray, site_ids: List[int], labels: List[int], mask: np.ndarray, subset: int):
    if subset and subset < len(site_ids):
        idx = np.random.choice(len(site_ids), size=subset, replace=False)
        return X[idx], [site_ids[i] for i in idx], [labels[i] for i in idx], mask[idx]
    return X, site_ids, labels, mask
