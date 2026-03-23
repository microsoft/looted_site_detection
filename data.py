"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
from .utils import load_features, build_temporal_matrix, aggregate_features, extract_class, subset_sites, compute_monthly_stats, apply_monthly_normalization, fit_pca_concat, transform_pca_concat
from .splits import get_site_ids
import numpy as np
from typing import Tuple, List

class SiteFeatureDataset:
    def __init__(self, feature_type: str, aggregation: str, split: str, fold: int = 1, subset: int = None, year: int | None = None, site_ids_dict=None,
                 normalize: bool = False, norm_method: str = 'standard', norm_stats: dict | None = None,
                 pca_model=None):
        self.feature_type = feature_type
        self.aggregation = aggregation
        self.split = split
        self.fold = fold
        self.normalize = normalize
        self.norm_method = norm_method
        raw_df = load_features(feature_type)
        raw_df['site_id'] = raw_df['site_name'].apply(lambda s: int(s.split('_')[-1]) if '_' in s else -1)
        if site_ids_dict is None:
            # Legacy fold-based IDs (numeric) retained for backward compatibility
            looted_ids, preserved_ids = get_site_ids(split, fold)
            keep_ids_numeric = set(looted_ids) | set(preserved_ids)
            raw_df = raw_df[raw_df['site_id'].isin(keep_ids_numeric)].copy()
        else:
            split_entries = site_ids_dict[split]
            if not split_entries:
                raw_df = raw_df.iloc[0:0].copy()
            else:
                first = split_entries[0]
                if isinstance(first, str):  # New dynamic split returns site_name strings
                    keep_names = set(split_entries)
                    raw_df = raw_df[raw_df['site_name'].isin(keep_names)].copy()
                else:  # Fallback: assume numeric site IDs as before
                    keep_ids_numeric = set(split_entries)
                    raw_df = raw_df[raw_df['site_id'].isin(keep_ids_numeric)].copy()
        all_site_names = sorted(raw_df['site_name'].unique())
        if subset and subset < len(all_site_names):
            chosen = set(np.random.choice(all_site_names, size=subset, replace=False))
            raw_df = raw_df[raw_df['site_name'].isin(chosen)].copy()
        X, site_ids, months, feature_cols, mask = build_temporal_matrix(raw_df, year=year)
        self.mask = mask
        site_names = sorted(raw_df['site_name'].unique())
        labels = [extract_class(s) for s in site_names]
        # Normalization (before aggregation)
        if normalize:
            if norm_stats is None:
                self.norm_stats = compute_monthly_stats(X, mask, months, method=norm_method)
            else:
                self.norm_stats = norm_stats
            X = apply_monthly_normalization(X, mask, self.norm_stats)
        else:
            self.norm_stats = None
        # PCA aggregation: flatten temporal dimension then reduce back to original feature size
        if aggregation == 'pca':
            n_components = X.shape[2]  # target original monthly feature dimensionality
            if pca_model is None:
                self.pca_model = fit_pca_concat(X, n_components)
            else:
                self.pca_model = pca_model
            self.X = transform_pca_concat(X, self.pca_model)
        elif aggregation == 'none':
            self.pca_model = None
            self.X = X
        else:
            self.pca_model = None
            self.X = aggregate_features(X, aggregation, mask)
        self.y = np.array(labels, dtype=np.int64)
        self.site_ids = site_ids
        self.months = months
        if aggregation == 'none':
            self.num_features = X.shape[-1]
        else:
            self.num_features = self.X.shape[-1]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        return self.X, self.y, self.site_ids

    def get_norm_stats(self):
        return self.norm_stats

    def get_pca_model(self):
        return self.pca_model
