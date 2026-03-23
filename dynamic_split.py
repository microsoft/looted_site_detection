"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Dynamic stratified split generation for feature datasets.
"""
from typing import Dict, List
import numpy as np
from .utils import load_features, extract_class
from sklearn.model_selection import train_test_split

def generate_stratified_site_splits(feature_type: str,
                                    test_size: float = 0.2,
                                    val_size: float = 0.1,
                                    seed: int = 42) -> Dict[str, List[str]]:
    """Generate disjoint stratified splits based on unique site_name.

    Returns lists of site_name strings ensuring:
        - No overlap among train/val/test
        - Class balance preserved via stratification on looted vs preserved
    """
    assert 0 < test_size < 0.5, "test_size should be in (0,0.5)"
    assert 0 < val_size < 0.5, "val_size should be in (0,0.5)"
    df = load_features(feature_type)
    sites = np.array(sorted(df['site_name'].unique()))  # length ~1943
    labels = np.array([extract_class(s) for s in sites])
    indices = np.arange(len(sites))
    idx_temp, idx_test, y_temp, y_test = train_test_split(indices, labels, test_size=test_size, stratify=labels, random_state=seed)
    val_rel = val_size / (1 - test_size)  # fraction of remaining data for validation
    idx_train, idx_val, y_train, y_val = train_test_split(idx_temp, y_temp, test_size=val_rel, stratify=y_temp, random_state=seed)
    return {
        'train': sites[idx_train].tolist(),
        'val': sites[idx_val].tolist(),
        'test': sites[idx_test].tolist()
    }
