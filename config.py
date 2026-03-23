# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

# Root paths
REPO_ROOT = Path(__file__).resolve().parent.parent

# Feature configuration
# Allow experiment scripts (e.g., tmux launchers) to override feature location/type
# without requiring manual edits to this file.
_FEATURE_VARIANT = os.environ.get('LOOTED_FEATURE_VARIANT', '').strip().lower()  # 'with_mask' | 'without_mask'
_FEATURE_ROOT_OVERRIDE = os.environ.get('LOOTED_FEATURE_ROOT', '').strip()

if _FEATURE_ROOT_OVERRIDE:
    FEATURE_ROOT = Path(_FEATURE_ROOT_OVERRIDE)
else:
    if _FEATURE_VARIANT == 'without_mask':
        FEATURE_ROOT = REPO_ROOT / 'looted_site_detection' / 'data' / 'features_without_mask_2023'
    else:
        # Default stays masked to match prior behavior in this repo.
        FEATURE_ROOT = REPO_ROOT / 'looted_site_detection' / 'data' / 'features_with_mask_2023'
FOLD_DICT_PATH = REPO_ROOT / 'change_detection' / 'datasets' / 'fold_dict.json'

# Months available (2016_01 .. 2023_12)
MONTHS = [f'{year}_{month}' for year in range(2016, 2024) for month in ['01','02','03','04','05','06','07','08','09','10','11','12']]
CUTOFF_MONTH = '2023_12'
MONTHS_UP_TO_CUTOFF = [m for m in MONTHS if int(m.split('_')[0]) < 2024 and m <= CUTOFF_MONTH]

CLASS_MAP = {'looted':1,'preserved':0}
INVERSE_CLASS_MAP = {v:k for k,v in CLASS_MAP.items()}

_EMBEDDING_2023_FILES_WITHOUT_MASK = {
    'satclip': 'features_satclip_2023.csv',
    'satmae': 'features_satmae_2023.csv',
    'satlaspretrain': 'features_satlaspretrain_2023.csv',
    'prithvi-eo-2.0': 'features_prithvi-eo-2.0_2023.csv',
    'georsclip': 'features_georsclip_2023.csv',
    'dinov3': 'features_dinov3_2023.csv',
}

_EMBEDDING_2023_FILES_WITH_MASK = {
    'satclip': 'features_satclip_2023_masked.csv',
    'satmae': 'features_satmae_2023_masked.csv',
    'satlaspretrain': 'features_satlaspretrain_2023_masked.csv',
    'prithvi-eo-2.0': 'features_prithvi-eo-2.0_2023_masked.csv',
    'georsclip': 'features_georsclip_2023_masked.csv',
    'dinov3': 'features_dinov3_2023_masked.csv',
}

FEATURE_FILE_MAP = {
    # Kept for backward compatibility (not present in 2023-only embedding dirs)
    'handcrafted': 'features_handcrafted_monthly.csv',
}

if _FEATURE_VARIANT == 'without_mask':
    FEATURE_FILE_MAP.update(_EMBEDDING_2023_FILES_WITHOUT_MASK)
else:
    FEATURE_FILE_MAP.update(_EMBEDDING_2023_FILES_WITH_MASK)

RANDOM_SEED = 42

PLANETSCOPE_MEAN = [172.39825689, 149.42724701, 111.42677006]
PLANETSCOPE_STD = [42.36875904, 40.11172176, 42.71382535]
