"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
from .config import FOLD_DICT_PATH, CLASS_MAP

def load_fold_dict(path: Path = FOLD_DICT_PATH) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def get_site_ids(split: str, fold: int = 1, fold_dict: Dict = None) -> Tuple[List[int], List[int]]:
    if fold_dict is None:
        fold_dict = load_fold_dict()
    assert split in {'train','val','test'}, f"split must be one of train|val|test not {split}"
    if split == 'test':
        d = fold_dict['test']
        return d['looted'], d['preserved']
    if split == 'val':
        key = f'val_{fold}'
        d = fold_dict[key]
        return d['looted'], d['preserved']
    train_looted = set(fold_dict['train']['looted'])
    train_preserved = set(fold_dict['train']['preserved'])
    for k in range(1,6):
        if k != fold:
            vk = fold_dict[f'val_{k}']
            train_looted.update(vk['looted'])
            train_preserved.update(vk['preserved'])
    return sorted(train_looted), sorted(train_preserved)

def assign_split(site_id: int, looted_ids: List[int], preserved_ids: List[int]) -> int:
    if site_id in looted_ids:
        return CLASS_MAP['looted']
    if site_id in preserved_ids:
        return CLASS_MAP['preserved']
    return -1
