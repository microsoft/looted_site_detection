# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Dynamic dataset splitting for CNN image-based training.
Creates train/val/test splits with no site overlap, using only 2023 images.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_image_based_splits(
    data_root: str,
    labels_csv: str = None,  # Optional: can extract from site names
    test_size: float = 0.2,
    val_size: float = 0.1,
    year: int = None,  # None = all years, int = specific year
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create stratified train/val/test splits for image-based CNN training.
    
    If year is specified (e.g., 2023), uses only images from that year to ensure label consistency.
    If year is None, uses all available images from 2017-2023, which may create temporal label noise.
    
    Expected structure:
        data_root/
        ├── looted/
        │   ├── 0/
        │   │   ├── 2016_01.jpg
        │   │   ├── 2023_12.jpg
        │   │   └── mask.png
        │   └── ...
        └── preserved/
            ├── 0/
            └── ...
    
    Args:
        data_root: Root directory containing looted/ and preserved/ subdirectories
        labels_csv: Not used (labels extracted from directory structure)
        test_size: Fraction of data for test set (default 0.2)
        val_size: Fraction of total data for validation set (default 0.1)
        year: Year to use for images (default 2023 - most recent/final state)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing site ID lists (e.g., "looted_0", "preserved_123")
    """
    data_root = Path(data_root)
    
    print(f"Looking for images in: {data_root}")
    
    # Find sites in looted/ and preserved/ directories
    available_sites = {}  # site_id -> count of year images
    if year is not None:
        year_pattern = f"{year}_"
        year_desc = f"from year {year}"
    else:
        year_pattern = None  # Match all years
        year_desc = "from all years (2017-2023)"
    
    for class_name in ['looted', 'preserved']:
        class_dir = data_root / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
        
        # Iterate through site directories
        for site_dir in class_dir.iterdir():
            if not site_dir.is_dir():
                continue
            
            site_num = site_dir.name
            site_id = f"{class_name}_{site_num}"
            
            # Count images from specified year (or all years if year=None)
            if year_pattern is not None:
                # Filter by specific year
                year_images = [f for f in site_dir.iterdir()
                              if f.suffix.lower() in ['.jpg', '.tif', '.png']
                              and f.stem.startswith(year_pattern)
                              and f.stem != 'mask']
            else:
                # Use all images (any year from 2017-2023)
                year_images = [f for f in site_dir.iterdir()
                              if f.suffix.lower() in ['.jpg', '.tif', '.png']
                              and f.stem != 'mask'
                              and any(f.stem.startswith(f"{y}_") for y in range(2017, 2024))]

            if len(year_images) > 0:
                available_sites[site_id] = len(year_images)
    
    print(f"Found {len(available_sites)} sites with images {year_desc}")
    if len(available_sites) == 0:
        # Fallback attempt: common situation is running from project root while
        # actual dataset resides under 'looted_site_detection/datasets'. If the
        # provided data_root is relative and lacks class subdirectories, try
        # prefixing with 'looted_site_detection/'. Avoid infinite recursion by
        # checking if we've already attempted the fallback.
        if 'looted_site_detection' not in str(data_root):
            candidate = Path('looted_site_detection') / data_root.name
            looted_dir = candidate / 'looted'
            preserved_dir = candidate / 'preserved'
            if looted_dir.exists() and preserved_dir.exists():
                print(f"[dynamic_split_images] Fallback data_root resolved to: {candidate}")
                return create_image_based_splits(
                    data_root=str(candidate),
                    labels_csv=labels_csv,
                    test_size=test_size,
                    val_size=val_size,
                    year=year,
                    seed=seed
                )
        raise ValueError(f"No sites found with images {year_desc}. Checked root: {data_root}")
    
    # Show stats
    total_year_images = sum(available_sites.values())
    print(f"Total images {year_desc}: {total_year_images}")
    print(f"Average images per site: {total_year_images/len(available_sites):.1f}")
    
    # Extract labels from site IDs (ensure deterministic ordering by sorting)
    print("Extracting labels from directory structure (sorted for determinism)...")
    site_labels = {}
    for site_id in sorted(available_sites.keys()):  # sorted for stable cross-method consistency
        if site_id.startswith('looted_'):
            site_labels[site_id] = 1
        elif site_id.startswith('preserved_'):
            site_labels[site_id] = 0
        else:
            print(f"Warning: Cannot determine label for site '{site_id}'")
            continue

    site_ids = np.array(sorted(site_labels.keys()))  # second sort defensively
    labels = np.array([site_labels[sid] for sid in site_ids])
    
    if len(site_ids) == 0:
        raise ValueError(f"No sites found with both labels and images in year {year}")
    
    # Count class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution in {year}:")
    for lbl, cnt in zip(unique_labels, counts):
        class_name = "Preserved" if lbl == 0 else "Looted"
        print(f"  {class_name} ({lbl}): {cnt}")
    
    # Extract site IDs and labels
    # Already have them from above
    
    # Stratified split: train+val vs test
    np.random.seed(seed)
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        site_ids, labels, 
        test_size=test_size, 
        stratify=labels,
        random_state=seed
    )
    
    # Stratified split: train vs val (from train_val set)
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to train_val set
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_labels,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=seed
    )
    
    # Create split dictionary
    splits = {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
    
    # Verify no overlap
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])
    
    assert len(train_set & val_set) == 0, "Train and val sets overlap!"
    assert len(train_set & test_set) == 0, "Train and test sets overlap!"
    assert len(val_set & test_set) == 0, "Val and test sets overlap!"
    
    # Print split statistics
    print(f"\n{'='*60}")
    print(f"Split Statistics (Year {year}):")
    print(f"{'='*60}")
    print(f"Train: {len(train_ids)} sites ({len(train_ids)/len(site_ids)*100:.1f}%)")
    print(f"  - Preserved: {(train_labels == 0).sum()}")
    print(f"  - Looted: {(train_labels == 1).sum()}")
    print(f"Val: {len(val_ids)} sites ({len(val_ids)/len(site_ids)*100:.1f}%)")
    print(f"  - Preserved: {(val_labels == 0).sum()}")
    print(f"  - Looted: {(val_labels == 1).sum()}")
    print(f"Test: {len(test_ids)} sites ({len(test_ids)/len(site_ids)*100:.1f}%)")
    print(f"  - Preserved: {(test_labels == 0).sum()}")
    print(f"  - Looted: {(test_labels == 1).sum()}")
    print(f"{'='*60}")
    
    return splits


def save_splits(splits: Dict[str, List[str]], output_path: str):
    """Save splits to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Splits saved to: {output_path}")


def load_splits(splits_path: str) -> Dict[str, List[str]]:
    """Load splits from JSON file."""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    print(f"✓ Loaded splits from: {splits_path}")
    print(f"  Train: {len(splits['train'])} sites")
    print(f"  Val: {len(splits['val'])} sites")
    print(f"  Test: {len(splits['test'])} sites")
    
    return splits

# Alias to mirror feature-based naming for external uniformity/import convenience.
def generate_stratified_site_splits_images(**kwargs) -> Dict[str, List[str]]:
    """Alias wrapper for create_image_based_splits providing a parallel name to
    generate_stratified_site_splits in feature-based workflow.
    Accepts the same keyword arguments as create_image_based_splits.
    """
    return create_image_based_splits(**kwargs)


if __name__ == '__main__':
    """
    Example usage:
    
    python -m looted_site_detection.dynamic_split_images \
        --data_root change_detection/planet_mosaics_final_4bands \
        --labels_csv looted_site_detection/data/site_labels.csv \
        --year 2023 \
        --output splits_2023.json
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Create stratified splits for image-based CNN training')
    parser.add_argument('--data_root', type=str, 
                       default='change_detection/planet_mosaics_final_4bands',
                       help='Root directory with monthly image folders')
    parser.add_argument('--labels_csv', type=str,
                       default=None,
                       help='Optional CSV file with site_id and label columns (if not provided, extracts from site names)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction for test set (default 0.2)')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Fraction for validation set (default 0.1)')
    parser.add_argument('--year', type=int, default=2023,
                       help='Year to use for images (default 2023)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default 42)')
    parser.add_argument('--output', type=str, default='splits_2023.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Create splits
    splits = create_image_based_splits(
        data_root=args.data_root,
        labels_csv=args.labels_csv,
        test_size=args.test_size,
        val_size=args.val_size,
        year=args.year,
        seed=args.seed
    )
    
    # Save splits
    save_splits(splits, args.output)
