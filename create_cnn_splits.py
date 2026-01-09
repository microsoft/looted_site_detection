#!/usr/bin/env python
"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Helper script to generate CNN image-based stratified splits.

Run from inside the looted_site_detection directory without needing the package import path.
Example:
    python create_cnn_splits.py \
        --data_root ../change_detection/planet_mosaics_final_4bands/datasets \
        --year 2023 \
        --output tmp_cnn_splits.json
"""
import argparse
import json
from pathlib import Path

# Use relative import so this works when executed from within the package dir
from dynamic_split_images import create_image_based_splits

def parse_args():
    p = argparse.ArgumentParser(description="Generate year-filtered CNN splits")
    p.add_argument('--data_root', type=str, required=True,
                   help='Root directory containing looted/ and preserved/ subdirectories')
    p.add_argument('--year', type=int, default=2023,
                   help='Year to filter images (default 2023)')
    p.add_argument('--test_size', type=float, default=0.2,
                   help='Fraction for test set (default 0.2)')
    p.add_argument('--val_size', type=float, default=0.1,
                   help='Fraction for validation set (default 0.1)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default 42)')
    p.add_argument('--output', type=str, default='tmp_cnn_splits.json',
                   help='Output JSON file (default tmp_cnn_splits.json)')
    return p.parse_args()

def main():
    args = parse_args()
    splits = create_image_based_splits(
        data_root=args.data_root,
        labels_csv=None,
        test_size=args.test_size,
        val_size=args.val_size,
        year=args.year,
        seed=args.seed
    )
    out_path = Path(args.output)
    out_path.write_text(json.dumps(splits, indent=2))
    print(f"Saved splits to {out_path}:")
    for k in ['train','val','test']:
        print(f"  {k}: {len(splits[k])} sites")

if __name__ == '__main__':
    main()
