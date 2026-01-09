#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Batch runner for visualization experiments.

Runs:
- Time-series (month=7) for random looted and preserved sites
- Mask overlays (year=2020, month=12) for random looted (excluding 'looted_0') and preserved sites

Outputs saved to results/sites.
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import List

from visualize_sites import (
    plot_time_series,
    plot_mask_overlay,
    list_site_ids,
    get_base_dir,
)
import argparse

def pick_random(items: List[str], k: int, exclude: List[str] | None = None) -> List[str]:
    pool = [x for x in items if (exclude is None or x not in exclude)]
    if len(pool) < k:
        raise ValueError(f"Not enough items to pick {k} random elements. Available: {len(pool)}")
    random.shuffle(pool)
    return pool[:k]

def main():
    parser = argparse.ArgumentParser(description='Batch visualization runner')
    parser.add_argument('--n_looted_ts', type=int, default=10, help='Number of looted time-series to generate')
    parser.add_argument('--n_preserved_ts', type=int, default=10, help='Number of preserved time-series to generate')
    parser.add_argument('--n_looted_overlay', type=int, default=8, help='Number of looted mask overlays to generate (excluding looted_0)')
    parser.add_argument('--n_preserved_overlay', type=int, default=5, help='Number of preserved mask overlays to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--base_dir', type=Path, default=None, help='Root directory of mosaics (images/masks/masks_buffered)')
    args = parser.parse_args()

    random.seed(args.seed)

    # Initialize visualize_sites module globals for directory resolution
    base_dir = get_base_dir(args.base_dir)
    import visualize_sites as vs
    vs.IMAGES_DIR = base_dir / 'images'
    vs.MASKS_DIR = base_dir / 'masks'
    vs.MASKS_BUFFERED_DIR = base_dir / 'masks_buffered'

    # Collect site ids
    looted_sites = list_site_ids('looted')
    preserved_sites = list_site_ids('preserved')

    # time-series for looted (month=7)
    for site_id in pick_random(looted_sites, args.n_looted_ts):
        out = plot_time_series(site_id, 7)
        print(f"[BATCH] time_series looted: {out}")

    # time-series for preserved (month=7)
    for site_id in pick_random(preserved_sites, args.n_preserved_ts):
        out = plot_time_series(site_id, 7)
        print(f"[BATCH] time_series preserved: {out}")

    # mask overlays for looted excluding 'looted_0' (2020-12)
    for site_id in pick_random(looted_sites, args.n_looted_overlay, exclude=['looted_0']):
        out = plot_mask_overlay(site_id, 2020, 12, no_image=True)
        print(f"[BATCH] mask_overlay looted: {out}")

    # mask overlays for preserved (2020-12)
    for site_id in pick_random(preserved_sites, args.n_preserved_overlay):
        out = plot_mask_overlay(site_id, 2020, 12, no_image=True)
        print(f"[BATCH] mask_overlay preserved: {out}")

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
