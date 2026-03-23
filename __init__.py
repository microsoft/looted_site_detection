"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Looted Site Detection package.

Exports a unified dynamic splitting interface for both feature-based and CNN image-based training so that
downstream code (and interactive usage) can access consistent function names.

Feature-based splitting:
	generate_stratified_site_splits(feature_type, test_size=0.2, val_size=0.1, seed=42)

CNN image-based splitting (year-filtered for label consistency):
	generate_stratified_site_splits_images(data_root, test_size=0.2, val_size=0.1, year=2023, seed=42)

Both return a dict with keys: 'train', 'val', 'test' mapping to lists of site identifiers.
"""

from .dynamic_split import generate_stratified_site_splits  # feature-based
from .dynamic_split_images import create_image_based_splits
from .dynamic_split_images import create_image_based_splits as generate_stratified_site_splits_images

__all__ = [
	'generate_stratified_site_splits',
	'generate_stratified_site_splits_images',
	'create_image_based_splits'
]
