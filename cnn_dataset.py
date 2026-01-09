"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.

Image-based Dataset for CNN models - Simplified for data/datasets/ directory structure.
Loads raw satellite imagery from: data/datasets/{looted|preserved}/{site_num}/{YYYY_MM}.jpg
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import os
from scipy.ndimage import binary_dilation
import logging

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
except ImportError:
    raise ImportError("PyTorch required for CNN models. Install: pip install torch torchvision")


class RandomRotation90:
    """Randomly rotate image by 0, 90, 180, or 270 degrees."""
    def __call__(self, img):
        k = np.random.randint(0, 4)
        return torch.rot90(img, k, dims=[-2, -1])


class ImageDataset(Dataset):
    """
    PyTorch Dataset that loads raw satellite imagery for CNN models.
    
    Expected structure:
        data_root/
        ├── looted/
        │   ├── 0/
        │   │   ├── 2016_01.jpg
        │   │   ├── ...
        │   │   ├── 2023_12.jpg
        │   │   └── mask.png
        │   ├── 1/
        │   └── ...
        └── preserved/
            ├── 0/
            │   ├── 2016_01.jpg
            │   ├── ...
            │   ├── 2023_12.jpg
            │   └── mask.png
            └── ...
    
    Args:
        data_root: Root directory containing looted/ and preserved/ subdirectories
        mask_root: Not used (masks are in site directories)
        labels_csv: Not used (labels extracted from directory names)
        site_ids: List of site IDs (e.g., "looted_0", "preserved_123")
        image_size: Resize images to this size (default 224 for ResNet/UNet)
        num_temporal_steps: Number of temporal images to load per site
        mask_mode: How to apply masks ('multiply', 'concat', 'none')
        augment: Apply data augmentation (training only)
        normalize: Apply normalization
        year_filter: Only use images from this year (e.g., 2023)
    """
    
    def __init__(
        self,
        data_root: str,
        mask_root: str,  # Not used but kept for compatibility
        labels_csv: str,  # Not used but kept for compatibility
        site_ids: List[str],
        image_size: int = 224,
        num_temporal_steps: int = 1,
        mask_mode: str = 'multiply',
        augment: bool = False,
        normalize: bool = True,
        year_filter: Optional[int] = None,
        mask_selection: str = 'auto',  # 'auto' (prefer buffered), 'plain', or 'buffered'
        enforce_min_area: bool = False,
        min_area_pixels: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.num_temporal_steps = num_temporal_steps
        self.mask_mode = mask_mode
        self.augment = augment
        self.normalize = normalize
        self.year_filter = year_filter
        self.mask_selection = mask_selection
        self.enforce_min_area = enforce_min_area
        self.min_area_pixels = min_area_pixels
        
        # Build mapping of site_id -> (class_dir, site_num, site_path)
        self.site_info = {}
        self.labels = {}
        
        for site_id in site_ids:
            # Extract class and number from site_id
            if site_id.startswith('looted_'):
                class_dir = 'looted'
                site_num = site_id.replace('looted_', '')
                label = 1
            elif site_id.startswith('preserved_'):
                class_dir = 'preserved'
                site_num = site_id.replace('preserved_', '')
                label = 0
            else:
                continue
            
            site_path = self.data_root / class_dir / site_num
            if site_path.exists():
                self.site_info[site_id] = {
                    'path': site_path,
                    'class': class_dir,
                    'number': site_num
                }
                self.labels[site_id] = label
        
        # Filter to valid sites only
        self.valid_sites = list(self.site_info.keys())
        
        # Setup transforms
        self._setup_transforms()
        
        year_msg = f" (year {year_filter})" if year_filter else ""
        print(f"ImageDataset: {len(self.valid_sites)} sites{year_msg}, "
              f"image_size={image_size}, temporal_steps={num_temporal_steps}, "
              f"mask_mode={mask_mode}, augment={augment}")
    
    def _setup_transforms(self):
        """Setup image transforms (augmentation and normalization)."""
        # PlanetScope dataset statistics
        mean = [172.39825689 / 255.0, 149.42724701 / 255.0, 111.42677006 / 255.0]
        std = [42.36875904 / 255.0, 40.11172176 / 255.0, 42.71382535 / 255.0]
        
        self.to_tensor = transforms.ToTensor()
        
        if self.augment:
            self.aug_flip_h = transforms.RandomHorizontalFlip(p=0.5)
            self.aug_flip_v = transforms.RandomVerticalFlip(p=0.5)
            self.aug_rotate = RandomRotation90()
            self.aug_crop = transforms.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.3, 1.0),
                ratio=(0.75, 1.33)
            )
        else:
            self.resize = transforms.Resize((self.image_size, self.image_size))
        
        if self.normalize:
            self.normalizer = transforms.Normalize(mean=mean, std=std)
    
    def _get_available_images(self, site_id: str) -> List[str]:
        """Get list of available image files for a site, filtered by year if specified."""
        site_path = self.site_info[site_id]['path']
        
        # Get all image files
        all_images = []
        for ext in ['.jpg', '.tif', '.png']:
            all_images.extend(site_path.glob(f'*{ext}'))
        
        # Filter by year if specified
        if self.year_filter is not None:
            year_pattern = f"{self.year_filter}_"
            filtered = [img for img in all_images if img.stem.startswith(year_pattern) and img.stem != 'mask']
        else:
            filtered = [img for img in all_images if img.stem != 'mask']
        
        return sorted(filtered, key=lambda x: x.stem)
    
    def _load_image(self, img_path: Path) -> Optional[np.ndarray]:
        """Load a single image."""
        try:
            img = Image.open(img_path).convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def _load_mask(self, site_id: str) -> Optional[np.ndarray]:
        """Load site boundary mask."""
        site_path = self.site_info[site_id]['path']
        buffered_path = site_path / 'mask_buffered.png'
        plain_path = site_path / 'mask.png'

        chosen = None
        sel = (self.mask_selection or 'auto').lower()
        if sel == 'buffered':
            chosen = buffered_path if buffered_path.exists() else (plain_path if plain_path.exists() else None)
        elif sel == 'plain':
            chosen = plain_path if plain_path.exists() else (buffered_path if buffered_path.exists() else None)
        else:  # auto: prefer buffered
            if buffered_path.exists():
                chosen = buffered_path
            elif plain_path.exists():
                chosen = plain_path
        
        if chosen is not None:
            try:
                mask = Image.open(chosen).convert('L')
                # Optional debug: print which mask file was selected
                if os.environ.get('DEBUG_MASK_SELECTION', '').strip().lower() in ('1','true','yes'):
                    print(f"[mask] Using: {chosen}")
                mask_arr = np.array(mask)
                if self.enforce_min_area and self.min_area_pixels is not None:
                    # Ensure binary mask (0/1)
                    bin_mask = (mask_arr > 0).astype(np.uint8)
                    current_area = int(bin_mask.sum())
                    if current_area < self.min_area_pixels and current_area > 0:
                        # Iteratively dilate until area >= min_area_pixels
                        # Use a 3x3 structuring element for controlled growth
                        selem = np.ones((3,3), dtype=bool)
                        safety_counter = 0
                        while bin_mask.sum() < self.min_area_pixels and safety_counter < 2048:
                            bin_mask = binary_dilation(bin_mask.astype(bool), structure=selem).astype(np.uint8)
                            safety_counter += 1
                        return (bin_mask * 255).astype(np.uint8)
                    else:
                        return mask_arr
                return mask_arr
            except Exception as e:
                print(f"Error loading mask {chosen}: {e}")
        return None
    
    def __len__(self) -> int:
        return len(self.valid_sites)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            image: Tensor of shape [C, H, W] for single image
            label: 0 (preserved) or 1 (looted)
            site_id: Site identifier string
        """
        site_id = self.valid_sites[idx]
        label = self.labels[site_id]
        
        # Get available images
        available_images = self._get_available_images(site_id)
        
        if len(available_images) == 0:
            logging.warning(f"No images found for site {site_id}")
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
            return img_tensor, label, site_id
        
        # Sample image(s)
        if self.num_temporal_steps == 1:
            # Single random image
            img_path = available_images[np.random.randint(0, len(available_images))]
        else:
            # Sample uniformly across available images
            indices = np.linspace(0, len(available_images) - 1, self.num_temporal_steps, dtype=int)
            img_path = available_images[indices[0]]  # For now, just use first
        
        # Load image
        img_array = self._load_image(img_path)
        if img_array is None:
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
            return img_tensor, label, site_id
        
        # Convert to PIL for transforms
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        
        # Apply augmentation
        if self.augment:
            img_pil = self.aug_crop(img_pil)
            img_pil = self.aug_flip_h(img_pil)
            img_pil = self.aug_flip_v(img_pil)
        else:
            img_pil = self.resize(img_pil)
        
        # Convert to tensor
        img_tensor = self.to_tensor(img_pil)  # [C, H, W]
        
        # Apply rotation augmentation (after tensor conversion)
        if self.augment:
            img_tensor = self.aug_rotate(img_tensor)
        
        # Apply mask
        if self.mask_mode != 'none':
            mask_array = self._load_mask(site_id)
            if mask_array is not None:
                mask_pil = Image.fromarray(mask_array.astype(np.uint8))
                if self.augment:
                    mask_pil = self.aug_crop(mask_pil)
                    mask_pil = self.aug_flip_h(mask_pil)
                    mask_pil = self.aug_flip_v(mask_pil)
                else:
                    mask_pil = self.resize(mask_pil)
                
                mask_tensor = self.to_tensor(mask_pil)[0:1, :, :]  # [1, H, W]
                
                if self.augment:
                    mask_tensor = self.aug_rotate(mask_tensor)
                
                if self.mask_mode == 'multiply':
                    img_tensor = img_tensor * mask_tensor
                elif self.mask_mode == 'concat':
                    img_tensor = torch.cat([img_tensor, mask_tensor], dim=0)
        
        # Normalize
        if self.normalize and self.normalizer:
            # Only normalize first 3 channels (RGB), not mask channel if concatenated
            if self.mask_mode == 'concat':
                img_tensor[:3] = self.normalizer(img_tensor[:3])
            else:
                img_tensor = self.normalizer(img_tensor)
        
        return img_tensor, label, site_id


def create_image_datasets(
    data_root: str,
    mask_root: str,
    labels_csv: str,
    site_ids_dict: Dict[str, List[str]],
    image_size: int = 224,
    num_temporal_steps: int = 1,
    mask_mode: str = 'multiply',
    year_filter: Optional[int] = None,
    test_use_buffered_masks: bool = False,
    enforce_test_min_area: bool = False,
) -> Tuple[ImageDataset, ImageDataset, ImageDataset]:
    """
    Create train, val, test ImageDatasets.
    
    Args:
        data_root: Root directory with looted/ and preserved/ subdirectories
        mask_root: Not used (kept for compatibility)
        labels_csv: Not used (kept for compatibility)
        site_ids_dict: Dict with 'train', 'val', 'test' keys containing site ID lists
        image_size: Image size (default 224)
        num_temporal_steps: Number of temporal images per site
        mask_mode: Mask application mode
        year_filter: Only use images from this year (e.g., 2023 for final state)
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # Compute median area from training plain masks if needed
    median_area = None
    if enforce_test_min_area:
        areas: List[int] = []
        for sid in site_ids_dict.get('train', []):
            # Build site path
            if sid.startswith('looted_'):
                class_dir = 'looted'
                site_num = sid.replace('looted_', '')
            elif sid.startswith('preserved_'):
                class_dir = 'preserved'
                site_num = sid.replace('preserved_', '')
            else:
                continue
            site_path = Path(data_root) / class_dir / site_num
            plain_path = site_path / 'mask.png'
            if plain_path.exists():
                try:
                    arr = np.array(Image.open(plain_path).convert('L'))
                    areas.append(int((arr > 0).sum()))
                except Exception:
                    pass
        if areas:
            median_area = int(np.median(np.array(areas)))

    train_ds = ImageDataset(
        data_root=data_root,
        mask_root=mask_root,
        labels_csv=labels_csv,
        site_ids=site_ids_dict['train'],
        image_size=image_size,
        num_temporal_steps=num_temporal_steps,
        mask_mode=mask_mode,
        augment=True,
        normalize=True,
        year_filter=year_filter,
        mask_selection='plain',
        enforce_min_area=False,
    )
    
    val_ds = ImageDataset(
        data_root=data_root,
        mask_root=mask_root,
        labels_csv=labels_csv,
        site_ids=site_ids_dict['val'],
        image_size=image_size,
        num_temporal_steps=num_temporal_steps,
        mask_mode=mask_mode,
        augment=False,
        normalize=True,
        year_filter=year_filter,
        mask_selection='plain',
        enforce_min_area=False,
    )
    
    test_ds = ImageDataset(
        data_root=data_root,
        mask_root=mask_root,
        labels_csv=labels_csv,
        site_ids=site_ids_dict['test'],
        image_size=image_size,
        num_temporal_steps=num_temporal_steps,
        mask_mode=mask_mode,
        augment=False,
        normalize=True,
        year_filter=year_filter,
        mask_selection=('buffered' if test_use_buffered_masks else 'plain'),
        enforce_min_area=enforce_test_min_area,
        min_area_pixels=median_area if enforce_test_min_area else None,
    )
    
    return train_ds, val_ds, test_ds
