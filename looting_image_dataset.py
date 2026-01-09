# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Image-based Dataset for CNN training on raw satellite imagery

"""

import os
import json
import numpy as np
import torch
import torch.utils.data as tdata
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from datetime import datetime
from torchvision.utils import _log_api_usage_once
import rasterio
from pathlib import Path


class RandomRotation90(torch.nn.Module):
    """Random rotation by 0, 90, 180, or 270 degrees"""
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    @staticmethod
    def get_params():
        angle = float([0, 90, 180, 270][np.random.randint(4)])
        return angle

    def forward(self, img):
        angle = self.get_params()
        return TF.rotate(img, angle)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class LootingImageDataset(tdata.Dataset):
    """
    PyTorch Dataset for loading raw satellite imagery for looting detection.
    
    Args:
        data_dir: Root directory containing processed_sites/
        metadata_path: Path to metadata CSV file
        split: 'train', 'val', or 'test'
        image_size: Tuple of (height, width) for resizing images (default: 224x224)
        norm_stats: Normalization statistics ('planetscope', 'imagenet', or 'planetscope')
        augment: Whether to apply data augmentation (only for training)
        single: If True, load single images; if False, load temporal sequences
        mask_mode: How to apply masks - 'none', 'multiply', or 'channel'
        use_buffered_masks: Whether to use buffered masks (default: True)
        temporal_sampling: Number of temporal samples to use during training (default: 24)
    """
    
    def __init__(
        self,
        data_dir,
        metadata_path,
        split='train',
        image_size=(224, 224),
        norm_stats='planetscope',
        augment=True,
        single=False,
        mask_mode='multiply',
        use_buffered_masks=True,
        temporal_sampling=24,
    ):
        super(LootingImageDataset, self).__init__()
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.single = single
        self.mask_mode = mask_mode
        self.temporal_sampling = temporal_sampling
        
        # Set normalization statistics
        if norm_stats == 'planetscope':
            # From reference_benchmark dataset
            self.mean = torch.tensor([172.39825689, 149.42724701, 111.42677006])
            self.std = torch.tensor([42.36875904, 40.11172176, 42.71382535])
        elif norm_stats == 'imagenet':
            # ImageNet statistics
            self.mean = torch.tensor([123.675, 116.28, 103.53])
            self.std = torch.tensor([58.395, 57.12, 57.375])
        elif norm_stats == 'planetscope':
            # PlanetScope statistics (to be computed from data)
            # Using reference_benchmark as default for now
            self.mean = torch.tensor([172.39825689, 149.42724701, 111.42677006])
            self.std = torch.tensor([42.36875904, 40.11172176, 42.71382535])
        else:
            raise ValueError(f'norm_stats must be "planetscope", "imagenet", or "planetscope", not {norm_stats}')
        
        # Set paths
        self.image_dir = self.data_dir / 'planet_mosaics_final_4bands' / 'images'
        if use_buffered_masks:
            self.mask_dir = self.data_dir / 'planet_mosaics_final_4bands' / 'masks_buffered'
        else:
            self.mask_dir = self.data_dir / 'planet_mosaics_final_4bands' / 'masks'
        
        # Load metadata and split data
        self._load_metadata(metadata_path, split)
        
        # Setup augmentation transforms
        self._setup_transforms()
        
        print(f'{split} dataset: {len(self)} samples ({self.num_looted} looted, {self.num_preserved} preserved)')
    
    def _load_metadata(self, metadata_path, split):
        """Load metadata and create train/val/test splits"""
        import pandas as pd
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        # Get unique sites
        sites = df['site_id'].unique()
        n_sites = len(sites)
        
        # Create stratified split based on labels
        # Get labels for stratification
        site_labels = {}
        for site in sites:
            site_data = df[df['site_id'] == site]
            # Use the most common label (or could use final label)
            site_labels[site] = site_data['label'].mode()[0]
        
        # Split sites (70% train, 10% val, 20% test)
        from sklearn.model_selection import train_test_split
        
        looted_sites = [s for s in sites if site_labels[s] == 1]
        preserved_sites = [s for s in sites if site_labels[s] == 0]
        
        # First split: separate test set (20%)
        looted_train_val, looted_test = train_test_split(
            looted_sites, test_size=0.2, random_state=42
        )
        preserved_train_val, preserved_test = train_test_split(
            preserved_sites, test_size=0.2, random_state=42
        )
        
        # Second split: separate validation from training (10/70 = 1/7)
        looted_train, looted_val = train_test_split(
            looted_train_val, test_size=1/7, random_state=42
        )
        preserved_train, preserved_val = train_test_split(
            preserved_train_val, test_size=1/7, random_state=42
        )
        
        # Combine splits
        if split == 'train':
            selected_sites = looted_train + preserved_train
        elif split == 'val':
            selected_sites = looted_val + preserved_val
        elif split == 'test':
            selected_sites = looted_test + preserved_test
        else:
            raise ValueError(f'split must be "train", "val", or "test", not {split}')
        
        # Filter dataframe to selected sites
        self.df = df[df['site_id'].isin(selected_sites)].reset_index(drop=True)
        
        # Create sample list
        self.samples = []
        self.labels = []
        
        if self.single:
            # Single-image mode: each (site, month) is a separate sample
            for _, row in self.df.iterrows():
                site_id = row['site_id']
                year_month = row['year_month']
                label = row['label']
                
                # Check if image exists
                image_path = self.image_dir / f"{site_id}_{year_month}.tif"
                if image_path.exists():
                    self.samples.append({
                        'site_id': site_id,
                        'year_month': year_month,
                        'image_path': image_path,
                        'label': label
                    })
                    self.labels.append(label)
        else:
            # Temporal sequence mode: each site is a sample with all its monthly images
            for site_id in selected_sites:
                site_data = self.df[self.df['site_id'] == site_id]
                label = site_labels[site_id]
                
                # Get all available months for this site
                available_months = []
                for _, row in site_data.iterrows():
                    year_month = row['year_month']
                    image_path = self.image_dir / f"{site_id}_{year_month}.tif"
                    if image_path.exists():
                        available_months.append({
                            'year_month': year_month,
                            'image_path': image_path,
                            'temporal_idx': self._year_month_to_idx(year_month)
                        })
                
                if available_months:
                    self.samples.append({
                        'site_id': site_id,
                        'months': available_months,
                        'label': label
                    })
                    self.labels.append(label)
        
        # Count class distribution
        self.num_looted = sum(self.labels)
        self.num_preserved = len(self.labels) - self.num_looted
    
    def _year_month_to_idx(self, year_month):
        """Convert year_month string to temporal index (0-indexed from 2016-01)"""
        year, month = map(int, year_month.split('-'))
        base_year, base_month = 2016, 1
        return (year - base_year) * 12 + (month - base_month)
    
    def _setup_transforms(self):
        """Setup data augmentation transforms"""
        self.resize_transform = torchvision.transforms.Resize(self.image_size)
        
        if self.augment:
            self.random_vflip = torchvision.transforms.RandomVerticalFlip(p=0.5)
            self.random_hflip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
            self.random_rotate = RandomRotation90()
            self.random_resize_crop = torchvision.transforms.RandomResizedCrop(
                size=self.image_size,
                scale=(0.3, 1.0)
            )
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, image_path):
        """Load image from .tif file (first 3 bands as RGB)"""
        with rasterio.open(image_path) as src:
            # Read first 3 bands (RGB)
            img = src.read([1, 2, 3])  # Shape: (3, H, W)
            img = torch.from_numpy(img).float()
        return img
    
    def _load_mask(self, site_id):
        """Load site mask"""
        mask_path = self.mask_dir / f"{site_id}.tif"
        
        if not mask_path.exists():
            # If mask doesn't exist, return all ones
            return torch.ones((1, *self.image_size))
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Shape: (H, W)
            mask = torch.from_numpy(mask).float()
            mask = (mask > 0).float()  # Binarize
        
        return mask
    
    def _apply_augmentation(self, data):
        """Apply augmentation to data tensor"""
        data = self.random_resize_crop(data)
        data = self.random_hflip(data)
        data = self.random_vflip(data)
        data = self.random_rotate(data)
        return data
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = torch.tensor([sample['label']], dtype=torch.long)
        
        if self.single:
            # Single-image mode
            image = self._load_image(sample['image_path'])
            mask = self._load_mask(sample['site_id'])
            
            # Resize
            image = self.resize_transform(image)
            mask = self.resize_transform(mask.unsqueeze(0)).squeeze(0)
            
            # Normalize
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]
            
            # Apply augmentation (with mask)
            if self.augment:
                # Stack image and mask for consistent augmentation
                combined = torch.cat([image, mask.unsqueeze(0).expand(3, -1, -1)], dim=0)
                combined = self._apply_augmentation(combined)
                image, mask = combined[:3], combined[3]
            
            # Apply mask
            if self.mask_mode == 'multiply':
                image = image * mask.unsqueeze(0)
            elif self.mask_mode == 'channel':
                image = torch.cat([image, mask.unsqueeze(0)], dim=0)
            
            temporal_idx = torch.tensor([self._year_month_to_idx(sample['year_month'])])
            
            return image, temporal_idx, mask, label
        
        else:
            # Temporal sequence mode
            months = sample['months']
            site_id = sample['site_id']
            
            # Load mask once for the site
            mask = self._load_mask(site_id)
            
            # Sample temporal frames
            if self.split == 'train' and len(months) > self.temporal_sampling:
                # Random sampling strategy similar to reference_benchmark
                n_bins = 8
                samples_per_bin = 3
                bin_size = len(months) // n_bins
                
                sampled_indices = []
                for k in range(n_bins):
                    bin_start = k * bin_size
                    bin_end = min((k + 1) * bin_size, len(months))
                    if bin_end > bin_start:
                        bin_indices = np.random.choice(
                            range(bin_start, bin_end),
                            size=min(samples_per_bin, bin_end - bin_start),
                            replace=False
                        )
                        sampled_indices.extend(bin_indices)
                
                sampled_indices = sorted(sampled_indices)[:self.temporal_sampling]
                selected_months = [months[i] for i in sampled_indices]
            else:
                # Use all available months
                selected_months = months
            
            # Load images
            images = []
            temporal_indices = []
            for month_data in selected_months:
                img = self._load_image(month_data['image_path'])
                images.append(img)
                temporal_indices.append(month_data['temporal_idx'])
            
            # Stack images: (T, C, H, W)
            images = torch.stack(images, dim=0)
            temporal_indices = torch.tensor(temporal_indices)
            
            # Resize
            T, C, H, W = images.shape
            images = images.view(T * C, H, W)
            images = self.resize_transform(images)
            images = images.view(T, C, *self.image_size)
            
            mask = self.resize_transform(mask.unsqueeze(0)).squeeze(0)
            
            # Normalize
            images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
            
            # Apply augmentation (with mask)
            if self.augment:
                # Stack all images and mask for consistent augmentation
                T, C, H, W = images.shape
                combined = torch.cat([
                    images.view(T * C, H, W),
                    mask.unsqueeze(0).expand(3, -1, -1)
                ], dim=0)
                combined = self._apply_augmentation(combined)
                images = combined[:T * C].view(T, C, *self.image_size)
                mask = combined[T * C]
            
            # Apply mask
            if self.mask_mode == 'multiply':
                images = images * mask[None, None, :, :]
            elif self.mask_mode == 'channel':
                # Add mask as extra channel to each temporal frame
                mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(T, 1, -1, -1)
                images = torch.cat([images, mask_expanded], dim=1)
            
            return images, temporal_indices, mask, label


def compute_dataset_statistics(data_dir, metadata_path, num_samples=1000):
    """
    Compute mean and std statistics from a sample of the dataset.
    
    Args:
        data_dir: Root directory containing images
        metadata_path: Path to metadata CSV
        num_samples: Number of images to sample for statistics
    
    Returns:
        mean: (3,) tensor of mean values per channel
        std: (3,) tensor of std values per channel
    """
    import pandas as pd
    
    df = pd.read_csv(metadata_path)
    image_dir = Path(data_dir) / 'planet_mosaics_final_4bands' / 'images'
    
    # Sample random images
    sample_rows = df.sample(min(num_samples, len(df)))
    
    means = []
    stds = []
    
    print(f"Computing dataset statistics from {len(sample_rows)} samples...")
    
    for _, row in sample_rows.iterrows():
        site_id = row['site_id']
        year_month = row['year_month']
        image_path = image_dir / f"{site_id}_{year_month}.tif"
        
        if image_path.exists():
            with rasterio.open(image_path) as src:
                img = src.read([1, 2, 3]).astype(np.float32)
                
                # Compute per-channel statistics
                means.append(img.reshape(3, -1).mean(axis=1))
                stds.append(img.reshape(3, -1).std(axis=1))
    
    # Aggregate statistics
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")
    
    return torch.tensor(mean), torch.tensor(std)


if __name__ == '__main__':
    # Test dataset loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.environ.get('CHANGE_DETECTION_DIR', 'change_detection'))
    parser.add_argument('--metadata_path', type=str, 
                        default=os.environ.get('LOOTED_METADATA_PATH', 'data/metadata.csv'))
    parser.add_argument('--compute_stats', action='store_true', help='Compute dataset statistics')
    args = parser.parse_args()
    
    if args.compute_stats:
        mean, std = compute_dataset_statistics(args.data_dir, args.metadata_path)
        print("\nUse these values in your dataset config:")
        print(f"mean = torch.tensor({mean.tolist()})")
        print(f"std = torch.tensor({std.tolist()})")
    else:
        # Test dataset loading
        dataset = LootingImageDataset(
            data_dir=args.data_dir,
            metadata_path=args.metadata_path,
            split='train',
            single=False,
            augment=True
        )
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Looted: {dataset.num_looted}, Preserved: {dataset.num_preserved}")
        
        # Load a sample
        images, temporal_indices, mask, label = dataset[0]
        print(f"\nSample 0:")
        print(f"  Images shape: {images.shape}")
        print(f"  Temporal indices: {temporal_indices}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Label: {label.item()}")
