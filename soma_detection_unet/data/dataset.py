"""
Dataset classes for soma segmentation training.
"""

import os
import numpy as np
import nrrd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
from scipy import ndimage
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SomaDataset(Dataset):
    """
    Dataset class for soma segmentation.
    Handles NRRD files and corresponding binary masks.
    """
    
    def __init__(self, 
                 nrrd_dir: str,
                 mask_dir: str,
                 transform: bool = True,
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 normalize: bool = True,
                 augment: bool = True):
        """
        Args:
            nrrd_dir: Directory containing NRRD files
            mask_dir: Directory containing mask files (should match NRRD filenames)
            transform: Whether to apply transformations
            target_size: Target size for cropping/padding
            normalize: Whether to normalize intensity values
            augment: Whether to apply data augmentation
        """
        self.nrrd_dir = nrrd_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        
        # Get list of files
        self.nrrd_files = self._get_nrrd_files()
        self.mask_files = self._get_mask_files()
        
        # Validate that we have matching files
        self._validate_files()
        
        logger.info(f"Found {len(self.nrrd_files)} NRRD files and {len(self.mask_files)} mask files")
    
    def _get_nrrd_files(self) -> List[str]:
        """Get all NRRD files from the directory"""
        files = []
        for file in os.listdir(self.nrrd_dir):
            if file.endswith('.nrrd') or file.endswith('.nhdr'):
                files.append(file)
        return sorted(files)
    
    def _get_mask_files(self) -> List[str]:
        """Get all mask files from the directory"""
        files = []
        for file in os.listdir(self.mask_dir):
            if file.endswith('.nrrd') or file.endswith('.nhdr') or file.endswith('.nii.gz'):
                files.append(file)
        return sorted(files)
    
    def _validate_files(self):
        """Validate that NRRD and mask files match"""
        if len(self.nrrd_files) != len(self.mask_files):
            logger.warning(f"Number of NRRD files ({len(self.nrrd_files)}) "
                         f"doesn't match number of mask files ({len(self.mask_files)})")
        
        # Check if filenames match (without extension)
        nrrd_names = [os.path.splitext(f)[0] for f in self.nrrd_files]
        mask_names = [os.path.splitext(f)[0] for f in self.mask_files]
        
        # Remove .nii from mask names if present
        mask_names = [name.replace('.nii', '') for name in mask_names]
        
        if set(nrrd_names) != set(mask_names):
            logger.warning("Some NRRD files don't have corresponding masks or vice versa")
    
    def _load_nrrd(self, file_path: str) -> np.ndarray:
        """Load NRRD file and return numpy array"""
        try:
            data, header = nrrd.read(file_path)
            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading NRRD file {file_path}: {e}")
            raise
    
    def _load_mask(self, file_path: str) -> np.ndarray:
        """Load mask file and return binary numpy array"""
        try:
            if file_path.endswith('.nii.gz'):
                # Load NIfTI file
                nii = nib.load(file_path)
                data = nii.get_fdata()
            else:
                # Load NRRD file
                data, header = nrrd.read(file_path)
            
            # Ensure binary mask
            return (data > 0).astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading mask file {file_path}: {e}")
            raise
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1] range"""
        # Remove outliers (top and bottom 1%)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        return volume
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize volume to target size using interpolation"""
        current_size = volume.shape
        
        # Calculate resize factors
        resize_factors = [target_size[i] / current_size[i] for i in range(3)]
        
        # Resize using scipy
        resized = ndimage.zoom(volume, resize_factors, order=1)
        
        # Ensure exact target size
        if resized.shape != target_size:
            # Pad or crop to exact size
            pad_width = []
            crop_slices = []
            
            for i in range(3):
                if resized.shape[i] < target_size[i]:
                    # Pad
                    pad_before = (target_size[i] - resized.shape[i]) // 2
                    pad_after = target_size[i] - resized.shape[i] - pad_before
                    pad_width.append((pad_before, pad_after))
                    crop_slices.append(slice(None))
                else:
                    # Crop
                    crop_start = (resized.shape[i] - target_size[i]) // 2
                    crop_end = crop_start + target_size[i]
                    crop_slices.append(slice(crop_start, crop_end))
                    pad_width.append((0, 0))
            
            if any(pad[0] > 0 or pad[1] > 0 for pad in pad_width):
                resized = np.pad(resized, pad_width, mode='constant', constant_values=0)
            
            resized = resized[tuple(crop_slices)]
        
        return resized
    
    def _apply_augmentation(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation"""
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            axes = np.random.choice([0, 1, 2], 2, replace=False)
            volume = ndimage.rotate(volume, angle, axes=axes, reshape=False, order=1)
            mask = ndimage.rotate(mask, angle, axes=axes, reshape=False, order=0)
        
        # Random flip
        if np.random.random() > 0.5:
            axis = np.random.choice([0, 1, 2])
            volume = np.flip(volume, axis=axis)
            mask = np.flip(mask, axis=axis)
        
        # Random intensity adjustment
        if np.random.random() > 0.5:
            volume = volume * np.random.uniform(0.8, 1.2)
            volume = np.clip(volume, 0, 1)
        
        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.02, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        
        return volume, mask
    
    def __len__(self) -> int:
        return len(self.nrrd_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample"""
        # Load NRRD file
        nrrd_file = self.nrrd_files[idx]
        nrrd_path = os.path.join(self.nrrd_dir, nrrd_file)
        volume = self._load_nrrd(nrrd_path)
        
        # Load corresponding mask
        base_name = os.path.splitext(nrrd_file)[0]
        mask_file = None
        for mf in self.mask_files:
            if base_name in mf:
                mask_file = mf
                break
        
        if mask_file is None:
            raise ValueError(f"No mask file found for {nrrd_file}")
        
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = self._load_mask(mask_path)
        
        # Preprocessing
        if self.normalize:
            volume = self._normalize_intensity(volume)
        
        # Resize to target size
        volume = self._resize_volume(volume, self.target_size)
        mask = self._resize_volume(mask, self.target_size)
        
        # Ensure mask is binary
        mask = (mask > 0.5).astype(np.float32)
        
        # Data augmentation
        if self.augment and np.random.random() > 0.5:
            volume, mask = self._apply_augmentation(volume, mask)
        
        # Convert to tensors
        volume = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)      # Add channel dimension
        
        return volume, mask


class SomaDataModule:
    """
    Data module for managing train/validation/test splits.
    """
    
    def __init__(self,
                 nrrd_dir: str,
                 mask_dir: str,
                 batch_size: int = 4,
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15):
        """
        Args:
            nrrd_dir: Directory containing NRRD files
            mask_dir: Directory containing mask files
            batch_size: Batch size for data loaders
            target_size: Target size for volumes
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
        """
        self.nrrd_dir = nrrd_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets with train/val/test splits"""
        # Create full dataset
        full_dataset = SomaDataset(
            nrrd_dir=self.nrrd_dir,
            mask_dir=self.mask_dir,
            target_size=self.target_size,
            augment=True  # Enable augmentation for training
        )
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        logger.info(f"Dataset split - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


def create_sample_data(nrrd_dir: str, mask_dir: str, num_samples: int = 10):
    """
    Create sample data for testing the dataset.
    This is a utility function to generate synthetic data.
    """
    import numpy as np
    
    os.makedirs(nrrd_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create synthetic volume (random noise with some structure)
        volume = np.random.randn(100, 100, 80) * 100 + 500
        
        # Add some spherical structures to simulate somas
        for _ in range(np.random.randint(1, 4)):
            center = np.random.randint(20, 80, 3)
            radius = np.random.randint(5, 15)
            
            # Create sphere
            z, y, x = np.ogrid[:100, :100, :80]
            mask_sphere = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
            volume[mask_sphere] += np.random.randint(200, 500)
        
        # Create corresponding mask
        mask = (volume > 600).astype(np.uint8)
        
        # Save files
        nrrd.write(os.path.join(nrrd_dir, f'sample_{i:03d}.nrrd'), volume)
        nrrd.write(os.path.join(mask_dir, f'sample_{i:03d}.nrrd'), mask)
    
    logger.info(f"Created {num_samples} sample data files")


if __name__ == '__main__':
    # Test the dataset
    import tempfile
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        mask_dir = os.path.join(temp_dir, 'masks')
        
        # Create sample data
        create_sample_data(nrrd_dir, mask_dir, num_samples=5)
        
        # Test dataset
        dataset = SomaDataset(nrrd_dir, mask_dir, target_size=(64, 64, 64))
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test getting an item
        volume, mask = dataset[0]
        print(f"Volume shape: {volume.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
        print(f"Mask unique values: {torch.unique(mask)}")