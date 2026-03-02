# Soma Segmentation with U-Net/MONAI - Complete Guide

## Overview

This guide explains how to train a 3D U-Net using MONAI for automatic soma segmentation from your high-resolution soma blocks (360×360×90 voxels, 0.65µm × 0.65µm × 3µm resolution).

## Your Data Structure

### Input Data (Raw Soma Blocks)
- **Location**: `resource/segmented_cubes/{sample_id}/{sample_id}_{neuron_id}_SomaBlock.nii.gz`
- **Dimensions**: 360 × 360 × 90 voxels
- **Resolution**: 0.65µm × 0.65µm × 3µm (X × Y × Z)
- **Data Type**: uint16 (0-65535)
- **Format**: NIfTI (.nii.gz)

### Ground Truth Masks (From 3D Slicer)
- **Expected Location**: `resource/segmented_cubes/{sample_id}/masks/{neuron_id}_soma_mask.nii.gz`
- **Format**: Binary NIfTI mask (0 = background, 1 = soma)
- **Should Match**: Same dimensions and affine as input data

## Directory Setup

```
projectome_analysis/
├── resource/
│   └── segmented_cubes/
│       └── 251637/                    # Sample ID
│           ├── 251637_003.swc_SomaBlock.nii.gz    # Input: Raw soma block
│           ├── 251637_004.swc_SomaBlock.nii.gz
│           ├── ...
│           └── masks/                 # Ground truth from 3D Slicer
│               ├── 003_soma_mask.nii.gz           # Output: Binary mask
│               ├── 004_soma_mask.nii.gz
│               └── ...
├── soma_detection_unet/
│   ├── data/
│   │   ├── soma_dataset.py           # Dataset loader for your format
│   │   └── preprocessing.py          # Preprocessing pipeline
│   ├── models/
│   │   └── unet_3d.py                # 3D U-Net architecture
│   ├── training/
│   │   └── train_monai.py            # MONAI training script
│   ├── inference/
│   │   └── predict.py                # Inference on new data
│   └── configs/
│       └── soma_config.yaml          # Configuration file
```

## Step 1: Creating Ground Truth Masks in 3D Slicer

### 1.1 Load Soma Block in 3D Slicer
1. Open 3D Slicer
2. File → Add Data → Select `251637_003.swc_SomaBlock.nii.gz`
3. The volume will appear in the scene

### 1.2 Create Segmentation
1. Go to **Segment Editor** module
2. Click **Add** to create a new segmentation
3. Select **Add segment** for soma
4. Use **Paint** or **Draw** tools to segment the soma:
   - **Threshold**: Set range to capture soma intensity (e.g., 5000-65535)
   - **Scissors**: Remove non-soma regions
   - **Smoothing**: Apply Gaussian smoothing (optional)

### 1.3 Export Segmentation
1. Go to **Data** module
2. Right-click on segmentation → **Export visible segments to binary labelmap**
3. Save as: `{neuron_id}_soma_mask.nii.gz`
4. Move to `resource/segmented_cubes/{sample_id}/masks/`

## Step 2: MONAI-Based Training Pipeline

### 2.1 Installation

```bash
# Create conda environment
conda create -n soma_seg python=3.9
conda activate soma_seg

# Install MONAI and dependencies
pip install monai[all]
pip install nibabel scipy scikit-image matplotlib pandas tensorboard
```

### 2.2 Dataset Configuration

Create `configs/soma_config.yaml`:

```yaml
# Soma Segmentation Configuration

# Data paths
data:
  base_dir: "../resource/segmented_cubes"
  sample_ids: ["251637"]  # Add more samples as needed
  train_val_test_split: [0.7, 0.15, 0.15]
  
# Preprocessing
preprocessing:
  # Input volume size: 360 x 360 x 90
  target_size: [360, 360, 90]  # Keep original or crop/pad
  patch_size: [128, 128, 64]   # For patch-based training
  patch_overlap: 0.5
  
  # Intensity normalization
  intensity:
    method: "percentile"  # "zscore" or "minmax" or "percentile"
    lower_percentile: 1
    upper_percentile: 99
    
  # Augmentation
  augmentation:
    enabled: true
    rotation_range: [-10, 10]  # degrees
    flip_prob: 0.5
    intensity_scale: [0.9, 1.1]
    gaussian_noise_std: 0.01

# Model
model:
  name: "UNet3D"
  in_channels: 1
  out_channels: 1  # Binary segmentation
  features: [32, 64, 128, 256, 512]
  dropout: 0.1
  
# Training
training:
  batch_size: 2
  epochs: 200
  learning_rate: 1e-4
  weight_decay: 1e-5
  optimizer: "Adam"
  scheduler: "ReduceLROnPlateau"
  patience: 20
  
  # Loss function
  loss:
    name: "DiceCELoss"  # Combined Dice + Cross-Entropy
    dice_weight: 0.5
    ce_weight: 0.5
    
  # Metrics
  metrics:
    - "DiceMetric"
    - "HausdorffDistanceMetric"
    - "SurfaceDistanceMetric"

# Inference
inference:
  sliding_window: true
  window_size: [128, 128, 64]
  overlap: 0.5
  confidence_threshold: 0.5
```

### 2.3 MONAI Dataset Loader

Create `data/soma_monai_dataset.py`:

```python
"""
MONAI-based dataset loader for soma segmentation.
Optimized for 3D Slicer exported masks and NIfTI format.
"""

import os
import glob
import numpy as np
import nibabel as nib
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader

from monai.data import Dataset, CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, NormalizeIntensityd,
    RandRotated, RandFlipd, RandScaleIntensityd,
    RandGaussianNoised, RandGaussianSmoothd,
    Resized, SpatialPadd, CropForegroundd,
    RandCropByPosNegLabeld, EnsureTyped,
    Invertd, SaveImaged
)
from monai.utils import first


class SomaSegmentationDataset:
    """
    Dataset manager for soma segmentation using MONAI.
    Handles NIfTI files from 3D Slicer.
    """
    
    def __init__(self, 
                 base_dir: str,
                 sample_ids: List[str],
                 patch_size: Tuple[int, int, int] = (128, 128, 64),
                 batch_size: int = 2,
                 num_workers: int = 4):
        """
        Args:
            base_dir: Base directory containing segmented_cubes
            sample_ids: List of sample IDs (e.g., ['251637'])
            patch_size: Size of patches for training
            batch_size: Batch size
            num_workers: Number of data loading workers
        """
        self.base_dir = base_dir
        self.sample_ids = sample_ids
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Build data dictionary
        self.data_dicts = self._build_data_dict()
        
    def _build_data_dict(self) -> List[Dict]:
        """Build list of data dictionaries for MONAI."""
        data_dicts = []
        
        for sample_id in self.sample_ids:
            sample_dir = os.path.join(self.base_dir, sample_id)
            mask_dir = os.path.join(sample_dir, "masks")
            
            if not os.path.exists(mask_dir):
                print(f"Warning: Mask directory not found: {mask_dir}")
                continue
            
            # Find all soma block files
            soma_files = glob.glob(
                os.path.join(sample_dir, f"{sample_id}_*.swc_SomaBlock.nii.gz")
            )
            
            for soma_file in soma_files:
                # Extract neuron ID from filename
                # Format: 251637_003.swc_SomaBlock.nii.gz -> 003
                basename = os.path.basename(soma_file)
                parts = basename.split('_')
                if len(parts) >= 2:
                    neuron_id = parts[1].replace('.swc', '')
                    
                    # Look for corresponding mask
                    mask_file = os.path.join(mask_dir, f"{neuron_id}_soma_mask.nii.gz")
                    
                    if os.path.exists(mask_file):
                        data_dicts.append({
                            "image": soma_file,
                            "label": mask_file,
                            "sample_id": sample_id,
                            "neuron_id": neuron_id
                        })
                    else:
                        print(f"Warning: Mask not found for {basename}")
        
        print(f"Found {len(data_dicts)} valid image-mask pairs")
        return data_dicts
    
    def get_train_transforms(self) -> Compose:
        """Get training transforms with augmentation."""
        return Compose([
            # Load NIfTI files
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            
            # Intensity normalization (percentile-based for uint16 data)
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0, a_max=65535,  # uint16 range
                b_min=0, b_max=1,
                clip=True
            ),
            
            # Optional: Z-score normalization within foreground
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True
            ),
            
            # Crop foreground to remove empty regions
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=self.patch_size
            ),
            
            # Random crop with positive/negative sampling
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.patch_size,
                pos=1, neg=1,
                num_samples=4,  # 4 patches per volume
                image_key="image",
                image_threshold=0
            ),
            
            # Data augmentation
            RandRotated(
                keys=["image", "label"],
                range_x=0.174,  # ~10 degrees in radians
                range_y=0.174,
                range_z=0.174,
                prob=0.5,
                mode=["bilinear", "nearest"]
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0, 1, 2],
                prob=0.5
            ),
            RandScaleIntensityd(
                keys=["image"],
                factors=0.1,
                prob=0.5
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.3,
                std=0.01
            ),
            
            # Convert to tensor
            EnsureTyped(keys=["image", "label"])
        ])
    
    def get_val_transforms(self) -> Compose:
        """Get validation transforms (no augmentation)."""
        return Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0, a_max=65535,
                b_min=0, b_max=1,
                clip=True
            ),
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True
            ),
            # Pad to ensure divisibility
            SpatialPadd(
                keys=["image", "label"],
                k_divisible=self.patch_size
            ),
            EnsureTyped(keys=["image", "label"])
        ])
    
    def get_test_transforms(self) -> Compose:
        """Get test/inference transforms."""
        return Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0, a_max=65535,
                b_min=0, b_max=1,
                clip=True
            ),
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True
            ),
            EnsureTyped(keys=["image"])
        ])
    
    def get_data_loaders(self, 
                         train_split: float = 0.7,
                         val_split: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train/val/test data loaders."""
        
        # Split data
        n_total = len(self.data_dicts)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_files = self.data_dicts[:n_train]
        val_files = self.data_dicts[n_train:n_train + n_val]
        test_files = self.data_dicts[n_train + n_val:]
        
        print(f"Data split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Create datasets
        train_ds = CacheDataset(
            data=train_files,
            transform=self.get_train_transforms(),
            cache_rate=1.0,
            num_workers=self.num_workers
        )
        
        val_ds = CacheDataset(
            data=val_files,
            transform=self.get_val_transforms(),
            cache_rate=1.0,
            num_workers=self.num_workers
        )
        
        test_ds = Dataset(
            data=test_files,
            transform=self.get_val_transforms()
        )
        
        # Create data loaders
        train_loader = MonaiDataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = MonaiDataLoader(
            val_ds,
            batch_size=1,  # Full volume for validation
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = MonaiDataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader


def verify_data_pair(image_path: str, label_path: str) -> bool:
    """Verify that image and label match in dimensions and affine."""
    try:
        img = nib.load(image_path)
        lbl = nib.load(label_path)
        
        # Check shape
        if img.shape != lbl.shape:
            print(f"Shape mismatch: {img.shape} vs {lbl.shape}")
            return False
        
        # Check affine (with tolerance)
        if not np.allclose(img.affine, lbl.affine, atol=1e-3):
            print(f"Affine mismatch detected")
            print(f"Image affine:\n{img.affine}")
            print(f"Label affine:\n{lbl.affine}")
            return False
        
        return True
    except Exception as e:
        print(f"Error loading files: {e}")
        return False


if __name__ == "__main__":
    # Test the dataset
    dataset = SomaSegmentationDataset(
        base_dir="../resource/segmented_cubes",
        sample_ids=["251637"],
        patch_size=(128, 128, 64),
        batch_size=2
    )
    
    train_loader, val_loader, test_loader = dataset.get_data_loaders()
    
    # Test one batch
    batch = first(train_loader)
    print(f"Image batch shape: {batch['image'].shape}")
    print(f"Label batch shape: {batch['label'].shape}")
    print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"Label unique: {torch.unique(batch['label'])}")
```

### 2.4 MONAI Training Script

Create `training/train_monai.py`:

```python
"""
MONAI-based training script for 3D soma segmentation.
"""

import os
import time
import logging
from typing import Dict
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# MONAI imports
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete
from monai.utils import set_determinism

# Import custom dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.soma_monai_dataset import SomaSegmentationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SomaSegmentationTrainer:
    """Trainer class for soma segmentation using MONAI."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set determinism for reproducibility
        set_determinism(seed=42)
        
        # Create output directories
        self.output_dir = "results"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)
        
        # Best model tracking
        self.best_dice = 0.0
        self.patience_counter = 0
        
    def _setup_data(self):
        """Setup data loaders."""
        data_config = self.config['data']
        prep_config = self.config['preprocessing']
        
        dataset = SomaSegmentationDataset(
            base_dir=data_config['base_dir'],
            sample_ids=data_config['sample_ids'],
            patch_size=tuple(prep_config['patch_size']),
            batch_size=self.config['training']['batch_size']
        )
        
        self.train_loader, self.val_loader, self.test_loader = dataset.get_data_loaders(
            train_val_test_split=data_config['train_val_test_split'][0],
            val_split=data_config['train_val_test_split'][1]
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
    
    def _setup_model(self):
        """Setup U-Net model."""
        model_config = self.config['model']
        
        self.model = UNet(
            spatial_dims=3,
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            channels=model_config['features'],
            strides=[2, 2, 2, 2],
            num_res_units=2,
            dropout=model_config['dropout']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_training(self):
        """Setup loss, optimizer, and scheduler."""
        train_config = self.config['training']
        
        # Loss function
        loss_name = train_config['loss']['name']
        if loss_name == "DiceCELoss":
            self.criterion = DiceCELoss(
                sigmoid=True,
                dice_weight=train_config['loss'].get('dice_weight', 0.5),
                ce_weight=train_config['loss'].get('ce_weight', 0.5)
            )
        elif loss_name == "DiceLoss":
            self.criterion = DiceLoss(sigmoid=True)
        elif loss_name == "TverskyLoss":
            self.criterion = TverskyLoss(sigmoid=True)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        
        # Optimizer
        if train_config['optimizer'] == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'] == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                momentum=0.9,
                weight_decay=train_config['weight_decay']
            )
        
        # Scheduler
        if train_config['scheduler'] == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=train_config['patience'] // 2,
                verbose=True
            )
        elif train_config['scheduler'] == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs']
            )
        else:
            self.scheduler = None
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False)
        
        # Post-processing transforms
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.post_label = AsDiscrete(to_onehot=None)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = epoch_loss / num_batches
        return {"loss": avg_loss}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        # Reset metrics
        self.dice_metric.reset()
        self.hausdorff_metric.reset()
        
        for batch_data in self.val_loader:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            
            # Sliding window inference for large volumes
            if self.config['inference']['sliding_window']:
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=self.config['inference']['window_size'],
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=self.config['inference']['overlap']
                )
            else:
                outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            val_loss += loss.item()
            
            # Post-process predictions
            val_outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
            val_labels = [self.post_label(i) for i in decollate_batch(labels)]
            
            # Update metrics
            self.dice_metric(y_pred=val_outputs, y=val_labels)
            self.hausdorff_metric(y_pred=val_outputs, y=val_labels)
        
        # Aggregate metrics
        avg_loss = val_loss / len(self.val_loader)
        dice_score = self.dice_metric.aggregate().item()
        hausdorff_dist = self.hausdorff_metric.aggregate().item()
        
        return {
            "loss": avg_loss,
            "dice": dice_score,
            "hausdorff": hausdorff_dist
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_dice': self.best_dice
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self):
        """Full training loop."""
        num_epochs = self.config['training']['epochs']
        patience = self.config['training']['patience']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            elif self.scheduler:
                self.scheduler.step()
            
            # Logging
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val Hausdorff: {val_metrics['hausdorff']:.2f}"
            )
            
            # Tensorboard
            self.writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
            self.writer.add_scalar("Loss/Validation", val_metrics['loss'], epoch)
            self.writer.add_scalar("Metrics/Dice", val_metrics['dice'], epoch)
            self.writer.add_scalar("Metrics/Hausdorff", val_metrics['hausdorff'], epoch)
            self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            is_best = val_metrics['dice'] > self.best_dice
            if is_best:
                self.best_dice = val_metrics['dice']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation Dice: {self.best_dice:.4f}")
        
        self.writer.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train 3D U-Net for soma segmentation")
    parser.add_argument("--config", type=str, default="configs/soma_config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = SomaSegmentationTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
```

## Step 3: Running the Training

### 3.1 Prepare Your Data

```bash
# Create mask directory structure
mkdir -p resource/segmented_cubes/251637/masks

# Export masks from 3D Slicer and place them in the masks directory
# Each mask should be named: {neuron_id}_soma_mask.nii.gz
# Example: 003_soma_mask.nii.gz corresponds to 251637_003.swc_SomaBlock.nii.gz
```

### 3.2 Train the Model

```bash
cd soma_detection_unet

# Activate environment
conda activate soma_seg

# Run training
python training/train_monai.py --config configs/soma_config.yaml
```

### 3.3 Monitor Training

```bash
# In a separate terminal, launch TensorBoard
tensorboard --logdir results/logs

# Open browser at http://localhost:6006
```

## Step 4: Inference on New Data

Create `inference/predict_monai.py`:

```python
"""
Inference script for soma segmentation using trained MONAI model.
"""

import os
import argparse
import logging
from typing import List, Dict
import yaml

import numpy as np
import nibabel as nib
import torch

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, NormalizeIntensityd,
    EnsureTyped, Activations, AsDiscrete,
    SaveImaged, Invertd
)
from monai.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SomaSegmentor:
    """Soma segmentation inference class."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """Initialize segmentor with trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Build model
        model_config = self.config.get('model', {})
        self.model = UNet(
            spatial_dims=3,
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            channels=model_config.get('features', [32, 64, 128, 256, 512]),
            strides=[2, 2, 2, 2],
            num_res_units=2,
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Preprocessing transforms
        self.preprocess = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0, a_max=65535,
                b_min=0, b_max=1,
                clip=True
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"])
        ])
        
        # Post-processing
        self.post_process = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])
    
    def segment(self, image_path: str, output_path: str = None) -> np.ndarray:
        """Segment soma in a single image."""
        # Load and preprocess
        data = {"image": image_path}
        data = self.preprocess(data)
        
        input_tensor = data["image"].unsqueeze(0).to(self.device)
        
        # Inference
        logger.info(f"Running inference on {image_path}")
        with torch.no_grad():
            # Use sliding window for large volumes
            output = sliding_window_inference(
                input_tensor,
                roi_size=(128, 128, 64),
                sw_batch_size=4,
                predictor=self.model,
                overlap=0.5
            )
            
            # Post-process
            output = self.post_process(output)
        
        # Convert to numpy
        segmentation = output.squeeze().cpu().numpy().astype(np.uint8)
        
        # Save if output path provided
        if output_path:
            # Load original to get affine
            original = nib.load(image_path)
            seg_nii = nib.Nifti1Image(segmentation, original.affine, original.header)
            nib.save(seg_nii, output_path)
            logger.info(f"Segmentation saved to {output_path}")
        
        return segmentation
    
    def batch_segment(self, image_paths: List[str], output_dir: str):
        """Segment multiple images."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")
            
            basename = os.path.basename(image_path).replace('.nii.gz', '')
            output_path = os.path.join(output_dir, f"{basename}_pred.nii.gz")
            
            try:
                segmentation = self.segment(image_path, output_path)
                
                # Calculate statistics
                num_voxels = np.sum(segmentation)
                results.append({
                    'file': image_path,
                    'output': output_path,
                    'soma_voxels': int(num_voxels),
                    'success': True
                })
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'file': image_path,
                    'error': str(e),
                    'success': False
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Soma segmentation inference")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Input image path or directory")
    parser.add_argument("--output", type=str, default="predictions",
                       help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    args = parser.parse_args()
    
    # Create segmentor
    segmentor = SomaSegmentor(args.model, args.config)
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        segmentor.segment(args.input, os.path.join(args.output, "prediction.nii.gz"))
    else:
        # Directory
        import glob
        image_paths = glob.glob(os.path.join(args.input, "*_SomaBlock.nii.gz"))
        results = segmentor.batch_segment(image_paths, args.output)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Processed {len(results)} images, {successful} successful")


if __name__ == "__main__":
    main()
```

## Step 5: Usage Examples

### 5.1 Segment a Single Image

```bash
python inference/predict_monai.py \
    --model results/checkpoints/best_model.pth \
    --input resource/segmented_cubes/251637/251637_005.swc_SomaBlock.nii.gz \
    --output predictions/
```

### 5.2 Batch Process

```bash
python inference/predict_monai.py \
    --model results/checkpoints/best_model.pth \
    --input resource/segmented_cubes/251637/ \
    --output predictions/251637/
```

## Key Considerations

### Data Quality
1. **Consistent labeling**: Ensure all somas are labeled consistently in 3D Slicer
2. **Complete somas**: Include the entire soma, not just the center
3. **Avoid partial somas**: Don't include blocks where soma is at the edge

### Training Tips
1. **Start small**: Begin with 10-20 labeled somas
2. **Augmentation**: Heavy augmentation helps with small datasets
3. **Patch size**: 128³ works well; adjust based on GPU memory
4. **Learning rate**: Start with 1e-4, reduce if unstable

### Common Issues
1. **Out of memory**: Reduce batch size or patch size
2. **Poor segmentation**: Check label quality, increase training data
3. **Overfitting**: Increase augmentation, add dropout

## MONAI vs. Custom U-Net

| Feature | MONAI | Custom U-Net |
|---------|-------|--------------|
| Pre-built models | ✅ Yes | ❌ Manual |
| Data augmentation | ✅ Advanced | ⚠️ Basic |
| Sliding window | ✅ Built-in | ⚠️ Manual |
| Flexibility | ⚠️ Standard | ✅ Full |
| Learning curve | ⚠️ Steeper | ✅ Gentler |

For your use case, **MONAI is recommended** due to:
- Optimized 3D operations
- Built-in sliding window inference
- Advanced augmentation
- Production-ready code
