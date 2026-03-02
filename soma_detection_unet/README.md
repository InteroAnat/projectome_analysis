# 3D U-Net for Soma Detection and Volume Measurement

This module provides a complete pipeline for training a 3D U-Net to automatically detect soma segments and measure their volumes from NRRD files.

## Overview

The system consists of:
- **Data preprocessing pipeline** for NRRD files and ground truth masks
- **3D U-Net architecture** optimized for soma segmentation
- **Training pipeline** with data augmentation and validation
- **Inference pipeline** for soma detection and volume measurement
- **Evaluation metrics** for segmentation quality assessment

## Directory Structure

```
soma_detection_unet/
├── data/                    # Data preprocessing and loading
├── models/                  # U-Net architecture definitions
├── training/                # Training scripts and utilities
├── inference/               # Inference and evaluation
├── utils/                   # Helper functions and utilities
├── configs/                 # Configuration files
└── results/                 # Training results and model checkpoints
```

## Installation

```bash
pip install torch torchvision nibabel nrrd scipy scikit-learn matplotlib seaborn tensorboard
```

## Usage

### 1. Data Preparation
```python
from data.dataset import SomaDataset
from data.preprocessing import prepare_training_data

# Prepare your NRRD files and ground truth masks
dataset = SomaDataset(nrrd_dir='path/to/nrrd/files', 
                     mask_dir='path/to/masks',
                     transform=True)
```

### 2. Training
```python
from training.train import train_unet

model = train_unet(
    train_dataset=dataset,
    val_split=0.2,
    epochs=100,
    batch_size=4,
    learning_rate=1e-4
)
```

### 3. Inference and Volume Measurement
```python
from inference.predict import SomaDetector

detector = SomaDetector(model_path='results/best_model.pth')
volumes = detector.detect_and_measure('path/to/new/nrrd/file.nrrd')
```

## Model Architecture

The 3D U-Net consists of:
- **Encoder path**: 4 downsampling blocks with convolution + max pooling
- **Decoder path**: 4 upsampling blocks with transposed convolution
- **Skip connections**: Concatenation of encoder and decoder features
- **Output**: Sigmoid activation for binary segmentation

## Performance Metrics

- **Dice coefficient**: Overall segmentation overlap
- **Precision/Recall**: Detection accuracy
- **Volume correlation**: Measured vs. ground truth volumes
- **Hausdorff distance**: Boundary accuracy