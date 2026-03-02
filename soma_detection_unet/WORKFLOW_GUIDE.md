# 3D U-Net Soma Detection - Complete Workflow Guide

## Overview

This guide provides a comprehensive walkthrough for using the 3D U-Net system to automatically detect soma segments and measure their volumes from NRRD files. The system is designed specifically for your neuron analysis project that already has NRRD files containing soma segments.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Training the Model](#training-the-model)
5. [Running Inference](#running-inference)
6. [Evaluation and Validation](#evaluation-and-validation)
7. [Batch Processing](#batch-processing)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting)

## System Architecture

### Core Components

```
soma_detection_unet/
├── models/              # U-Net architectures
│   ├── unet_3d.py     # Basic and enhanced 3D U-Net
├── data/                # Data handling
│   ├── dataset.py       # Dataset classes and preprocessing
├── training/            # Training pipeline
│   ├── train.py         # Training loop and utilities
├── inference/           # Inference pipeline
│   ├── predict.py       # Soma detection and volume measurement
├── utils/               # Utilities
│   ├── metrics.py       # Evaluation metrics
│   ├── visualization.py # Plotting and visualization
├── configs/             # Configuration files
│   └── config.yaml      # Main configuration
└── main.py              # Command-line interface
```

### Key Features

- **3D U-Net Architecture**: Optimized for soma segmentation with skip connections
- **Data Augmentation**: Rotation, flipping, intensity adjustment, noise injection
- **Sliding Window Inference**: For processing large volumes
- **Comprehensive Metrics**: Dice, IoU, Hausdorff distance, volume accuracy
- **Batch Processing**: Process multiple volumes efficiently
- **Visualization**: Rich plotting capabilities for results analysis

## Installation

### Prerequisites

```bash
# Create virtual environment
python -m venv soma_env
source soma_env/bin/activate  # On Windows: soma_env\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Install Package

```bash
# Install from requirements
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Verify Installation

```bash
python -c "from soma_detection_unet.models.unet_3d import SomaUNet3D; print('Installation successful!')"
```

## Data Preparation

### Expected Data Format

Your NRRD files should contain:
- **3D image volumes**: Grayscale intensity data (typically uint16)
- **Soma segments**: Binary masks where 1 = soma, 0 = background
- **Resolution**: 0.65×0.65×3.0 microns (configurable)

### Directory Structure

```
your_data/
├── volumes/          # NRRD files with image data
│   ├── neuron_001.nrrd
│   ├── neuron_002.nrrd
│   └── ...
└── masks/            # Corresponding binary masks
    ├── neuron_001.nrrd
    ├── neuron_002.nrrd
    └── ...
```

### Quick Data Check

```python
import nrrd
import numpy as np

# Load and inspect your data
data, header = nrrd.read('your_volume.nrrd')
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Range: [{data.min()}, {data.max()}]")

# Check mask
mask, mask_header = nrrd.read('your_mask.nrrd')
print(f"Mask unique values: {np.unique(mask)}")
print(f"Soma voxels: {np.sum(mask > 0)}")
```

## Training the Model

### Method 1: Command Line

```bash
# Create demo data (optional, for testing)
python main.py demo --num_samples 50

# Train model
python main.py train \
    --config configs/config.yaml \
    --epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4
```

### Method 2: Programmatic

```python
from soma_detection_unet.training.train import train_unet

# Train model
history = train_unet(
    nrrd_dir='path/to/your/volumes',
    mask_dir='path/to/your/masks',
    model_type='soma_unet',
    epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    checkpoint_dir='results/checkpoints',
    log_dir='results/logs'
)

print(f"Final validation Dice: {history['val_dice_scores'][-1]:.4f}")
```

### Training Configuration

Key parameters in `configs/config.yaml`:

```yaml
model:
  type: 'soma_unet'  # or 'unet'
  dropout_rate: 0.1

training:
  epochs: 100
  batch_size: 4
  learning_rate: 1e-4
  loss_type: 'bce'  # or 'dice', 'combined'
  
data:
  target_size: [128, 128, 128]
  augment: true
  normalize: true
```

### Monitoring Training

Training progress is automatically logged to TensorBoard:

```bash
tensorboard --logdir results/logs
```

Key metrics to monitor:
- **Training/Validation Loss**: Should decrease steadily
- **Dice Score**: Target > 0.85 for good performance
- **Precision/Recall**: Balance based on your needs

## Running Inference

### Single Volume

```bash
python main.py infer \
    --config configs/config.yaml \
    --model_path results/checkpoints/best_model.pth \
    --volume_path path/to/your/volume.nrrd
```

### Batch Processing

```bash
python main.py infer \
    --config configs/config.yaml \
    --model_path results/checkpoints/best_model.pth \
    --volume_path path/to/volume/directory/
```

### Programmatic Inference

```python
from soma_detection_unet.inference.predict import SomaDetector

# Create detector
detector = SomaDetector(
    model_path='results/checkpoints/best_model.pth',
    model_type='soma_unet',
    confidence_threshold=0.5,
    min_soma_volume=100,  # voxels
    max_soma_volume=100000,  # voxels
    voxel_spacing=(0.65, 0.65, 3.0)  # microns
)

# Detect somas
result = detector.detect_somas(
    'your_volume.nrrd',
    use_sliding_window=False,  # Set True for large volumes
    visualize=True
)

# Access results
print(f"Detected {result['num_somas']} somas")
for soma in result['soma_properties']:
    print(f"Volume: {soma['volume_um3']:.2f} μm³")
    print(f"Center: {soma['center_physical']}")
    print(f"Sphericity: {soma['sphericity']:.3f}")
```

## Evaluation and Validation

### Against Ground Truth

```bash
python main.py evaluate \
    --config configs/config.yaml \
    --model_path results/checkpoints/best_model.pth \
    --volume_path your_volume.nrrd \
    --mask_path your_mask.nrrd
```

### Comprehensive Evaluation

```python
from soma_detection_unet.utils.metrics import comprehensive_evaluation, print_metrics

# Get prediction and ground truth
pred_mask = result['segmentation_mask']
gt_mask, _ = nrrd.read('your_mask.nrrd')

# Evaluate
metrics = comprehensive_evaluation(
    pred_mask, 
    gt_mask,
    voxel_spacing=(0.65, 0.65, 3.0)
)

# Print results
print_metrics(metrics, "Evaluation Results")
```

### Key Metrics Explained

- **Dice Coefficient**: Overlap between prediction and ground truth (0-1, higher is better)
- **Hausdorff Distance**: Maximum surface distance (lower is better)
- **Volume Error**: Relative volume difference (lower is better)
- **Object-level F1**: Detection accuracy for individual somas

## Batch Processing

### Efficient Processing

```python
# Process multiple volumes
volume_files = ['vol1.nrrd', 'vol2.nrrd', 'vol3.nrrd']

results = detector.batch_process(
    volume_files,
    output_dir='results/batch_output',
    save_segmentations=True,
    save_visualizations=True
)

# Analyze results
for vol_path, result in results.items():
    if 'error' not in result:
        print(f"{vol_path}: {result['num_somas']} somas")
```

### Parallel Processing

For large datasets, consider using the sliding window approach:

```python
result = detector.detect_somas(
    'large_volume.nrrd',
    use_sliding_window=True,
    window_size=(128, 128, 128),
    overlap=0.5
)
```

## Advanced Configuration

### Custom Model Architecture

Modify `models/unet_3d.py` to:
- Add attention mechanisms
- Change network depth
- Modify convolution parameters

### Data Augmentation

Customize augmentation in `data/dataset.py`:

```python
def _apply_augmentation(self, volume, mask):
    # Add custom augmentation here
    if np.random.random() > 0.5:
        # Your custom transformation
        volume = your_transformation(volume)
        mask = your_transformation(mask)
    return volume, mask
```

### Loss Functions

Available loss functions:
- `bce`: Binary Cross Entropy
- `dice`: Dice Loss
- `combined`: Weighted combination

Configure in `configs/config.yaml`:

```yaml
training:
  loss_type: 'combined'
  dice_weight: 0.5
```

## Integration with Your Existing Pipeline

### Adapting to Your NRRD Files

Since you already have NRRD files with soma segments, here's how to integrate:

1. **Organize your data**:
   ```bash
   mkdir -p soma_detection_data/volumes
   mkdir -p soma_detection_data/masks
   cp your_existing_files/*.nrrd soma_detection_data/volumes/
   cp your_ground_truth_masks/*.nrrd soma_detection_data/masks/
   ```

2. **Update configuration**:
   ```yaml
   paths:
     nrrd_dir: 'soma_detection_data/volumes'
     mask_dir: 'soma_detection_data/masks'
   ```

3. **Train on your data**:
   ```bash
   python main.py train --config configs/config.yaml
   ```

### Custom Preprocessing

If your NRRD files need special handling:

```python
class YourCustomDataset(SomaDataset):
    def _load_nrrd(self, file_path):
        # Custom loading logic
        data, header = nrrd.read(file_path)
        # Apply your preprocessing
        data = your_preprocessing(data)
        return data
```

## Performance Optimization

### Memory Management

- **Batch Size**: Reduce if running out of GPU memory
- **Target Size**: Smaller input size (96³ vs 128³) for faster training
- **Mixed Precision**: Enable for faster training on modern GPUs

### Speed Optimization

- **Sliding Window**: Use for large volumes to avoid memory issues
- **GPU Utilization**: Monitor with `nvidia-smi`
- **Data Loading**: Use multiple workers for faster data loading

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size
   - Use smaller target size
   - Enable gradient checkpointing

2. **Poor Segmentation Results**:
   - Check data quality and alignment
   - Increase training epochs
   - Adjust learning rate
   - Try different loss functions

3. **Slow Training**:
   - Use GPU acceleration
   - Optimize data loading
   - Reduce model complexity if needed

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your code with debug output
```

### Validation Checklist

Before training:
- [ ] Data loaded correctly
- [ ] Masks align with volumes
- [ ] Augmentation working properly
- [ ] Model architecture suitable

During training:
- [ ] Loss decreasing
- [ ] Metrics improving
- [ ] No overfitting
- [ ] GPU utilization good

After training:
- [ ] Test on unseen data
- [ ] Visual inspection of results
- [ ] Quantitative evaluation
- [ ] Volume measurements reasonable

## Next Steps

1. **Train on your data**: Start with a small subset to validate the pipeline
2. **Tune hyperparameters**: Adjust learning rate, batch size, architecture
3. **Validate results**: Compare with manual annotations
4. **Scale up**: Process your full dataset
5. **Integrate**: Add to your existing analysis pipeline

For questions or issues, please refer to the documentation or create an issue in the repository.