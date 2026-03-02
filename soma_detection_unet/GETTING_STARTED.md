# Getting Started with 3D U-Net Soma Detection

## For Your Existing NRRD Files

Since you already have NRRD files containing soma segments, this guide will help you quickly set up and use the 3D U-Net system for automated soma detection and volume measurement.

## Quick Start (5 minutes)

### 1. Test the System with Demo Data

```bash
# Run the quick start script with demo data
python quick_start.py --demo
```

This will:
- Create synthetic training data
- Train a model for 10 epochs
- Test inference on sample volumes
- Show you the complete workflow

### 2. Process Your NRRD Files

Organize your data and run:

```bash
python quick_start.py \
    --nrrd_dir path/to/your/nrrd/files \
    --mask_dir path/to/your/ground_truth/masks \
    --output_dir results \
    --epochs 50
```

## Detailed Setup

### Step 1: Installation

```bash
# Clone or copy the soma_detection_unet directory
# Install dependencies
pip install torch torchvision nibabel nrrd scipy scikit-learn matplotlib seaborn
```

### Step 2: Organize Your Data

```
your_project/
├── soma_detection_unet/     # This system
├── your_nrrd_files/         # Your existing NRRD files
└── results/                 # Output directory
```

### Step 3: Prepare Training Data

You need two sets of files:
- **Volumes**: Your NRRD files with neuron data
- **Masks**: Binary masks showing where somas are located

If you don't have ground truth masks, you can:
1. Manually annotate a few examples
2. Use existing segmentations
3. Start with synthetic data to test the system

### Step 4: Train the Model

```python
from soma_detection_unet.training.train import train_unet

# Train on your data
history = train_unet(
    nrrd_dir='your_nrrd_files',
    mask_dir='your_masks', 
    model_type='soma_unet',
    epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    checkpoint_dir='results/checkpoints',
    log_dir='results/logs'
)
```

### Step 5: Detect Somas in New Data

```python
from soma_detection_unet.inference.predict import SomaDetector

# Create detector with trained model
detector = SomaDetector(
    model_path='results/checkpoints/best_model.pth',
    model_type='soma_unet'
)

# Process your NRRD files
result = detector.detect_somas('your_volume.nrrd')

# Get volume measurements
for soma in result['soma_properties']:
    print(f"Soma volume: {soma['volume_um3']:.2f} μm³")
    print(f"Center position: {soma['center_physical']}")
```

## Adapting to Your Specific Data

### Understanding Your NRRD Files

First, inspect your existing NRRD files:

```python
import nrrd
import numpy as np

# Load one of your files
data, header = nrrd.read('your_file.nrrd')

print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Voxel spacing: {header.get('spacings', 'Not found')}")
print(f"Value range: [{data.min()}, {data.max()}]")
```

### Customizing for Your Data

Update the configuration file `configs/config.yaml`:

```yaml
data:
  voxel_spacing: [0.65, 0.65, 3.0]  # Change to your actual spacing
  target_size: [128, 128, 128]      # Adjust based on your soma sizes

inference:
  min_soma_volume: 100              # Adjust minimum expected soma size
  max_soma_volume: 100000           # Adjust maximum expected soma size
```

### Handling Different Data Formats

If your NRRD files have different characteristics:

```python
class YourCustomDataset(SomaDataset):
    def _load_nrrd(self, file_path):
        data, header = nrrd.read(file_path)
        
        # Apply your custom preprocessing
        if data.dtype == np.uint16:
            data = data.astype(np.float32) / 65535.0
        
        # Handle your specific spacing
        spacing = header.get('spacings', [1.0, 1.0, 1.0])
        
        return data
```

## Common Use Cases

### Case 1: You Have Ground Truth Masks

Perfect! Use them to train the model:

```bash
python main.py train \
    --nrrd_dir your_volumes \
    --mask_dir your_masks \
    --epochs 100
```

### Case 2: You Need to Create Training Masks

If you have NRRD files but no masks:

1. Manually annotate a few examples (5-10)
2. Train initial model
3. Use model to help annotate more data
4. Retrain with larger dataset

### Case 3: You Want to Measure Volumes Automatically

After training:

```python
# Process all your NRRD files
volume_files = ['vol1.nrrd', 'vol2.nrrd', ...]

results = detector.batch_process(
    volume_files,
    output_dir='volume_measurements',
    save_segmentations=True
)

# Extract volume data
volumes = []
for result in results:
    for soma in result['soma_properties']:
        volumes.append(soma['volume_um3'])

print(f"Average soma volume: {np.mean(volumes):.2f} μm³")
```

## Expected Results

### Training Progress

You should see:
- Training loss decreasing
- Validation Dice score increasing (target: >0.85)
- Stable precision/recall values

### Inference Output

For each volume, you'll get:
- **Segmentation mask**: Binary mask showing detected somas
- **Volume measurements**: Physical volumes in μm³
- **Position data**: Center coordinates of each soma
- **Shape metrics**: Sphericity, surface area

### Typical Performance

On good quality data:
- Dice coefficient: 0.85-0.95
- Volume error: <10%
- Processing time: ~1-5 seconds per volume

## Troubleshooting

### "No somas detected"
- Lower the confidence threshold
- Check if your data format matches training data
- Verify voxel spacing is correct

### "Poor segmentation results"
- Train for more epochs
- Check if training data is representative
- Adjust preprocessing parameters

### "Out of memory"
- Reduce batch size
- Use smaller target size
- Process volumes one at a time

## Next Steps

1. **Validate on small subset**: Test on 5-10 volumes first
2. **Adjust parameters**: Fine-tune based on your results
3. **Scale up**: Process your full dataset
4. **Integrate**: Add to your existing analysis pipeline

## Getting Help

- Check the example scripts in `example_usage.py`
- Review the workflow guide in `WORKFLOW_GUIDE.md`
- Run the demo mode first to understand the system
- Start with default parameters and adjust as needed

The system is designed to work with your existing NRRD files with minimal setup. Start simple, validate the results, and then customize as needed for your specific data and requirements.