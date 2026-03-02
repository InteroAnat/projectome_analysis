"""
Example usage scripts for the 3D U-Net soma detection pipeline.

This script demonstrates how to use the soma detection system programmatically.
"""

import os
import tempfile
import numpy as np
from pathlib import Path

# Import the modules
from soma_detection_unet.data.dataset import create_sample_data, SomaDataModule
from soma_detection_unet.training.train import train_unet
from soma_detection_unet.inference.predict import SomaDetector
from soma_detection_unet.utils.metrics import comprehensive_evaluation, print_metrics
from soma_detection_unet.utils.visualization import visualize_soma_detection, plot_soma_properties


def example_1_create_demo_data():
    """Example 1: Create demo data for testing."""
    print("=== Example 1: Creating Demo Data ===")
    
    # Create temporary directory for demo data
    with tempfile.TemporaryDirectory() as temp_dir:
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        mask_dir = os.path.join(temp_dir, 'masks')
        
        # Create 20 sample volumes with somas
        create_sample_data(nrrd_dir, mask_dir, num_samples=20)
        
        print(f"Created 20 sample volumes in {nrrd_dir}")
        print(f"Created corresponding masks in {mask_dir}")
        
        # List created files
        nrrd_files = [f for f in os.listdir(nrrd_dir) if f.endswith('.nrrd')]
        print(f"Created NRRD files: {nrrd_files[:5]}...")  # Show first 5
        
        return temp_dir, nrrd_dir, mask_dir


def example_2_train_model():
    """Example 2: Train a 3D U-Net model."""
    print("\n=== Example 2: Training 3D U-Net Model ===")
    
    # Create demo data
    temp_dir, nrrd_dir, mask_dir = example_1_create_demo_data()
    
    try:
        # Train model
        history = train_unet(
            nrrd_dir=nrrd_dir,
            mask_dir=mask_dir,
            model_type='soma_unet',
            epochs=10,  # Small number for demo
            batch_size=2,
            learning_rate=1e-4,
            checkpoint_dir=os.path.join(temp_dir, 'checkpoints'),
            log_dir=os.path.join(temp_dir, 'logs')
        )
        
        print("Training completed!")
        print(f"Final validation Dice: {history['val_dice_scores'][-1]:.4f}")
        
        # Return paths for next example
        model_path = os.path.join(temp_dir, 'checkpoints', 'best_model.pth')
        return model_path, temp_dir
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None, temp_dir


def example_3_run_inference():
    """Example 3: Run inference on a new volume."""
    print("\n=== Example 3: Running Inference ===")
    
    # Train model first
    model_path, temp_dir = example_2_train_model()
    
    if model_path is None or not os.path.exists(model_path):
        print("Model not available, skipping inference example")
        return
    
    try:
        # Create detector
        detector = SomaDetector(
            model_path=model_path,
            model_type='soma_unet',
            confidence_threshold=0.5,
            min_soma_volume=50,
            max_soma_volume=10000,
            voxel_spacing=(0.65, 0.65, 3.0)
        )
        
        # Create a test volume
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        test_volume = os.path.join(nrrd_dir, 'sample_000.nrrd')
        
        if not os.path.exists(test_volume):
            print(f"Test volume not found: {test_volume}")
            return
        
        # Run inference
        result = detector.detect_somas(
            test_volume,
            use_sliding_window=False,
            visualize=True  # This will show plots
        )
        
        print(f"Inference completed!")
        print(f"Detected {result['num_somas']} somas")
        print(f"Inference time: {result['inference_time']:.2f} seconds")
        
        if result['soma_properties']:
            print(f"Soma volumes (μm³): {[prop['volume_um3'] for prop in result['soma_properties']]}")
            
            # Show soma properties plot
            plot_soma_properties(result['soma_properties'])
        
    except Exception as e:
        print(f"Inference failed: {e}")


def example_4_evaluate_performance():
    """Example 4: Evaluate model performance against ground truth."""
    print("\n=== Example 4: Evaluating Model Performance ===")
    
    # Train model first
    model_path, temp_dir = example_2_train_model()
    
    if model_path is None or not os.path.exists(model_path):
        print("Model not available, skipping evaluation example")
        return
    
    try:
        # Create detector
        detector = SomaDetector(
            model_path=model_path,
            model_type='soma_unet',
            voxel_spacing=(0.65, 0.65, 3.0)
        )
        
        # Get test volume and ground truth
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        mask_dir = os.path.join(temp_dir, 'masks')
        test_volume = os.path.join(nrrd_dir, 'sample_000.nrrd')
        test_mask = os.path.join(mask_dir, 'sample_000.nrrd')
        
        if not (os.path.exists(test_volume) and os.path.exists(test_mask)):
            print("Test data not available")
            return
        
        # Evaluate
        metrics = detector.evaluate_on_ground_truth(
            test_volume,
            test_mask,
            visualize=True
        )
        
        print("Evaluation completed!")
        print_metrics(metrics, "Evaluation Results")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")


def example_5_batch_processing():
    """Example 5: Batch processing multiple volumes."""
    print("\n=== Example 5: Batch Processing ===")
    
    # Train model first
    model_path, temp_dir = example_2_train_model()
    
    if model_path is None or not os.path.exists(model_path):
        print("Model not available, skipping batch processing example")
        return
    
    try:
        # Create detector
        detector = SomaDetector(
            model_path=model_path,
            model_type='soma_unet',
            voxel_spacing=(0.65, 0.65, 3.0)
        )
        
        # Get multiple test volumes
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        volume_files = [os.path.join(nrrd_dir, f) for f in os.listdir(nrrd_dir) 
                       if f.endswith('.nrrd')][:5]  # Use first 5 volumes
        
        if len(volume_files) == 0:
            print("No volume files found")
            return
        
        print(f"Processing {len(volume_files)} volumes...")
        
        # Batch process
        results = detector.batch_process(
            volume_files,
            output_dir=os.path.join(temp_dir, 'batch_results'),
            save_segmentations=True,
            save_visualizations=False  # Set to True to save plots
        )
        
        print("Batch processing completed!")
        
        # Summary
        total_somas = 0
        for vol_path, result in results.items():
            if 'error' not in result:
                num_somas = result['num_somas']
                total_somas += num_somas
                print(f"{Path(vol_path).name}: {num_somas} somas")
        
        print(f"Total somas detected: {total_somas}")
        
    except Exception as e:
        print(f"Batch processing failed: {e}")


def example_6_custom_workflow():
    """Example 6: Custom workflow with data module."""
    print("\n=== Example 6: Custom Workflow ===")
    
    # Create demo data
    temp_dir, nrrd_dir, mask_dir = example_1_create_demo_data()
    
    try:
        # Create data module
        data_module = SomaDataModule(
            nrrd_dir=nrrd_dir,
            mask_dir=mask_dir,
            batch_size=2,
            target_size=(64, 64, 64),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15
        )
        data_module.setup()
        
        print(f"Dataset split:")
        print(f"  Training: {len(data_module.train_dataset)} samples")
        print(f"  Validation: {len(data_module.val_dataset)} samples")
        print(f"  Test: {len(data_module.test_dataset)} samples")
        
        # Get a sample batch
        train_loader = data_module.train_dataloader()
        volumes, masks = next(iter(train_loader))
        
        print(f"Batch shape - Volumes: {volumes.shape}, Masks: {masks.shape}")
        print(f"Volume range: [{volumes.min():.3f}, {volumes.max():.3f}]")
        print(f"Mask unique values: {torch.unique(masks)}")
        
        # Train a simple model
        print("\nTraining small model...")
        history = train_unet(
            nrrd_dir=nrrd_dir,
            mask_dir=mask_dir,
            model_type='soma_unet',
            epochs=3,  # Very small for quick demo
            batch_size=2,
            learning_rate=1e-3,
            checkpoint_dir=os.path.join(temp_dir, 'custom_checkpoints')
        )
        
        print(f"Training completed with final Dice: {history['val_dice_scores'][-1]:.4f}")
        
    except Exception as e:
        print(f"Custom workflow failed: {e}")


def run_all_examples():
    """Run all examples in sequence."""
    print("=" * 60)
    print("3D U-Net Soma Detection - Complete Example Workflow")
    print("=" * 60)
    
    examples = [
        ("Create Demo Data", example_1_create_demo_data),
        ("Train Model", example_2_train_model),
        ("Run Inference", example_3_run_inference),
        ("Evaluate Performance", example_4_evaluate_performance),
        ("Batch Processing", example_5_batch_processing),
        ("Custom Workflow", example_6_custom_workflow)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            example_func()
        except Exception as e:
            print(f"Example '{name}' failed: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    # You can run individual examples or all examples
    
    # Run a specific example
    # example_1_create_demo_data()
    # example_2_train_model()
    # example_3_run_inference()
    
    # Run all examples
    run_all_examples()