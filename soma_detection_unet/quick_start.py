#!/usr/bin/env python3
"""
Quick Start Script for 3D U-Net Soma Detection

This script provides a complete, automated workflow for soma detection
starting from your existing NRRD files.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SomaDataset, create_sample_data
from training.train import train_unet
from inference.predict import SomaDetector
from utils.metrics import comprehensive_evaluation, print_metrics
from utils.visualization import visualize_soma_detection, plot_soma_properties

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_train_and_test(nrrd_dir, mask_dir, output_dir, epochs=50, quick_mode=False):
    """
    Quick training and testing workflow.
    
    Args:
        nrrd_dir: Directory containing NRRD files
        mask_dir: Directory containing mask files  
        output_dir: Output directory for results
        epochs: Number of training epochs
        quick_mode: If True, use reduced settings for faster execution
    """
    logger.info("="*60)
    logger.info("3D U-Net Soma Detection - Quick Start")
    logger.info("="*60)
    
    # Create output directories
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    results_dir = os.path.join(output_dir, 'results')
    
    for dir_path in [checkpoint_dir, log_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Adjust settings for quick mode
    if quick_mode:
        epochs = min(epochs, 10)
        batch_size = 1
        target_size = (64, 64, 64)
        logger.info("Running in quick mode with reduced settings")
    else:
        batch_size = 2
        target_size = (128, 128, 128)
    
    # Step 1: Check data availability
    logger.info(f"Checking data in {nrrd_dir}")
    nrrd_files = [f for f in os.listdir(nrrd_dir) if f.endswith(('.nrrd', '.nhdr'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.nrrd', '.nhdr', '.nii.gz'))]
    
    if not nrrd_files:
        logger.error(f"No NRRD files found in {nrrd_dir}")
        return False
    
    if not mask_files:
        logger.error(f"No mask files found in {mask_dir}")
        return False
    
    logger.info(f"Found {len(nrrd_files)} NRRD files and {len(mask_files)} mask files")
    
    # Step 2: Train model
    logger.info(f"Training model for {epochs} epochs...")
    try:
        history = train_unet(
            nrrd_dir=nrrd_dir,
            mask_dir=mask_dir,
            model_type='soma_unet',
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=1e-4,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir
        )
        
        final_dice = history['val_dice_scores'][-1]
        logger.info(f"Training completed! Final validation Dice: {final_dice:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False
    
    # Step 3: Test inference
    logger.info("Testing inference on training data...")
    try:
        # Find best model
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if not os.path.exists(best_model_path):
            # Use last checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
            if checkpoints:
                best_model_path = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
            else:
                logger.error("No model checkpoints found")
                return False
        
        # Create detector
        detector = SomaDetector(
            model_path=best_model_path,
            model_type='soma_unet',
            confidence_threshold=0.5,
            min_soma_volume=50,
            max_soma_volume=50000,
            voxel_spacing=(0.65, 0.65, 3.0)
        )
        
        # Test on a few volumes
        test_volumes = nrrd_files[:min(3, len(nrrd_files))]
        all_results = []
        
        for vol_file in test_volumes:
            vol_path = os.path.join(nrrd_dir, vol_file)
            logger.info(f"Testing on {vol_file}")
            
            result = detector.detect_somas(
                vol_path,
                use_sliding_window=False,
                visualize=False  # Set to True to see plots
            )
            
            all_results.append({
                'volume': vol_file,
                'num_somas': result['num_somas'],
                'inference_time': result['inference_time']
            })
            
            if result['soma_properties']:
                volumes = [prop['volume_um3'] for prop in result['soma_properties']]
                logger.info(f"  Detected {result['num_somas']} somas")
                logger.info(f"  Volumes: {[f'{v:.1f}' for v in volumes]} μm³")
            else:
                logger.info(f"  No somas detected")
        
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return False
    
    # Step 4: Summary
    logger.info("="*60)
    logger.info("Quick Start Summary")
    logger.info("="*60)
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Final validation Dice: {final_dice:.4f}")
    logger.info(f"Tested on {len(test_volumes)} volumes")
    
    total_somas = sum(r['num_somas'] for r in all_results)
    avg_time = np.mean([r['inference_time'] for r in all_results])
    
    logger.info(f"Total somas detected: {total_somas}")
    logger.info(f"Average inference time: {avg_time:.2f} seconds")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info(f"Results saved to: {output_dir}")
    
    return True


def create_demo_workflow():
    """Create a complete demo workflow with synthetic data."""
    logger.info("Creating demo workflow with synthetic data...")
    
    # Create temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        mask_dir = os.path.join(temp_dir, 'masks')
        output_dir = os.path.join(temp_dir, 'output')
        
        # Create synthetic data
        logger.info("Creating synthetic training data...")
        create_sample_data(nrrd_dir, mask_dir, num_samples=20)
        
        # Run quick training and testing
        success = quick_train_and_test(
            nrrd_dir, mask_dir, output_dir, 
            epochs=10, quick_mode=True
        )
        
        if success:
            logger.info("Demo workflow completed successfully!")
        else:
            logger.error("Demo workflow failed")
        
        return success


def process_existing_data(nrrd_dir, mask_dir, output_dir, epochs=50):
    """Process your existing NRRD data."""
    logger.info(f"Processing existing data from {nrrd_dir}")
    
    if not os.path.exists(nrrd_dir):
        logger.error(f"NRRD directory does not exist: {nrrd_dir}")
        return False
    
    if not os.path.exists(mask_dir):
        logger.error(f"Mask directory does not exist: {mask_dir}")
        return False
    
    return quick_train_and_test(nrrd_dir, mask_dir, output_dir, epochs, quick_mode=False)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Quick start script for 3D U-Net soma detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data
  python quick_start.py --demo
  
  # Process your existing data
  python quick_start.py --nrrd_dir path/to/volumes --mask_dir path/to/masks --output_dir results
  
  # Quick test with reduced settings
  python quick_start.py --demo --quick --epochs 5
        """
    )
    
    # Demo mode
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with synthetic data')
    
    # Data directories
    parser.add_argument('--nrrd_dir', type=str,
                       help='Directory containing NRRD volume files')
    parser.add_argument('--mask_dir', type=str,
                       help='Directory containing mask files')
    parser.add_argument('--output_dir', type=str, default='quick_start_results',
                       help='Output directory for results (default: quick_start_results)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced settings for faster execution')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.demo:
        # Run demo mode
        success = create_demo_workflow()
    elif args.nrrd_dir and args.mask_dir:
        # Process existing data
        success = process_existing_data(
            args.nrrd_dir, 
            args.mask_dir, 
            args.output_dir,
            epochs=args.epochs
        )
    else:
        parser.print_help()
        print("\nError: Either --demo or both --nrrd_dir and --mask_dir must be specified")
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())