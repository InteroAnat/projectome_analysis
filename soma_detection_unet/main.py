"""
Main script for 3D U-Net soma detection and volume measurement.

This script provides a command-line interface for:
1. Training 3D U-Net models for soma segmentation
2. Running inference on new volumes
3. Evaluating model performance
4. Batch processing multiple volumes
"""

import os
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np

# Import custom modules
from models.unet_3d import UNet3D, SomaUNet3D
from data.dataset import SomaDataset, SomaDataModule, create_sample_data
from training.train import SomaTrainer, train_unet
from inference.predict import SomaDetector
from utils.metrics import comprehensive_evaluation, print_metrics
from utils.visualization import create_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('soma_detection.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories."""
    paths = config['paths']
    for path_name, path_value in paths.items():
        if path_name != 'nrrd_dir' and path_name != 'mask_dir':  # Don't create data directories
            Path(path_value).mkdir(parents=True, exist_ok=True)
    logger.info("Directories setup completed")


def train_model(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Train the 3D U-Net model."""
    logger.info("Starting model training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data module
    data_config = config['data']
    data_module = SomaDataModule(
        nrrd_dir=config['paths']['nrrd_dir'],
        mask_dir=config['paths']['mask_dir'],
        batch_size=config['training']['batch_size'],
        target_size=tuple(data_config['target_size']),
        train_split=data_config['train_split'],
        val_split=data_config['val_split'],
        test_split=data_config['test_split']
    )
    data_module.setup()
    
    # Create model
    model_config = config['model']
    if model_config['type'] == 'unet':
        model = UNet3D(
            n_channels=model_config['n_channels'],
            n_classes=model_config['n_classes']
        )
    elif model_config['type'] == 'soma_unet':
        model = SomaUNet3D(
            n_channels=model_config['n_channels'],
            n_classes=model_config['n_classes'],
            dropout_rate=model_config['dropout_rate'],
            use_attention=model_config['use_attention']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    model = model.to(device)
    
    # Create optimizer
    train_config = config['training']
    if train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=train_config['momentum'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
    
    # Create scheduler
    scheduler_config = train_config['scheduler']
    if scheduler_config == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=train_config['scheduler_factor'],
            patience=train_config['scheduler_patience'],
            verbose=True
        )
    elif scheduler_config == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config['step_size'],
            gamma=train_config['gamma']
        )
    elif scheduler_config == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['epochs']
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = SomaTrainer(
        model=model,
        device=device,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir']
    )
    
    # Train model
    history = trainer.train(train_config['epochs'])
    
    # Plot training curves
    trainer.plot_training_curves(
        save_path=os.path.join(config['paths']['log_dir'], 'training_curves.png')
    )
    
    logger.info("Training completed!")


def run_inference(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Run inference on volumes."""
    logger.info("Starting inference...")
    
    # Create detector
    inference_config = config['inference']
    detector = SomaDetector(
        model_path=args.model_path,
        model_type=config['model']['type'],
        confidence_threshold=inference_config['confidence_threshold'],
        min_soma_volume=inference_config['min_soma_volume'],
        max_soma_volume=inference_config['max_soma_volume'],
        voxel_spacing=tuple(config['data']['voxel_spacing'])
    )
    
    # Process single volume or batch
    if args.volume_path:
        if os.path.isfile(args.volume_path):
            # Single volume
            logger.info(f"Processing single volume: {args.volume_path}")
            result = detector.detect_somas(
                args.volume_path,
                use_sliding_window=inference_config['use_sliding_window'],
                visualize=config['logging']['save_visualizations']
            )
            
            # Save results
            output_dir = config['paths']['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            if config['logging']['save_predictions']:
                import nrrd
                base_name = Path(args.volume_path).stem
                seg_path = os.path.join(output_dir, f'{base_name}_segmentation.nrrd')
                nrrd.write(seg_path, result['segmentation_mask'])
                logger.info(f"Segmentation saved to {seg_path}")
            
            # Print summary
            logger.info(f"Detected {result['num_somas']} somas")
            logger.info(f"Inference time: {result['inference_time']:.2f} seconds")
            
            if result['soma_properties']:
                volumes = [prop['volume_um3'] for prop in result['soma_properties']]
                logger.info(f"Soma volumes (μm³): {volumes}")
                
        elif os.path.isdir(args.volume_path):
            # Batch processing
            logger.info(f"Processing directory: {args.volume_path}")
            
            # Get all volume files
            volume_files = []
            for ext in ['*.nrrd', '*.nhdr', '*.nii.gz']:
                volume_files.extend(Path(args.volume_path).glob(ext))
            
            if not volume_files:
                logger.error("No volume files found in directory")
                return
            
            # Batch process
            volume_paths = [str(f) for f in volume_files]
            results = detector.batch_process(
                volume_paths,
                output_dir=config['paths']['output_dir'],
                save_segmentations=config['logging']['save_predictions'],
                save_visualizations=config['logging']['save_visualizations']
            )
            
            logger.info(f"Batch processing completed for {len(volume_paths)} volumes")
            
    else:
        logger.error("No volume path provided")


def evaluate_model(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Evaluate model performance."""
    logger.info("Starting model evaluation...")
    
    # Create detector
    detector = SomaDetector(
        model_path=args.model_path,
        model_type=config['model']['type'],
        voxel_spacing=tuple(config['data']['voxel_spacing'])
    )
    
    # Evaluate on ground truth
    if args.volume_path and args.mask_path:
        metrics = detector.evaluate_on_ground_truth(
            args.volume_path,
            args.mask_path,
            visualize=config['logging']['save_visualizations']
        )
        
        # Print metrics
        print_metrics(metrics, "Evaluation Results")
        
        # Create evaluation report
        if config['logging']['save_visualizations']:
            create_evaluation_report(
                metrics,
                output_dir=os.path.join(config['paths']['output_dir'], 'evaluation'),
                volume_path=args.volume_path
            )
    else:
        logger.error("Both volume_path and mask_path are required for evaluation")


def create_demo_data(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Create demo data for testing."""
    logger.info("Creating demo data...")
    
    nrrd_dir = config['paths']['nrrd_dir']
    mask_dir = config['paths']['mask_dir']
    
    # Create directories
    Path(nrrd_dir).mkdir(parents=True, exist_ok=True)
    Path(mask_dir).mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    create_sample_data(nrrd_dir, mask_dir, num_samples=args.num_samples)
    
    logger.info(f"Demo data created: {args.num_samples} samples in {nrrd_dir} and {mask_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='3D U-Net for Soma Detection and Volume Measurement')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='configs/config.yaml',
                             help='Path to configuration file')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--config', type=str, default='configs/config.yaml',
                             help='Path to configuration file')
    infer_parser.add_argument('--model_path', type=str, required=True,
                             help='Path to trained model')
    infer_parser.add_argument('--volume_path', type=str, required=True,
                             help='Path to volume file or directory')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--config', type=str, default='configs/config.yaml',
                            help='Path to configuration file')
    eval_parser.add_argument('--model_path', type=str, required=True,
                            help='Path to trained model')
    eval_parser.add_argument('--volume_path', type=str, required=True,
                            help='Path to volume file')
    eval_parser.add_argument('--mask_path', type=str, required=True,
                            help='Path to ground truth mask')
    
    # Demo data command
    demo_parser = subparsers.add_parser('demo', help='Create demo data')
    demo_parser.add_argument('--config', type=str, default='configs/config.yaml',
                            help='Path to configuration file')
    demo_parser.add_argument('--num_samples', type=int, default=20,
                            help='Number of demo samples to create')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if hasattr(args, 'epochs') and args.epochs:
        config['training']['epochs'] = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Setup directories
    setup_directories(config)
    
    # Execute command
    if args.command == 'train':
        train_model(config, args)
    elif args.command == 'infer':
        run_inference(config, args)
    elif args.command == 'evaluate':
        evaluate_model(config, args)
    elif args.command == 'demo':
        create_demo_data(config, args)
    else:
        logger.error(f"Unknown command: {args.command}")


if __name__ == '__main__':
    main()