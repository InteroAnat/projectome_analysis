"""
Inference pipeline for soma detection and volume measurement.
"""

import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import nrrd
import nibabel as nib
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import custom modules
from models.unet_3d import UNet3D, SomaUNet3D
from utils.metrics import compute_volume_metrics, dice_coefficient
from utils.visualization import visualize_soma_detection, plot_volume_comparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SomaDetector:
    """
    Soma detection and volume measurement using trained 3D U-Net.
    """
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'soma_unet',
                 device: Optional[torch.device] = None,
                 confidence_threshold: float = 0.5,
                 min_soma_volume: float = 100,  # minimum volume in voxels
                 max_soma_volume: float = 100000,  # maximum volume in voxels
                 voxel_spacing: Tuple[float, float, float] = (0.65, 0.65, 3.0)):
        """
        Args:
            model_path: Path to trained model checkpoint
            model_type: Type of model ('unet' or 'soma_unet')
            device: Device to run inference on
            confidence_threshold: Threshold for binary segmentation
            min_soma_volume: Minimum soma volume in voxels
            max_soma_volume: Maximum soma volume in voxels
            voxel_spacing: Physical spacing of voxels (x, y, z) in microns
        """
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.min_soma_volume = min_soma_volume
        self.max_soma_volume = max_soma_volume
        self.voxel_spacing = voxel_spacing
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load model
        self.model = self._load_model()
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model
        if self.model_type == 'unet':
            model = UNet3D(n_channels=1, n_classes=1)
        elif self.model_type == 'soma_unet':
            model = SomaUNet3D(n_channels=1, n_classes=1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _preprocess_volume(self, volume: np.ndarray, 
                          target_size: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Preprocess volume for inference.
        
        Args:
            volume: Input volume as numpy array
            target_size: Target size for model input
        
        Returns:
            Preprocessed volume and resize factors
        """
        original_shape = volume.shape
        
        # Normalize intensity
        # Remove outliers (top and bottom 1%)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Calculate resize factors
        resize_factors = tuple(target_size[i] / original_shape[i] for i in range(3))
        
        # Resize volume
        if volume.shape != target_size:
            volume = ndimage.zoom(volume, resize_factors, order=1)
        
        return volume, resize_factors
    
    def _postprocess_prediction(self, prediction: np.ndarray, 
                               original_shape: Tuple[int, int, int],
                               resize_factors: Tuple[float, float, float]) -> np.ndarray:
        """
        Postprocess model prediction.
        
        Args:
            prediction: Model prediction (probabilities)
            original_shape: Original volume shape
            resize_factors: Resize factors used in preprocessing
        
        Returns:
            Binary segmentation mask in original resolution
        """
        # Apply confidence threshold
        binary_mask = (prediction > self.confidence_threshold).astype(np.uint8)
        
        # Resize back to original resolution
        inverse_factors = tuple(1.0 / f for f in resize_factors)
        binary_mask = ndimage.zoom(binary_mask, inverse_factors, order=0)
        
        # Ensure exact original shape
        if binary_mask.shape != original_shape:
            # Pad or crop to exact size
            pad_width = []
            crop_slices = []
            
            for i in range(3):
                if binary_mask.shape[i] < original_shape[i]:
                    # Pad
                    pad_before = (original_shape[i] - binary_mask.shape[i]) // 2
                    pad_after = original_shape[i] - binary_mask.shape[i] - pad_before
                    pad_width.append((pad_before, pad_after))
                    crop_slices.append(slice(None))
                else:
                    # Crop
                    crop_start = (binary_mask.shape[i] - original_shape[i]) // 2
                    crop_end = crop_start + original_shape[i]
                    crop_slices.append(slice(crop_start, crop_end))
                    pad_width.append((0, 0))
            
            if any(pad[0] > 0 or pad[1] > 0 for pad in pad_width):
                binary_mask = np.pad(binary_mask, pad_width, mode='constant', constant_values=0)
            
            binary_mask = binary_mask[tuple(crop_slices)]
        
        return binary_mask
    
    def _sliding_window_inference(self, volume: np.ndarray, 
                                 window_size: Tuple[int, int, int] = (128, 128, 128),
                                 overlap: float = 0.5) -> np.ndarray:
        """
        Perform sliding window inference for large volumes.
        
        Args:
            volume: Input volume
            window_size: Size of sliding window
            overlap: Overlap factor between windows
        
        Returns:
            Full resolution prediction
        """
        volume_shape = volume.shape
        prediction = np.zeros(volume_shape, dtype=np.float32)
        weight_map = np.zeros(volume_shape, dtype=np.float32)
        
        # Calculate step size based on overlap
        step_size = tuple(int(w * (1 - overlap)) for w in window_size)
        
        # Process each window
        for z in range(0, volume_shape[0] - window_size[0] + 1, step_size[0]):
            for y in range(0, volume_shape[1] - window_size[1] + 1, step_size[1]):
                for x in range(0, volume_shape[2] - window_size[2] + 1, step_size[2]):
                    # Extract window
                    window = volume[z:z+window_size[0], 
                                  y:y+window_size[1], 
                                  x:x+window_size[2]]
                    
                    # Pad if necessary
                    pad_width = []
                    for i in range(3):
                        if window.shape[i] < window_size[i]:
                            pad_before = window_size[i] - window.shape[i]
                            pad_width.append((0, pad_before))
                        else:
                            pad_width.append((0, 0))
                    
                    if any(pad[1] > 0 for pad in pad_width):
                        window = np.pad(window, pad_width, mode='constant', constant_values=0)
                    
                    # Convert to tensor and add batch/channel dimensions
                    window_tensor = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).float()
                    window_tensor = window_tensor.to(self.device)
                    
                    # Predict
                    with torch.no_grad():
                        window_pred = self.model(window_tensor)
                        window_pred = torch.sigmoid(window_pred)
                    
                    # Convert back to numpy
                    window_pred = window_pred.squeeze().cpu().numpy()
                    
                    # Remove padding if applied
                    if any(pad[1] > 0 for pad in pad_width):
                        crop_slices = tuple(slice(0, window_size[i] - pad_width[i][1]) 
                                          for i in range(3))
                        window_pred = window_pred[crop_slices]
                    
                    # Add to prediction and weight map
                    pred_slice = tuple(slice(z, z + window_pred.shape[i]) for i in range(3))
                    prediction[pred_slice] += window_pred
                    weight_map[pred_slice] += 1
        
        # Normalize by weight map
        prediction = prediction / (weight_map + 1e-8)
        
        return prediction
    
    def detect_somas(self, volume_path: str, 
                    use_sliding_window: bool = False,
                    visualize: bool = False) -> Dict[str, Union[np.ndarray, List[Dict]]]:
        """
        Detect somas in a volume and measure their properties.
        
        Args:
            volume_path: Path to NRRD or NIfTI file
            use_sliding_window: Whether to use sliding window for large volumes
            visualize: Whether to generate visualization
        
        Returns:
            Dictionary containing segmentation mask and soma properties
        """
        start_time = time.time()
        
        # Load volume
        logger.info(f"Loading volume from {volume_path}")
        if volume_path.endswith('.nii.gz'):
            nii = nib.load(volume_path)
            volume = nii.get_fdata()
            affine = nii.affine
        else:
            volume, header = nrrd.read(volume_path)
            affine = None
        
        original_shape = volume.shape
        logger.info(f"Volume shape: {original_shape}")
        
        # Preprocess
        processed_volume, resize_factors = self._preprocess_volume(volume)
        
        # Inference
        logger.info("Running inference...")
        if use_sliding_window and any(s > 128 for s in original_shape):
            # Use sliding window for large volumes
            prediction = self._sliding_window_inference(processed_volume)
        else:
            # Single pass inference
            volume_tensor = torch.from_numpy(processed_volume).unsqueeze(0).unsqueeze(0).float()
            volume_tensor = volume_tensor.to(self.device)
            
            with torch.no_grad():
                prediction = self.model(volume_tensor)
                prediction = torch.sigmoid(prediction)
                prediction = prediction.squeeze().cpu().numpy()
        
        # Postprocess
        logger.info("Postprocessing prediction...")
        binary_mask = self._postprocess_prediction(prediction, original_shape, resize_factors)
        
        # Extract soma properties
        soma_properties = self._extract_soma_properties(binary_mask)
        
        # Filter by volume
        filtered_properties = [
            prop for prop in soma_properties 
            if self.min_soma_volume <= prop['volume_voxels'] <= self.max_soma_volume
        ]
        
        inference_time = time.time() - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        logger.info(f"Detected {len(filtered_properties)} somas")
        
        # Visualization
        if visualize:
            self._visualize_detection(volume, binary_mask, filtered_properties)
        
        return {
            'segmentation_mask': binary_mask,
            'soma_properties': filtered_properties,
            'inference_time': inference_time,
            'num_somas': len(filtered_properties)
        }
    
    def _extract_soma_properties(self, binary_mask: np.ndarray) -> List[Dict]:
        """
        Extract properties of detected somas.
        
        Args:
            binary_mask: Binary segmentation mask
        
        Returns:
            List of soma properties dictionaries
        """
        # Label connected components
        labeled_mask, num_features = ndimage.label(binary_mask)
        
        properties = []
        
        for label_id in range(1, num_features + 1):
            # Extract individual soma
            soma_mask = (labeled_mask == label_id)
            
            # Calculate properties
            volume_voxels = np.sum(soma_mask)
            volume_um3 = volume_voxels * np.prod(self.voxel_spacing)
            
            # Get bounding box
            coords = np.argwhere(soma_mask)
            if len(coords) > 0:
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                center = (min_coords + max_coords) / 2
                
                # Calculate physical coordinates
                center_physical = center * np.array(self.voxel_spacing)
                
                # Calculate surface area (approximation)
                surface_voxels = np.sum(soma_mask) - np.sum(ndimage.binary_erosion(soma_mask))
                surface_area = surface_voxels * np.prod(self.voxel_spacing[:2])  # Approximate
                
                # Calculate sphericity
                volume_sphere = (4/3) * np.pi * (volume_um3 / (4/3 * np.pi))**(2/3)
                sphericity = volume_sphere / surface_area if surface_area > 0 else 0
                
                properties.append({
                    'label_id': label_id,
                    'volume_voxels': int(volume_voxels),
                    'volume_um3': float(volume_um3),
                    'surface_area_um2': float(surface_area),
                    'sphericity': float(sphericity),
                    'center_voxel': center.tolist(),
                    'center_physical': center_physical.tolist(),
                    'bounding_box': {
                        'min': min_coords.tolist(),
                        'max': max_coords.tolist()
                    }
                })
        
        return properties
    
    def _visualize_detection(self, volume: np.ndarray, 
                           segmentation_mask: np.ndarray, 
                           soma_properties: List[Dict]):
        """
        Create visualization of soma detection results.
        
        Args:
            volume: Original volume
            segmentation_mask: Binary segmentation mask
            soma_properties: List of soma properties
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Find middle slices
        mid_slices = [s // 2 for s in volume.shape]
        
        # Original volume slices
        axes[0, 0].imshow(volume[mid_slices[0], :, :], cmap='gray')
        axes[0, 0].set_title(f'Original Volume (Z={mid_slices[0]})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(volume[:, mid_slices[1], :], cmap='gray')
        axes[0, 1].set_title(f'Original Volume (Y={mid_slices[1]})')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(volume[:, :, mid_slices[2]], cmap='gray')
        axes[0, 2].set_title(f'Original Volume (X={mid_slices[2]})')
        axes[0, 2].axis('off')
        
        # Segmentation overlay slices
        axes[1, 0].imshow(volume[mid_slices[0], :, :], cmap='gray')
        axes[1, 0].imshow(segmentation_mask[mid_slices[0], :, :], cmap='Reds', alpha=0.5)
        axes[1, 0].set_title(f'Segmentation Overlay (Z={mid_slices[0]})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(volume[:, mid_slices[1], :], cmap='gray')
        axes[1, 1].imshow(segmentation_mask[:, mid_slices[1], :], cmap='Reds', alpha=0.5)
        axes[1, 1].set_title(f'Segmentation Overlay (Y={mid_slices[1]})')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(volume[:, :, mid_slices[2]], cmap='gray')
        axes[1, 2].imshow(segmentation_mask[:, :, mid_slices[2]], cmap='Reds', alpha=0.5)
        axes[1, 2].set_title(f'Segmentation Overlay (X={mid_slices[2]})')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Volume distribution plot
        if soma_properties:
            volumes = [prop['volume_um3'] for prop in soma_properties]
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.hist(volumes, bins=20, edgecolor='black')
            plt.xlabel('Volume (μm³)')
            plt.ylabel('Count')
            plt.title('Soma Volume Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.boxplot(volumes)
            plt.ylabel('Volume (μm³)')
            plt.title('Soma Volume Box Plot')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def batch_process(self, 
                     volume_paths: List[str], 
                     output_dir: str,
                     save_segmentations: bool = True,
                     save_visualizations: bool = True) -> Dict[str, Dict]:
        """
        Process multiple volumes and save results.
        
        Args:
            volume_paths: List of paths to volume files
            output_dir: Directory to save results
            save_segmentations: Whether to save segmentation masks
            save_visualizations: Whether to save visualization plots
        
        Returns:
            Dictionary mapping volume paths to their results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        for i, volume_path in enumerate(volume_paths):
            logger.info(f"Processing volume {i+1}/{len(volume_paths)}: {volume_path}")
            
            try:
                # Detect somas
                result = self.detect_somas(volume_path, visualize=False)
                
                # Store results
                all_results[volume_path] = result
                
                # Save segmentation mask
                if save_segmentations:
                    base_name = os.path.splitext(os.path.basename(volume_path))[0]
                    if volume_path.endswith('.nii.gz'):
                        base_name = base_name.replace('.nii', '')
                    
                    seg_path = os.path.join(output_dir, f'{base_name}_segmentation.nrrd')
                    nrrd.write(seg_path, result['segmentation_mask'])
                    logger.info(f"Saved segmentation mask: {seg_path}")
                
                # Save visualization
                if save_visualizations:
                    # Load original volume for visualization
                    if volume_path.endswith('.nii.gz'):
                        volume = nib.load(volume_path).get_fdata()
                    else:
                        volume, _ = nrrd.read(volume_path)
                    
                    self._visualize_detection(volume, 
                                            result['segmentation_mask'], 
                                            result['soma_properties'])
                    
                    viz_path = os.path.join(output_dir, f'{base_name}_visualization.png')
                    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved visualization: {viz_path}")
                
                # Save summary CSV
                if result['soma_properties']:
                    import pandas as pd
                    df = pd.DataFrame(result['soma_properties'])
                    csv_path = os.path.join(output_dir, f'{base_name}_soma_properties.csv')
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved soma properties: {csv_path}")
                
            except Exception as e:
                logger.error(f"Error processing {volume_path}: {e}")
                all_results[volume_path] = {'error': str(e)}
        
        # Save overall summary
        summary_path = os.path.join(output_dir, 'batch_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Soma Detection Batch Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            
            total_somas = 0
            for vol_path, result in all_results.items():
                if 'error' in result:
                    f.write(f"{vol_path}: ERROR - {result['error']}\n")
                else:
                    num_somas = result['num_somas']
                    total_somas += num_somas
                    f.write(f"{vol_path}: {num_somas} somas detected\n")
            
            f.write(f"\nTotal somas detected: {total_somas}\n")
        
        logger.info(f"Batch processing completed. Results saved to {output_dir}")
        return all_results
    
    def evaluate_on_ground_truth(self, 
                               volume_path: str, 
                               gt_mask_path: str,
                               visualize: bool = True) -> Dict[str, float]:
        """
        Evaluate model predictions against ground truth mask.
        
        Args:
            volume_path: Path to volume file
            gt_mask_path: Path to ground truth mask
            visualize: Whether to generate visualization
        
        Returns:
            Evaluation metrics dictionary
        """
        # Detect somas
        result = self.detect_somas(volume_path, visualize=False)
        pred_mask = result['segmentation_mask']
        
        # Load ground truth
        if gt_mask_path.endswith('.nii.gz'):
            gt_mask = nib.load(gt_mask_path).get_fdata()
        else:
            gt_mask, _ = nrrd.read(gt_mask_path)
        
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # Ensure same shape
        if pred_mask.shape != gt_mask.shape:
            logger.warning(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
            # Resize prediction to match ground truth
            resize_factors = tuple(gt_mask.shape[i] / pred_mask.shape[i] for i in range(3))
            pred_mask = ndimage.zoom(pred_mask, resize_factors, order=0)
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
        # Convert to tensors for metric calculation
        pred_tensor = torch.from_numpy(pred_mask).float().unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0).unsqueeze(0)
        
        # Calculate metrics
        dice = dice_coefficient(pred_tensor, gt_tensor)
        
        # Precision and recall
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        tp = np.sum(pred_flat * gt_flat)
        fp = np.sum(pred_flat * (1 - gt_flat))
        fn = np.sum((1 - pred_flat) * gt_flat)
        tn = np.sum((1 - pred_flat) * (1 - gt_flat))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Volume metrics
        pred_volume = np.sum(pred_mask) * np.prod(self.voxel_spacing)
        gt_volume = np.sum(gt_mask) * np.prod(self.voxel_spacing)
        volume_error = abs(pred_volume - gt_volume) / (gt_volume + 1e-8)
        
        metrics = {
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'pred_volume_um3': float(pred_volume),
            'gt_volume_um3': float(gt_volume),
            'volume_error': float(volume_error)
        }
        
        # Visualization
        if visualize:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Find middle slice
            mid_slice = volume_shape[0] // 2 if len(volume_shape := pred_mask.shape) > 2 else 0
            
            # Load original volume for visualization
            if volume_path.endswith('.nii.gz'):
                volume = nib.load(volume_path).get_fdata()
            else:
                volume, _ = nrrd.read(volume_path)
            
            # Original
            axes[0].imshow(volume[mid_slice], cmap='gray')
            axes[0].set_title('Original Volume')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(volume[mid_slice], cmap='gray')
            axes[1].imshow(gt_mask[mid_slice], cmap='Reds', alpha=0.5)
            axes[1].set_title(f'Ground Truth (Dice: {dice:.3f})')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(volume[mid_slice], cmap='gray')
            axes[2].imshow(pred_mask[mid_slice], cmap='Greens', alpha=0.5)
            axes[2].set_title(f'Prediction (Vol Error: {volume_error:.1%})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return metrics


def create_demo_data():
    """Create demo data for testing the detector"""
    import tempfile
    from data.dataset import create_sample_data
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    nrrd_dir = os.path.join(temp_dir, 'nrrd')
    mask_dir = os.path.join(temp_dir, 'masks')
    
    # Create sample data
    create_sample_data(nrrd_dir, mask_dir, num_samples=3)
    
    return nrrd_dir, mask_dir


if __name__ == '__main__':
    # Example usage
    import tempfile
    from data.dataset import create_sample_data
    from training.train import train_unet
    
    # Create sample data and train a model
    with tempfile.TemporaryDirectory() as temp_dir:
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        mask_dir = os.path.join(temp_dir, 'masks')
        
        # Create sample data
        create_sample_data(nrrd_dir, mask_dir, num_samples=10)
        
        # Train a model
        history = train_unet(
            nrrd_dir=nrrd_dir,
            mask_dir=mask_dir,
            model_type='soma_unet',
            epochs=5,
            batch_size=2,
            checkpoint_dir=os.path.join(temp_dir, 'checkpoints'),
            log_dir=os.path.join(temp_dir, 'logs')
        )
        
        # Create detector
        model_path = os.path.join(temp_dir, 'checkpoints', 'best_model.pth')
        detector = SomaDetector(model_path=model_path)
        
        # Test detection
        test_volume = os.path.join(nrrd_dir, 'sample_000.nrrd')
        result = detector.detect_somas(test_volume, visualize=True)
        
        print(f"Detected {result['num_somas']} somas")
        print(f"Soma volumes: {[prop['volume_um3'] for prop in result['soma_properties']]}")
        
        # Test evaluation
        test_mask = os.path.join(mask_dir, 'sample_000.nrrd')
        metrics = detector.evaluate_on_ground_truth(test_volume, test_mask)
        print(f"Evaluation metrics: {metrics}")