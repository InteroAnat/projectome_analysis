"""
Visualization utilities for soma segmentation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import logging
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        val_dice_scores: List[float],
                        val_precisions: List[float],
                        val_recalls: List[float],
                        save_path: Optional[str] = None,
                        show_plot: bool = True) -> None:
    """
    Plot training curves for model training history.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        val_dice_scores: Validation Dice scores per epoch
        val_precisions: Validation precision per epoch
        val_recalls: Validation recall per epoch
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice score
    axes[0, 1].plot(epochs, val_dice_scores, 'g-', linewidth=2)
    axes[0, 1].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)
    
    # Precision and Recall
    axes[1, 0].plot(epochs, val_precisions, 'm-', label='Precision', linewidth=2)
    axes[1, 0].plot(epochs, val_recalls, 'c-', label='Recall', linewidth=2)
    axes[1, 0].set_title('Validation Precision and Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    # Combined metrics
    axes[1, 1].plot(epochs, val_dice_scores, label='Dice', linewidth=2)
    axes[1, 1].plot(epochs, val_precisions, label='Precision', linewidth=2)
    axes[1, 1].plot(epochs, val_recalls, label='Recall', linewidth=2)
    axes[1, 1].set_title('All Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_soma_detection(volume: np.ndarray,
                           segmentation_mask: np.ndarray,
                           soma_properties: List[Dict],
                           slice_indices: Optional[Tuple[int, int, int]] = None,
                           figsize: Tuple[int, int] = (15, 10),
                           save_path: Optional[str] = None,
                           show_plot: bool = True) -> None:
    """
    Visualize soma detection results with 3D volume slices and segmentation overlay.
    
    Args:
        volume: Original 3D volume
        segmentation_mask: Binary segmentation mask
        soma_properties: List of soma properties dictionaries
        slice_indices: Specific slice indices to visualize (z, y, x)
        figsize: Figure size
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    if slice_indices is None:
        # Use middle slices
        slice_indices = tuple(s // 2 for s in volume.shape)
    
    z_slice, y_slice, x_slice = slice_indices
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Original volume slices
    im1 = axes[0, 0].imshow(volume[z_slice, :, :], cmap='gray', aspect='auto')
    axes[0, 0].set_title(f'Original Volume (Z={z_slice})', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 1].imshow(volume[:, y_slice, :], cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Original Volume (Y={y_slice})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[0, 2].imshow(volume[:, :, x_slice], cmap='gray', aspect='auto')
    axes[0, 2].set_title(f'Original Volume (X={x_slice})', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Segmentation overlay slices
    axes[1, 0].imshow(volume[z_slice, :, :], cmap='gray', aspect='auto')
    axes[1, 0].imshow(segmentation_mask[z_slice, :, :], cmap='Reds', alpha=0.6, aspect='auto')
    axes[1, 0].set_title(f'Segmentation Overlay (Z={z_slice})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(volume[:, y_slice, :], cmap='gray', aspect='auto')
    axes[1, 1].imshow(segmentation_mask[:, y_slice, :], cmap='Reds', alpha=0.6, aspect='auto')
    axes[1, 1].set_title(f'Segmentation Overlay (Y={y_slice})', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(volume[:, :, x_slice], cmap='gray', aspect='auto')
    axes[1, 2].imshow(segmentation_mask[:, :, x_slice], cmap='Reds', alpha=0.6, aspect='auto')
    axes[1, 2].set_title(f'Segmentation Overlay (X={x_slice})', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Add soma count to figure
    fig.suptitle(f'Soma Detection Results - {len(soma_properties)} Somas Detected', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Soma detection visualization saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_volume_comparison(volume: np.ndarray,
                         pred_mask: np.ndarray,
                         gt_mask: np.ndarray,
                         metrics: Optional[Dict] = None,
                         slice_indices: Optional[Tuple[int, int, int]] = None,
                         figsize: Tuple[int, int] = (15, 12),
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """
    Plot comparison between predicted and ground truth segmentation.
    
    Args:
        volume: Original 3D volume
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        metrics: Evaluation metrics dictionary
        slice_indices: Specific slice indices to visualize
        figsize: Figure size
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    if slice_indices is None:
        slice_indices = tuple(s // 2 for s in volume.shape)
    
    z_slice, y_slice, x_slice = slice_indices
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # Original volume
    axes[0, 0].imshow(volume[z_slice, :, :], cmap='gray', aspect='auto')
    axes[0, 0].set_title('Original Volume (Z)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(volume[:, y_slice, :], cmap='gray', aspect='auto')
    axes[0, 1].set_title('Original Volume (Y)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(volume[:, :, x_slice], cmap='gray', aspect='auto')
    axes[0, 2].set_title('Original Volume (X)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Ground truth
    axes[1, 0].imshow(volume[z_slice, :, :], cmap='gray', aspect='auto')
    axes[1, 0].imshow(gt_mask[z_slice, :, :], cmap='Reds', alpha=0.6, aspect='auto')
    axes[1, 0].set_title('Ground Truth (Z)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(volume[:, y_slice, :], cmap='gray', aspect='auto')
    axes[1, 1].imshow(gt_mask[:, y_slice, :], cmap='Reds', alpha=0.6, aspect='auto')
    axes[1, 1].set_title('Ground Truth (Y)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(volume[:, :, x_slice], cmap='gray', aspect='auto')
    axes[1, 2].imshow(gt_mask[:, :, x_slice], cmap='Reds', alpha=0.6, aspect='auto')
    axes[1, 2].set_title('Ground Truth (X)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Prediction
    axes[2, 0].imshow(volume[z_slice, :, :], cmap='gray', aspect='auto')
    axes[2, 0].imshow(pred_mask[z_slice, :, :], cmap='Greens', alpha=0.6, aspect='auto')
    axes[2, 0].set_title('Prediction (Z)', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(volume[:, y_slice, :], cmap='gray', aspect='auto')
    axes[2, 1].imshow(pred_mask[:, y_slice, :], cmap='Greens', alpha=0.6, aspect='auto')
    axes[2, 1].set_title('Prediction (Y)', fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(volume[:, :, x_slice], cmap='gray', aspect='auto')
    axes[2, 2].imshow(pred_mask[:, :, x_slice], cmap='Greens', alpha=0.6, aspect='auto')
    axes[2, 2].set_title('Prediction (X)', fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')
    
    # Add metrics to title if provided
    if metrics:
        title_text = f"Segmentation Comparison"
        if 'dice' in metrics:
            title_text += f" - Dice: {metrics['dice']:.3f}"
        if 'volume_error' in metrics:
            title_text += f" - Vol Error: {metrics['volume_error']:.1%}"
        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Segmentation Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Volume comparison saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_soma_properties(soma_properties: List[Dict],
                        metrics: Optional[Dict] = None,
                        figsize: Tuple[int, int] = (15, 10),
                        save_path: Optional[str] = None,
                        show_plot: bool = True) -> None:
    """
    Plot soma properties including volume distribution, sphericity, etc.
    
    Args:
        soma_properties: List of soma properties dictionaries
        metrics: Optional evaluation metrics
        figsize: Figure size
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    if not soma_properties:
        logger.warning("No soma properties provided for plotting")
        return
    
    # Extract properties
    volumes = [prop['volume_um3'] for prop in soma_properties]
    sphericities = [prop['sphericity'] for prop in soma_properties]
    surface_areas = [prop['surface_area_um2'] for prop in soma_properties]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Volume distribution
    axes[0, 0].hist(volumes, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Volume (μm³)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Soma Volume Distribution', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Volume box plot
    axes[0, 1].boxplot(volumes)
    axes[0, 1].set_ylabel('Volume (μm³)')
    axes[0, 1].set_title('Volume Box Plot', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Volume vs Surface Area scatter
    axes[0, 2].scatter(volumes, surface_areas, alpha=0.7, color='coral')
    axes[0, 2].set_xlabel('Volume (μm³)')
    axes[0, 2].set_ylabel('Surface Area (μm²)')
    axes[0, 2].set_title('Volume vs Surface Area', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Sphericity distribution
    axes[1, 0].hist(sphericities, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].set_xlabel('Sphericity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Sphericity Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sphericity box plot
    axes[1, 1].boxplot(sphericities)
    axes[1, 1].set_ylabel('Sphericity')
    axes[1, 1].set_title('Sphericity Box Plot', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Volume vs Sphericity scatter
    axes[1, 2].scatter(volumes, sphericities, alpha=0.7, color='gold')
    axes[1, 2].set_xlabel('Volume (μm³)')
    axes[1, 2].set_ylabel('Sphericity')
    axes[1, 2].set_title('Volume vs Sphericity', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add summary statistics
    total_somas = len(soma_properties)
    mean_volume = np.mean(volumes)
    std_volume = np.std(volumes)
    mean_sphericity = np.mean(sphericities)
    
    summary_text = f"Total Somas: {total_somas}\n"
    summary_text += f"Mean Volume: {mean_volume:.1f} ± {std_volume:.1f} μm³\n"
    summary_text += f"Mean Sphericity: {mean_sphericity:.3f}"
    
    if metrics and 'dice' in metrics:
        summary_text += f"\nDice Score: {metrics['dice']:.3f}"
    
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.suptitle('Soma Properties Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Soma properties plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_3d_soma_mesh(volume: np.ndarray,
                     segmentation_mask: np.ndarray,
                     soma_properties: List[Dict],
                     figsize: Tuple[int, int] = (12, 10),
                     save_path: Optional[str] = None,
                     show_plot: bool = True) -> None:
    """
    Create 3D mesh visualization of detected somas.
    
    Args:
        volume: Original 3D volume
        segmentation_mask: Binary segmentation mask
        soma_properties: List of soma properties
        figsize: Figure size
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    try:
        # Generate mesh using marching cubes
        from skimage import measure
        
        # Get the largest soma for visualization
        if soma_properties:
            largest_soma = max(soma_properties, key=lambda x: x['volume_voxels'])
            label_id = largest_soma['label_id']
            
            # Create mask for this soma
            soma_mask = (segmentation_mask == label_id)
        else:
            soma_mask = segmentation_mask > 0
        
        # Generate mesh
        verts, faces, normals, values = measure.marching_cubes(soma_mask.astype(float), 
                                                              level=0.5)
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh
        mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                              cmap='viridis', alpha=0.8, linewidth=0.2)
        
        # Customize plot
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        ax.set_zlabel('Z (voxels)')
        ax.set_title('3D Soma Mesh Visualization', fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(mesh, ax=ax, shrink=0.5, aspect=5)
        
        # Set equal aspect ratio
        ax.set_box_aspect([np.ptp(verts[:, 0]), np.ptp(verts[:, 1]), np.ptp(verts[:, 2])])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D mesh visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except ImportError:
        logger.warning("skimage.measure not available for 3D mesh visualization")
    except Exception as e:
        logger.error(f"Error creating 3D mesh visualization: {e}")


def plot_precision_recall_curve(precisions: np.ndarray,
                              recalls: np.ndarray,
                              pr_auc: float,
                              figsize: Tuple[int, int] = (8, 6),
                              save_path: Optional[str] = None,
                              show_plot: bool = True) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        precisions: Precision values
        recalls: Recall values
        pr_auc: Area under the precision-recall curve
        figsize: Figure size
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=figsize)
    
    # Plot PR curve
    plt.plot(recalls, precisions, 'b-', linewidth=2, 
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    
    # Add baseline
    baseline = np.mean(precisions)
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.7,
                label=f'Baseline ({baseline:.3f})')
    
    # Customize plot
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-recall curve saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_evaluation_report(metrics: Dict,
                           soma_properties: Optional[List[Dict]] = None,
                           output_dir: str = 'evaluation_report',
                           volume_path: str = '') -> None:
    """
    Create a comprehensive evaluation report with multiple visualizations.
    
    Args:
        metrics: Evaluation metrics dictionary
        soma_properties: Optional soma properties list
        output_dir: Directory to save the report
        volume_path: Path to the evaluated volume (for reference)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary text file
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Soma Segmentation Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Volume: {volume_path}\n")
        f.write(f"Evaluation Date: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Metrics Summary:\n")
        f.write("-" * 20 + "\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key in ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy', 'specificity']:
                    f.write(f"{key}: {value:.4f}\n")
                elif 'volume' in key and 'um3' in key:
                    f.write(f"{key}: {value:.2f} μm³\n")
                elif 'distance' in key:
                    f.write(f"{key}: {value:.2f} μm\n")
                elif 'error' in key:
                    f.write(f"{key}: {value:.1%}\n")
                else:
                    f.write(f"{key}: {value}\n")
    
    logger.info(f"Evaluation summary saved to {summary_path}")
    
    # Create metrics bar chart
    if any(key in metrics for key in ['dice', 'precision', 'recall', 'f1']):
        metric_names = ['dice', 'precision', 'recall', 'f1']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.ylabel('Score')
        plt.title('Segmentation Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create soma properties plots if available
    if soma_properties:
        plot_soma_properties(soma_properties, 
                           metrics=metrics,
                           save_path=os.path.join(output_dir, 'soma_properties.png'),
                           show_plot=False)
    
    logger.info(f"Evaluation report saved to {output_dir}")


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization functions...")
    
    # Create sample data
    np.random.seed(42)
    volume = np.random.randn(64, 64, 64) * 100 + 500
    
    # Create synthetic soma
    z, y, x = np.ogrid[:64, :64, :64]
    soma_mask = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 10**2
    segmentation_mask = soma_mask.astype(np.uint8)
    
    # Add some noise to segmentation
    noise = np.random.random(segmentation_mask.shape) < 0.05
    segmentation_mask[noise] = 1 - segmentation_mask[noise]
    
    # Create soma properties
    soma_properties = [{
        'label_id': 1,
        'volume_voxels': np.sum(soma_mask),
        'volume_um3': np.sum(soma_mask) * 0.65 * 0.65 * 3.0,
        'surface_area_um2': 500.0,
        'sphericity': 0.85,
        'center_voxel': [32, 32, 32],
        'center_physical': [20.8, 20.8, 96.0],
        'bounding_box': {'min': [22, 22, 22], 'max': [42, 42, 42]}
    }]
    
    # Test training curves
    train_losses = [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.13, 0.12, 0.11]
    val_losses = [0.45, 0.35, 0.28, 0.24, 0.21, 0.19, 0.17, 0.16, 0.15, 0.14]
    val_dice = [0.75, 0.80, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]
    val_precision = [0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
    val_recall = [0.80, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93]
    
    plot_training_curves(train_losses, val_losses, val_dice, val_precision, val_recall, 
                        show_plot=False)
    
    # Test soma detection visualization
    visualize_soma_detection(volume, segmentation_mask, soma_properties, show_plot=False)
    
    # Test soma properties plot
    plot_soma_properties(soma_properties, show_plot=False)
    
    print("Visualization testing completed!")
    print("Check the generated plots to verify functionality.")