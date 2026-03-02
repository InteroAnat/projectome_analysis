"""
Metrics for evaluating soma segmentation performance.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from typing import Dict, List, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                    smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient between prediction and target.
    
    Args:
        pred: Predicted segmentation (binary or probabilities)
        target: Ground truth segmentation (binary)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient (0-1)
    """
    # Ensure tensors are binary
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat)
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.item()


def iou_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                   smooth: float = 1e-6) -> float:
    """
    Calculate Intersection over Union (IoU) coefficient.
    
    Args:
        pred: Predicted segmentation (binary or probabilities)
        target: Ground truth segmentation (binary)
        smooth: Smoothing factor
    
    Returns:
        IoU coefficient (0-1)
    """
    # Ensure tensors are binary
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate intersection and union
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat) - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def precision_recall_f1(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        pred: Predicted segmentation (binary or probabilities)
        target: Ground truth segmentation (binary)
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Ensure tensors are binary
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    
    # Calculate confusion matrix elements
    tp = np.sum(pred_flat * target_flat)
    fp = np.sum(pred_flat * (1 - target_flat))
    fn = np.sum((1 - pred_flat) * target_flat)
    
    # Calculate metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return precision, recall, f1


def hausdorff_distance_3d(pred: np.ndarray, target: np.ndarray, 
                         voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    Calculate Hausdorff distance between two 3D binary masks.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        voxel_spacing: Physical spacing of voxels
    
    Returns:
        Hausdorff distance in physical units
    """
    # Get surface points
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    # Calculate distances
    pred_to_target = calculate_surface_distances(pred_surface, target_surface, voxel_spacing)
    target_to_pred = calculate_surface_distances(target_surface, pred_surface, voxel_spacing)
    
    # Hausdorff distance is the maximum of the minimum distances
    hausdorff = max(np.max(pred_to_target), np.max(target_to_pred))
    
    return hausdorff


def get_surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface points from a 3D binary mask.
    
    Args:
        mask: 3D binary mask
    
    Returns:
        Array of surface point coordinates
    """
    # Calculate gradient magnitude
    gradient = np.gradient(mask.astype(float))
    gradient_magnitude = np.sqrt(sum(g**2 for g in gradient))
    
    # Threshold to get surface points
    surface_mask = gradient_magnitude > 0.1
    surface_points = np.argwhere(surface_mask)
    
    return surface_points


def calculate_surface_distances(surface1: np.ndarray, surface2: np.ndarray, 
                               voxel_spacing: Tuple[float, float, float]) -> np.ndarray:
    """
    Calculate minimum distances from points on surface1 to surface2.
    
    Args:
        surface1: Coordinates of surface 1 points
        surface2: Coordinates of surface 2 points
        voxel_spacing: Physical spacing of voxels
    
    Returns:
        Array of minimum distances
    """
    if len(surface1) == 0 or len(surface2) == 0:
        return np.array([float('inf')])
    
    # Convert to physical coordinates
    surface1_physical = surface1 * np.array(voxel_spacing)
    surface2_physical = surface2 * np.array(voxel_spacing)
    
    # Calculate pairwise distances
    distances = []
    for point1 in surface1_physical:
        point_distances = np.sqrt(np.sum((surface2_physical - point1)**2, axis=1))
        min_distance = np.min(point_distances)
        distances.append(min_distance)
    
    return np.array(distances)


def average_surface_distance(pred: np.ndarray, target: np.ndarray, 
                           voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    Calculate average surface distance between two 3D binary masks.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        voxel_spacing: Physical spacing of voxels
    
    Returns:
        Average surface distance in physical units
    """
    # Get surface points
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float('inf')
    
    # Calculate distances
    pred_to_target = calculate_surface_distances(pred_surface, target_surface, voxel_spacing)
    target_to_pred = calculate_surface_distances(target_surface, pred_surface, voxel_spacing)
    
    # Average surface distance is the average of all minimum distances
    asd = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2
    
    return asd


def surface_overlap_metrics(pred: np.ndarray, target: np.ndarray, 
                           tolerance: float = 1.0,
                           voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Calculate surface overlap metrics.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        tolerance: Tolerance distance for surface overlap
        voxel_spacing: Physical spacing of voxels
    
    Returns:
        Dictionary containing surface overlap metrics
    """
    # Get surface points
    pred_surface = get_surface_points(pred)
    target_surface = get_surface_points(target)
    
    if len(pred_surface) == 0 or len(target_surface) == 0:
        return {'surface_overlap': 0.0, 'pred_surface_covered': 0.0, 'target_surface_covered': 0.0}
    
    # Convert to physical coordinates
    pred_surface_physical = pred_surface * np.array(voxel_spacing)
    target_surface_physical = target_surface * np.array(voxel_spacing)
    
    # Calculate distances
    pred_to_target = calculate_surface_distances(pred_surface, target_surface, voxel_spacing)
    target_to_pred = calculate_surface_distances(target_surface, pred_surface, voxel_spacing)
    
    # Surface overlap within tolerance
    pred_overlap = np.sum(pred_to_target <= tolerance) / len(pred_to_target)
    target_overlap = np.sum(target_to_pred <= tolerance) / len(target_to_pred)
    
    # Average surface overlap
    surface_overlap = (pred_overlap + target_overlap) / 2
    
    return {
        'surface_overlap': float(surface_overlap),
        'pred_surface_covered': float(pred_overlap),
        'target_surface_covered': float(target_overlap)
    }


def compute_volume_metrics(pred: np.ndarray, target: np.ndarray, 
                          voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Compute volume-based metrics for soma segmentation.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        voxel_spacing: Physical spacing of voxels
    
    Returns:
        Dictionary containing volume metrics
    """
    # Calculate volumes
    pred_volume_voxels = np.sum(pred)
    target_volume_voxels = np.sum(target)
    
    pred_volume_um3 = pred_volume_voxels * np.prod(voxel_spacing)
    target_volume_um3 = target_volume_voxels * np.prod(voxel_spacing)
    
    # Volume difference
    volume_diff = abs(pred_volume_um3 - target_volume_um3)
    volume_error = volume_diff / (target_volume_um3 + 1e-8)
    
    # Volume overlap
    intersection = np.sum(pred * target)
    union = np.sum(np.maximum(pred, target))
    
    volume_overlap = intersection / (union + 1e-8)
    volume_dice = 2 * intersection / (pred_volume_voxels + target_volume_voxels + 1e-8)
    
    return {
        'pred_volume_voxels': int(pred_volume_voxels),
        'target_volume_voxels': int(target_volume_voxels),
        'pred_volume_um3': float(pred_volume_um3),
        'target_volume_um3': float(target_volume_um3),
        'volume_difference_um3': float(volume_diff),
        'volume_error': float(volume_error),
        'volume_overlap': float(volume_overlap),
        'volume_dice': float(volume_dice)
    }


def compute_object_level_metrics(pred: np.ndarray, target: np.ndarray, 
                                min_object_size: int = 10) -> Dict[str, Union[float, int]]:
    """
    Compute object-level metrics for soma detection.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        min_object_size: Minimum size for objects to be considered
    
    Returns:
        Dictionary containing object-level metrics
    """
    # Label connected components
    pred_labels, num_pred = ndimage.label(pred)
    target_labels, num_target = ndimage.label(target)
    
    # Filter small objects
    pred_objects = []
    for i in range(1, num_pred + 1):
        obj_mask = (pred_labels == i)
        if np.sum(obj_mask) >= min_object_size:
            pred_objects.append(obj_mask)
    
    target_objects = []
    for i in range(1, num_target + 1):
        obj_mask = (target_labels == i)
        if np.sum(obj_mask) >= min_object_size:
            target_objects.append(obj_mask)
    
    # Calculate object-level metrics
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    # For each predicted object, find best matching target object
    used_target_objects = set()
    
    for pred_obj in pred_objects:
        best_iou = 0
        best_target_idx = -1
        
        for i, target_obj in enumerate(target_objects):
            if i in used_target_objects:
                continue
            
            # Calculate IoU
            intersection = np.sum(pred_obj * target_obj)
            union = np.sum(np.maximum(pred_obj, target_obj))
            iou = intersection / (union + 1e-8)
            
            if iou > best_iou:
                best_iou = iou
                best_target_idx = i
        
        # Consider it a match if IoU > 0.5
        if best_iou > 0.5:
            tp += 1
            used_target_objects.add(best_target_idx)
        else:
            fp += 1
    
    # Remaining target objects are false negatives
    fn = len(target_objects) - len(used_target_objects)
    
    # Calculate metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'num_pred_objects': len(pred_objects),
        'num_target_objects': len(target_objects),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'object_precision': float(precision),
        'object_recall': float(recall),
        'object_f1': float(f1)
    }


def compute_auc_metrics(pred_probs: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute AUC metrics for probabilistic predictions.
    
    Args:
        pred_probs: Predicted probabilities
        target: Ground truth binary mask
    
    Returns:
        Dictionary containing AUC metrics
    """
    # Flatten arrays
    pred_flat = pred_probs.flatten()
    target_flat = target.flatten()
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(target_flat, pred_flat)
    except ValueError:
        roc_auc = 0.0  # Handle case where only one class is present
    
    # Precision-Recall AUC
    try:
        pr_auc = average_precision_score(target_flat, pred_flat)
    except ValueError:
        pr_auc = 0.0
    
    # Precision-Recall curve
    precisions, recalls, _ = precision_recall_curve(target_flat, pred_flat)
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precisions': precisions,
        'recalls': recalls
    }


def comprehensive_evaluation(pred: Union[np.ndarray, torch.Tensor], 
                           target: Union[np.ndarray, torch.Tensor],
                           pred_probs: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                           min_object_size: int = 10) -> Dict[str, Union[float, int]]:
    """
    Perform comprehensive evaluation of soma segmentation.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        pred_probs: Predicted probabilities (optional)
        voxel_spacing: Physical spacing of voxels
        min_object_size: Minimum size for objects to be considered
    
    Returns:
        Comprehensive evaluation metrics
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Ensure binary masks
    pred_binary = (pred > 0.5).astype(np.uint8)
    target_binary = (target > 0.5).astype(np.uint8)
    
    # Convert to tensors for some metrics
    pred_tensor = torch.from_numpy(pred_binary).float()
    target_tensor = torch.from_numpy(target_binary).float()
    
    # Calculate all metrics
    metrics = {}
    
    # Basic segmentation metrics
    dice = dice_coefficient(pred_tensor, target_tensor)
    iou = iou_coefficient(pred_tensor, target_tensor)
    precision, recall, f1 = precision_recall_f1(pred_tensor, target_tensor)
    
    metrics.update({
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    })
    
    # Surface distance metrics
    try:
        hausdorff = hausdorff_distance_3d(pred_binary, target_binary, voxel_spacing)
        avg_surface_dist = average_surface_distance(pred_binary, target_binary, voxel_spacing)
        surface_overlap = surface_overlap_metrics(pred_binary, target_binary, voxel_spacing=voxel_spacing)
        
        metrics.update({
            'hausdorff_distance': float(hausdorff),
            'avg_surface_distance': float(avg_surface_dist),
            **surface_overlap
        })
    except Exception as e:
        logger.warning(f"Error calculating surface distance metrics: {e}")
        metrics.update({
            'hausdorff_distance': float('inf'),
            'avg_surface_distance': float('inf'),
            'surface_overlap': 0.0,
            'pred_surface_covered': 0.0,
            'target_surface_covered': 0.0
        })
    
    # Volume metrics
    volume_metrics = compute_volume_metrics(pred_binary, target_binary, voxel_spacing)
    metrics.update(volume_metrics)
    
    # Object-level metrics
    object_metrics = compute_object_level_metrics(pred_binary, target_binary, min_object_size)
    metrics.update(object_metrics)
    
    # AUC metrics (if probabilities provided)
    if pred_probs is not None:
        if isinstance(pred_probs, torch.Tensor):
            pred_probs = pred_probs.cpu().numpy()
        
        auc_metrics = compute_auc_metrics(pred_probs, target_binary)
        metrics.update({
            'roc_auc': auc_metrics['roc_auc'],
            'pr_auc': auc_metrics['pr_auc']
        })
    
    return metrics


def print_metrics(metrics: Dict[str, Union[float, int]], title: str = "Evaluation Metrics"):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # Basic metrics
    print(f"Dice Coefficient: {metrics.get('dice', 'N/A'):.4f}")
    print(f"IoU: {metrics.get('iou', 'N/A'):.4f}")
    print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
    print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
    print(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}")
    
    # Surface metrics
    if 'hausdorff_distance' in metrics and metrics['hausdorff_distance'] != float('inf'):
        print(f"Hausdorff Distance: {metrics['hausdorff_distance']:.2f} μm")
        print(f"Average Surface Distance: {metrics['avg_surface_distance']:.2f} μm")
        print(f"Surface Overlap: {metrics.get('surface_overlap', 'N/A'):.4f}")
    
    # Volume metrics
    if 'pred_volume_um3' in metrics:
        print(f"Predicted Volume: {metrics['pred_volume_um3']:.2f} μm³")
        print(f"Target Volume: {metrics['target_volume_um3']:.2f} μm³")
        print(f"Volume Error: {metrics.get('volume_error', 'N/A'):.1%}")
    
    # Object metrics
    if 'object_f1' in metrics:
        print(f"Object-level F1: {metrics['object_f1']:.4f}")
        print(f"Detected Objects: {metrics.get('num_pred_objects', 'N/A')}")
        print(f"Target Objects: {metrics.get('num_target_objects', 'N/A')}")
    
    # AUC metrics
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")


if __name__ == '__main__':
    # Test the metrics
    print("Testing metrics functions...")
    
    # Create sample data
    np.random.seed(42)
    target = np.zeros((64, 64, 64))
    # Create a sphere in the center
    z, y, x = np.ogrid[:64, :64, :64]
    mask = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 15**2
    target[mask] = 1
    
    # Create prediction with some noise
    pred = target.copy()
    noise = np.random.random(target.shape) < 0.1
    pred[noise] = 1 - pred[noise]
    
    # Add some random blobs
    for _ in range(3):
        center = np.random.randint(10, 54, 3)
        radius = np.random.randint(3, 8)
        blob_mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
        pred[blob_mask] = 1
    
    # Convert to tensors
    pred_tensor = torch.from_numpy(pred).float()
    target_tensor = torch.from_numpy(target).float()
    
    # Test individual metrics
    dice = dice_coefficient(pred_tensor, target_tensor)
    iou = iou_coefficient(pred_tensor, target_tensor)
    precision, recall, f1 = precision_recall_f1(pred_tensor, target_tensor)
    
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Test comprehensive evaluation
    metrics = comprehensive_evaluation(pred, target, voxel_spacing=(0.65, 0.65, 3.0))
    print_metrics(metrics, "Comprehensive Evaluation")
    
    print("\nMetrics testing completed!")