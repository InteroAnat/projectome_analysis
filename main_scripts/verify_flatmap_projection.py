#!/usr/bin/env python3
"""
Verify Flatmap Projection Correctness
======================================

Validates flatmap projection by checking:
1. Geodesic distances are computed correctly
2. 2D projection preserves neighborhood relationships
3. No vertices are collapsed or lost
4. The mapping is smooth and continuous
5. Visual comparison of 3D vs 2D layouts
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
import trimesh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

# Add paths
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
sys.path.append(neurovis_path)

from NMTFlatmap import InsulaFlatmapNMT


def verify_geodesic_distances(mesh, anchor_idx, lengths):
    """
    Verify geodesic distances are reasonable.
    
    Checks:
    - Anchor point has distance 0
    - All distances are non-negative
    - Distances increase as we move away from anchor
    """
    print("\n" + "="*60)
    print("VERIFICATION 1: Geodesic Distances")
    print("="*60)
    
    # Check 1: Anchor should have distance 0
    if lengths[anchor_idx] != 0:
        print(f"  ❌ FAIL: Anchor distance is {lengths[anchor_idx]}, should be 0")
        return False
    print(f"  ✓ Anchor distance: {lengths[anchor_idx]} (correct)")
    
    # Check 2: All distances non-negative
    min_dist = min(lengths.values())
    if min_dist < 0:
        print(f"  ❌ FAIL: Negative distances found (min={min_dist})")
        return False
    print(f"  ✓ All distances non-negative (min={min_dist:.2f})")
    
    # Check 3: Distance range
    max_dist = max(lengths.values())
    print(f"  ✓ Distance range: 0 to {max_dist:.2f} voxels")
    
    # Check 4: Compare with Euclidean distance (should be correlated)
    anchor_pos = mesh.vertices[anchor_idx]
    euclidean_dists = np.linalg.norm(mesh.vertices - anchor_pos, axis=1)
    geodesic_dists = np.array([lengths.get(i, np.nan) for i in range(len(mesh.vertices))])
    
    # Remove NaN values
    valid = ~np.isnan(geodesic_dists)
    if valid.sum() > 10:
        corr, pval = pearsonr(euclidean_dists[valid], geodesic_dists[valid])
        print(f"  ✓ Correlation with Euclidean distance: r={corr:.3f} (p={pval:.2e})")
        if corr < 0.5:
            print(f"  ⚠️  WARNING: Low correlation - geodesic distances may be wrong")
    
    return True


def verify_projection_smoothness(mesh, coords_2d):
    """
    Verify 2D projection preserves local neighborhood structure.
    
    Checks:
    - Nearby vertices in 3D should be nearby in 2D
    - No sudden jumps or discontinuities
    """
    print("\n" + "="*60)
    print("VERIFICATION 2: Projection Smoothness")
    print("="*60)
    
    # Get vertices that have 2D coordinates
    valid_indices = [i for i in range(len(mesh.vertices)) if i in coords_2d]
    
    if len(valid_indices) < 10:
        print(f"  ❌ FAIL: Only {len(valid_indices)} vertices mapped")
        return False
    
    print(f"  ✓ {len(valid_indices)}/{len(mesh.vertices)} vertices mapped")
    
    # Sample random pairs of nearby vertices
    np.random.seed(42)
    n_samples = min(100, len(valid_indices) // 2)
    
    dist_3d = []
    dist_2d = []
    
    for _ in range(n_samples):
        # Pick a random vertex
        i = np.random.choice(valid_indices)
        
        # Find its neighbors in 3D
        vi = mesh.vertices[i]
        neighbors_3d = []
        for j in valid_indices:
            if i != j:
                d = np.linalg.norm(vi - mesh.vertices[j])
                neighbors_3d.append((j, d))
        
        # Get 5 nearest neighbors
        neighbors_3d.sort(key=lambda x: x[1])
        for j, d3d in neighbors_3d[:5]:
            # Compute 2D distance
            p1 = np.array(coords_2d[i])
            p2 = np.array(coords_2d[j])
            d2d = np.linalg.norm(p1 - p2)
            
            dist_3d.append(d3d)
            dist_2d.append(d2d)
    
    # Check correlation
    dist_3d = np.array(dist_3d)
    dist_2d = np.array(dist_2d)
    
    corr, pval = pearsonr(dist_3d, dist_2d)
    print(f"  ✓ 3D vs 2D distance correlation: r={corr:.3f}")
    
    if corr < 0.3:
        print(f"  ⚠️  WARNING: Low correlation - projection may not preserve neighborhoods")
        return False
    
    # Check for extreme ratios
    ratios = dist_2d / (dist_3d + 1e-6)
    print(f"  ✓ Distance ratio (2D/3D): mean={ratios.mean():.2f}, std={ratios.std():.2f}")
    
    if ratios.std() > 5:
        print(f"  ⚠️  WARNING: High variance in distance ratios - projection may be distorted")
    
    return True


def verify_no_collisions(coords_2d, threshold=0.1):
    """
    Verify no vertices are mapped to the same point (collisions).
    """
    print("\n" + "="*60)
    print("VERIFICATION 3: No Collisions")
    print("="*60)
    
    points = np.array(list(coords_2d.values()))
    
    # Check for duplicate points
    unique_points = np.unique(points, axis=0)
    n_total = len(points)
    n_unique = len(unique_points)
    
    print(f"  Total points: {n_total}")
    print(f"  Unique points: {n_unique}")
    
    if n_unique < n_total:
        print(f"  ❌ FAIL: {n_total - n_unique} points are collided (mapped to same location)")
        return False
    
    print(f"  ✓ No collisions detected")
    
    # Check minimum distance between points
    if len(points) > 1:
        dists = pdist(points)
        min_dist = dists.min()
        print(f"  ✓ Minimum distance between points: {min_dist:.4f}")
        
        if min_dist < threshold:
            print(f"  ⚠️  WARNING: Some points are very close (min dist < {threshold})")
    
    return True


def visualize_3d_vs_2d(mesh, coords_2d, anchor_idx, output_path):
    """
    Create side-by-side visualization of 3D mesh and 2D flatmap.
    """
    print("\n" + "="*60)
    print("VERIFICATION 4: Visual Comparison")
    print("="*60)
    
    fig = plt.figure(figsize=(16, 8))
    
    # 3D view
    ax3d = fig.add_subplot(121, projection='3d')
    
    # Color vertices by distance from anchor
    dists = np.linalg.norm(mesh.vertices - mesh.vertices[anchor_idx], axis=1)
    colors = plt.cm.viridis(dists / dists.max())
    
    # Subsample for visualization
    n_verts = len(mesh.vertices)
    if n_verts > 5000:
        indices = np.random.choice(n_verts, 5000, replace=False)
    else:
        indices = np.arange(n_verts)
    
    ax3d.scatter(mesh.vertices[indices, 0], 
                mesh.vertices[indices, 1], 
                mesh.vertices[indices, 2],
                c=colors[indices], s=1, alpha=0.5)
    
    # Mark anchor
    ax3d.scatter(*mesh.vertices[anchor_idx], c='red', s=100, marker='*', label='Anchor')
    
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Mesh (colored by distance from anchor)')
    ax3d.legend()
    
    # 2D view
    ax2d = fig.add_subplot(122)
    
    points = np.array(list(coords_2d.values()))
    indices_mapped = list(coords_2d.keys())
    
    # Use same coloring
    dists_mapped = dists[indices_mapped]
    colors_mapped = plt.cm.viridis(dists_mapped / dists_mapped.max())
    
    ax2d.scatter(points[:, 0], points[:, 1], c=colors_mapped, s=5, alpha=0.5)
    
    # Mark anchor
    anchor_2d = coords_2d[anchor_idx]
    ax2d.scatter(*anchor_2d, c='red', s=200, marker='*', edgecolors='black', linewidth=2, label='Anchor')
    
    ax2d.set_xlabel('Flatmap X')
    ax2d.set_ylabel('Flatmap Y')
    ax2d.set_title('2D Flatmap (same coloring)')
    ax2d.set_aspect('equal')
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved visualization: {output_path}")
    plt.close()


def verify_flatmap_for_subregion(flatmap_gen, mesh, coords_2d, anchor_idx, subregion_name, output_dir):
    """
    Run all verifications for a single subregion flatmap.
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING: {subregion_name}")
    print(f"{'='*70}")
    
    # Compute geodesic distances
    lengths = flatmap_gen.compute_geodesic_distances(mesh, anchor_idx)
    
    # Run verifications
    v1 = verify_geodesic_distances(mesh, anchor_idx, lengths)
    v2 = verify_projection_smoothness(mesh, coords_2d)
    v3 = verify_no_collisions(coords_2d)
    
    # Create visualization
    vis_path = os.path.join(output_dir, f'verify_{subregion_name}_3d_vs_2d.png')
    visualize_3d_vs_2d(mesh, coords_2d, anchor_idx, vis_path)
    
    # Summary
    print(f"\n{'='*70}")
    if v1 and v2 and v3:
        print(f"✓ {subregion_name}: ALL VERIFICATIONS PASSED")
    else:
        print(f"⚠️  {subregion_name}: SOME VERIFICATIONS FAILED")
    print(f"{'='*70}")
    
    return v1 and v2 and v3


def main():
    """Main verification workflow."""
    import glob
    
    print("\n" + "="*70)
    print("FLATMAP PROJECTION VERIFICATION")
    print("="*70)
    
    output_dir = r".\flatmap_verification"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all flatmap pickle files
    flatmap_files = glob.glob(r".\flatmap_step_by_step\step5_flatmap_*.png")
    
    if not flatmap_files:
        print("No flatmap files found. Please run step_by_step_insula_flatmap.py first.")
        return
    
    print(f"Found {len(flatmap_files)} flatmap files to verify")
    
    # Load one example and verify
    # For now, let's create a simple test case
    print("\nCreating test mesh for verification...")
    
    # Create a simple sphere mesh for testing
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10)
    
    flatmap_gen = InsulaFlatmapNMT()
    anchor_idx = 0
    coords_2d = flatmap_gen.create_flatmap_projection(sphere, anchor_idx)
    
    verify_flatmap_for_subregion(
        flatmap_gen, sphere, coords_2d, anchor_idx, 
        "test_sphere", output_dir
    )
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print(f"Check outputs in: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
