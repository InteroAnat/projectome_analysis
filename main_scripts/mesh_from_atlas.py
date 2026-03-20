#!/usr/bin/env python3
"""
mesh_from_atlas.py - Wrapper for Monkey Atlas Mesh Generation

A simplified interface for generating 3D mesh files from volumetric atlases
using the specialized monkey atlas mesh generator.

Usage:
    # Generate mesh from default ARM atlas at hierarchy level 6
    python mesh_from_atlas.py
    
    # Generate mesh with specific parameters
    python mesh_from_atlas.py --atlas_path /path/to/atlas.nii.gz --level 3 --output_dir ./meshes
    
    # Use with custom colors and naming
    python mesh_from_atlas.py --level 3 --colors palette.txt --output_name my_atlas.mz3

Author: Wrapper around volume_to_mesh_mz3_MONKEY.py
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the submodule path to import the monkey atlas generator
SCRIPT_DIR = Path(__file__).parent.resolve()
MONKEY_MESH_PATH = SCRIPT_DIR / ".." / "subcortex_visualization" / "monkey_atlas_guide" / "volume_to_mesh_mz3_MONKEY.py"

# Default ARM atlas path (relative to project structure)
DEFAULT_ARM_ATLAS = SCRIPT_DIR / ".." / "atlas" / "ARM" / "ARM_in_NMT_v2.1_sym.nii.gz"


def run_mesh_generation(
    atlas_path=None,
    hierarchy_level=6,
    output_dir="./mesh_outputs",
    output_name="monkey_atlas.mz3",
    colors_file="colors.txt",
    min_voxels=100,
    delete_intermediate=False,
    verbose=True
):
    """
    Generate 3D mesh from volumetric atlas.
    
    This is a wrapper around volume_to_mesh_mz3_MONKEY.py that provides
    a simplified interface with sensible defaults.
    
    Parameters
    ----------
    atlas_path : str or Path, optional
        Path to the atlas NIfTI file. Default is ARM atlas.
    hierarchy_level : int, default 6
        Which hierarchy level to use (1-6 for ARM atlas).
        Level 1 = coarsest (fewer regions)
        Level 6 = finest (most regions)
    output_dir : str or Path, default "./mesh_outputs"
        Directory to save output mesh files.
    output_name : str, default "monkey_atlas.mz3"
        Name for the final combined mesh file.
    colors_file : str or Path, default "colors.txt"
        Path to color palette file (RGB values, one per line).
    min_voxels : int, default 100
        Minimum voxel count for a region to be processed.
        Smaller regions often fail mesh generation.
    delete_intermediate : bool, default False
        If True, delete individual region meshes after combining.
    verbose : bool, default True
        Print progress information.
    
    Returns
    -------
    Path
        Path to the generated combined mesh file.
    
    Raises
    ------
    FileNotFoundError
        If atlas file or mesh generator script not found.
    RuntimeError
        If mesh generation fails.
    
    Examples
    --------
    >>> # Quick start with defaults (ARM atlas, level 6)
    >>> mesh_file = run_mesh_generation()
    
    >>> # Coarser atlas with fewer regions
    >>> mesh_file = run_mesh_generation(hierarchy_level=3)
    
    >>> # Custom atlas and output location
    >>> mesh_file = run_mesh_generation(
    ...     atlas_path="/data/my_atlas.nii.gz",
    ...     output_dir="/output/meshes",
    ...     output_name="my_mesh.mz3"
    ... )
    """
    
    # Resolve atlas path
    if atlas_path is None:
        atlas_path = DEFAULT_ARM_ATLAS
    atlas_path = Path(atlas_path).resolve()
    
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    # Resolve mesh generator script path
    mesh_script = MONKEY_MESH_PATH.resolve()
    if not mesh_script.exists():
        raise FileNotFoundError(f"Mesh generator script not found: {mesh_script}")
    
    # Resolve output directory
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve colors file
    colors_path = Path(colors_file)
    if not colors_path.is_absolute():
        # Look in output directory first, then script directory
        if (output_dir / colors_file).exists():
            colors_path = output_dir / colors_file
        elif (mesh_script.parent / colors_file).exists():
            colors_path = mesh_script.parent / colors_file
        else:
            # Create a default colors.txt in output directory
            colors_path = output_dir / "colors.txt"
            _create_default_colors(colors_path)
    
    # Build command
    cmd = [
        sys.executable,
        str(mesh_script),
        "--input_volume", str(atlas_path),
        "--hierarchy_level", str(hierarchy_level),
        "--output_path", str(output_dir),
        "--out_file", output_name,
        "--colors", str(colors_path),
        "--min-voxels", str(min_voxels),
    ]
    
    if delete_intermediate:
        cmd.append("--delete_mz3")
    
    if verbose:
        print("=" * 60)
        print("Monkey Atlas Mesh Generation")
        print("=" * 60)
        print(f"Atlas: {atlas_path}")
        print(f"Hierarchy Level: {hierarchy_level}")
        print(f"Output: {output_dir / output_name}")
        print(f"Min Voxels: {min_voxels}")
        print(f"Colors: {colors_path}")
        print("-" * 60)
    
    # Run mesh generation
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Mesh generation failed:\n{result.stderr}")
    
    output_file = output_dir / output_name
    
    if verbose:
        print(f"\n✓ Mesh generated successfully: {output_file}")
    
    return output_file


def _create_default_colors(colors_path, n_colors=256):
    """Create a default color palette if none exists."""
    import numpy as np
    
    # Generate distinct colors using HSV color space
    colors = []
    for i in range(n_colors):
        hue = (i * 137.508) % 360  # Golden angle for good distribution
        saturation = 0.7 + (i % 3) * 0.1  # 0.7-0.9
        value = 0.8 + (i % 2) * 0.1     # 0.8-0.9
        
        # Convert HSV to RGB
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        colors.append(f"{r} {g} {b}")
    
    with open(colors_path, 'w') as f:
        f.write('\n'.join(colors))
    
    print(f"Created default color palette: {colors_path} ({n_colors} colors)")


def get_atlas_info(atlas_path=None):
    """
    Get information about an atlas without generating mesh.
    
    Parameters
    ----------
    atlas_path : str or Path, optional
        Path to atlas file. Default is ARM atlas.
    
    Returns
    -------
    dict
        Dictionary with atlas information.
    """
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        raise ImportError("nibabel is required for atlas info. Install with: pip install nibabel")
    
    if atlas_path is None:
        atlas_path = DEFAULT_ARM_ATLAS
    atlas_path = Path(atlas_path)
    
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")
    
    img = nib.load(atlas_path)
    data = img.get_fdata()
    
    info = {
        "path": str(atlas_path),
        "shape": data.shape,
        "dimensions": len(data.shape),
    }
    
    # Analyze hierarchy levels
    if len(data.shape) == 5:
        info["n_hierarchy_levels"] = data.shape[4]
        info["levels_info"] = []
        for level in range(data.shape[4]):
            level_data = data[:, :, :, 0, level]
            unique = np.unique(level_data)
            n_regions = len([x for x in unique if x > 0])
            info["levels_info"].append({
                "level": level,
                "n_regions": n_regions,
                "index_range": (int(unique[1]), int(unique[-1])) if len(unique) > 1 else None
            })
    elif len(data.shape) == 4 and data.shape[3] <= 10:
        info["n_hierarchy_levels"] = data.shape[3]
    else:
        info["n_hierarchy_levels"] = 1
        unique = np.unique(data)
        info["n_regions"] = len([x for x in unique if x > 0])
    
    return info


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate 3D mesh from monkey atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default ARM atlas at finest level (6)
  %(prog)s
  
  # Coarser level with fewer regions
  %(prog)s --level 3
  
  # Custom atlas
  %(prog)s --atlas /path/to/my_atlas.nii.gz --output_dir ./my_meshes
  
  # Include only larger regions
  %(prog)s --min-voxels 500 --level 4
  
  # Clean up intermediate files
  %(prog)s --delete-intermediate --level 6

Hierarchy Levels (ARM atlas):
  Level 1: Coarsest (~10-20 large regions)
  Level 3: Medium (~100-200 regions)
  Level 6: Finest (~200+ detailed regions)
        """
    )
    
    parser.add_argument("--atlas", "--atlas_path", "--input_volume",
                        type=str, default=None,
                        help="Path to atlas NIfTI file (default: ARM atlas)")
    
    parser.add_argument("--level", "--hierarchy_level",
                        type=int, default=6,
                        help="Hierarchy level 1-6 (default: 6, finest)")
    
    parser.add_argument("--output_dir", "-o",
                        type=str, default="./mesh_outputs",
                        help="Output directory (default: ./mesh_outputs)")
    
    parser.add_argument("--output_name", "-n",
                        type=str, default="monkey_atlas.mz3",
                        help="Output filename (default: monkey_atlas.mz3)")
    
    parser.add_argument("--colors",
                        type=str, default="colors.txt",
                        help="Color palette file (default: colors.txt)")
    
    parser.add_argument("--min-voxels",
                        type=int, default=100,
                        help="Minimum voxels per region (default: 100)")
    
    parser.add_argument("--delete-intermediate", "--delete_mz3",
                        action="store_true",
                        help="Delete intermediate mesh files after combining")
    
    parser.add_argument("--info", "--atlas-info",
                        action="store_true",
                        help="Show atlas information and exit")
    
    parser.add_argument("--quiet", "-q",
                        action="store_true",
                        help="Suppress output messages")
    
    args = parser.parse_args()
    
    # Show info and exit if requested
    if args.info:
        info = get_atlas_info(args.atlas)
        print("\nAtlas Information:")
        print("=" * 50)
        print(f"Path: {info['path']}")
        print(f"Shape: {info['shape']}")
        print(f"Dimensions: {info['dimensions']}D")
        print(f"Hierarchy Levels: {info['n_hierarchy_levels']}")
        if 'levels_info' in info:
            print("\nRegions per level:")
            for li in info['levels_info']:
                print(f"  Level {li['level']}: {li['n_regions']} regions "
                      f"(indices {li['index_range'][0]}-{li['index_range'][1]})")
        elif 'n_regions' in info:
            print(f"Total Regions: {info['n_regions']}")
        print()
        return
    
    # Run mesh generation
    try:
        output_file = run_mesh_generation(
            atlas_path=args.atlas,
            hierarchy_level=args.level,
            output_dir=args.output_dir,
            output_name=args.output_name,
            colors_file=args.colors,
            min_voxels=args.min_voxels,
            delete_intermediate=args.delete_intermediate,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\nOutput: {output_file}")
            print("Done!")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
