#!/usr/bin/env python3
"""
volume_to_mesh_mz3_MONKEY.py - Monkey Atlas Mesh Generator with Hierarchy Support

Specialized version for monkey atlases (like ARM) that have multiple hierarchy levels.

Usage:
    python volume_to_mesh_mz3_MONKEY.py \
        --input_volume ARM_in_NMT_v2.1_sym.nii.gz \
        --hierarchy_level 6 \
        --output_path ./mesh_outputs/ \
        --out_file ARM_h6.mz3

Author: Modified from Chris Rorden's niiAtlas2mesh.py
"""

import argparse
import glob
import gzip
import io
import nibabel as nib
import numpy as np
import os
import re
import shutil
import struct
import subprocess
import sys


def check_niimath():
    if not shutil.which("niimath"):
        raise RuntimeError("Error: 'niimath' not found in your PATH. Please install or add it.")


def check_file_exists(fname):
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Error: File not found: {fname}")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")


def get_hierarchy_info(nifti_file):
    """Analyze hierarchy structure of a multi-level atlas."""
    img = nib.load(nifti_file)
    data = img.get_fdata()
    
    print(f"\nAtlas shape: {data.shape}")
    
    # Handle different dimensionality
    if len(data.shape) == 5:
        # Shape: (x, y, z, 1, n_hierarchies)
        squeezed = data.squeeze()  # Remove singleton dimension
        n_levels = squeezed.shape[3] if len(squeezed.shape) == 4 else 1
        print(f"Detected 5D atlas with {n_levels} hierarchy levels")
        return n_levels, img
    elif len(data.shape) == 4:
        # Shape: (x, y, z, n_hierarchies) or (x, y, z, 1)
        if data.shape[3] <= 10:  # Likely hierarchy levels
            n_levels = data.shape[3]
            print(f"Detected 4D atlas with {n_levels} hierarchy levels")
            return n_levels, img
        else:
            print("Detected 4D atlas (single volume)")
            return 1, img
    elif len(data.shape) == 3:
        print("Detected 3D atlas (single volume)")
        return 1, img
    else:
        sys.exit(f"Error: Unsupported atlas dimensionality: {data.shape}")


def get_unique_indices_at_level(img, level):
    """Get unique region indices at a specific hierarchy level."""
    data = img.get_fdata()
    
    # Extract data at specific level
    if len(data.shape) == 5:
        level_data = data[:, :, :, 0, level-1]
    elif len(data.shape) == 4:
        if data.shape[3] > 1:
            level_data = data[:, :, :, level-1]
        else:
            level_data = data[:, :, :, 0]
    else:
        level_data = data
    
    unique = np.unique(level_data)
    # Return only non-zero indices (exclude background)
    indices = sorted([int(x) for x in unique if x > 0])
    return indices, level_data


def filter_non_empty_regions(img, indices, level, min_voxels=10):
    """
    Filter regions to only include those with actual voxel data.
    
    Args:
        img: NIfTI image object
        indices: List of region indices to check
        level: Hierarchy level
        min_voxels: Minimum voxel count to consider region valid
    
    Returns:
        List of (index, voxel_count) tuples for non-empty regions
    """
    data = img.get_fdata()
    
    # Extract data at specific level
    if len(data.shape) == 5:
        level_data = data[:, :, :, 0, level-1]
    elif len(data.shape) == 4:
        if data.shape[3] > 1:
            level_data = data[:, :, :, level-1]
        else:
            level_data = data[:, :, :, 0]
    else:
        level_data = data
    
    valid_regions = []
    print(f"\nChecking {len(indices)} regions for voxel content...")
    
    for idx in indices:
        voxel_count = np.sum(level_data == idx)
        if voxel_count >= min_voxels:
            valid_regions.append((idx, voxel_count))
    
    print(f"Found {len(valid_regions)} non-empty regions (min {min_voxels} voxels)")
    return valid_regions


def run_niimath(nifti_file, idx, outpath=None, level=0, temp_3d_file=None):
    """
    Run niimath to convert a single region to mesh.
    
    Args:
        nifti_file: Original nifti file (for naming)
        idx: Region index to extract
        outpath: Output directory
        level: Hierarchy level (for naming)
        temp_3d_file: Path to temporary 3D volume at specific hierarchy level
    """
    def strip_extensions(filename):
        """Remove all extensions (.nii.gz -> base name)."""
        name = filename
        while '.' in name:
            name, ext = os.path.splitext(name)
            if not ext:
                break
        return name
    
    outname = (
        f"{strip_extensions(nifti_file)}_{idx}"
        if outpath is None
        else os.path.join(
            outpath,
            f"{strip_extensions(os.path.basename(nifti_file))}_L{level}_{idx}"
        )
    )
    
    # Use the 3D temp file if provided, otherwise fall back to original (for 3D atlases)
    input_file = temp_3d_file if temp_3d_file else nifti_file
    
    cmd = [
        "niimath", input_file,
        "-thr", str(idx),
        "-uthr", str(idx),
        "-bin",
        "-s", "1.2",           # one smoothing step only
        "-mesh",
        "-i", "0.5",
        "-r", "0.5",
        "-q", "b",             # quality: binary mesh output
        outname
    ]
    
    print(f"  Processing region {idx}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    Warning: Region {idx} mesh generation failed")
        if result.stderr:
            print(f"    Error: {result.stderr[:200]}")
        return result.returncode
    else:
        print(f"    ✓ Saved")
        return 0


def read_mz3(filename):
    with open(filename, 'rb') as f:
        magic = f.read(2)
    if magic == b'\x1f\x8b':
        with gzip.open(filename, 'rb') as gz:
            raw = gz.read()
        f = io.BytesIO(raw)
    else:
        f = open(filename, 'rb')

    with f:
        hdr = f.read(16)
        magic, attr, nface, nvert, nskip = struct.unpack('<HHIII', hdr)
        if magic != 23117:
            raise ValueError(f"{filename} is not a valid MZ3 file.")
        f.read(nskip)
        faces = np.frombuffer(f.read(nface * 12), dtype=np.int32).reshape((-1, 3)) if attr & 1 else []
        verts = np.frombuffer(f.read(nvert * 12), dtype=np.float32).reshape((-1, 3)) if attr & 2 else []
        f.read(nvert * 4) if attr & 4 else None
        f.read(nvert * 4) if attr & 8 else None
        return faces, verts


def write_mz3(filename, faces, verts, rgba, scalars):
    attr = 1 | 2 | 4 | 8
    nface = len(faces)
    nvert = len(verts)
    with gzip.open(filename, 'wb') as f:
        f.write(struct.pack('<HHIII', 0x5A4D, attr, nface, nvert, 0))
        f.write(faces.astype(np.int32).tobytes())
        f.write(verts.astype(np.float32).tobytes())
        f.write(rgba.astype(np.uint8).tobytes())
        f.write(scalars.astype(np.float32).tobytes())


def load_colors(color_file):
    color_map = {}
    with open(color_file, 'r') as f:
        for idx, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            r, g, b = map(int, parts)
            color_map[idx] = np.array([r, g, b, 255], dtype=np.uint8)
    return color_map


def combine_mz3(mesh_glob='*.mz3', input_path='.',
                color_file='colors.txt', 
                out_file='combined_atlas.mz3', 
                delete_mz3=False):
    """Combine individual mesh files into a single colored atlas."""
    if input_path == '.':
        input_path = os.getcwd()

    print(f"Using input path: {input_path}")
    print(f"Looking for files matching: {mesh_glob}")

    # Find mesh files - exclude intermediate files with _L{n}_ pattern
    all_files = glob.glob(f'{input_path}/{mesh_glob}')
    mesh_files = [f for f in sorted(all_files)
                  if re.match(r'^(.+?)_(\d+)\.mz3$', os.path.basename(f)) 
                  and '_L' not in os.path.basename(f)]
    
    if not mesh_files:
        print(f"Found {len(all_files)} total .mz3 files")
        print(f"Filtered to {len(mesh_files)} valid files")
        raise FileNotFoundError("No valid *_n.mz3 files found (excluding intermediate _L files).")

    print(f"Found {len(mesh_files)} mesh files")

    # Extract prefix and index
    parsed = [re.match(r'^(.+?)_(\d+)\.mz3$', os.path.basename(f)) for f in mesh_files]
    prefixes = set(m.group(1) for m in parsed)
    if len(prefixes) != 1:
        raise ValueError(f"Multiple filename prefixes found: {sorted(prefixes)}")

    prefix = prefixes.pop()
    indices = [int(m.group(2)) for m in parsed]

    color_map = load_colors(color_file)
    n_colors = len(color_map)
    
    print(f"Color file has {n_colors} colors")
    print(f"Processing {len(indices)} regions - will cycle through colors if needed")

    all_faces = []
    all_verts = []
    all_rgba = []
    all_scalar = []
    vert_offset = 0

    # Sort by region index for consistent ordering
    sorted_pairs = sorted(zip(indices, mesh_files))
    
    for seq_num, (region_id, f) in enumerate(sorted_pairs, 1):
        # Assign color sequentially (cycle through available colors)
        color_idx = ((seq_num - 1) % n_colors) + 1
        color = color_map[color_idx]
        
        faces, verts = read_mz3(f)
        nvert = verts.shape[0]
        all_faces.append(faces + vert_offset)
        all_verts.append(verts)
        all_rgba.append(np.tile(color, (nvert, 1)))
        all_scalar.append(np.full(nvert, region_id, dtype=np.float32))
        vert_offset += nvert
        
        if seq_num <= 5 or seq_num == len(sorted_pairs):
            print(f"  Region {region_id} -> Color {color_idx} ({color[:3]})")

    write_mz3(f'{input_path}/{out_file}',
              np.vstack(all_faces),
              np.vstack(all_verts),
              np.vstack(all_rgba),
              np.concatenate(all_scalar))
    print(f"Combined mesh saved as: {out_file}")

    if delete_mz3:
        for f in mesh_files:
            os.remove(f)
        print("Deleted intermediate mesh files.")

def volume_to_mesh_mz3_monkey_atlas(
    input_volume,
    hierarchy_level=6,
    output_path=None,
    out_file='monkey_atlas.mz3',
    colors='colors.txt',
    delete_mz3=False,
    min_voxels=100,
):
    """
    Convert a (potentially multi-hierarchy) monkey atlas NIfTI volume into a combined MZ3 mesh.

    This wraps the script's original CLI workflow so it can be called from Python code.
    """
    # Setup
    check_niimath()
    check_file_exists(input_volume)

    outpath = output_path
    combine_input_path = outpath if outpath is not None else '.'

    if outpath:
        ensure_dir(outpath)

    # Analyze hierarchy structure
    n_levels, img = get_hierarchy_info(input_volume)

    # Validate hierarchy level
    if hierarchy_level < 0 or hierarchy_level >= n_levels:
        raise ValueError(
            f"Error: Invalid hierarchy level {hierarchy_level}. Atlas has levels 0-{n_levels-1}"
        )

    print(f"\nUsing hierarchy level {hierarchy_level}")

    # For multi-level atlases, extract 3D volume at specific hierarchy level
    # Niimath requires a 3D input, not 4D/5D
    temp_3d_file = None
    if n_levels > 1:
        print(f"\nExtracting 3D volume at hierarchy level {hierarchy_level}...")
        data = img.get_fdata()

        # Extract the specific level
        if len(data.shape) == 5:
            level_data_3d = data[:, :, :, 0, hierarchy_level - 1]
        elif len(data.shape) == 4:
            level_data_3d = data[:, :, :, hierarchy_level - 1]
        else:
            level_data_3d = data

        # Save as temporary NIfTI
        temp_3d_file = os.path.join(
            outpath if outpath else '.',
            f"temp_level_{hierarchy_level}.nii.gz",
        )
        level_img = nib.Nifti1Image(level_data_3d.astype(np.uint16), img.affine, img.header)
        nib.save(level_img, temp_3d_file)
        print(f"  Saved temporary 3D volume: {temp_3d_file}")

    # Get regions at this level
    existing_indices, _level_data = get_unique_indices_at_level(img, hierarchy_level)

    if not existing_indices:
        raise RuntimeError(f"Error: No labeled regions found at level {hierarchy_level}")

    print(f"Found {len(existing_indices)} labeled regions at level {hierarchy_level}")
    print(f"Index range: {existing_indices[0]} - {existing_indices[-1]}")

    if len(existing_indices) <= 20:
        print(f"All indices: {existing_indices}")
    else:
        print(f"Sample indices: {existing_indices[:10]} ... {existing_indices[-10:]}")

    # Filter to only non-empty regions (saves processing time)
    valid_regions = filter_non_empty_regions(
        img,
        existing_indices,
        hierarchy_level,
        min_voxels=min_voxels,
    )

    if not valid_regions:
        raise RuntimeError("Error: No valid regions found with sufficient voxels")

    # Sort by voxel count (process largest first for better progress visibility)
    valid_regions.sort(key=lambda x: x[1], reverse=True)

    print(f"\nProcessing {len(valid_regions)} regions with sufficient voxels...")
    print(f"Largest region: {valid_regions[0][0]} ({valid_regions[0][1]} voxels)")
    print(f"Smallest region: {valid_regions[-1][0]} ({valid_regions[-1][1]} voxels)")
    print()

    failed_indices = []
    success_indices = []  # Track successful region indices for combining

    # Get base name for files (strip all extensions like .nii.gz)
    def strip_extensions(filename):
        name = filename
        while '.' in name:
            name, ext = os.path.splitext(name)
            if not ext:
                break
        return name

    base_name = strip_extensions(os.path.basename(input_volume))

    for idx, voxel_count in valid_regions:
        result = run_niimath(input_volume, idx, outpath, hierarchy_level, temp_3d_file)
        if result != 0:
            failed_indices.append((idx, voxel_count))
            continue

        success_indices.append(idx)

        # When output_path is provided, run_niimath includes the hierarchy level in filenames.
        # combine_mz3 expects {base_name}_{idx}.mz3, so rename those intermediate files.
        if outpath is not None:
            try:
                old_name = os.path.join(outpath, f"{base_name}_L{hierarchy_level}_{idx}.mz3")
                new_name = os.path.join(outpath, f"{base_name}_{idx}.mz3")
                if os.path.exists(old_name):
                    if os.path.exists(new_name):
                        os.remove(new_name)
                    os.rename(old_name, new_name)
            except Exception as e:
                print(f"    Note: Rename issue (ok if already correct): {e}")

    # Summary
    total_processed = len(valid_regions) - len(failed_indices)
    print(f"\n{'='*60}")
    print(f"Summary: {total_processed}/{len(valid_regions)} regions converted")
    print(f"  - Skipped (empty): {len(existing_indices) - len(valid_regions)}")
    print(f"  - Failed (niimath): {len(failed_indices)}")
    if failed_indices:
        failed_ids = [str(idx) for idx, _ in failed_indices[:20]]
        print(f"  - Failed IDs: {', '.join(failed_ids)}")
    print(f"{'='*60}\n")

    # Cleanup temp file
    if temp_3d_file and os.path.exists(temp_3d_file):
        os.remove(temp_3d_file)
        print(f"Cleaned up temporary file: {temp_3d_file}")

    # Combine meshes
    if total_processed > 0:
        print("Combining into single mesh...")
        mesh_pattern = f"{base_name}_*.mz3"
        combine_mz3(
            mesh_glob=mesh_pattern,
            color_file=colors,
            input_path=combine_input_path,
            out_file=out_file,
            delete_mz3=delete_mz3,
        )
    else:
        raise RuntimeError("Error: No regions were successfully processed")

    # Return useful info to callers
    output_dir = os.getcwd() if combine_input_path == '.' else combine_input_path
    out_path_full = os.path.abspath(os.path.join(output_dir, out_file))
    return {
        'out_file': out_path_full,
        'total_processed': total_processed,
        'failed_indices': failed_indices,
        'success_indices': success_indices,
        'hierarchy_level': hierarchy_level,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert monkey atlas with hierarchies to 3D mesh'
    )
    parser.add_argument('--input_volume', type=str, required=True,
                        help='Input volumetric segmentation (NIfTI)')
    parser.add_argument('--hierarchy_level', type=int, default=6,
                        help='Which hierarchy level to use (1-6 for ARM, default: 6)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output directory for meshes')
    parser.add_argument('--out_file', type=str, default='monkey_atlas.mz3',
                        help='Name for combined output file')
    parser.add_argument('--colors', type=str, default='colors.txt',
                        help='Path to color file')
    parser.add_argument('--delete_mz3', action='store_true',
                        help='Delete individual mesh files after combining')
    parser.add_argument('--min-voxels', type=int, default=100,
                        help='Minimum voxel count for region to be processed (default: 100)')

    args = parser.parse_args()

    try:
        volume_to_mesh_mz3_monkey_atlas(
            input_volume=args.input_volume,
            hierarchy_level=args.hierarchy_level,
            output_path=args.output_path,
            out_file=args.out_file,
            colors=args.colors,
            delete_mz3=args.delete_mz3,
            min_voxels=args.min_voxels,
        )
    except Exception as e:
        sys.exit(str(e))
