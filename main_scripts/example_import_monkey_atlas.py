#!/usr/bin/env python3
"""
Example: How to import and use volume_to_mesh_mz3_MONKEY from main_scripts

This demonstrates multiple ways to import and use the monkey atlas mesh generator.

Quick Reference:
---------------

1. Direct import (add script dir to path):
   sys.path.insert(0, r'..\subcortex_visualization\monkey_atlas_guide')
   from volume_to_mesh_mz3_MONKEY import volume_to_mesh_mz3_monkey_atlas

2. Package import (add project root to path):
   sys.path.insert(0, PROJECT_ROOT)
   from subcortex_visualization.monkey_atlas_guide import volume_to_mesh_mz3_monkey_atlas

3. Import specific functions:
   from subcortex_visualization.monkey_atlas_guide.volume_to_mesh_mz3_MONKEY import (
       volume_to_mesh_mz3_monkey_atlas,
       generate_distinct_colors,
       load_colors,
       combine_mz3,
       main
   )
"""

import sys
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def print_import_examples():
    """Print import code examples"""
    print("\n" + "="*60)
    print("Import Code Examples (Copy these to your script)")
    print("="*60)
    
    print(f"""
# ============================================================================
# Method 1: Direct import (add script directory to sys.path)
# ============================================================================

import sys
import os

# Add the monkey_atlas_guide directory to path
monkey_atlas_path = os.path.join(r'{PROJECT_ROOT}', 
                                 'subcortex_visualization', 'monkey_atlas_guide')
if monkey_atlas_path not in sys.path:
    sys.path.insert(0, monkey_atlas_path)

# Now import directly
from volume_to_mesh_mz3_MONKEY import (
    volume_to_mesh_mz3_monkey_atlas,
    generate_distinct_colors,
    load_colors,
    combine_mz3,
    main
)

# Use the functions
result = volume_to_mesh_mz3_monkey_atlas(
    input_volume="ARM_in_NMT_v2.1_sym.nii.gz",
    hierarchy_level=6,
    output_path="./output",
    out_file="ARM_h6.mz3",
    colors=None,  # Auto-generate colors
    delete_mz3=False,
    min_voxels=100
)


# ============================================================================
# Method 2: Package import (add project root to sys.path) - RECOMMENDED
# ============================================================================

import sys

PROJECT_ROOT = r'{PROJECT_ROOT}'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import using package notation
from subcortex_visualization.monkey_atlas_guide.volume_to_mesh_mz3_MONKEY import (
    volume_to_mesh_mz3_monkey_atlas,
    generate_distinct_colors,
    load_colors,
    combine_mz3,
    main
)


# ============================================================================
# Method 3: Import from package __init__ (cleanest)
# ============================================================================

import sys

PROJECT_ROOT = r'{PROJECT_ROOT}'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from subcortex_visualization.monkey_atlas_guide import (
    volume_to_mesh_mz3_monkey_atlas,
    generate_distinct_colors,
)


# ============================================================================
# Method 4: Run as subprocess (CLI mode)
# ============================================================================

import subprocess
import os

script_path = os.path.join(r'{PROJECT_ROOT}', 
    'subcortex_visualization', 'monkey_atlas_guide', 
    'volume_to_mesh_mz3_MONKEY.py')

result = subprocess.run([
    'python', script_path,
    '--input_volume', 'ARM_in_NMT_v2.1_sym.nii.gz',
    '--hierarchy_level', '6',
    '--out_file', 'ARM_h6.mz3'
], capture_output=True, text=True)

print(result.stdout)
""")


def method1_direct_import():
    """Add the script directory directly to sys.path"""
    
    # Add the monkey_atlas_guide directory to path
    monkey_atlas_path = os.path.join(PROJECT_ROOT, 'subcortex_visualization', 'monkey_atlas_guide')
    if monkey_atlas_path not in sys.path:
        sys.path.insert(0, monkey_atlas_path)
    
    # Now import directly
    from volume_to_mesh_mz3_MONKEY import (
        volume_to_mesh_mz3_monkey_atlas,
        generate_distinct_colors,
        load_colors,
        main
    )
    
    print("Method 1: Direct import successful!")
    print(f"  Functions available: {volume_to_mesh_mz3_monkey_atlas.__name__}")
    
    # Example: Generate some colors
    colors = generate_distinct_colors(n_colors=10, seed=42)
    print(f"  Generated {len(colors)} colors")
    
    return volume_to_mesh_mz3_monkey_atlas


def method2_package_import():
    """Import using the package structure"""
    
    # Add project root to path
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # Import using package notation
    from subcortex_visualization.monkey_atlas_guide.volume_to_mesh_mz3_MONKEY import (
        volume_to_mesh_mz3_monkey_atlas,
        generate_distinct_colors,
        load_colors,
        combine_mz3,
        main
    )
    
    print("\nMethod 2: Package import successful!")
    print(f"  All functions imported: {[volume_to_mesh_mz3_monkey_atlas, generate_distinct_colors, load_colors, combine_mz3, main]}")
    
    return volume_to_mesh_mz3_monkey_atlas


def method3_init_import():
    """Import from package __init__.py"""
    
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # Import from package init
    from subcortex_visualization.monkey_atlas_guide import (
        volume_to_mesh_mz3_monkey_atlas,
        generate_distinct_colors,
    )
    
    print("\nMethod 3: Package init import successful!")
    
    return volume_to_mesh_mz3_monkey_atlas


def method4_subprocess():
    """Run the script as a subprocess"""
    import subprocess
    
    script_path = os.path.join(PROJECT_ROOT, 'subcortex_visualization', 'monkey_atlas_guide', 
                               'volume_to_mesh_mz3_MONKEY.py')
    
    # Example: Show help
    result = subprocess.run(
        ['python', script_path, '--help'],
        capture_output=True,
        text=True
    )
    
    print("\nMethod 4: Subprocess (CLI mode)")
    print("  Help output:")
    print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)


def example_usage():
    """Example of how to use the imported functions"""
    
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    from subcortex_visualization.monkey_atlas_guide.volume_to_mesh_mz3_MONKEY import (
        volume_to_mesh_mz3_monkey_atlas,
        generate_distinct_colors
    )
    
    print("\n" + "="*60)
    print("Example Usage")
    print("="*60)
    
    # Example 1: Generate colors programmatically
    print("\n1. Generate distinct colors:")
    colors = generate_distinct_colors(n_colors=5, seed=42)
    for i, (idx, color) in enumerate(colors.items(), 1):
        print(f"   Color {idx}: RGB({color[0]}, {color[1]}, {color[2]})")
    
    # Example 2: Call the main function (commented out as it requires actual data)
    print("\n2. Convert atlas to mesh (example call):")
    print("""
    result = volume_to_mesh_mz3_monkey_atlas(
        input_volume="path/to/ARM_in_NMT_v2.1_sym.nii.gz",
        hierarchy_level=6,
        output_path="./output",
        out_file="ARM_h6.mz3",
        colors=None,  # Auto-generate colors
        delete_mz3=False,
        min_voxels=100
    )
    print(f"Output file: {result['out_file']}")
    print(f"Total regions processed: {result['total_processed']}")
    """)


if __name__ == "__main__":
    print("Monkey Atlas Mesh Generator - Import Examples")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print()
    
    # Check if we have required dependencies
    try:
        import nibabel
        HAS_NIBABEL = True
    except ImportError:
        HAS_NIBABEL = False
        print("Note: nibabel not installed. Showing import code examples only.")
        print("Install with: pip install nibabel numpy\n")
    
    if HAS_NIBABEL:
        # Try all import methods
        try:
            fn1 = method1_direct_import()
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        try:
            fn2 = method2_package_import()
        except Exception as e:
            print(f"Method 2 failed: {e}")
        
        try:
            fn3 = method3_init_import()
        except Exception as e:
            print(f"Method 3 failed: {e}")
        
        try:
            method4_subprocess()
        except Exception as e:
            print(f"Method 4 failed: {e}")
        
        # Show usage examples
        example_usage()
    else:
        # Just print the import examples
        print_import_examples()
