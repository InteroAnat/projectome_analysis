#!/usr/bin/env python3
"""
Simple wrapper to import monkey atlas mesh generator from main_scripts.

Usage:
    from monkey_atlas_mesh import volume_to_mesh_mz3_monkey_atlas, generate_distinct_colors
    
    result = volume_to_mesh_mz3_monkey_atlas(
        input_volume="ARM_in_NMT_v2.1_sym.nii.gz",
        hierarchy_level=6,
        out_file="output.mz3"
    )
"""

import sys
import os

# Add monkey_atlas_guide to path
_atlas_path = os.path.join(os.path.dirname(__file__), 
                           '..', 'subcortex_visualization', 'monkey_atlas_guide')
if _atlas_path not in sys.path:
    sys.path.insert(0, _atlas_path)

# Import all public functions
from volume_to_mesh_mz3_MONKEY import (
    volume_to_mesh_mz3_monkey_atlas,
    generate_distinct_colors,
    load_colors,
    combine_mz3,
    main,
)

__all__ = [
    'volume_to_mesh_mz3_monkey_atlas',
    'generate_distinct_colors',
    'load_colors',
    'combine_mz3',
    'main',
]
