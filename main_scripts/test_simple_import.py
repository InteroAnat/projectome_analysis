#!/usr/bin/env python3
"""Test simple import of monkey atlas mesh generator."""

# Simple import - just one line!
from monkey_atlas_mesh import volume_to_mesh_mz3_monkey_atlas, generate_distinct_colors

print("Import successful!")
print(f"Functions: volume_to_mesh_mz3_monkey_atlas, generate_distinct_colors")

# Test color generation
colors = generate_distinct_colors(n_colors=5, seed=42)
print(f"\nGenerated {len(colors)} colors:")
for i, (idx, color) in enumerate(colors.items(), 1):
    print(f"  Color {idx}: RGB({color[0]}, {color[1]}, {color[2]})")
