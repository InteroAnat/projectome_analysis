#!/usr/bin/env python3
"""
Create a mindmap/flowchart of the projectome analysis pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure with a clean white background
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')
ax.set_facecolor('white')

# Color scheme
colors = {
    'step1': '#4A90D9',      # Blue
    'step2': '#5CB85C',      # Green  
    'step3': '#F0AD4E',      # Orange
    'data': '#D9534F',       # Red
    'tool': '#6C757D',       # Gray
    'output': '#9B59B6',     # Purple
    'atlas': '#1ABC9C',      # Teal
    'bg': '#F8F9FA',         # Light gray background
}

def draw_box(ax, x, y, width, height, text, color, fontsize=9, alpha=0.9):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=color, edgecolor='black', alpha=alpha, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', wrap=True)
    return box

def draw_data_box(ax, x, y, width, height, text, fontsize=8):
    """Draw a data file box."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor='#FFF3CD', edgecolor='#F0AD4E', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#856404')
    return box

def draw_tool_box(ax, x, y, width, height, text, fontsize=8):
    """Draw a tool/module box."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor='#E2E3E5', edgecolor='#6C757D', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#383D41')
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', lw=1.5):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, linewidth=lw,
                           connectionstyle="arc3,rad=0", mutation_scale=15)
    ax.add_patch(arrow)

def draw_dashed_arrow(ax, x1, y1, x2, y2, color='gray', lw=1):
    """Draw a dashed arrow for dependencies."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls='--'))

# Title
ax.text(10, 13.5, 'Projectome Analysis Pipeline', ha='center', va='center',
        fontsize=20, fontweight='bold', color='#2C3E50')
ax.text(10, 13.0, 'Main Processing Flow & Module Dependencies', ha='center', va='center',
        fontsize=12, color='#7F8C8D')

# ==================== STEP 1: Region Analysis ====================
draw_box(ax, 3.5, 10.5, 5, 1.2, 'STEP 1\nRegion Analysis', colors['step1'], fontsize=11)

# Step 1 inputs
draw_data_box(ax, 1.2, 10.5, 1.8, 0.8, 'Neuron Tables\n(.xlsx/.csv)', fontsize=8)
draw_arrow(ax, 2.2, 10.5, 1.0, 10.5, color=colors['step1'], lw=2)

# Step 1 dependencies (vertical stack on right)
draw_tool_box(ax, 6.5, 12, 2.5, 0.7, 'PopulationRegionAnalysis', fontsize=8)
draw_tool_box(ax, 6.5, 11.2, 2.5, 0.7, 'Cortex/Subcortex\nHierarchy', fontsize=8)
draw_tool_box(ax, 6.5, 10.4, 2.5, 0.7, 'ARM/CHARM/SARM\nAtlas', fontsize=8)
draw_tool_box(ax, 6.5, 9.6, 2.5, 0.7, 'Laterality Analysis', fontsize=8)

draw_dashed_arrow(ax, 6.0, 10.5, 5.0, 11.5)
draw_dashed_arrow(ax, 6.0, 10.5, 5.0, 10.9)
draw_dashed_arrow(ax, 6.0, 10.5, 5.0, 10.3)
draw_dashed_arrow(ax, 6.0, 10.5, 5.0, 9.7)

# Step 1 outputs
draw_data_box(ax, 3.5, 7.8, 4.5, 0.9, 'Output: Region Analysis Results\n(projection_strength, soma_region, terminal_regions)', fontsize=8)
draw_arrow(ax, 3.5, 9.9, 3.5, 8.4, color=colors['step1'], lw=2)

# ==================== STEP 2: FNT Distance Pipeline ====================
draw_box(ax, 10, 10.5, 5, 1.2, 'STEP 2\nFNT Distance Pipeline', colors['step2'], fontsize=11)

# Step 2 inputs
draw_arrow(ax, 6.0, 7.8, 8.5, 10.0, color='gray', lw=1.5)
ax.text(7.0, 8.5, 'neuron_ids', ha='center', va='center', fontsize=8, 
        style='italic', color='#666')

# Step 2 dependencies
draw_tool_box(ax, 14, 12, 2.5, 0.7, 'fnt-from-swc', fontsize=8)
draw_tool_box(ax, 14, 11.2, 2.5, 0.7, 'fnt-decimate', fontsize=8)
draw_tool_box(ax, 14, 10.4, 2.5, 0.7, 'fnt-join', fontsize=8)
draw_tool_box(ax, 14, 9.6, 2.5, 0.7, 'fnt-dist', fontsize=8)
draw_tool_box(ax, 14, 8.8, 2.5, 0.7, 'IONData (SWC fetch)', fontsize=8)

draw_dashed_arrow(ax, 12.5, 10.5, 12.8, 11.7)
draw_dashed_arrow(ax, 12.5, 10.5, 12.8, 10.9)
draw_dashed_arrow(ax, 12.5, 10.5, 12.8, 10.1)
draw_dashed_arrow(ax, 12.5, 10.5, 12.8, 9.3)
draw_dashed_arrow(ax, 12.5, 10.0, 12.8, 8.5)

# Step 2 outputs
draw_data_box(ax, 10, 7.8, 4.5, 0.9, 'Output: FNT Files + Distance Matrix\n(*_joined.fnt, *_dist.txt)', fontsize=8)
draw_arrow(ax, 10, 9.9, 10, 8.4, color=colors['step2'], lw=2)

# ==================== STEP 3: Bulk Visualization ====================
draw_box(ax, 16.5, 10.5, 5, 1.2, 'STEP 3\nBulk Visualization', colors['step3'], fontsize=11)

# Step 3 inputs
draw_arrow(ax, 12.3, 7.8, 14.5, 10.0, color='gray', lw=1.5)
ax.text(13.2, 8.5, 'neuron_tables', ha='center', va='center', fontsize=8,
        style='italic', color='#666')

# Step 3 dependencies
draw_tool_box(ax, 18.5, 12, 2.5, 0.7, 'Visual_toolkit', fontsize=8)
draw_tool_box(ax, 18.5, 11.2, 2.5, 0.7, 'IONData', fontsize=8)
draw_tool_box(ax, 18.5, 10.4, 2.5, 0.7, 'High/Low Res\nBlocks', fontsize=8)

draw_dashed_arrow(ax, 19.0, 10.5, 17.8, 11.7)
draw_dashed_arrow(ax, 19.0, 10.5, 17.8, 10.9)
draw_dashed_arrow(ax, 19.0, 10.5, 17.8, 10.1)

# Step 3 outputs
draw_data_box(ax, 16.5, 7.8, 4.5, 0.9, 'Output: Plots & NIfTI\n(HighRes Soma + LowRes WideField)', fontsize=8)
draw_arrow(ax, 16.5, 9.9, 16.5, 8.4, color=colors['step3'], lw=2)

# ==================== Key Data Files ====================
ax.text(10, 6.5, 'Key Data Files', ha='center', va='center', fontsize=12, 
        fontweight='bold', color='#2C3E50')

# Data files row
data_files = [
    ('251637_INS.xlsx\n251637_ACC.xlsx\n251637_M1.xlsx', 3),
    ('ARM_key_all.txt\nCHARM_key_table_v2.csv\nSARM_key_table_v2.csv', 7.5),
    ('*_joined.fnt\n*_dist.txt', 12),
    ('Cube cache\nresource/cubes/', 16.5),
]
for text, x in data_files:
    draw_data_box(ax, x, 5.2, 3.5, 1.3, text, fontsize=8)

# ==================== Supporting Tools ====================
ax.text(10, 3.8, 'Supporting Analysis Tools', ha='center', va='center', fontsize=12,
        fontweight='bold', color='#2C3E50')

tools = [
    ('FNTCubeVis.py\nFNT 3D Visualization', 2.5, colors['tool']),
    ('fnt_dist_clustering.py\nDistance Analysis', 6, colors['tool']),
    ('brain_mesh_viz.py\nBrain Surface Viz', 9.5, colors['tool']),
    ('neuro_tracer.py\nNeuron Tracing', 13, colors['tool']),
    ('fnt_tools.py\nSWC/FNT Utils', 16.5, colors['tool']),
]

for text, x, color in tools:
    draw_box(ax, x, 2.5, 3.2, 1.0, text, color, fontsize=8, alpha=0.8)

# ==================== Generic Pipeline Variant ====================
ax.text(10, 1.3, 'Generic Pipeline: fnt-dist_pipeline_generic.py (works with any neuron table)',
        ha='center', va='center', fontsize=10, style='italic', color='#E74C3C')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors['step1'], label='Step 1: Region Analysis', alpha=0.9),
    mpatches.Patch(facecolor=colors['step2'], label='Step 2: FNT Processing', alpha=0.9),
    mpatches.Patch(facecolor=colors['step3'], label='Step 3: Visualization', alpha=0.9),
    mpatches.Patch(facecolor='#FFF3CD', edgecolor='#F0AD4E', label='Data Files'),
    mpatches.Patch(facecolor='#E2E3E5', edgecolor='#6C757D', label='Dependencies'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.95)

plt.tight_layout()

# Save the figure
output_path = '/mnt/d/projectome_analysis/figures_charts/processing_pipeline_mindmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
print(f"Saved: {output_path.replace('.png', '.pdf')}")

plt.close()

# Also copy to main_scripts for repo
import shutil
shutil.copy(output_path, '/mnt/d/projectome_analysis/main_scripts/pipeline_mindmap.png')
print("Copied to main_scripts/pipeline_mindmap.png")
