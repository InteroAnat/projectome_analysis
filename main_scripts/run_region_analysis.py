"""
run_region_analysis.py - Example workflow
"""

import pandas as pd
from region_analysis import (
    load_hierarchy, 
    add_hierarchy_levels,
    reorder_columns,
    hierarchy_summary,
    plot_level_comparison,
    plot_stacked_by_type,
    plot_stacked_by_cluster,
    export_results
)

# ==========================================
# CONFIGURATION
# ==========================================
NEURON_TABLE = r'neuron_tables\INS_df_v3_clustered.xlsx'
ARM_KEY_TABLE = r'atlas\arm_key.xlsx'
OUTPUT_PREFIX = 'INS_region'

# ==========================================
# LOAD DATA
# ==========================================
print("Loading data...")

# Load neuron table (with your existing columns)
neuron_df = pd.read_excel(NEURON_TABLE)
print(f"Loaded {len(neuron_df)} neurons")
print(f"Columns: {list(neuron_df.columns)}")

# Load hierarchy
hierarchy = load_hierarchy(ARM_KEY_TABLE)

# ==========================================
# ADD HIERARCHY LEVELS
# ==========================================
# This adds Region_L1, Region_L2, ..., Region_L6 columns
neuron_df = add_hierarchy_levels(
    neuron_df, 
    hierarchy,
    region_col='Region',  # Your native region column
    max_level=6
)

# Reorder columns nicely
neuron_df = reorder_columns(neuron_df)

print(f"\nUpdated columns: {list(neuron_df.columns)}")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "=" * 50)
print("  HIERARCHY SUMMARY")
print("=" * 50)

summary_df = hierarchy_summary(neuron_df)
print(summary_df.to_string(index=False))

# ==========================================
# VISUALIZATIONS
# ==========================================
print("\nGenerating visualizations...")

# 1. Compare levels 2, 3, 4
plot_level_comparison(
    neuron_df, 
    levels=[2, 3, 4],
    save_path=f'{OUTPUT_PREFIX}_level_comparison.png'
)

# 2. By neuron type at level 3
if 'Neuron_Type' in neuron_df.columns:
    plot_stacked_by_type(
        neuron_df,
        level=3,
        save_path=f'{OUTPUT_PREFIX}_by_type_L3.png'
    )

# 3. By morph cluster at level 3
if 'Morph_Cluster' in neuron_df.columns:
    plot_stacked_by_cluster(
        neuron_df,
        level=3,
        save_path=f'{OUTPUT_PREFIX}_by_cluster_L3.png'
    )

# ==========================================
# EXPORT
# ==========================================
export_results(
    neuron_df,
    hierarchy,
    output_prefix=OUTPUT_PREFIX
)

# Save updated neuron table
neuron_df.to_excel(f'{OUTPUT_PREFIX}_neurons_full.xlsx', index=False)
print(f"\nSaved: {OUTPUT_PREFIX}_neurons_full.xlsx")

print("\nDone!")
