"""
Usage Examples for fnt_pipeline.py

This file demonstrates how to use the FNTNeuronPipeline class.
"""

from fnt_pipeline import FNTNeuronPipeline, run_pipeline

# ==============================================================================
# Example 1: Simple usage with defaults
# ==============================================================================
print("="*70)
print("Example 1: Simple Usage")
print("="*70)

pipeline = FNTNeuronPipeline('neuron_tables/251637_INS.xlsx')
# results = pipeline.run()  # Uncomment to run

# ==============================================================================
# Example 2: Custom sample_id
# ==============================================================================
print("\n" + "="*70)
print("Example 2: Custom Sample ID")
print("="*70)

pipeline = FNTNeuronPipeline(
    table_path='neuron_tables/251637_INS.xlsx',
    sample_id='123456'
)
# results = pipeline.run()  # Uncomment to run

# ==============================================================================
# Example 3: Custom neuron ID column
# ==============================================================================
print("\n" + "="*70)
print("Example 3: Custom Neuron ID Column")
print("="*70)

pipeline = FNTNeuronPipeline(
    table_path='neuron_tables/my_neurons.csv',
    sample_id='251637',
    neuron_id_column='CellID'  # If your column is named differently
)
# results = pipeline.run()  # Uncomment to run

# ==============================================================================
# Example 4: Using the convenience function
# ==============================================================================
print("\n" + "="*70)
print("Example 4: Convenience Function")
print("="*70)

# results = run_pipeline(
#     table_path='neuron_tables/251637_INS.xlsx',
#     sample_id='251637'
# )

# ==============================================================================
# Example 5: Step-by-step processing (for debugging)
# ==============================================================================
print("\n" + "="*70)
print("Example 5: Step-by-Step Processing")
print("="*70)

pipeline = FNTNeuronPipeline('neuron_tables/251637_INS.xlsx')

# Load table manually
# df = pipeline._load_neuron_table()
# print(f"Loaded {len(pipeline.neuron_ids)} neurons")

# Process single neuron
# success = pipeline.process_single_neuron('001.swc')

# Process all neurons
# success_count, failed = pipeline.process_all()

# Join FNT files
# joined_file = pipeline.join_fnt_files()

# Compute distances
# dist_file = pipeline.compute_distances()

# ==============================================================================
# Example 6: Accessing results
# ==============================================================================
print("\n" + "="*70)
print("Example 6: Accessing Results")
print("="*70)

# results = pipeline.run()
# print(f"Total: {results['total']}")
# print(f"Success: {results['success']}")
# print(f"Failed: {results['failed']}")
# print(f"Joined file: {results['joined_file']}")
# print(f"Distance file: {results['dist_file']}")
# print(f"Output directory: {results['output_dir']}")

# ==============================================================================
# Expected Output Structure
# ==============================================================================
print("\n" + "="*70)
print("Expected Output Structure")
print("="*70)

print("""
processed_neurons/{sample_id}/
├── raw_swcs/                     # Downloaded SWCs
└── fnt_processed/
    └── {batch_name}/
        ├── 001.swc.processed.swc
        ├── 001.swc.fnt
        ├── 001.swc.decimate.fnt
        ├── ... (individual files for each neuron)
        ├── {batch_name}_joined.fnt      # Single joined file
        └── {batch_name}_dist.txt        # Distance matrix
""")

print("\n" + "="*70)
print("Examples loaded successfully!")
print("Uncomment the relevant lines to run the pipeline.")
print("="*70)
