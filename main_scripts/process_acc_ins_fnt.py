#%%
"""
process_acc_ins_fnt.py - FNT Pipeline for ACC and INS Neurons

This script processes ACC (cingulate) and INS (insula) neurons through the FNT pipeline:
1. Load neuron lists from ACC_df_v2.xlsx and INS_df_v2.xlsx
2. Convert SWC to FNT using fnt-from-swc
3. Decimate FNT files using fnt-decimate
4. Update FNT neuron names to match filenames
5. Join decimated FNT files into combined files for ACC and INS
6. Compute FNT distance matrix using fnt-dist

Requirements:
- FNT tools (fnt-from-swc, fnt-decimate, fnt-join, fnt-dist) in PATH
- ACC_df_v2.xlsx and INS_df_v2.xlsx in working directory
- Raw SWC files in processed_neurons/251637/

Author: Assistant
Date: 2026-02-26
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
import glob
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
SAMPLE_ID = '251637'
SWC_DIR = f'processed_neurons/{SAMPLE_ID}'
OUTPUT_DIR = f'processed_neurons/{SAMPLE_ID}/fnt_processed'
ACC_DF_FILE = 'ACC_df_v2.xlsx'
INS_DF_FILE = 'INS_df_v2.xlsx'

# FNT tool parameters
DECIMATE_D = 5000  # Distance parameter for decimation
DECIMATE_A = 5000  # Angle parameter for decimation

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def execute_command(command, verbose=True):
    """Execute a shell command and capture output."""
    if verbose:
        print(f"  Executing: {command}")
    
    process = subprocess.Popen(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        shell=True
    )
    output, error = process.communicate()
    output = output.decode('utf-8').strip()
    error = error.decode('utf-8').strip()
    
    if output and verbose:
        print(f"    Output: {output[:200]}..." if len(output) > 200 else f"    Output: {output}")
    
    if error and verbose:
        print(f"    Error: {error[:200]}..." if len(error) > 200 else f"    Error: {error}")
    
    return process.returncode == 0


# def check_fnt_tools():
#     """Check if required FNT tools are available."""
#     tools = ['fnt-from-swc', 'fnt-decimate', 'fnt-join', 'fnt-dist']
#     missing = []
    
#     for tool in tools:
#         result = subprocess.run(['which', tool], capture_output=True)
#         if result.returncode != 0:
#             # Try with .exe extension (Windows)
#             result = subprocess.run(['which', f'{tool}.exe'], capture_output=True)
#             if result.returncode != 0:
#                 missing.append(tool)
    
#     if missing:
#         print("WARNING: Missing FNT tools:")
#         for tool in missing:
#             print(f"  - {tool}")
#         print("\nPlease ensure FNT tools are installed and in PATH.")
#         print("FNT tools should be available from the neuron-vis package.")
#         return False
    
#     print("✓ All FNT tools found in PATH")
#     return True


def load_neuron_dataframes():
    """Load ACC and INS neuron dataframes."""
    print("\n" + "="*70)
    print("LOADING NEURON DATAFRAMES")
    print("="*70)
    
    data = {}
    
    for name, file_path in [('ACC', ACC_DF_FILE), ('INS', INS_DF_FILE)]:
        if not os.path.exists(file_path):
            print(f"✗ {file_path} not found!")
            continue
        
        df = pd.read_excel(file_path,index_col=0)
        
        # Parse Terminal_Regions to ensure it's a list
        if 'Terminal_Regions' in df.columns:
            import ast
            def parse_regions(x):
                if isinstance(x, (list, tuple)):
                    return list(x)
                if isinstance(x, str):
                    try:
                        if x.startswith('[') and x.endswith(']'):
                            return ast.literal_eval(x)
                    except:
                        pass
                return []
            df['Terminal_Regions'] = df['Terminal_Regions'].apply(parse_regions)
        
        neuron_ids = df['NeuronID'].tolist()
        print(f"✓ {name}: Loaded {len(neuron_ids)} neurons")
        
        data[name] = {
            'df': df,
            'neuron_ids': neuron_ids
        }
    
    return data


# ==============================================================================
# FNT PROCESSING FUNCTIONS
# ==============================================================================

def swc_to_fnt(swc_file, output_fnt=None):
    """Convert SWC file to FNT format."""
    if output_fnt is None:
        output_fnt = f"{swc_file}.fnt"
    
    # Check if already exists
    if os.path.exists(output_fnt):
        print(f"    FNT already exists: {os.path.basename(output_fnt)}")
        return True
    
    command = f'fnt-from-swc "{swc_file}" "{output_fnt}"'
    return execute_command(command)


def decimate_fnt(fnt_file, output_decimate=None):
    """Decimate FNT file."""
    if output_decimate is None:
        output_decimate = f"{fnt_file}.decimate.fnt"
    
    # Check if already exists
    if os.path.exists(output_decimate):
        print(f"    Decimated FNT already exists: {os.path.basename(output_decimate)}")
        return True
    
    command = f'fnt-decimate -d {DECIMATE_D} -a {DECIMATE_A} "{fnt_file}" "{output_decimate}"'
    return execute_command(command)


def update_fnt_neuron_name(fnt_file, neuron_name):
    """Update the last line of FNT file to match the neuron name.
    
    Converts the file to LF line endings (Unix style) which is required
    by fnt-join.exe. Also ensures the file ends with a proper newline.
    """
    try:
        # Read file in binary mode to detect line endings
        with open(fnt_file, 'rb') as f:
            content = f.read()
        
        if not content:
            return False
        
        # Decode content to string, normalizing line endings to LF
        text = content.decode('utf-8').replace('\r\n', '\n')
        lines = text.splitlines()
        
        if not lines:
            return False
        
        # Update last line to "0 Neuron <neuron_name>"
        new_last_line = f"0 Neuron {neuron_name}"
        
        # Always rewrite the file to ensure LF line endings and proper ending
        if lines[-1].strip() != new_last_line.strip():
            lines[-1] = new_last_line
        
        # Write back with LF line endings (required by fnt-join.exe)
        with open(fnt_file, 'w', encoding='utf-8', newline='\n') as f:
            for line in lines:
                f.write(line + '\n')
        
        return True
    except Exception as e:
        print(f"    Error updating FNT name: {e}")
        return False


def process_single_neuron(neuron_id, swc_dir, output_dir):
    """Process a single neuron through the FNT pipeline."""
    swc_file = os.path.join(swc_dir, neuron_id)
    
    if not os.path.exists(swc_file):
        print(f"  ✗ SWC file not found: {swc_file}")
        return False
    
    # Step 1: Convert SWC to FNT
    fnt_file = os.path.join(output_dir, f"{neuron_id}.fnt")
    if not swc_to_fnt(swc_file, fnt_file):
        print(f"  ✗ Failed to convert SWC to FNT: {neuron_id}")
        return False
    
    # Step 2: Decimate FNT
    decimate_file = os.path.join(output_dir, f"{neuron_id}.decimate.fnt")
    if not decimate_fnt(fnt_file, decimate_file):
        print(f"  ✗ Failed to decimate FNT: {neuron_id}")
        return False
    
    # Step 3: Update neuron name in decimated FNT
    neuron_name = neuron_id.replace('.swc', '')
    if not update_fnt_neuron_name(decimate_file, neuron_name):
        print(f"  ✗ Failed to update FNT name: {neuron_id}")
        return False
    
    return True


def process_neuron_group(group_name, neuron_ids, swc_dir, output_dir):
    """Process a group of neurons (ACC or INS)."""
    print("\n" + "="*70)
    print(f"PROCESSING {group_name} NEURONS ({len(neuron_ids)} total)")
    print("="*70)
    
    # Create output subdirectory for this group
    group_output_dir = os.path.join(output_dir, group_name.lower())
    os.makedirs(group_output_dir, exist_ok=True)
    
    success_count = 0
    failed_neurons = []
    
    for i, neuron_id in enumerate(neuron_ids, 1):
        print(f"\n[{i}/{len(neuron_ids)}] Processing {neuron_id}...")
        
        if process_single_neuron(neuron_id, swc_dir, group_output_dir):
            success_count += 1
            print(f"  ✓ Success")
        else:
            failed_neurons.append(neuron_id)
            print(f"  ✗ Failed")
    
    print(f"\n{group_name} Processing Complete:")
    print(f"  Success: {success_count}/{len(neuron_ids)}")
    print(f"  Failed: {len(failed_neurons)}")
    if failed_neurons:
        print(f"  Failed neurons: {failed_neurons}")
    
    return success_count, failed_neurons, group_output_dir


def join_fnt_files(group_name, group_output_dir):
    """Join all decimated FNT files for a group into a single file."""
    print("\n" + "-"*70)
    print(f"JOINING {group_name} FNT FILES")
    print("-"*70)
    
    # Find all decimate.fnt files
    decimate_files = sorted(glob.glob(os.path.join(group_output_dir, "*.decimate.fnt")))
    
    if not decimate_files:
        print(f"  ✗ No decimate.fnt files found in {group_output_dir}")
        return None
    
    print(f"  Found {len(decimate_files)} decimate.fnt files")
    
    # Create joined file
    joined_file = os.path.join(group_output_dir, f"{group_name.lower()}_joined.fnt")
    
    # Build command - use wildcard for joining
    file_pattern = os.path.join(group_output_dir, "*.decimate.fnt")
    command = f'fnt-join.exe {file_pattern} -o "{joined_file}"'
    
    if execute_command(command):
        print(f"  ✓ Joined FNT created: {joined_file}")
        return joined_file
    else:
        print(f"  ✗ Failed to join FNT files")
        return None


def compute_fnt_distances(group_name, group_output_dir):
    """Compute FNT distance matrix for a group."""
    print("\n" + "-"*70)
    print(f"COMPUTING {group_name} FNT DISTANCES")
    print("-"*70)
    
    # Find all decimate.fnt files
    decimate_files = sorted(glob.glob(os.path.join(group_output_dir, "*.decimate.fnt")))
    
    if len(decimate_files) < 2:
        print(f"  ✗ Need at least 2 neurons for distance computation, found {len(decimate_files)}")
        return None
    
    print(f"  Computing distances for {len(decimate_files)} neurons...")
    
    # Output distance file
    dist_file = os.path.join(group_output_dir, f"{group_name.lower()}_dist.txt")
    
    # Build command
    file_pattern = os.path.join(group_output_dir, "*.decimate.fnt")
    command = f'fnt-dist {file_pattern} > "{dist_file}"'
    
    if execute_command(command):
        print(f"  ✓ Distance matrix saved: {dist_file}")
        
        # Check file size
        if os.path.exists(dist_file):
            size = os.path.getsize(dist_file)
            print(f"  File size: {size:,} bytes")
        
        return dist_file
    else:
        print(f"  ✗ Failed to compute distances")
        return None


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
#%%

"""Main workflow for FNT processing of ACC and INS neurons."""
print("="*70)
print("FNT PIPELINE FOR ACC AND INS NEURONS")
print("="*70)
print(f"Sample ID: {SAMPLE_ID}")
print(f"SWC Directory: {SWC_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

# # Check FNT tools
# if not check_fnt_tools():
#     print("\n✗ Cannot proceed without FNT tools. Exiting.")
#     return 1

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load neuron dataframes
neuron_data = load_neuron_dataframes()

# if not neuron_data:
#     print("\n✗ No neuron data loaded. Exiting.")
#     return 1

# Process each group
results = {}

for group_name, data in neuron_data.items():
    neuron_ids = data['neuron_ids']
    
    # Process neurons
    success_count, failed_neurons, group_output_dir = process_neuron_group(
        group_name, neuron_ids, SWC_DIR, OUTPUT_DIR
    )
    
    if success_count == 0:
        print(f"\n✗ No {group_name} neurons processed successfully. Skipping join/dist.")
        continue
    
    # Join FNT files
    joined_file = join_fnt_files(group_name, group_output_dir)
    
    # Compute distances
    # dist_file = compute_fnt_distances(group_name, group_output_dir)
    
    # results[group_name] = {
    #     'total': len(neuron_ids),
    #     'success': success_count,
    #     'failed': failed_neurons,
    #     'output_dir': group_output_dir,
    #     'joined_file': joined_file,
    #     'dist_file': dist_file
    # }

# # Summary
# print("\n" + "="*70)
# print("PROCESSING SUMMARY")
# print("="*70)

# for group_name, result in results.items():
#     print(f"\n{group_name}:")
#     print(f"  Neurons processed: {result['success']}/{result['total']}")
#     print(f"  Output directory: {result['output_dir']}")
#     print(f"  Joined FNT: {result['joined_file'] or 'N/A'}")
#     print(f"  Distance matrix: {result['dist_file'] or 'N/A'}")

# print("\n" + "="*70)
# print("FNT PIPELINE COMPLETE")
# print("="*70)

# return 0


# if __name__ == "__main__":
#     main()

# %%
