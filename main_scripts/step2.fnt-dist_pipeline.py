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

# Setup path for IONData
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)
try:
    import IONData
    IONDATA_AVAILABLE = True
except ImportError:
    IONDATA_AVAILABLE = False
    print("Warning: IONData module not available. SWC download functionality disabled.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Paths
SAMPLE_ID = '251637'
SWC_DIR = f'processed_neurons/{SAMPLE_ID}'
RAW_SWC_DIR = f'processed_neurons/{SAMPLE_ID}/raw_swcs'  # Separate folder for downloaded raw SWCs
OUTPUT_DIR = f'processed_neurons/{SAMPLE_ID}/fnt_processed'
ACC_DF_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\ACC_df_v3.xlsx'
INS_DF_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\INS_df_v3.xlsx'

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
    # error = error.decode('utf-8').strip()
    
    if output and verbose:
        print(f"    Output: {output[:200]}..." if len(output) > 200 else f"    Output: {output}")
    
    # if error and verbose:
    #     print(f"    Error: {error[:200]}..." if len(error) > 200 else f"    Error: {error}")
    
    return process.returncode == 0


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

def preprocess_swc_coordinates(swc_file, output_swc=None):
    """Preprocess SWC file to flip x-coordinate to the left side (x < 32000).
    
    SWC format columns:
    1: node ID
    2: structure type
    3: x coordinate (column index 2, 0-indexed)
    4: y coordinate
    5: z coordinate
    6: radius
    7: parent node ID
    
    The midline is at x = 32000. Neurons on the right side (x > 32000) are
    mirrored to the left side using: new_x = 64000 - x
    This ensures proper reflection across the midline.
    
    Example: x = 48000 (right side, 16000 from midline) 
             -> new_x = 64000 - 48000 = 16000 (left side, 16000 from midline)
    """
    MIDLINE = 32000
    
    if output_swc is None:
        output_swc = f"{swc_file}.processed.swc"
    
    # Check if already exists
    if os.path.exists(output_swc):
        print(f"    Preprocessed SWC already exists: {os.path.basename(output_swc)}")
        return output_swc
    
    try:
        with open(swc_file, 'r') as f:
            lines = f.readlines()
        
        processed_lines = []
        flipped_count = 0
        already_left_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Keep comment lines unchanged
            if not line_stripped or line_stripped.startswith('#'):
                processed_lines.append(line)
                continue
            
            # Split the line into columns
            parts = line_stripped.split()
            
            if len(parts) >= 3:
                try:
                    x = float(parts[2])
                    if x > MIDLINE:
                        # Mirror across the midline
                        x = 2 * MIDLINE - x  # = 64000 - x
                        parts[2] = str(x)
                        flipped_count += 1
                    else:
                        already_left_count += 1
                except ValueError:
                    pass  # Keep original if conversion fails
            
            processed_lines.append(' '.join(parts) + '\n')
        
        with open(output_swc, 'w') as f:
            f.writelines(processed_lines)
        
        if flipped_count > 0:
            print(f"    Flipped {flipped_count} nodes from right to left (x > {MIDLINE})")
        if already_left_count > 0:
            print(f"    {already_left_count} nodes already on left side (x <= {MIDLINE})")
        
        return output_swc
    except Exception as e:
        print(f"    Error preprocessing SWC: {e}")
        return None


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


def download_swc_if_missing(neuron_id, sample_id=SAMPLE_ID):
    """Download SWC file using IONDATA if not found locally.
    
    Downloads are saved to RAW_SWC_DIR to keep raw data separate from processed files.
    Note: IONDATA saves to ../resource/swc_raw/{sample_id}/ by default, so we move it.
    
    Args:
        neuron_id: Neuron ID (e.g., '001.swc')
        sample_id: Sample ID for IONDATA lookup
    
    Returns:
        str: Path to SWC file if successful, None otherwise
    """
    # Target location for raw SWCs
    raw_swc_file = os.path.join(RAW_SWC_DIR, neuron_id)
    
    # Check if already exists in raw folder
    if os.path.exists(raw_swc_file):
        print(f"    Found existing raw SWC: {raw_swc_file}")
        return raw_swc_file
    
    # Try to download using IONDATA
    if not IONDATA_AVAILABLE:
        print(f"    IONData not available, cannot download {neuron_id}")
        return None
    
    try:
        print(f"    SWC not found locally, attempting download via IONDATA...")
        os.makedirs(RAW_SWC_DIR, exist_ok=True)
        
        # Initialize IONData and download neuron
        # Note: neuron_id should include .swc extension for IONDATA
        iondata = IONData.IONData()
        iondata.getRawNeuronTreeByID(sample_id, neuron_id)
        
        # IONDATA saves to ../resource/swc_raw/{sample_id}/{neuron_id}
        iondata_default_path = f'../resource/swc_raw/{sample_id}/{neuron_id}'
        
        # Check if download was successful at IONDATA's default location
        if os.path.exists(iondata_default_path):
            # Move to our raw_swcs folder
            shutil.move(iondata_default_path, raw_swc_file)
            print(f"    ✓ Successfully downloaded and moved {neuron_id} to {RAW_SWC_DIR}")
            return raw_swc_file
        # Also check if it was somehow already in the right place
        elif os.path.exists(raw_swc_file):
            print(f"    ✓ Successfully downloaded {neuron_id} to {RAW_SWC_DIR}")
            return raw_swc_file
        else:
            print(f"    ✗ Download failed: {neuron_id} not found at {iondata_default_path}")
            return None
    except Exception as e:
        print(f"    ✗ Error downloading {neuron_id}: {e}")
        return None


def find_swc_file(neuron_id, swc_dir):
    """Find SWC file in either the original directory or the raw_swcs folder.
    
    Priority:
    1. Check RAW_SWC_DIR first (downloaded raw data)
    2. Check swc_dir (original location)
    3. Try to download if not found
    
    Args:
        neuron_id: Neuron ID (e.g., '001.swc')
        swc_dir: Original SWC directory to check
    
    Returns:
        str: Path to SWC file if found/downloaded, None otherwise
    """
    # Priority 1: Check raw_swcs folder
    raw_path = os.path.join(RAW_SWC_DIR, neuron_id)
    if os.path.exists(raw_path):
        return raw_path
    
    # Priority 2: Check original swc_dir
    original_path = os.path.join(swc_dir, neuron_id)
    if os.path.exists(original_path):
        return original_path
    
    # Priority 3: Try to download
    return download_swc_if_missing(neuron_id)


def process_single_neuron(neuron_id, swc_dir, output_dir):
    """Process a single neuron through the FNT pipeline."""
    # Find SWC file (check raw_swcs first, then original dir, then download)
    swc_file = find_swc_file(neuron_id, swc_dir)
    
    if swc_file is None:
        print(f"  ✗ SWC file not found and could not download: {neuron_id}")
        return False
    
    print(f"  Using SWC: {swc_file}")
    
    # Step 1: Preprocess SWC to flip x-coordinate if > 32000
    preprocessed_swc = os.path.join(output_dir, f"{neuron_id}.processed.swc")
    preprocessed_swc = preprocess_swc_coordinates(swc_file, preprocessed_swc)
    if preprocessed_swc is None:
        print(f"  ✗ Failed to preprocess SWC coordinates: {neuron_id}")
        return False
    
    # Step 2: Convert preprocessed SWC to FNT
    fnt_file = os.path.join(output_dir, f"{neuron_id}.fnt")
    if not swc_to_fnt(preprocessed_swc, fnt_file):
        print(f"  ✗ Failed to convert SWC to FNT: {neuron_id}")
        return False
    
    # Step 3: Decimate FNT
    decimate_file = os.path.join(output_dir, f"{neuron_id}.decimate.fnt")
    if not decimate_fnt(fnt_file, decimate_file):
        print(f"  ✗ Failed to decimate FNT: {neuron_id}")
        return False
    
    # Step 4: Update neuron name in decimated FNT
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
    group_output_dir = output_dir + '/'+ group_name.lower()
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
    joined_file = group_output_dir + '/' +f"{group_name.lower()}_joined.fnt"
    
    # Build command using bash -c to handle wildcard expansion
    # This avoids "command line too long" error on Windows
    # Note: No quotes around wildcard pattern to allow bash expansion
    file_pattern = group_output_dir + "/*.decimate.fnt"
    command = f"bash -c 'fnt-join.exe {file_pattern} -o {joined_file}'"
    
    if execute_command(command):
        print(f"  ✓ Joined FNT created: {joined_file}")
        return joined_file
    else:
        print(f"  ✗ Failed to join FNT files")
        return None


def compute_fnt_distances(group_name, group_output_dir, joined_file=None):
    """Compute FNT distance matrix for a group.
    
    Note: fnt-dist requires a single joined FNT file, not multiple individual files.
    If joined_file is not provided, it will look for {group_name}_joined.fnt
    """
    print("\n" + "-"*70)
    print(f"COMPUTING {group_name} FNT DISTANCES")
    print("-"*70)
    
    # Find the joined FNT file
    if joined_file is None:
        joined_file = os.path.join(group_output_dir, f"{group_name.lower()}_joined.fnt")
    
    if not os.path.exists(joined_file):
        print(f"  ✗ Joined FNT file not found: {joined_file}")
        print(f"     Please run join_fnt_files first.")
        return None
    
    # Output distance file
    dist_file = os.path.join(group_output_dir, f"{group_name.lower()}_dist.txt")
    
    print(f"  Computing distances from joined file...")
    
    # Build command - fnt-dist takes a single joined FNT file
    command = f'fnt-dist.exe "{joined_file}" -o "{dist_file}"'
    
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
def main():
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
        
        # Compute distances (requires joined file)
        dist_file = compute_fnt_distances(group_name, group_output_dir, joined_file)
        
        results[group_name] = {
            'total': len(neuron_ids),
            'success': success_count,
            'failed': failed_neurons,
            'output_dir': group_output_dir,
            'joined_file': joined_file,
            'dist_file': dist_file
        }
        # Summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)

    for group_name, result in results.items():
        print(f"\n{group_name}:")
        print(f"  Neurons processed: {result['success']}/{result['total']}")
        print(f"  Output directory: {result['output_dir']}")
        print(f"  Joined FNT: {result['joined_file'] or 'N/A'}")
        print(f"  Distance matrix: {result['dist_file'] or 'N/A'}")

    print("\n" + "="*70)
    print("FNT PIPELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
    
    


