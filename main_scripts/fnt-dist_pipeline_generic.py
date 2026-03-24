#%%
"""
fnt-dist_pipeline_generic.py - Generic FNT Pipeline for Any Neuron Table

This script processes neurons from any neuron table through the FNT pipeline:
1. Load neuron list from a user-specified table (Excel or CSV)
2. Convert SWC to FNT using fnt-from-swc
3. Decimate FNT files using fnt-decimate
4. Update FNT neuron names to match filenames
5. Join decimated FNT files into a single combined file
6. Compute FNT distance matrix using fnt-dist

Requirements:
- FNT tools (fnt-from-swc, fnt-decimate, fnt-join, fnt-dist) in PATH
- Neuron table file (Excel .xlsx or CSV .csv) with a neuron ID column
- Raw SWC files in processed_neurons/{SAMPLE_ID}/

Author: Assistant
Date: 2026-03-24
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
# USER CONFIGURATION - Modify these variables
# ==============================================================================

# Sample ID for SWC lookup
SAMPLE_ID = '251637'

# Path to neuron table file (Excel .xlsx or CSV .csv)
NEURON_TABLE_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\251637_INS.xlsx'

# Column name containing neuron IDs
NEURON_ID_COLUMN = 'NeuronID'

# FNT tool parameters
DECIMATE_D = 5000  # Distance parameter for decimation
DECIMATE_A = 5000  # Angle parameter for decimation

# ==============================================================================
# DERIVED CONFIGURATION (Auto-generated from above, usually no need to modify)
# ==============================================================================

def get_paths(sample_id=SAMPLE_ID):
    """Get directory paths based on sample_id."""
    swc_dir = f'processed_neurons/{sample_id}'
    raw_swc_dir = f'processed_neurons/{sample_id}/raw_swcs'
    output_dir = f'processed_neurons/{sample_id}/fnt_processed'
    return swc_dir, raw_swc_dir, output_dir

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
    
    if output and verbose:
        print(f"    Output: {output[:200]}..." if len(output) > 200 else f"    Output: {output}")
    
    return process.returncode == 0


def load_neuron_table(table_path, neuron_id_column=NEURON_ID_COLUMN):
    """Load neuron table from Excel or CSV file."""
    print("\n" + "="*70)
    print("LOADING NEURON TABLE")
    print("="*70)
    
    if not os.path.exists(table_path):
        print(f"✗ Table file not found: {table_path}")
        return None
    
    # Load based on file extension
    file_ext = Path(table_path).suffix.lower()
    
    try:
        if file_ext == '.csv':
            df = pd.read_csv(table_path)
            print(f"✓ Loaded CSV file: {table_path}")
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(table_path, index_col=0)
            print(f"✓ Loaded Excel file: {table_path}")
        else:
            print(f"✗ Unsupported file format: {file_ext}")
            return None
    except Exception as e:
        print(f"✗ Error loading table: {e}")
        return None
    
    # Check if neuron ID column exists
    if neuron_id_column not in df.columns:
        print(f"✗ Column '{neuron_id_column}' not found in table.")
        print(f"  Available columns: {list(df.columns)}")
        return None
    
    # Parse Terminal_Regions to ensure it's a list (if column exists)
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
    
    neuron_ids = df[neuron_id_column].tolist()
    print(f"✓ Loaded {len(neuron_ids)} neurons from column '{neuron_id_column}'")
    
    return df, neuron_ids


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
    
    Downloads are saved to raw_swcs folder to keep raw data separate from processed files.
    Note: IONDATA saves to ../resource/swc_raw/{sample_id}/ by default, so we move it.
    
    Args:
        neuron_id: Neuron ID (e.g., '001.swc')
        sample_id: Sample ID for IONDATA lookup
    
    Returns:
        str: Path to SWC file if successful, None otherwise
    """
    # Get paths based on sample_id
    _, raw_swc_dir, _ = get_paths(sample_id)
    
    # Target location for raw SWCs
    raw_swc_file = os.path.join(raw_swc_dir, neuron_id)
    
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
        os.makedirs(raw_swc_dir, exist_ok=True)
        
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
            print(f"    ✓ Successfully downloaded and moved {neuron_id} to {raw_swc_dir}")
            return raw_swc_file
        # Also check if it was somehow already in the right place
        elif os.path.exists(raw_swc_file):
            print(f"    ✓ Successfully downloaded {neuron_id} to {raw_swc_dir}")
            return raw_swc_file
        else:
            print(f"    ✗ Download failed: {neuron_id} not found at {iondata_default_path}")
            return None
    except Exception as e:
        print(f"    ✗ Error downloading {neuron_id}: {e}")
        return None


def find_swc_file(neuron_id, swc_dir, raw_swc_dir):
    """Find SWC file in either the original directory or the raw_swcs folder.
    
    Priority:
    1. Check raw_swcs folder first (downloaded raw data)
    2. Check swc_dir (original location)
    3. Try to download if not found
    
    Args:
        neuron_id: Neuron ID (e.g., '001.swc')
        swc_dir: Original SWC directory to check
        raw_swc_dir: Raw SWC directory for downloaded files
    
    Returns:
        str: Path to SWC file if found/downloaded, None otherwise
    """
    # Priority 1: Check raw_swcs folder
    raw_path = os.path.join(raw_swc_dir, neuron_id)
    if os.path.exists(raw_path):
        return raw_path
    
    # Priority 2: Check original swc_dir
    original_path = os.path.join(swc_dir, neuron_id)
    if os.path.exists(original_path):
        return original_path
    
    # Priority 3: Try to download (extract sample_id from raw_swc_dir path)
    sample_id = os.path.basename(os.path.dirname(raw_swc_dir))
    return download_swc_if_missing(neuron_id, sample_id=sample_id)


def process_single_neuron(neuron_id, swc_dir, raw_swc_dir, output_dir):
    """Process a single neuron through the FNT pipeline."""
    # Find SWC file (check raw_swcs first, then original dir, then download)
    swc_file = find_swc_file(neuron_id, swc_dir, raw_swc_dir)
    
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


def process_neuron_group(neuron_ids, swc_dir, raw_swc_dir, output_dir, output_name):
    """Process a group of neurons."""
    print("\n" + "="*70)
    print(f"PROCESSING NEURONS ({len(neuron_ids)} total)")
    print("="*70)
    
    # Create output subdirectory
    group_output_dir = output_dir + '/' + output_name
    os.makedirs(group_output_dir, exist_ok=True)
    
    success_count = 0
    failed_neurons = []
    
    for i, neuron_id in enumerate(neuron_ids, 1):
        print(f"\n[{i}/{len(neuron_ids)}] Processing {neuron_id}...")
        
        if process_single_neuron(neuron_id, swc_dir, raw_swc_dir, group_output_dir):
            success_count += 1
            print(f"  ✓ Success")
        else:
            failed_neurons.append(neuron_id)
            print(f"  ✗ Failed")
    
    print(f"\nProcessing Complete:")
    print(f"  Success: {success_count}/{len(neuron_ids)}")
    print(f"  Failed: {len(failed_neurons)}")
    if failed_neurons:
        print(f"  Failed neurons: {failed_neurons}")
    
    return success_count, failed_neurons, group_output_dir


def join_fnt_files(group_output_dir, output_name):
    """Join all decimated FNT files into a single file."""
    print("\n" + "-"*70)
    print(f"JOINING FNT FILES")
    print("-"*70)
    
    # Find all decimate.fnt files
    decimate_files = sorted(glob.glob(os.path.join(group_output_dir, "*.decimate.fnt")))
    
    if not decimate_files:
        print(f"  ✗ No decimate.fnt files found in {group_output_dir}")
        return None
    
    print(f"  Found {len(decimate_files)} decimate.fnt files")
    
    # Create joined file named after the output_name
    joined_file = group_output_dir + '/' + f"{output_name}_joined.fnt"
    
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


def compute_fnt_distances(group_output_dir, output_name, joined_file=None):
    """Compute FNT distance matrix.
    
    Note: fnt-dist requires a single joined FNT file, not multiple individual files.
    If joined_file is not provided, it will look for {output_name}_joined.fnt
    """
    print("\n" + "-"*70)
    print(f"COMPUTING FNT DISTANCES")
    print("-"*70)
    
    # Find the joined FNT file
    if joined_file is None:
        joined_file = os.path.join(group_output_dir, f"{output_name}_joined.fnt")
    
    if not os.path.exists(joined_file):
        print(f"  ✗ Joined FNT file not found: {joined_file}")
        print(f"     Please run join_fnt_files first.")
        return None
    
    # Output distance file named after output_name
    dist_file = os.path.join(group_output_dir, f"{output_name}_dist.txt")
    
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
def run_pipeline(neuron_table_file=None, sample_id=None, neuron_id_column=None):
    """Main workflow for FNT processing of neurons from a table.
    
    Args:
        neuron_table_file: Path to neuron table file (uses NEURON_TABLE_FILE if None)
        sample_id: Sample ID for SWC lookup (uses SAMPLE_ID if None)
        neuron_id_column: Column name for neuron IDs (uses NEURON_ID_COLUMN if None)
    
    Returns:
        dict: Results containing output paths and statistics
    """
    # Use defaults from configuration if not provided
    if neuron_table_file is None:
        neuron_table_file = NEURON_TABLE_FILE
    if sample_id is None:
        sample_id = SAMPLE_ID
    if neuron_id_column is None:
        neuron_id_column = NEURON_ID_COLUMN
    
    # Derive output name from table filename
    output_name = Path(neuron_table_file).stem
    
    # Get paths based on sample_id (single source of truth)
    swc_dir, raw_swc_dir, output_dir = get_paths(sample_id)
    
    print("="*70)
    print("FNT PIPELINE FOR NEURON TABLE")
    print("="*70)
    print(f"Sample ID: {sample_id}")
    print(f"Table File: {neuron_table_file}")
    print(f"Neuron ID Column: {neuron_id_column}")
    print(f"Output Name: {output_name}")
    print(f"SWC Directory: {swc_dir}")
    print(f"Output Directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load neuron table
    result = load_neuron_table(neuron_table_file, neuron_id_column)
    
    if result is None:
        print("\n✗ No neuron data loaded. Exiting.")
        return None
    
    df, neuron_ids = result

    # Process neurons
    success_count, failed_neurons, group_output_dir = process_neuron_group(
        neuron_ids, swc_dir, raw_swc_dir, output_dir, output_name
    )
    
    if success_count == 0:
        print(f"\n✗ No neurons processed successfully. Skipping join/dist.")
        return {
            'total': len(neuron_ids),
            'success': 0,
            'failed': neuron_ids,
            'output_dir': group_output_dir,
            'joined_file': None,
            'dist_file': None
        }
    
    # Join FNT files
    joined_file = join_fnt_files(group_output_dir, output_name)
    
    # Compute distances (requires joined file)
    dist_file = compute_fnt_distances(group_output_dir, output_name, joined_file)
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)

    print(f"\nResults:")
    print(f"  Neurons processed: {success_count}/{len(neuron_ids)}")
    print(f"  Output directory: {group_output_dir}")
    print(f"  Joined FNT: {joined_file or 'N/A'}")
    print(f"  Distance matrix: {dist_file or 'N/A'}")

    print("\n" + "="*70)
    print("FNT PIPELINE COMPLETE")
    print("="*70)
    
    return {
        'total': len(neuron_ids),
        'success': success_count,
        'failed': failed_neurons,
        'output_dir': group_output_dir,
        'joined_file': joined_file,
        'dist_file': dist_file
    }


def main():
    """Main entry point - uses configuration from top of file."""
    return run_pipeline()


if __name__ == "__main__":
    main()
