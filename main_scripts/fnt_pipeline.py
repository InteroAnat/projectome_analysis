#!/usr/bin/env python3
"""
fnt_pipeline.py - Simplified FNT Pipeline for Neuron Analysis

A clean, class-based interface for processing neurons through the FNT pipeline.

Usage Example:
    from fnt_pipeline import FNTNeuronPipeline
    
    # Simple usage with defaults
    pipeline = FNTNeuronPipeline('neuron_tables/my_neurons.xlsx')
    pipeline.run()
    
    # With custom sample_id
    pipeline = FNTNeuronPipeline(
        table_path='neuron_tables/my_neurons.xlsx',
        sample_id='123456'
    )
    pipeline.run()

Author: Assistant
Date: 2026-03-31
"""

import os
import sys
import shutil
import subprocess
import pandas as pd
import glob
from pathlib import Path
from typing import List, Optional, Dict, Tuple

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


class FNTNeuronPipeline:
    """
    FNT Pipeline for processing neuron morphologies.
    
    Processes neurons through:
    1. SWC coordinate preprocessing (flip to left hemisphere)
    2. SWC to FNT conversion
    3. FNT decimation
    4. FNT joining
    5. Distance matrix computation
    
    Parameters
    ----------
    table_path : str
        Path to neuron table file (Excel .xlsx or CSV .csv)
    sample_id : str, default '251637'
        Sample ID for SWC lookup and output organization
    neuron_id_column : str, default 'NeuronID'
        Column name containing neuron IDs in the table
    decimate_d : int, default 5000
        Distance parameter for FNT decimation
    decimate_a : int, default 5000
        Angle parameter for FNT decimation
    """
    
    def __init__(
        self,
        table_path: str,
        sample_id: str = '251637',
        neuron_id_column: str = 'NeuronID',
        decimate_d: int = 5000,
        decimate_a: int = 5000
    ):
        self.table_path = table_path
        self.sample_id = sample_id
        self.neuron_id_column = neuron_id_column
        self.decimate_d = decimate_d
        self.decimate_a = decimate_a
        
        # Derive paths from sample_id
        self.swc_dir = f'processed_neurons/{sample_id}'
        self.raw_swc_dir = f'processed_neurons/{sample_id}/raw_swcs'
        self.output_dir = f'processed_neurons/{sample_id}/fnt_processed'
        
        # Derive batch name from table filename
        self.batch_name = Path(table_path).stem
        self.batch_output_dir = os.path.join(self.output_dir, self.batch_name)
        
        # Results storage
        self.neuron_ids: List[str] = []
        self.success_count = 0
        self.failed_neurons: List[str] = []
        self.joined_file: Optional[str] = None
        self.dist_file: Optional[str] = None
        
    def _execute_command(self, command: str, verbose: bool = True) -> bool:
        """Execute a shell command."""
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
        
        if output and verbose and len(output) < 500:
            print(f"    Output: {output}")
        
        return process.returncode == 0
    
    def _load_neuron_table(self) -> pd.DataFrame:
        """Load neuron table from Excel or CSV."""
        print(f"\nLoading neuron table: {self.table_path}")
        
        if not os.path.exists(self.table_path):
            raise FileNotFoundError(f"Table file not found: {self.table_path}")
        
        ext = Path(self.table_path).suffix.lower()
        
        if ext == '.csv':
            df = pd.read_csv(self.table_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(self.table_path, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        if self.neuron_id_column not in df.columns:
            raise ValueError(f"Column '{self.neuron_id_column}' not found. Available: {list(df.columns)}")
        
        self.neuron_ids = df[self.neuron_id_column].tolist()
        print(f"Loaded {len(self.neuron_ids)} neurons from column '{self.neuron_id_column}'")
        
        return df
    
    def _preprocess_swc(self, swc_file: str, output_swc: str) -> bool:
        """Preprocess SWC to flip x-coordinate to left hemisphere."""
        MIDLINE = 32000
        
        if os.path.exists(output_swc):
            print(f"    Preprocessed SWC already exists")
            return True
        
        try:
            with open(swc_file, 'r') as f:
                lines = f.readlines()
            
            processed_lines = []
            flipped = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    processed_lines.append(line)
                    continue
                
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[2])
                        if x > MIDLINE:
                            x = 2 * MIDLINE - x
                            parts[2] = str(x)
                            flipped += 1
                    except ValueError:
                        pass
                
                processed_lines.append(' '.join(parts) + '\n')
            
            with open(output_swc, 'w') as f:
                f.writelines(processed_lines)
            
            if flipped > 0:
                print(f"    Flipped {flipped} nodes from right to left")
            
            return True
            
        except Exception as e:
            print(f"    Error preprocessing SWC: {e}")
            return False
    
    def _swc_to_fnt(self, swc_file: str, fnt_file: str) -> bool:
        """Convert SWC to FNT format."""
        if os.path.exists(fnt_file):
            print(f"    FNT already exists")
            return True
        
        command = f'fnt-from-swc "{swc_file}" "{fnt_file}"'
        return self._execute_command(command)
    
    def _decimate_fnt(self, fnt_file: str, decimated_file: str) -> bool:
        """Decimate FNT file."""
        if os.path.exists(decimated_file):
            print(f"    Decimated FNT already exists")
            return True
        
        command = f'fnt-decimate -d {self.decimate_d} -a {self.decimate_a} "{fnt_file}" "{decimated_file}"'
        return self._execute_command(command)
    
    def _update_fnt_name(self, fnt_file: str, neuron_name: str) -> bool:
        """Update neuron name in FNT file."""
        try:
            with open(fnt_file, 'rb') as f:
                content = f.read()
            
            if not content:
                return False
            
            text = content.decode('utf-8').replace('\r\n', '\n')
            lines = text.splitlines()
            
            if not lines:
                return False
            
            lines[-1] = f"0 Neuron {neuron_name}"
            
            with open(fnt_file, 'w', encoding='utf-8', newline='\n') as f:
                for line in lines:
                    f.write(line + '\n')
            
            return True
            
        except Exception as e:
            print(f"    Error updating FNT name: {e}")
            return False
    
    def _download_swc(self, neuron_id: str) -> Optional[str]:
        """Download SWC using IONData if missing."""
        if not IONDATA_AVAILABLE:
            return None
        
        raw_swc_file = os.path.join(self.raw_swc_dir, neuron_id)
        
        if os.path.exists(raw_swc_file):
            return raw_swc_file
        
        try:
            print(f"    Downloading {neuron_id} via IONDATA...")
            os.makedirs(self.raw_swc_dir, exist_ok=True)
            
            iondata = IONData.IONData()
            iondata.getRawNeuronTreeByID(self.sample_id, neuron_id)
            
            iondata_path = f'../resource/swc_raw/{self.sample_id}/{neuron_id}'
            
            if os.path.exists(iondata_path):
                shutil.move(iondata_path, raw_swc_file)
                print(f"    Downloaded successfully")
                return raw_swc_file
            
            return None
            
        except Exception as e:
            print(f"    Download failed: {e}")
            return None
    
    def _find_swc(self, neuron_id: str) -> Optional[str]:
        """Find SWC file locally or download."""
        # Check raw folder
        raw_path = os.path.join(self.raw_swc_dir, neuron_id)
        if os.path.exists(raw_path):
            return raw_path
        
        # Check swc dir
        original_path = os.path.join(self.swc_dir, neuron_id)
        if os.path.exists(original_path):
            return original_path
        
        # Download
        return self._download_swc(neuron_id)
    
    def process_single_neuron(self, neuron_id: str) -> bool:
        """
        Process a single neuron through the FNT pipeline.
        
        Parameters
        ----------
        neuron_id : str
            Neuron ID (e.g., '001.swc')
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        print(f"\nProcessing {neuron_id}...")
        
        # Find SWC
        swc_file = self._find_swc(neuron_id)
        if swc_file is None:
            print(f"  ✗ SWC not found: {neuron_id}")
            return False
        
        # Step 1: Preprocess
        preprocessed = os.path.join(self.batch_output_dir, f"{neuron_id}.processed.swc")
        if not self._preprocess_swc(swc_file, preprocessed):
            return False
        
        # Step 2: Convert to FNT
        fnt_file = os.path.join(self.batch_output_dir, f"{neuron_id}.fnt")
        if not self._swc_to_fnt(preprocessed, fnt_file):
            print(f"  ✗ Failed to convert SWC to FNT")
            return False
        
        # Step 3: Decimate
        decimated = os.path.join(self.batch_output_dir, f"{neuron_id}.decimate.fnt")
        if not self._decimate_fnt(fnt_file, decimated):
            print(f"  ✗ Failed to decimate FNT")
            return False
        
        # Step 4: Update name
        neuron_name = neuron_id.replace('.swc', '')
        if not self._update_fnt_name(decimated, neuron_name):
            print(f"  ✗ Failed to update FNT name")
            return False
        
        print(f"  ✓ Success")
        return True
    
    def process_all(self) -> Tuple[int, List[str]]:
        """
        Process all neurons in the table.
        
        Returns
        -------
        Tuple[int, List[str]]
            (success_count, failed_neurons)
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING {len(self.neuron_ids)} NEURONS")
        print(f"{'='*70}")
        
        os.makedirs(self.batch_output_dir, exist_ok=True)
        
        self.success_count = 0
        self.failed_neurons = []
        
        for i, neuron_id in enumerate(self.neuron_ids, 1):
            print(f"\n[{i}/{len(self.neuron_ids)}]", end="")
            
            if self.process_single_neuron(neuron_id):
                self.success_count += 1
            else:
                self.failed_neurons.append(neuron_id)
        
        print(f"\n{'='*70}")
        print(f"Processing Complete: {self.success_count}/{len(self.neuron_ids)} succeeded")
        if self.failed_neurons:
            print(f"Failed: {self.failed_neurons}")
        print(f"{'='*70}")
        
        return self.success_count, self.failed_neurons
    
    def join_fnt_files(self) -> Optional[str]:
        """
        Join all decimated FNT files into a single file.
        
        Returns
        -------
        str or None
            Path to joined file, or None if failed
        """
        print(f"\n{'-'*70}")
        print("JOINING FNT FILES")
        print(f"{'-'*70}")
        
        decimate_files = sorted(glob.glob(os.path.join(self.batch_output_dir, "*.decimate.fnt")))
        
        if not decimate_files:
            print(f"✗ No decimate.fnt files found")
            return None
        
        print(f"Found {len(decimate_files)} decimate.fnt files")
        
        self.joined_file = os.path.join(self.batch_output_dir, f"{self.batch_name}_joined.fnt")
        
        file_pattern = os.path.join(self.batch_output_dir, "*.decimate.fnt")
        command = f"bash -c 'fnt-join.exe \"{file_pattern}\" -o \"{self.joined_file}\"'"
        
        if self._execute_command(command):
            print(f"✓ Joined: {self.joined_file}")
            return self.joined_file
        else:
            print(f"✗ Failed to join FNT files")
            return None
    
    def compute_distances(self) -> Optional[str]:
        """
        Compute FNT distance matrix.
        
        Returns
        -------
        str or None
            Path to distance matrix file, or None if failed
        """
        print(f"\n{'-'*70}")
        print("COMPUTING DISTANCES")
        print(f"{'-'*70}")
        
        if self.joined_file is None:
            self.joined_file = os.path.join(self.batch_output_dir, f"{self.batch_name}_joined.fnt")
        
        if not os.path.exists(self.joined_file):
            print(f"✗ Joined FNT not found: {self.joined_file}")
            return None
        
        self.dist_file = os.path.join(self.batch_output_dir, f"{self.batch_name}_dist.txt")
        
        command = f'fnt-dist.exe "{self.joined_file}" -o "{self.dist_file}"'
        
        if self._execute_command(command):
            print(f"✓ Distance matrix: {self.dist_file}")
            if os.path.exists(self.dist_file):
                size = os.path.getsize(self.dist_file)
                print(f"  File size: {size:,} bytes")
            return self.dist_file
        else:
            print(f"✗ Failed to compute distances")
            return None
    
    def run(self) -> Dict:
        """
        Run the complete FNT pipeline.
        
        Returns
        -------
        dict
            Results with keys: total, success, failed, joined_file, dist_file, output_dir
        """
        print(f"{'='*70}")
        print("FNT NEURON PIPELINE")
        print(f"{'='*70}")
        print(f"Sample ID: {self.sample_id}")
        print(f"Table: {self.table_path}")
        print(f"Batch: {self.batch_name}")
        print(f"Output: {self.batch_output_dir}")
        
        # Load table
        self._load_neuron_table()
        
        if not self.neuron_ids:
            print("✗ No neurons to process")
            return {'total': 0, 'success': 0, 'failed': [], 'joined_file': None, 'dist_file': None}
        
        # Process neurons
        self.process_all()
        
        if self.success_count == 0:
            print("\n✗ No neurons processed successfully")
            return {
                'total': len(self.neuron_ids),
                'success': 0,
                'failed': self.neuron_ids,
                'joined_file': None,
                'dist_file': None,
                'output_dir': self.batch_output_dir
            }
        
        # Join FNT files
        self.join_fnt_files()
        
        # Compute distances
        self.compute_distances()
        
        # Summary
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Processed: {self.success_count}/{len(self.neuron_ids)}")
        print(f"Joined: {self.joined_file or 'N/A'}")
        print(f"Distance: {self.dist_file or 'N/A'}")
        
        return {
            'total': len(self.neuron_ids),
            'success': self.success_count,
            'failed': self.failed_neurons,
            'joined_file': self.joined_file,
            'dist_file': self.dist_file,
            'output_dir': self.batch_output_dir
        }


# Convenience function for simple usage
def run_pipeline(
    table_path: str,
    sample_id: str = '251637',
    neuron_id_column: str = 'NeuronID'
) -> Dict:
    """
    Run FNT pipeline with a single function call.
    
    Parameters
    ----------
    table_path : str
        Path to neuron table
    sample_id : str, default '251637'
        Sample ID
    neuron_id_column : str, default 'NeuronID'
        Column name for neuron IDs
    
    Returns
    -------
    dict
        Pipeline results
    """
    pipeline = FNTNeuronPipeline(
        table_path=table_path,
        sample_id=sample_id,
        neuron_id_column=neuron_id_column
    )
    return pipeline.run()


# Backward compatibility with fnt-dist_pipeline_generic.py
run_pipeline_generic = run_pipeline


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='FNT Neuron Pipeline')
    parser.add_argument('table_path', help='Path to neuron table (.xlsx or .csv)')
    parser.add_argument('--sample-id', default='251637', help='Sample ID')
    parser.add_argument('--neuron-id-column', default='NeuronID', help='Neuron ID column name')
    
    args = parser.parse_args()
    
    pipeline = FNTNeuronPipeline(
        table_path=args.table_path,
        sample_id=args.sample_id,
        neuron_id_column=args.neuron_id_column
    )
    pipeline.run()
