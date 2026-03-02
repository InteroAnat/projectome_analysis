#!/usr/bin/env python3
"""
SWC to FNT Conversion and Decimation Tool
==========================================

This script converts SWC files to FNT format and applies decimation for efficient
morphological analysis. It's a Python implementation of the original shell scripts
with enhanced flexibility and error handling.

Based on the FNT (Functional Neuroanatomy Toolbox) workflow for neuron morphology analysis.

Usage:
    python convert_swc_to_fnt_decimate.py --input_dir /path/to/swc/files --output_dir /path/to/output
    
Requirements:
    - FNT tools must be installed and available in PATH or specified via --fnt_dir
    - SWC files should follow standard neuron morphology format
"""

import os
import sys
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('swc_fnt_conversion.log')
    ]
)
logger = logging.getLogger(__name__)

class SWCtoFNTConverter:
    """
    Converts SWC files to FNT format and applies decimation for morphological analysis.
    """
    
    def __init__(self, fnt_tools_dir: Optional[str] = None, 
                 decimate_distance: int = 5000, 
                 decimate_angle: int = 5000,
                 n_workers: int = 4):
        """
        Initialize the converter.
        
        Args:
            fnt_tools_dir: Directory containing FNT tools (fnt-from-swc, fnt-decimate)
            decimate_distance: Distance threshold for decimation (default: 5000)
            decimate_angle: Angle threshold for decimation (default: 5000)
            n_workers: Number of parallel workers
        """
        self.fnt_tools_dir = Path(fnt_tools_dir) if fnt_tools_dir else None
        self.decimate_distance = decimate_distance
        self.decimate_angle = decimate_angle
        self.n_workers = n_workers
        
        # Verify FNT tools are available
        self._verify_fnt_tools()
    
    def _verify_fnt_tools(self):
        """Verify that required FNT tools are available."""
        required_tools = ['fnt-from-swc', 'fnt-decimate']
        
        for tool in required_tools:
            tool_path = self._get_tool_path(tool)
            if not tool_path.exists():
                raise FileNotFoundError(f"FNT tool not found: {tool}")
            
            # Check if executable
            if not os.access(tool_path, os.X_OK):
                raise PermissionError(f"FNT tool not executable: {tool_path}")
        
        logger.info("All FNT tools verified successfully")
    
    def _get_tool_path(self, tool_name: str) -> Path:
        """Get the full path to an FNT tool."""
        if self.fnt_tools_dir:
            tool_path = self.fnt_tools_dir / tool_name
            if tool_path.exists():
                return tool_path
        
        # Try to find in PATH
        try:
            result = subprocess.run(['which', tool_name], 
                                  capture_output=True, text=True, check=True)
            return Path(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try with .exe extension (Windows)
            if tool_name + '.exe' in tool_name:
                tool_path = self.fnt_tools_dir / (tool_name + '.exe') if self.fnt_tools_dir else Path(tool_name + '.exe')
                if tool_path.exists():
                    return tool_path
            
            raise FileNotFoundError(f"FNT tool '{tool_name}' not found in {self.fnt_tools_dir} or PATH")
    
    def convert_single_swc(self, swc_file: Path, output_dir: Path) -> Tuple[bool, str]:
        """
        Convert a single SWC file to FNT and apply decimation.
        
        Args:
            swc_file: Path to SWC file
            output_dir: Output directory for FNT files
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get base filename without extension
            base_name = swc_file.stem
            fnt_file = output_dir / f"{base_name}.fnt"
            decimate_file = output_dir / f"{base_name}.decimate.fnt"
            
            logger.info(f"Processing {swc_file.name}")
            
            # Step 1: Convert SWC to FNT
            cmd_convert = [
                str(self._get_tool_path('fnt-from-swc')),
                str(swc_file),
                str(fnt_file)
            ]
            
            result = subprocess.run(cmd_convert, capture_output=True, text=True, check=True)
            if result.returncode != 0:
                return False, f"SWC to FNT conversion failed for {swc_file.name}: {result.stderr}"
            
            logger.info(f"Successfully converted {swc_file.name} to FNT")
            
            # Step 2: Apply decimation
            cmd_decimate = [
                str(self._get_tool_path('fnt-decimate')),
                '-d', str(self.decimate_distance),
                '-a', str(self.decimate_angle),
                str(fnt_file),
                str(decimate_file)
            ]
            
            result = subprocess.run(cmd_decimate, capture_output=True, text=True, check=True)
            if result.returncode != 0:
                return False, f"Decimation failed for {fnt_file.name}: {result.stderr}"
            
            logger.info(f"Successfully decimated {fnt_file.name}")
            
            # Clean up intermediate FNT file if decimation was successful
            try:
                fnt_file.unlink()
            except OSError:
                pass  # Don't fail if cleanup doesn't work
            
            return True, f"Successfully processed {swc_file.name}"
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed for {swc_file.name}: {e.stderr if e.stderr else str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error processing {swc_file.name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_batch(self, swc_files: List[Path], output_dir: Path) -> List[Tuple[bool, str]]:
        """
        Process multiple SWC files in parallel.
        
        Args:
            swc_files: List of SWC file paths
            output_dir: Output directory for FNT files
            
        Returns:
            List of (success, message) tuples
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.convert_single_swc, swc_file, output_dir): swc_file 
                for swc_file in swc_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                swc_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_msg = f"Exception processing {swc_file.name}: {str(e)}"
                    logger.error(error_msg)
                    results.append((False, error_msg))
        
        return results
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         pattern: str = "*.swc") -> List[Tuple[bool, str]]:
        """
        Process all SWC files in a directory.
        
        Args:
            input_dir: Input directory containing SWC files
            output_dir: Output directory for FNT files
            pattern: File pattern to match (default: "*.swc")
            
        Returns:
            List of (success, message) tuples
        """
        # Find all SWC files
        swc_files = list(input_dir.glob(pattern))
        
        if not swc_files:
            logger.warning(f"No SWC files found in {input_dir} with pattern {pattern}")
            return []
        
        logger.info(f"Found {len(swc_files)} SWC files to process")
        
        # Process files
        return self.process_batch(swc_files, output_dir)

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert SWC files to FNT format and apply decimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single directory
    python convert_swc_to_fnt_decimate.py --input_dir /path/to/swc/files --output_dir /path/to/output
    
    # Specify FNT tools directory
    python convert_swc_to_fnt_decimate.py --input_dir /path/to/swc/files --fnt_dir /path/to/fnt/tools
    
    # Custom decimation parameters
    python convert_swc_to_fnt_decimate.py --input_dir /path/to/swc/files --decimate_distance 10000 --decimate_angle 10000
    
    # Use more parallel workers
    python convert_swc_to_fnt_decimate.py --input_dir /path/to/swc/files --workers 8
        """
    )
    
    parser.add_argument('--input_dir', '-i', required=True, type=str,
                       help='Input directory containing SWC files')
    parser.add_argument('--output_dir', '-o', required=True, type=str,
                       help='Output directory for FNT files')
    parser.add_argument('--fnt_dir', '-f', type=str, default=None,
                       help='Directory containing FNT tools (fnt-from-swc, fnt-decimate)')
    parser.add_argument('--decimate_distance', '-d', type=int, default=5000,
                       help='Distance threshold for decimation (default: 5000)')
    parser.add_argument('--decimate_angle', '-a', type=int, default=5000,
                       help='Angle threshold for decimation (default: 5000)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--pattern', '-p', type=str, default='*.swc',
                       help='File pattern to match (default: "*.swc")')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize converter
    try:
        converter = SWCtoFNTConverter(
            fnt_tools_dir=args.fnt_dir,
            decimate_distance=args.decimate_distance,
            decimate_angle=args.decimate_angle,
            n_workers=args.workers
        )
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Failed to initialize converter: {e}")
        sys.exit(1)
    
    # Find SWC files
    swc_files = list(input_dir.glob(args.pattern))
    
    if not swc_files:
        logger.error(f"No SWC files found in {input_dir} with pattern {args.pattern}")
        sys.exit(1)
    
    logger.info(f"Found {len(swc_files)} SWC files to process")
    
    if args.dry_run:
        logger.info("DRY RUN - Would process the following files:")
        for swc_file in swc_files:
            logger.info(f"  {swc_file.name}")
        return
    
    # Process files
    logger.info("Starting conversion process...")
    results = converter.process_directory(input_dir, output_dir, args.pattern)
    
    # Report results
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    
    logger.info(f"Conversion complete: {successful} successful, {failed} failed")
    
    if failed > 0:
        logger.error("Failed conversions:")
        for success, message in results:
            if not success:
                logger.error(f"  {message}")
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()