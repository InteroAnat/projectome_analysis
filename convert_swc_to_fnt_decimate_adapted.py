#!/usr/bin/env python3
"""
Adapted SWC to FNT Conversion and Decimation Tool
==================================================

This version uses the FNT Tools Adapter to work with existing FNT files or
actual FNT tools when available. It provides fallback functionality for testing
and development.

Usage:
    python convert_swc_to_fnt_decimate_adapted.py --input_dir /path/to/swc/files --output_dir /path/to/output
    
Requirements:
    - Either FNT tools installed, or existing FNT files for mock operations
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import our adapter
from fnt_tools_adapter import create_fnt_adapter, FNTToolsAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('swc_fnt_conversion_adapted.log')
    ]
)
logger = logging.getLogger(__name__)

class AdaptedSWCtoFNTConverter:
    """
    Converts SWC files to FNT format using the FNT Tools Adapter.
    """
    
    def __init__(self, fnt_tools_dir: Optional[str] = None, 
                 decimate_distance: int = 5000, 
                 decimate_angle: int = 5000,
                 n_workers: int = 4,
                 allow_mock: bool = True,
                 mock_data_dir: Optional[str] = None):
        """
        Initialize the converter with adapter.
        
        Args:
            fnt_tools_dir: Directory containing FNT tools
            decimate_distance: Distance threshold for decimation
            decimate_angle: Angle threshold for decimation
            n_workers: Number of parallel workers
            allow_mock: Whether to allow mock operations
            mock_data_dir: Directory with existing FNT files for mock operations
        """
        self.decimate_distance = decimate_distance
        self.decimate_angle = decimate_angle
        self.n_workers = n_workers
        
        # Initialize the FNT adapter
        try:
            self.fnt_adapter = create_fnt_adapter(
                fnt_tools_dir=fnt_tools_dir,
                allow_mock=allow_mock,
                mock_data_dir=mock_data_dir
            )
            logger.info("FNT Tools Adapter initialized successfully")
            logger.info(f"Available tools: {self.fnt_adapter.available_tools}")
            logger.info(f"Mock mode: {self.fnt_adapter.mock_mode}")
        except Exception as e:
            logger.error(f"Failed to initialize FNT adapter: {e}")
            raise
    
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
            intermediate_fnt = output_dir / f"{base_name}.fnt"
            final_fnt = output_dir / f"{base_name}.decimate.fnt"
            
            logger.info(f"Processing {swc_file.name}")
            
            # Step 1: Convert SWC to FNT
            success, message = self.fnt_adapter.run_fnt_from_swc(swc_file, intermediate_fnt)
            if not success:
                return False, f"SWC to FNT conversion failed for {swc_file.name}: {message}"
            
            logger.info(f"Successfully converted {swc_file.name} to FNT")
            
            # Step 2: Apply decimation
            success, message = self.fnt_adapter.run_fnt_decimate(
                intermediate_fnt, final_fnt, self.decimate_distance, self.decimate_angle
            )
            if not success:
                return False, f"Decimation failed for {intermediate_fnt.name}: {message}"
            
            logger.info(f"Successfully decimated {intermediate_fnt.name}")
            
            # Clean up intermediate FNT file if decimation was successful
            try:
                if intermediate_fnt.exists() and intermediate_fnt != final_fnt:
                    intermediate_fnt.unlink()
            except OSError:
                pass  # Don't fail if cleanup doesn't work
            
            return True, f"Successfully processed {swc_file.name}"
            
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
        description="Convert SWC files to FNT format and apply decimation (adapted version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process with automatic tool detection
    python convert_swc_to_fnt_decimate_adapted.py --input_dir /path/to/swc/files --output_dir /path/to/output
    
    # Specify FNT tools directory
    python convert_swc_to_fnt_decimate_adapted.py --input_dir /path/to/swc/files --fnt_dir /path/to/fnt/tools
    
    # Use mock mode with existing FNT files
    python convert_swc_to_fnt_decimate_adapted.py --input_dir /path/to/swc/files --mock_data_dir /path/to/existing/fnt/files
    
    # Custom decimation parameters
    python convert_swc_to_fnt_decimate_adapted.py --input_dir /path/to/swc/files --decimate_distance 10000 --decimate_angle 10000
    
    # Disable mock mode (require real tools)
    python convert_swc_to_fnt_decimate_adapted.py --input_dir /path/to/swc/files --no_mock
        """
    )
    
    parser.add_argument('--input_dir', '-i', required=True, type=str,
                       help='Input directory containing SWC files')
    parser.add_argument('--output_dir', '-o', required=True, type=str,
                       help='Output directory for FNT files')
    parser.add_argument('--fnt_dir', '-f', type=str, default=None,
                       help='Directory containing FNT tools')
    parser.add_argument('--decimate_distance', '-d', type=int, default=5000,
                       help='Distance threshold for decimation (default: 5000)')
    parser.add_argument('--decimate_angle', '-a', type=int, default=5000,
                       help='Angle threshold for decimation (default: 5000)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--pattern', '-p', type=str, default='*.swc',
                       help='File pattern to match (default: "*.swc")')
    parser.add_argument('--mock_data_dir', '-m', type=str, default=None,
                       help='Directory with existing FNT files for mock operations')
    parser.add_argument('--no_mock', action='store_true',
                       help='Disable mock mode (require real FNT tools)')
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
        converter = AdaptedSWCtoFNTConverter(
            fnt_tools_dir=args.fnt_dir,
            decimate_distance=args.decimate_distance,
            decimate_angle=args.decimate_angle,
            n_workers=args.workers,
            allow_mock=not args.no_mock,
            mock_data_dir=args.mock_data_dir
        )
    except Exception as e:
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
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using mock mode: {converter.fnt_adapter.mock_mode}")
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