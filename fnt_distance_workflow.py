#!/usr/bin/env python3
"""
FNT Distance Analysis Workflow
==============================

Complete workflow for preparing and running FNT distance analysis on neuron morphology data.
This script orchestrates the entire pipeline from SWC files to distance matrix calculation.

Workflow Steps:
1. Convert SWC files to FNT format
2. Apply decimation for efficient processing
3. Update neuron names in FNT files
4. Join FNT files into single file
5. Calculate distance matrix (optional)

Based on the FNT (Functional Neuroanatomy Toolbox) workflow for neuron morphology analysis.

Usage:
    python fnt_distance_workflow.py --input_dir /path/to/swc/files --output_dir /path/to/output
    
Requirements:
    - FNT tools must be installed and available in PATH or specified via --fnt_dir
    - SWC files should follow standard neuron morphology format
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict
import logging
import subprocess
import json
from datetime import datetime

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from convert_swc_to_fnt_decimate import SWCtoFNTConverter
from update_fnt_decimate_neuron_names import FNTNameUpdater
from join_fnt_decimate_files import FNTJoiner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fnt_distance_workflow.log')
    ]
)
logger = logging.getLogger(__name__)

class FNTDistanceWorkflow:
    """
    Complete workflow for FNT distance analysis preparation.
    """
    
    def __init__(self, fnt_tools_dir: Optional[str] = None, 
                 decimate_distance: int = 5000, 
                 decimate_angle: int = 5000,
                 n_workers: int = 4):
        """
        Initialize the workflow.
        
        Args:
            fnt_tools_dir: Directory containing FNT tools
            decimate_distance: Distance threshold for decimation
            decimate_angle: Angle threshold for decimation
            n_workers: Number of parallel workers
        """
        self.fnt_tools_dir = Path(fnt_tools_dir) if fnt_tools_dir else None
        self.decimate_distance = decimate_distance
        self.decimate_angle = decimate_angle
        self.n_workers = n_workers
        
        # Initialize components
        self.converter = SWCtoFNTConverter(
            fnt_tools_dir=fnt_tools_dir,
            decimate_distance=decimate_distance,
            decimate_angle=decimate_angle,
            n_workers=n_workers
        )
        
        self.name_updater = FNTNameUpdater(n_workers=n_workers)
        self.joiner = FNTJoiner(fnt_tools_dir=fnt_tools_dir)
    
    def run_conversion_step(self, input_dir: Path, temp_dir: Path) -> bool:
        """
        Run the SWC to FNT conversion and decimation step.
        
        Args:
            input_dir: Directory containing SWC files
            temp_dir: Temporary directory for intermediate files
            
        Returns:
            True if successful
        """
        logger.info("Step 1: Converting SWC files to FNT format and applying decimation...")
        
        # Create temp directory for this step
        conversion_dir = temp_dir / "fnt_decimate"
        conversion_dir.mkdir(parents=True, exist_ok=True)
        
        # Find SWC files
        swc_files = list(input_dir.glob("*.swc"))
        if not swc_files:
            logger.error("No SWC files found in input directory")
            return False
        
        logger.info(f"Found {len(swc_files)} SWC files to process")
        
        # Process files
        results = self.converter.process_batch(swc_files, conversion_dir)
        
        # Check results
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        logger.info(f"Conversion complete: {successful} successful, {failed} failed")
        
        if failed > 0:
            logger.error("Some conversions failed:")
            for success, message in results:
                if not success:
                    logger.error(f"  {message}")
            return False
        
        return True
    
    def run_name_update_step(self, temp_dir: Path) -> bool:
        """
        Run the neuron name update step.
        
        Args:
            temp_dir: Temporary directory containing FNT decimate files
            
        Returns:
            True if successful
        """
        logger.info("Step 2: Updating neuron names in FNT decimate files...")
        
        # Find FNT decimate files
        conversion_dir = temp_dir / "fnt_decimate"
        fnt_files = list(conversion_dir.glob("*.decimate.fnt"))
        
        if not fnt_files:
            logger.error("No FNT decimate files found for name updating")
            return False
        
        # Process files
        results = self.name_updater.process_batch(fnt_files)
        
        # Check results
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        logger.info(f"Name update complete: {successful} successful, {failed} failed")
        
        if failed > 0:
            logger.error("Some name updates failed:")
            for success, message in results:
                if not success:
                    logger.error(f"  {message}")
            return False
        
        # Verify updates
        logger.info("Verifying name updates...")
        verification_results = self.name_updater.verify_updates(conversion_dir)
        correct = sum(1 for _, _, is_correct in verification_results if is_correct)
        total = len(verification_results)
        
        logger.info(f"Verification: {correct}/{total} files have correct neuron names")
        
        if correct < total:
            logger.error("Some files have incorrect neuron names")
            return False
        
        return True
    
    def run_join_step(self, temp_dir: Path, output_file: Path) -> bool:
        """
        Run the FNT files joining step.
        
        Args:
            temp_dir: Temporary directory containing FNT decimate files
            output_file: Output file path for joined FNT
            
        Returns:
            True if successful
        """
        logger.info("Step 3: Joining FNT decimate files...")
        
        conversion_dir = temp_dir / "fnt_decimate"
        
        # Join files
        success = self.joiner.process_directory(conversion_dir, output_file)
        
        if success:
            logger.info(f"Successfully joined FNT files into {output_file}")
            
            # Verify output file
            if output_file.exists() and output_file.stat().st_size > 0:
                file_size = output_file.stat().st_size
                logger.info(f"Output file size: {file_size:,} bytes")
                return True
            else:
                logger.error("Output file was not created or is empty")
                return False
        else:
            logger.error("Failed to join FNT files")
            return False
    
    def run_distance_calculation(self, joined_fnt_file: Path, output_matrix: Path) -> bool:
        """
        Run the FNT distance calculation step.
        
        Args:
            joined_fnt_file: Joined FNT file
            output_matrix: Output file path for distance matrix
            
        Returns:
            True if successful
        """
        logger.info("Step 4: Calculating FNT distance matrix...")
        
        # Create output directory
        output_matrix.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get fnt-dist tool path
            dist_tool = self.joiner._get_tool_path('fnt-dist')
            
            # Build command
            cmd = [
                str(dist_tool),
                '-o', str(output_matrix),
                str(joined_fnt_file)
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode != 0:
                logger.error(f"FNT distance calculation failed: {result.stderr}")
                return False
            
            logger.info(f"Successfully calculated distance matrix: {output_matrix}")
            
            # Verify output
            if output_matrix.exists() and output_matrix.stat().st_size > 0:
                file_size = output_matrix.stat().st_size
                logger.info(f"Distance matrix size: {file_size:,} bytes")
                return True
            else:
                logger.error("Distance matrix file was not created or is empty")
                return False
                
        except FileNotFoundError:
            logger.error("fnt-dist tool not found - skipping distance calculation")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"FNT distance calculation failed: {e.stderr if e.stderr else str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during distance calculation: {str(e)}")
            return False
    
    def save_workflow_metadata(self, output_dir: Path, metadata: Dict):
        """
        Save workflow metadata for reproducibility.
        
        Args:
            output_dir: Directory to save metadata
            metadata: Workflow metadata
        """
        metadata_file = output_dir / "workflow_metadata.json"
        
        # Add timestamp and version info
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "workflow_version": "1.0.0",
            "fnt_tools_dir": str(self.fnt_tools_dir) if self.fnt_tools_dir else None,
            "decimate_distance": self.decimate_distance,
            "decimate_angle": self.decimate_angle,
            "n_workers": self.n_workers
        })
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved workflow metadata to {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save metadata: {str(e)}")
    
    def run_complete_workflow(self, input_dir: Path, output_dir: Path, 
                            run_distance_calc: bool = True) -> bool:
        """
        Run the complete FNT distance analysis workflow.
        
        Args:
            input_dir: Directory containing SWC files
            output_dir: Output directory for results
            run_distance_calc: Whether to run distance calculation step
            
        Returns:
            True if entire workflow was successful
        """
        start_time = datetime.now()
        logger.info("Starting complete FNT distance analysis workflow...")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Convert SWC to FNT and decimate
            if not self.run_conversion_step(input_dir, temp_dir):
                logger.error("Workflow failed at conversion step")
                return False
            
            # Step 2: Update neuron names
            if not self.run_name_update_step(temp_dir):
                logger.error("Workflow failed at name update step")
                return False
            
            # Step 3: Join FNT files
            joined_fnt_file = output_dir / "fnt_decimate_joined.fnt"
            if not self.run_join_step(temp_dir, joined_fnt_file):
                logger.error("Workflow failed at join step")
                return False
            
            # Step 4: Calculate distance matrix (optional)
            if run_distance_calc:
                distance_matrix_file = output_dir / "fnt_distance_matrix.txt"
                success = self.run_distance_calculation(joined_fnt_file, distance_matrix_file)
                if not success:
                    logger.warning("Distance calculation failed, but workflow continues")
            
            # Save metadata
            metadata = {
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "joined_fnt_file": str(joined_fnt_file),
                "distance_calculation": run_distance_calc
            }
            if run_distance_calc and success:
                metadata["distance_matrix_file"] = str(distance_matrix_file)
            
            self.save_workflow_metadata(output_dir, metadata)
            
            # Calculate and report total time
            end_time = datetime.now()
            total_time = end_time - start_time
            logger.info(f"Workflow completed successfully in {total_time}")
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow failed with unexpected error: {str(e)}")
            return False
        finally:
            # Clean up temporary directory
            try:
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    logger.info("Cleaned up temporary directory")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Complete FNT distance analysis workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete workflow
    python fnt_distance_workflow.py --input_dir /path/to/swc/files --output_dir /path/to/output
    
    # Specify FNT tools directory
    python fnt_distance_workflow.py --input_dir /path/to/swc/files --output_dir /path/to/output --fnt_dir /path/to/fnt/tools
    
    # Custom decimation parameters
    python fnt_distance_workflow.py --input_dir /path/to/swc/files --output_dir /path/to/output --decimate_distance 10000 --decimate_angle 10000
    
    # Skip distance calculation
    python fnt_distance_workflow.py --input_dir /path/to/swc/files --output_dir /path/to/output --no_distance_calc
    
    # Use more parallel workers
    python fnt_distance_workflow.py --input_dir /path/to/swc/files --output_dir /path/to/output --workers 8
        """
    )
    
    parser.add_argument('--input_dir', '-i', required=True, type=str,
                       help='Input directory containing SWC files')
    parser.add_argument('--output_dir', '-o', required=True, type=str,
                       help='Output directory for results')
    parser.add_argument('--fnt_dir', '-f', type=str, default=None,
                       help='Directory containing FNT tools')
    parser.add_argument('--decimate_distance', '-d', type=int, default=5000,
                       help='Distance threshold for decimation (default: 5000)')
    parser.add_argument('--decimate_angle', '-a', type=int, default=5000,
                       help='Angle threshold for decimation (default: 5000)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--no_distance_calc', action='store_true',
                       help='Skip the distance calculation step')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without executing')
    parser.add_argument('--step_only', choices=['convert', 'update_names', 'join', 'distance'],
                       help='Run only a specific step of the workflow')
    
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
    
    # Find SWC files
    swc_files = list(input_dir.glob("*.swc"))
    if not swc_files:
        logger.error("No SWC files found in input directory")
        sys.exit(1)
    
    logger.info(f"Found {len(swc_files)} SWC files to process")
    
    if args.dry_run:
        logger.info("DRY RUN - Would process the following files:")
        for swc_file in swc_files[:10]:  # Show first 10 files
            logger.info(f"  {swc_file.name}")
        if len(swc_files) > 10:
            logger.info(f"  ... and {len(swc_files) - 10} more files")
        return
    
    # Initialize workflow
    try:
        workflow = FNTDistanceWorkflow(
            fnt_tools_dir=args.fnt_dir,
            decimate_distance=args.decimate_distance,
            decimate_angle=args.decimate_angle,
            n_workers=args.workers
        )
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Failed to initialize workflow: {e}")
        sys.exit(1)
    
    # Run workflow
    run_distance_calc = not args.no_distance_calc
    
    if args.step_only:
        # Run only specific step
        logger.info(f"Running only the '{args.step_only}' step...")
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if args.step_only == 'convert':
            success = workflow.run_conversion_step(input_dir, temp_dir)
        elif args.step_only == 'update_names':
            success = workflow.run_name_update_step(temp_dir)
        elif args.step_only == 'join':
            joined_fnt_file = output_dir / "fnt_decimate_joined.fnt"
            success = workflow.run_join_step(temp_dir, joined_fnt_file)
        elif args.step_only == 'distance':
            joined_fnt_file = output_dir / "fnt_decimate_joined.fnt"
            distance_matrix_file = output_dir / "fnt_distance_matrix.txt"
            success = workflow.run_distance_calculation(joined_fnt_file, distance_matrix_file)
        else:
            logger.error(f"Unknown step: {args.step_only}")
            sys.exit(1)
    else:
        # Run complete workflow
        success = workflow.run_complete_workflow(input_dir, output_dir, run_distance_calc)
    
    if success:
        logger.info("Workflow completed successfully")
        sys.exit(0)
    else:
        logger.error("Workflow failed")
        sys.exit(1)

if __name__ == '__main__':
    main()