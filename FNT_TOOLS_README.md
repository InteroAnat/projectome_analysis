# FNT Distance Analysis Tools

This directory contains Python implementations of the FNT (Functional Neuroanatomy Toolbox) distance analysis workflow, originally implemented as shell scripts and R code.

## Overview

The FNT tools are used for analyzing neuron morphology data by calculating distances between neuronal structures. This implementation provides a more flexible and integrated approach compared to the original shell scripts.

## Tools Provided

### 1. `convert_swc_to_fnt_decimate.py`
Converts SWC files to FNT format and applies decimation for efficient processing.

**Functionality:**
- Converts SWC files to FNT format using `fnt-from-swc`
- Applies decimation using `fnt-decimate` with specified distance and angle thresholds
- Supports parallel processing for efficiency
- Provides detailed logging and error handling

**Usage:**
```bash
python convert_swc_to_fnt_decimate.py --input_dir /path/to/swc/files --output_dir /path/to/output
```

### 2. `update_fnt_decimate_neuron_names.py`
Updates neuron names in FNT decimate files to ensure proper identification.

**Functionality:**
- Reads FNT decimate files and extracts neuron names from filenames
- Updates the last line of each file with the correct neuron identifier
- Supports verification of existing updates
- Processes files in parallel for efficiency

**Usage:**
```bash
python update_fnt_decimate_neuron_names.py --input_dir /path/to/fnt/files
```

### 3. `join_fnt_decimate_files.py`
Joins multiple FNT decimate files into a single file for distance analysis.

**Functionality:**
- Combines multiple FNT decimate files using `fnt-join`
- Supports custom file patterns
- Option to remove original files after successful join
- Provides file size verification

**Usage:**
```bash
python join_fnt_decimate_files.py --input_dir /path/to/fnt/files --output_file joined.fnt
```

### 4. `fnt_distance_workflow.py`
Complete workflow that orchestrates all steps from SWC files to distance matrix.

**Functionality:**
- Runs the complete pipeline: conversion → name update → joining → distance calculation
- Supports running individual steps independently
- Provides comprehensive logging and metadata tracking
- Handles temporary file management and cleanup

**Usage:**
```bash
python fnt_distance_workflow.py --input_dir /path/to/swc/files --output_dir /path/to/output
```

## Original Scripts Reference

The original shell scripts and R code that these tools replace:

1. **`1_convert_swc_decimate_to_fnt_decimate.sh`**
   - Converts SWC files to FNT format
   - Applies decimation with distance=5000 and angle=5000 parameters
   - Processes files sequentially or in parallel

2. **`2_change_fnt_decimate_neuron_name.R`**
   - Updates neuron names in FNT decimate files
   - Extracts neuron name from filename (removes `.decimate.fnt`)
   - Replaces last line with format: `0 <neuron_name>`
   - Uses parallel processing via R's parallel/foreach packages

3. **`3_fnt_decimate_join.sh`**
   - Joins all FNT decimate files into a single file
   - Uses `fnt-join` tool with output file specification

4. **`4-2_[on_hpc]_sbatch_dist.slurm`**
   - SLURM job script for running `fnt-dist` on HPC cluster
   - Calculates distance matrix from joined FNT file
   - Configured for 96 tasks with 72-hour time limit

## FNT File Format

FNT (Functional Neuroanatomy Toolbox) files contain neuron morphology data with the following structure:

```
Fast Neurite Tracer Session File 1.0
[metadata lines...]
BEGIN_TRACING_DATA
[tracing data...]
0 <neuron_name>
```

The last line contains the neuron identifier, which is crucial for proper labeling in distance calculations.

## Installation Requirements

1. **FNT Tools**: The FNT toolkit must be installed and available in PATH or specified via `--fnt_dir`
   - `fnt-from-swc`: Convert SWC to FNT format
   - `fnt-decimate`: Apply decimation to reduce data density
   - `fnt-join`: Join multiple FNT files
   - `fnt-dist`: Calculate distance matrix (optional)

2. **Python Dependencies**:
   - Standard library modules (concurrent.futures, multiprocessing, pathlib, etc.)
   - No external Python packages required

3. **System Requirements**:
   - Python 3.6 or higher
   - Operating system: Linux/macOS/Windows
   - Sufficient disk space for temporary files

## Key Parameters

- **Decimation Distance** (`-d, --decimate_distance`): Spatial distance threshold for decimation (default: 5000)
- **Decimation Angle** (`-a, --decimate_angle`): Angular threshold for decimation (default: 5000)
- **Parallel Workers** (`-w, --workers`): Number of parallel processes (default: 4)
- **File Patterns**: Support for custom file matching patterns

## Error Handling

All tools provide comprehensive error handling:
- Verification of input files and directories
- Validation of FNT tool availability and permissions
- Detailed logging of processing steps and failures
- Graceful handling of partial failures in batch processing

## Output Files

The workflow generates:
1. **FNT decimate files**: Individual processed neuron files
2. **Joined FNT file**: Combined file for distance analysis
3. **Distance matrix**: Text file containing pairwise distances (optional)
4. **Metadata**: JSON file with workflow parameters and timing
5. **Log files**: Detailed processing logs for debugging

## Parallel Processing

All tools support parallel processing to handle large datasets efficiently:
- Configurable number of worker processes
- Automatic CPU core detection
- Load balancing across available cores
- Progress tracking for long-running operations

## Integration with Existing Code

These tools are designed to integrate seamlessly with the existing neuron analysis codebase:
- Compatible with existing file naming conventions
- Follow established directory structures
- Support for custom FNT tools installation paths
- Metadata preservation for downstream analysis

## Troubleshooting

Common issues and solutions:

1. **FNT tools not found**: Specify the FNT tools directory with `--fnt_dir`
2. **Permission errors**: Ensure FNT tools are executable
3. **Memory issues**: Reduce the number of parallel workers with `--workers`
4. **File format errors**: Verify SWC files follow standard format
5. **Large file processing**: Use appropriate decimation parameters

For detailed error information, check the log files generated by each tool.