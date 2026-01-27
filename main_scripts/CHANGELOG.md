# Main Scripts Changelog

All notable changes to major scripts in this directory.

## Format
- **Version**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Date**: YYYY-MM-DD format
- **Changes**: Bullet points of modifications

---

## Visual_toolkit.py

### [1.0.0] - 2026-01-27
- **Initial Release**
- Hybrid-resolution data retrieval (HTTP high-res + SSH low-res)
- Grid-based block acquisition for 0.65µm resolution
- Widefield slice acquisition for 5.0µm resolution
- NIfTI and TIFF export support
- SWC overlay visualization
- Soma block and widefield plotting functions

---

## Visual_toolkit_gui.py

### [1.0.0] - 2026-01-27
- **Initial Release**
- Tkinter-based GUI for Visual Toolkit
- Auto-fill soma coordinates from neuron trees
- Threaded processing with progress indicators
- Separate buttons for high-res/low-res/both processing
- Interactive parameter controls

---

## fnt_dist_clustering.py

### [1.2.0] - 2026-01-27
- Code improvements and refactoring
- Enhanced clustering algorithms

### [1.1.0] - 2026-01-26
- General updates and bug fixes

### [1.0.0] - 2026-01-20
- **Stable Release**
- FNT distance-based clustering implementation
- Support for hierarchical clustering
- Cluster visualization and export

---

## fnt_tools.py

### [2.1.0] - 2026-01-27
- Code improvements and optimizations

### [2.0.0] - 2026-01-26
- Major refactoring and updates

### [1.2.0] - 2026-01-15
- Bug fixes for FNT file handling

### [1.1.0] - 2026-01-15
- Added direct FNT opening with neuron ID and sample ID
- Integration with region analysis

### [1.0.0] - 2026-01-14
- **Initial Stable Release**
- Core FNT utility functions
- SWC to FNT conversion helpers

---

## region_analysis.py

### [3.2.0] - 2026-01-26
- General updates and improvements

### [3.1.0] - 2026-01-20
- Updated region analysis algorithms

### [3.0.0] - 2026-01-19
- **Major Update**
- Bug fixes for outlier plot handling
- Outlier plot now controlled by separate function
- Outlier count tracking and reporting

### [2.1.0] - 2026-01-15
- Integration with fnt-tools
- Direct FNT file opening support

### [2.0.0] - 2026-01-14
- Currently best working region analysis
- Optimized for macaque data

### [1.1.0] - 2026-01-13
- Hierarchical atlas support
- Replaced atlas with NII format
- Combined 5D CHARM+SARM atlas support

### [1.0.0] - 2026-01-12
- **Initial Release**
- Basic region analysis functionality
- NeuronVis tools integration

---

## monkey_936.py

### [1.1.0] - 2026-01-27
- Code improvements and optimizations

### [1.0.0] - 2026-01-26
- **Initial Release**
- Macaque 936 region-specific analysis
- Customized for 936-region atlas

---

## cube_analysis_per_neuron_terminal.py

### [1.1.0] - 2026-01-12
- NeuronVis tools integration
- Monkey settings compatibility

### [1.0.0] - 2026-01-08
- **Initial Release**
- Per-neuron terminal cube analysis
- Integration with neuro_tracer

---

## tiff_ds2m.py

### [1.1.0] - 2026-01-27
- Code improvements

### [1.0.0] - 2026-01-26
- **Initial Release**
- TIFF downsample and merge operations

---

## volume2.py

### [2.1.0] - 2026-01-27
- Code improvements and optimizations

### [2.0.0] - 2026-01-26
- Major refactoring

### [1.0.0] - 2026-01-12
- **Initial Release**
- Volume processing utilities
- 3D data manipulation functions

---

## ssh_cluster.py

### [1.1.0] - 2026-01-27
- Code improvements
- Enhanced cluster job management

### [1.0.0] - 2026-01-26
- **Initial Release**
- SSH-based cluster computing interface
- SLURM job submission helpers

---

## Version Numbering Guide

- **MAJOR**: Incompatible API changes or major feature additions
- **MINOR**: New functionality, backwards compatible
- **PATCH**: Bug fixes, minor improvements

---

## Unversioned Scripts

The following scripts are utility/test scripts without formal versioning:
- `learn.py` - Learning/testing script
- `plot_test.py` - Plotting tests
- `neuro_tracer.py` - Tracing utilities (legacy)

---

*Last Updated: 2026-01-27*
