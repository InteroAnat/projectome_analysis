"""
region_analysis.py - Monkey Projectome Region Analysis

Version: 3.5.0 (2026-03-04)
Author: [Your Name]

Key module for analysis of the projectome in the NMT space for region information.

Features:
- ARM atlas support with cortical laterality (CL/CR/SL/SR)
- Hierarchical atlas support (5D)
- Built-in hierarchy level aggregation (Region_L1 through Region_L6)
- Outlier plot control and tracking
- Direct FNT file opening
- Soma plotting functions
- Organized output folders (timestamp + sample_id)

Update Log:
v3.5 (2026-03-04): Added organized output folders with OutputManager
                   - Automatic folder creation with timestamp
                   - Organized plots, tables, and reports
v3.4 (2026-03-04): Integrated simplified hierarchy system
                   - Built-in RegionHierarchy class for multi-level analysis
                   - Automatic hierarchy column addition (Region_L1 through Region_L6)
                   - Seamless integration with existing workflow
v3.3 (2026-01-29): Updated default atlas to NMT_v2.1_sym with ARM
v3.0 (2026-01-19): Bug fixes for outlier plot, separate outlier control function
v2.0 (2026-01-14): Best working region analysis, optimized for macaque
v1.1 (2026-01-13): Hierarchical atlas support, NII format

See CHANGELOG.md for detailed version history.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import nibabel.affines
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from nilearn import plotting

neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

import IONData 
import neuro_tracer as nt

iondata = IONData.IONData()


# ==============================================================================
# OUTPUT MANAGER - Organized folder structure (NEW in v3.5)
# ==============================================================================

class OutputManager:
    """
    Manages organized output folders with timestamp and sample_id.
    
    Folder structure:
        output/
        └── {sample_id}_{timestamp}/
            ├── plots/
            ├── tables/
            ├── reports/
            └── debug/
    """
    
    def __init__(self, base_path: str = None, sample_id: str = None, 
                 create_timestamp: bool = True):
        """
        Initialize output manager.
        
        Args:
            base_path: Base directory for outputs (default: ./output)
            sample_id: Sample identifier
            create_timestamp: Add timestamp to folder name
        """
        self.base_path = Path(base_path or './output')
        self.sample_id = sample_id or 'analysis'
        
        # Create folder name
        if create_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            folder_name = f"{self.sample_id}_{timestamp}"
        else:
            folder_name = self.sample_id
        
        # Set up paths
        self.output_dir = self.base_path / folder_name
        self.plots_dir = self.output_dir / 'plots'
        self.tables_dir = self.output_dir / 'tables'
        self.reports_dir = self.output_dir / 'reports'
        self.debug_dir = self.output_dir / 'debug'
        
        self._created = False
    
    def create_dirs(self):
        """Create all output directories."""
        if self._created:
            return
        
        for d in [self.output_dir, self.plots_dir, self.tables_dir, 
                  self.reports_dir, self.debug_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self._created = True
        print(f"[OUTPUT] Created: {self.output_dir}")
    
    def get_plot_path(self, name: str, ext: str = 'png') -> Path:
        """Get path for plot file."""
        self.create_dirs()
        return self.plots_dir / f"{name}.{ext}"
    
    def get_table_path(self, name: str, ext: str = 'xlsx') -> Path:
        """Get path for table file."""
        self.create_dirs()
        return self.tables_dir / f"{name}.{ext}"
    
    def get_report_path(self, name: str, ext: str = 'txt') -> Path:
        """Get path for report file."""
        self.create_dirs()
        return self.reports_dir / f"{name}.{ext}"
    
    def get_debug_path(self, name: str, ext: str = 'png') -> Path:
        """Get path for debug file."""
        self.create_dirs()
        return self.debug_dir / f"{name}.{ext}"
    
    def print_summary(self):
        """Print output folder summary."""
        print("\n" + "="*60)
        print("OUTPUT SUMMARY")
        print("="*60)
        print(f"Location: {self.output_dir}")
        print(f"  Tables:  {self.tables_dir}")
        print(f"  Reports: {self.reports_dir}")
        print(f"  Plots:   {self.plots_dir}")
        print(f"  Debug:   {self.debug_dir}")
        print("="*60)


# ==============================================================================
# HIERARCHY SYSTEM (Integrated v3.4)
# ==============================================================================

class RegionHierarchy:
    """
    Manages hierarchical brain region relationships for multi-level analysis.
    """
    
    def __init__(self, arm_key_df: pd.DataFrame):
        """Initialize from arm_key DataFrame."""
        self.arm_key_df = arm_key_df.copy()
        self.parent_map: Dict[str, Optional[str]] = {}
        self.children_map: Dict[str, List[str]] = {}
        self.level_map: Dict[str, int] = {}
        self.name_map: Dict[str, str] = {}
        self._build_hierarchy()
    
    @classmethod
    def from_file(cls, arm_key_path: str) -> 'RegionHierarchy':
        """Load hierarchy from Excel/CSV/TXT file."""
        if arm_key_path.endswith('.xlsx'):
            arm_key_df = pd.read_excel(arm_key_path)
        elif arm_key_path.endswith('.csv'):
            arm_key_df = pd.read_csv(arm_key_path)
        else:
            arm_key_df = pd.read_csv(arm_key_path, sep='\t')
        
        required_cols = ['Index', 'Abbreviation', 'First_Level']
        missing = set(required_cols) - set(arm_key_df.columns)
        if missing:
            raise ValueError(f"arm_key missing columns: {missing}")
        
        print(f"    Loaded arm_key: {len(arm_key_df)} regions, levels 1-{arm_key_df['First_Level'].max()}")
        return cls(arm_key_df)
    
    def _build_hierarchy(self):
        """Build parent-child relationships from arm_key table."""
        df_sorted = self.arm_key_df.sort_values('Index')
        level_stack = {}
        
        for _, row in df_sorted.iterrows():
            abbrev = row['Abbreviation']
            level = row['First_Level']
            self.level_map[abbrev] = level
            self.name_map[abbrev] = row.get('Full_Name', abbrev)
            
            if level > 1 and (level - 1) in level_stack:
                parent = level_stack[level - 1]
                self.parent_map[abbrev] = parent
                if parent not in self.children_map:
                    self.children_map[parent] = []
                self.children_map[parent].append(abbrev)
            else:
                self.parent_map[abbrev] = None
            
            level_stack[level] = abbrev
            for lv in list(level_stack.keys()):
                if lv > level:
                    del level_stack[lv]
    
    def get_parent(self, region: str) -> Optional[str]:
        return self.parent_map.get(region)
    
    def get_children(self, region: str) -> List[str]:
        return self.children_map.get(region, [])
    
    def get_level(self, region: str) -> int:
        return self.level_map.get(region, -1)
    
    def get_ancestors(self, region: str) -> List[str]:
        ancestors = []
        current = self.get_parent(region)
        while current:
            ancestors.append(current)
            current = self.get_parent(current)
        return ancestors
    
    def get_descendants(self, region: str) -> List[str]:
        descendants = []
        for child in self.get_children(region):
            descendants.append(child)
            descendants.extend(self.get_descendants(child))
        return descendants
    
    def aggregate_to_level(self, region: str, target_level: int) -> Optional[str]:
        """Find ancestor at target level."""
        if pd.isna(region):
            return None
        current_level = self.get_level(region)
        if current_level == -1:
            return None
        if current_level == target_level:
            return region
        if current_level < target_level:
            return None
        current = region
        while current and self.get_level(current) > target_level:
            current = self.get_parent(current)
        return current
    
    def get_full_path(self, region: str) -> List[str]:
        if pd.isna(region):
            return []
        return [region] + self.get_ancestors(region)
    
    def get_regions_at_level(self, level: int) -> List[str]:
        return [r for r, lv in self.level_map.items() if lv == level]
    
    def print_tree(self, root: str = None, indent: int = 0, max_depth: int = 6):
        if indent > max_depth:
            return
        if root is None:
            roots = [r for r, p in self.parent_map.items() if p is None]
            for r in sorted(roots):
                self.print_tree(r, 0, max_depth)
            return
        prefix = "  " * indent + ("|- " if indent > 0 else "")
        label = root.replace('CL_', '')
        print(f"{prefix}{label} (L{self.get_level(root)})")
        for child in self.get_children(root):
            self.print_tree(child, indent + 1, max_depth)


def add_hierarchy_levels_to_df(df: pd.DataFrame, hierarchy: RegionHierarchy,
                                region_col: str = 'Soma_Region',
                                max_level: int = 6) -> pd.DataFrame:
    """Add hierarchy level columns to neuron DataFrame."""
    df = df.copy()
    for level in range(1, max_level + 1):
        col_name = f'Region_L{level}'
        df[col_name] = df[region_col].apply(lambda r: hierarchy.aggregate_to_level(r, level))
    return df


def hierarchy_summary(df: pd.DataFrame, max_level: int = 6) -> pd.DataFrame:
    """Summary statistics across all hierarchy levels."""
    results = []
    for level in range(1, max_level + 1):
        region_col = f'Region_L{level}'
        if region_col not in df.columns:
            continue
        valid = df[region_col].dropna()
        counts = valid.value_counts()
        results.append({
            'Level': level,
            'N_Regions': len(counts),
            'N_Neurons': len(valid),
            'Mean_per_Region': round(counts.mean(), 1) if len(counts) > 0 else 0,
            'Max_per_Region': counts.max() if len(counts) > 0 else 0,
            'Min_per_Region': counts.min() if len(counts) > 0 else 0,
            'Top_Region': counts.index[0] if len(counts) > 0 else None
        })
    return pd.DataFrame(results)


# ==============================================================================
# PROJECTION HIERARCHY - Map projection sites to hierarchy levels
# ==============================================================================

def add_projection_hierarchy(df: pd.DataFrame, hierarchy: RegionHierarchy,
                              max_level: int = 6) -> pd.DataFrame:
    """
    Add hierarchy columns for projection sites (Terminal_Regions).
    
    For each projection site in Terminal_Regions, creates columns:
    - Proj_L1, Proj_L2, ..., Proj_L6: List of regions at each level
    - Proj_Highest: List of highest (finest) level regions for each projection
    
    Args:
        df: DataFrame with 'Terminal_Regions' column (list of regions)
        hierarchy: RegionHierarchy object
        max_level: Maximum hierarchy level to add
    
    Returns:
        DataFrame with additional Proj_L* columns
    """
    df = df.copy()
    
    print(f"[PROJECTION HIERARCHY] Adding Proj_L1 to Proj_L{max_level}...")
    
    # Add projection hierarchy columns
    for level in range(1, max_level + 1):
        col_name = f'Proj_L{level}'
        df[col_name] = df['Terminal_Regions'].apply(
            lambda regions: _map_regions_to_level(regions, hierarchy, level)
        )
    
    # Add highest level column
    df['Proj_Highest'] = df['Terminal_Regions'].apply(
        lambda regions: _get_highest_level_regions(regions, hierarchy, max_level)
    )
    
    # Print summary
    total_projs = df['Terminal_Count'].sum()
    print(f"    Processed {total_projs} projections across {len(df)} neurons")
    
    return df


def _map_regions_to_level(regions: list, hierarchy: RegionHierarchy, 
                          level: int) -> list:
    """Map a list of regions to a specific hierarchy level."""
    if not isinstance(regions, (list, tuple)):
        regions = _parse_terminal_regions(regions)
    
    mapped = []
    for region in regions:
        mapped_region = hierarchy.aggregate_to_level(region, level)
        mapped.append(mapped_region)
    
    return mapped


def _get_highest_level_regions(regions: list, hierarchy: RegionHierarchy,
                                max_level: int = 6) -> list:
    """
    Get the highest (finest) level region for each projection.
    Returns the deepest level available for each region.
    """
    if not isinstance(regions, (list, tuple)):
        regions = _parse_terminal_regions(regions)
    
    highest_regions = []
    for region in regions:
        current_level = hierarchy.get_level(region)
        if current_level == -1:
            # Region not in hierarchy
            highest_regions.append(region)
        else:
            # Get the region at its current level (itself if at max_level or deeper)
            if current_level >= max_level:
                # Already at or below max_level
                highest_regions.append(region)
            else:
                # Get the finest level we can
                finest = hierarchy.aggregate_to_level(region, current_level)
                highest_regions.append(finest if finest else region)
    
    return highest_regions


# ==============================================================================
# HIERARCHY VISUALIZATION (Optional - can be called manually)
# ==============================================================================

def plot_region_distribution(df: pd.DataFrame, level: int, top_n: int = 15,
                              figsize: tuple = (10, 6), save_path: str = None):
    """Bar chart of top N regions at specified hierarchy level."""
    region_col = f'Region_L{level}'
    if region_col not in df.columns:
        print(f"Error: {region_col} not found. Run add_hierarchy_columns first.")
        return None
    counts = df[region_col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", len(counts))
    bars = ax.barh(range(len(counts)), counts.values, color=colors)
    labels = [str(r).replace('CL_', '').replace('CR_', '') for r in counts.index]
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    for i, (bar, v) in enumerate(zip(bars, counts.values)):
        ax.text(v + max(counts) * 0.01, i, f'{v}', va='center', fontsize=9)
    ax.set_xlabel('Neuron Count', fontsize=12)
    ax.set_title(f'Regional Distribution (Level {level})', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(counts) * 1.12)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig


# ==============================================================================
# HELPER: OUTLIER INSPECTOR
# ==============================================================================
def save_debug_snapshot(voxel_coords, neuron_name, template_img, point_type, folder="../resource/debug_outliers"):
    """Saves a snapshot to the resource folder."""
    if template_img is None:
        return
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as e:
            print(f"     [WARN] Could not create folder {folder}: {e}")
            return
    world_coords = nib.affines.apply_affine(template_img.affine, voxel_coords)
    try:
        display = plotting.plot_anat(template_img, cut_coords=world_coords, draw_cross=True,
                                      title=f"Outlier {point_type}: {neuron_name}")
        fname = f"{neuron_name}_{point_type}_ID0_{int(voxel_coords[0])}_{int(voxel_coords[1])}_{int(voxel_coords[2])}.png"
        full_path = os.path.join(folder, fname)
        display.savefig(full_path)
        display.close()
        print(f"     [DEBUG] Snapshot saved: {full_path}")
    except Exception as e:
        print(f"     [DEBUG] Failed to plot snapshot: {e}")


# ==============================================================================
# NEURON CLASSIFIER
# ==============================================================================
class NeuronClassifier:
    def __init__(self, atlas_table):
        try:
            ids = pd.to_numeric(atlas_table['Index'], errors='coerce')
        except KeyError:
            ids = pd.to_numeric(atlas_table.iloc[:, 0], errors='coerce')
        self.name_to_id = dict(zip(atlas_table['Abbreviation'], ids))
        if 'Full_Name' in atlas_table.columns:
            self.name_to_id.update(dict(zip(atlas_table['Full_Name'], ids)))

    def _get_detailed_category(self, region_identifier):
        rid = self.name_to_id.get(region_identifier)
        if rid is None:
            return 'Other'
        rid = int(rid)
        # PT TARGETS (Brainstem + Hypothalamus)
        if (1169 <= rid <= 1325) or (1669 <= rid <= 1825):
            return 'PT_Target'
        if (1083 <= rid <= 1107) or (1583 <= rid <= 1607):
            return 'PT_Target'
        # CT TARGETS (Thalamus Only)
        if (1111 <= rid <= 1168) or (1611 <= rid <= 1668):
            return 'Thalamus'
        # STRIATUM
        if (1051 <= rid <= 1061) or (1551 <= rid <= 1561):
            return 'Striatum'
        # CORTEX
        if 1 <= rid <= 500:
            return 'Cortex_L'
        if 501 <= rid <= 1000:
            return 'Cortex_R'
        return 'Other'

    def classify_single_neuron(self, terminal_list, soma_region):
        soma_side = 'Unknown'
        if 'CL_' in soma_region or 'SL_' in soma_region:
            soma_side = 'L'
        elif 'CR_' in soma_region or 'SR_' in soma_region:
            soma_side = 'R'
        is_PT = False
        is_CT = False
        has_striatum = False
        has_contra_cortex = False
        has_ipsi_cortex = False
        for t_name in terminal_list:
            cat = self._get_detailed_category(t_name)
            if cat == 'PT_Target':
                is_PT = True
            elif cat == 'Thalamus':
                is_CT = True
            elif cat == 'Striatum':
                has_striatum = True
            elif cat == 'Cortex_L':
                if soma_side == 'L':
                    has_ipsi_cortex = True
                elif soma_side == 'R':
                    has_contra_cortex = True
            elif cat == 'Cortex_R':
                if soma_side == 'R':
                    has_ipsi_cortex = True
                elif soma_side == 'L':
                    has_contra_cortex = True
        # Apply classification hierarchy
        if is_PT:
            return 'PT'
        if is_CT:
            return 'CT'
        if has_striatum:
            return 'ITs'
        if has_contra_cortex:
            return 'ITc'
        if has_ipsi_cortex:
            return 'ITi'
        return 'Unclassified'


# ==============================================================================
# PER NEURON ANALYSIS
# ==============================================================================
class region_analysis_per_neuron:
    def __init__(self, neuron_tracer_obj, atlas_volume, atlas_table):
        self.neuron = neuron_tracer_obj
        self.atlas = atlas_volume
        self.atlas_table = atlas_table
        self.brain_region_map = {row['Index']: row['Abbreviation'] for _, row in self.atlas_table.iterrows()}

    def region_analysis(self):
        self.mapped_brain_region_lengths, self.neuron_total_length = self._calculate_neuronal_branch_length()
        self.soma_region, self.terminal_regions = self._soma_and_terminal_region()

    def _distance(self, p1, p2):
        return np.sqrt((p1.x_nii - p2.x_nii)**2 + (p1.y_nii - p2.y_nii)**2 + (p1.z_nii - p2.z_nii)**2)

    def _calculate_neuronal_branch_length(self):
        brain_region_lengths = defaultdict(float)
        neuron_total_length = 0
        for branch in self.neuron.branches:
            for i in range(len(branch) - 1):
                current_node = branch[i]
                child_node = branch[i+1]
                edge_length = round(self._distance(current_node, child_node), 3)
                aa = tuple(np.round(np.array([current_node.x_nii, current_node.y_nii, current_node.z_nii])).astype(int).flatten())
                neuron_total_length += edge_length
                if (0 <= aa[0] < self.atlas.shape[0] and 0 <= aa[1] < self.atlas.shape[1] and 0 <= aa[2] < self.atlas.shape[2]):
                    brain_region_id = int(self.atlas[aa])
                    if brain_region_id > 0:
                        brain_region_lengths[brain_region_id] += edge_length
        mapped = {}
        for index, length in brain_region_lengths.items():
            region_name = self.brain_region_map.get(index, f'Unknown_{index}')
            mapped[region_name] = length
        return mapped, neuron_total_length

    def _soma_and_terminal_region(self):
        soma = self.neuron.root
        soma_pos = tuple(np.round(np.array([soma.x_nii, soma.y_nii, soma.z_nii])).astype(int).flatten())
        if (0 <= soma_pos[0] < self.atlas.shape[0] and 0 <= soma_pos[1] < self.atlas.shape[1] and 0 <= soma_pos[2] < self.atlas.shape[2]):
            s_id = int(self.atlas[soma_pos])
            soma_region = self.brain_region_map.get(s_id, f'Unknown_{s_id}')
        else:
            soma_region = "Out_of_Bounds"
        terminal_regions = []
        for node in self.neuron.terminal_nodes:
            node_pos = tuple(np.round(np.array([node.x_nii, node.y_nii, node.z_nii])).astype(int).flatten())
            if (0 <= node_pos[0] < self.atlas.shape[0] and 0 <= node_pos[1] < self.atlas.shape[1] and 0 <= node_pos[2] < self.atlas.shape[2]):
                t_id = int(self.atlas[node_pos])
                t_name = self.brain_region_map.get(t_id, f'Unknown_{t_id}')
                terminal_regions.append({'region': t_name, 'coords': node_pos})
        return soma_region, terminal_regions


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def _parse_terminal_regions(x):
    """Parse Terminal_Regions from various formats to list of UNIQUE regions."""
    if isinstance(x, (list, tuple)):
        seen = set()
        unique = []
        for item in x:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique
    if isinstance(x, str):
        import ast
        try:
            if x.startswith('[') and x.endswith(']'):
                parsed = ast.literal_eval(x)
                seen = set()
                unique = []
                for item in parsed:
                    if item not in seen:
                        seen.add(item)
                        unique.append(item)
                return unique
            if ',' in x:
                parts = [p.strip().strip("'\"") for p in x.split(',')]
                seen = set()
                unique = []
                for item in parts:
                    if item not in seen:
                        seen.add(item)
                        unique.append(item)
                return unique
            return [x.strip().strip("'\"")]
        except:
            return [x]
    return []


# ==============================================================================
# POPULATION MANAGER
# ==============================================================================
class PopulationRegionAnalysis:
    def __init__(self, sample_id, atlas, atlas_table, template_img=None, nii_space='monkey',
                 arm_key_path: str = None, auto_hierarchy: bool = True,
                 output_base: str = None, create_output_folder: bool = False):
        """
        Initialize PopulationRegionAnalysis.
        
        Args:
            sample_id: Sample ID to analyze
            atlas: Atlas volume data
            atlas_table: Atlas region table
            template_img: Template image for visualization
            nii_space: NII space identifier
            arm_key_path: Path to ARM_key file for hierarchy. If None, will auto-detect.
            auto_hierarchy: If True, automatically add hierarchy columns during process()
            output_base: Base path for organized output folders (NEW in v3.5)
            create_output_folder: If True, create organized output folders (NEW in v3.5)
        """
        self.sample_id = sample_id
        self.full_atlas = atlas
        self.atlas_table = atlas_table
        self.template_img = template_img
        self.nii_space = nii_space
        self.neuron_list = iondata.getNeuronListBySampleID(sample_id)
        self.classifier = NeuronClassifier(atlas_table)
        self.plot_dataframe = pd.DataFrame()
        self.neurons = {}
        
        # Hierarchy system (v3.4)
        self.auto_hierarchy = auto_hierarchy
        self.hierarchy = None
        self._arm_key_path = arm_key_path
        
        # Output manager (NEW in v3.5)
        self.output = None
        if create_output_folder:
            self.output = OutputManager(
                base_path=output_base,
                sample_id=sample_id,
                create_timestamp=True
            )
        
        # Load hierarchy
        if arm_key_path:
            self._load_hierarchy(arm_key_path)
        elif auto_hierarchy:
            self._detect_and_load_hierarchy()
        
        # Print hierarchy status
        if self.hierarchy:
            print(f"[INIT] Hierarchy loaded: {len(self.hierarchy.level_map)} regions")
        else:
            print("[INIT] Hierarchy not loaded (arm_key not found)")

    def _detect_and_load_hierarchy(self):
        """Auto-detect ARM key path and load hierarchy."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, '..', 'atlas', 'ARM_key_all.txt'),
            os.path.join(script_dir, 'atlas', 'ARM_key_all.txt'),
            r'D:\projectome_analysis\atlas\ARM_key_all.txt',
            r'D:\projectome_analysis\neuron_tables\arm_key.xlsx',
            r'D:\projectome_analysis\neuron_tables\ARM_key_all.txt',
            '../atlas/ARM_key_all.txt',
        ]
        print("[HIERARCHY] Auto-detecting ARM key file...")
        for path in possible_paths:
            print(f"  Checking: {path} ... {'FOUND' if os.path.exists(path) else 'not found'}")
            if os.path.exists(path):
                self._load_hierarchy(path)
                return
        print("[WARN] ARM key not auto-detected. Hierarchy not loaded.")
        print("[INFO] To use hierarchy features, provide arm_key_path when initializing:")
        print("  pop = PopulationRegionAnalysis(..., arm_key_path=r'D:\\path\\to\\arm_key.xlsx')")

    def _load_hierarchy(self, arm_key_path: str):
        """Load hierarchy from file."""
        try:
            print(f"[HIERARCHY] Loading from: {arm_key_path}")
            self.hierarchy = RegionHierarchy.from_file(arm_key_path)
            self._arm_key_path = arm_key_path
            print(f"[HIERARCHY] Ready with {len(self.hierarchy.level_map)} regions")
        except Exception as e:
            print(f"[HIERARCHY] Failed to load: {e}")
            self.hierarchy = None

    def process(self, limit=None, level=6, neuron_id=None, add_hierarchy: bool = None):
        """
        Process neurons and generate analysis dataframe.
        
        Args:
            limit: Maximum number of neurons to process
            level: Atlas hierarchy level (for 5D atlases)
            neuron_id: Specific neuron ID to process
            add_hierarchy: If True, add hierarchy columns. If None, uses auto_hierarchy setting.
        """
        if self.full_atlas.ndim == 5:
            current_atlas = self.full_atlas[:, :, :, 0, level-1]
        else:
            current_atlas = self.full_atlas

        if neuron_id:
            target_list = [n for n in self.neuron_list if n['name'] == neuron_id]
            if not target_list:
                print(f"Error: Neuron '{neuron_id}' not found.")
                return
        elif limit:
            target_list = self.neuron_list[:limit]
        else:
            target_list = self.neuron_list

        print(f"Processing {len(target_list)} neurons at Level {level}...")

        data = []
        for target_neuron in target_list:
            print(f"  -> {target_neuron['name']}")
            neuron = nt.neuro_tracer()
            try:
                neuron.process(target_neuron['sampleid'], target_neuron['name'], nii_space=self.nii_space)
            except:
                print("     Load failed.")
                continue

            analysis = region_analysis_per_neuron(neuron, current_atlas, self.atlas_table)
            analysis.region_analysis()
            
            neuron.outliers = []
            if "Unknown" in analysis.soma_region:
                root = neuron.root
                coords = (int(root.x_nii), int(root.y_nii), int(root.z_nii))
                neuron.outliers.append({'type': 'Soma', 'region': analysis.soma_region, 'coords': coords})
            
            for item in analysis.terminal_regions:
                t_name = item['region']
                coords = item['coords']
                if "Unknown" in t_name:
                    neuron.outliers.append({'type': 'Terminal', 'region': t_name, 'coords': coords})

            self.neurons[neuron.swc_filename] = neuron

            term_name_list_all = [item['region'] for item in analysis.terminal_regions]
            seen = set()
            term_name_list = []
            for region in term_name_list_all:
                if region not in seen:
                    seen.add(region)
                    term_name_list.append(region)

            n_type = self.classifier.classify_single_neuron(term_name_list, analysis.soma_region)
            
            # Get soma coordinates from root node
            root = neuron.root
            row = {
                'SampleID': self.sample_id,
                'NeuronID': neuron.swc_filename,
                'Neuron_Type': n_type,
                'Soma_Region': analysis.soma_region,
                'Soma_NII_X': round(root.x_nii, 4),
                'Soma_NII_Y': round(root.y_nii, 4),
                'Soma_NII_Z': round(root.z_nii, 4),
                'Soma_Phys_X': round(root.x, 4),
                'Soma_Phys_Y': round(root.y, 4),
                'Soma_Phys_Z': round(root.z, 4),
                'Total_Length': analysis.neuron_total_length,
                'Terminal_Count': len(term_name_list),
                'Terminal_Regions': term_name_list,
                'Region_projection_length': analysis.mapped_brain_region_lengths,
                'Outlier_Count': len(neuron.outliers),
                'Outlier_Details': neuron.outliers
            }
            data.append(row)

        self.plot_dataframe = pd.DataFrame(data)
        
        # AUTO ADD HIERARCHY COLUMNS (v3.4)
        should_add_hierarchy = add_hierarchy if add_hierarchy is not None else self.auto_hierarchy
        if should_add_hierarchy and self.hierarchy is not None and not self.plot_dataframe.empty:
            print("\n[HIERARCHY] Adding hierarchy columns...")
            self._apply_hierarchy_columns()

    def _apply_hierarchy_columns(self, max_level: int = 6):
        """Internal method to add hierarchy columns to plot_dataframe."""
        if self.hierarchy is None:
            print("[HIERARCHY] No hierarchy loaded. Cannot add columns.")
            return
        
        self.plot_dataframe = add_hierarchy_levels_to_df(
            self.plot_dataframe,
            self.hierarchy,
            region_col='Soma_Region',
            max_level=max_level
        )
        
        summary = hierarchy_summary(self.plot_dataframe, max_level)
        print("\n[HIERARCHY] Summary:")
        print(summary.to_string(index=False))

    def add_hierarchy_columns(self, arm_key_path: str = None, max_level: int = 6) -> pd.DataFrame:
        """
        Add hierarchy level columns (Region_L1 through Region_L6) to the dataframe.
        This is now automatically called during process() if auto_hierarchy=True.
        """
        if self.plot_dataframe.empty:
            print("Error: No data. Run process() or load_processed_dataframe() first.")
            return None
        
        if arm_key_path and (self._arm_key_path != arm_key_path or self.hierarchy is None):
            self._load_hierarchy(arm_key_path)
        
        if self.hierarchy is None:
            self._detect_and_load_hierarchy()
        
        if self.hierarchy is None:
            raise FileNotFoundError("ARM key not found. Please specify arm_key_path.")
        
        self._apply_hierarchy_columns(max_level)
        return self.plot_dataframe


    # ==========================================================================
    # SAVE METHODS (NEW in v3.5)
    # ==========================================================================
    
    def save_results(self, filename: str = None, include_details: bool = False):
        """
        Save DataFrame to Excel in organized tables folder.
        
        Args:
            filename: Output filename (default: {sample_id}_results.xlsx)
            include_details: Include Outlier_Details column
        """
        if self.plot_dataframe.empty:
            print("[ERROR] No data to save.")
            return
        
        if self.output is None:
            print("[WARN] Output manager not initialized. Set create_output_folder=True.")
            # Fallback: save to current directory
            filename = filename or f"{self.sample_id}_results.xlsx"
            path = Path(filename)
        else:
            filename = filename or f"{self.sample_id}_results"
            path = self.output.get_table_path(filename)
        
        df_out = self.plot_dataframe.copy()
        
        # Convert complex columns to strings for Excel
        list_columns = ['Terminal_Regions', 'Region_projection_length', 'Outlier_Details']
        # Also convert projection hierarchy columns (Proj_L1, Proj_L2, etc.)
        proj_cols = [c for c in df_out.columns if c.startswith('Proj_')]
        list_columns.extend(proj_cols)
        
        for col in list_columns:
            if col in df_out.columns:
                df_out[col] = df_out[col].apply(str)
        
        if not include_details and 'Outlier_Details' in df_out.columns:
            df_out = df_out.drop(columns=['Outlier_Details'])
        
        df_out.to_excel(path, index=False)
        print(f"[SAVED] {path}")
    
    def save_plot(self, fig, name: str):
        """Save figure to organized plots folder."""
        if self.output is None:
            plt.show()
            return
        
        path = self.output.get_plot_path(name)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {path}")
        plt.close(fig)
    
    def save_report(self, content: str, name: str):
        """Save text report to organized reports folder."""
        if self.output is None:
            print(content)
            return
        
        path = self.output.get_report_path(name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[SAVED] {path}")
    
    def save_all(self, include_details: bool = False):
        """
        Save everything: DataFrame and summary report to organized folders.
        
        Folder structure:
            output/
            └── {sample_id}_{timestamp}/
                ├── tables/
                │   ├── {sample_id}_results.xlsx
                │   └── {sample_id}_soma_coordinates.xlsx
                └── reports/
                    └── analysis_summary.txt
        
        Args:
            include_details: Include Outlier_Details in main table
        
        Returns:
            Path to output folder or None if output manager not initialized
        """
        if self.plot_dataframe.empty:
            print("[ERROR] No data to save. Run process() first.")
            return None
        
        if self.output is None:
            print("[WARN] Output manager not initialized. Set create_output_folder=True.")
            # Still save to current directory
            self.save_results(include_details=include_details)
            return None
        
        print("\n" + "="*60)
        print(f"SAVING ALL RESULTS: {self.sample_id}")
        print("="*60)
        
        # Save main table
        print("\n[TABLES]")
        self.save_results(include_details=include_details)
        
        # Save soma coordinates separately
        coord_path = self.output.get_table_path(f"{self.sample_id}_soma_coordinates")
        coord_cols = ['NeuronID', 'Neuron_Type', 'Soma_Region',
                      'Soma_NII_X', 'Soma_NII_Y', 'Soma_NII_Z',
                      'Soma_Phys_X', 'Soma_Phys_Y', 'Soma_Phys_Z']
        self.plot_dataframe[coord_cols].to_excel(coord_path, index=False)
        print(f"[SAVED] {coord_path}")
        
        # Save summary report
        print("\n[REPORTS]")
        self._save_summary_report()
        
        # Print summary
        self.output.print_summary()
        
        return str(self.output.output_dir)
    
    def add_projection_hierarchy_columns(self, max_level: int = 6, 
                                          arm_key_path: str = None) -> pd.DataFrame:
        """
        Add hierarchy columns for projection sites (Terminal_Regions).
        
        Adds columns:
        - Proj_L1 to Proj_L6: List of regions at each hierarchy level
        - Proj_Highest: List of highest (finest) level regions
        
        Args:
            max_level: Maximum hierarchy level to add
            arm_key_path: Optional path to ARM key file (if not already loaded)
        
        Returns:
            DataFrame with projection hierarchy columns
        """
        if self.plot_dataframe.empty:
            print("[ERROR] No data. Run process() first.")
            return None
        
        # Try to load hierarchy if not already loaded
        if self.hierarchy is None:
            if arm_key_path:
                print(f"[INFO] Loading hierarchy from: {arm_key_path}")
                self._load_hierarchy(arm_key_path)
            else:
                print("[INFO] Hierarchy not loaded. Attempting auto-detection...")
                self._detect_and_load_hierarchy()
        
        if self.hierarchy is None:
            print("[ERROR] No hierarchy loaded. Cannot add projection hierarchy columns.")
            print("[SOLUTION] Provide arm_key_path parameter:")
            print("  pop.add_projection_hierarchy_columns(arm_key_path=r'D:\\path\\to\\arm_key.xlsx')")
            return None
        
        self.plot_dataframe = add_projection_hierarchy(
            self.plot_dataframe, self.hierarchy, max_level
        )
        
        print("\n[PROJECTION HIERARCHY] Columns added:")
        proj_cols = [c for c in self.plot_dataframe.columns if c.startswith('Proj_')]
        for col in proj_cols:
            print(f"  - {col}")
        
        return self.plot_dataframe
    
    def _save_summary_report(self):
        """Generate and save summary report."""
        df = self.plot_dataframe
        
        lines = [
            "="*60,
            f"REGION ANALYSIS REPORT - {self.sample_id}",
            "="*60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "--- SUMMARY ---",
            f"Total neurons: {len(df)}",
            "",
            "--- Neuron Type Distribution ---"
        ]
        
        if 'Neuron_Type' in df.columns:
            type_counts = df['Neuron_Type'].value_counts()
            for ntype, count in type_counts.items():
                lines.append(f"  {ntype}: {count} ({count/len(df)*100:.1f}%)")
        
        lines.extend([
            "",
            "--- Soma Region Distribution (Top 10) ---"
        ])
        
        soma_counts = df['Soma_Region'].value_counts().head(10)
        for region, count in soma_counts.items():
            lines.append(f"  {region}: {count}")
        
        # Add projection hierarchy info if available
        proj_cols = [c for c in df.columns if c.startswith('Proj_L') and not c.startswith('Proj_Highest')]
        if proj_cols:
            lines.extend(["", "--- Projection Hierarchy ---"])
            lines.append(f"Available levels: {', '.join(sorted(proj_cols))}")
            if 'Proj_Highest' in df.columns:
                lines.append("Proj_Highest: Deepest level region for each projection")
        
        lines.extend([
            "",
            "--- Statistics ---",
            f"Total length - Mean: {df['Total_Length'].mean():.2f}, Std: {df['Total_Length'].std():.2f}",
            f"Terminal count - Mean: {df['Terminal_Count'].mean():.2f}, Max: {df['Terminal_Count'].max()}",
            f"Neurons with outliers: {(df['Outlier_Count'] > 0).sum()}",
            "",
            "="*60
        ])
        
        report = "\n".join(lines)
        self.save_report(report, 'analysis_summary')
        print(report)

    # ==========================================================================
    # EXISTING METHODS (unchanged)
    # ==========================================================================
    
    def get_region_matrix(self):
        """Get projection length matrix."""
        if self.plot_dataframe.empty:
            return pd.DataFrame()
        dict_list = self.plot_dataframe['Region_projection_length'].tolist()
        matrix = pd.DataFrame(dict_list)
        matrix.fillna(0, inplace=True)
        matrix.insert(0, 'NeuronID', self.plot_dataframe['NeuronID'])
        matrix.insert(1, 'Neuron_Type', self.plot_dataframe['Neuron_Type'])
        return matrix

    def export_outlier_snapshots(self, neuron_id, max_snapshots=3):
        """Generates images for outliers stored in a specific neuron object."""
        if neuron_id not in self.neurons:
            print(f"Neuron {neuron_id} not loaded.")
            return
        neuron = self.neurons[neuron_id]
        if not neuron.outliers:
            print(f"No outliers found for {neuron_id}.")
            return
        print(f"Exporting outliers for {neuron_id} (Limit: {max_snapshots})...")
        count = 0
        for error in neuron.outliers:
            if count >= max_snapshots:
                print(f"     [INFO] Max snapshots ({max_snapshots}) reached. Stopping.")
                break
            save_debug_snapshot(voxel_coords=error['coords'], neuron_name=neuron.swc_filename,
                                template_img=self.template_img, point_type=error['type'])
            count += 1

    def inspect_neuron(self, target_filename):
        """Display detailed report for a single neuron."""
        if self.plot_dataframe.empty:
            print("Error: No data. Run process() first.")
            return
        mask = self.plot_dataframe['NeuronID'] == target_filename
        if not mask.any():
            print(f"Error: '{target_filename}' not found.")
            return
        row = self.plot_dataframe[mask].iloc[0]
        
        print(f"\nREPORT: {row['NeuronID']}")
        print(f"  Type: {row['Neuron_Type']} | Soma: {row['Soma_Region']}")
        
        # Show hierarchy levels if available
        hierarchy_cols = [c for c in self.plot_dataframe.columns if c.startswith('Region_L')]
        if hierarchy_cols:
            hierarchy_info = " | ".join([f"{c}: {row[c]}" for c in sorted(hierarchy_cols) if pd.notna(row[c])])
            print(f"  Hierarchy: {hierarchy_info}")
        
        print(f"  Soma NII: ({row['Soma_NII_X']}, {row['Soma_NII_Y']}, {row['Soma_NII_Z']})")
        print(f"  Soma Phys: ({row['Soma_Phys_X']}, {row['Soma_Phys_Y']}, {row['Soma_Phys_Z']})")
        print(f"  Length: {row['Total_Length']:.3f}")
        print(f"  Targets: {row['Terminal_Regions']}")
        
        region_dict = row['Region_projection_length']
        if not region_dict or sum(region_dict.values()) == 0:
            return
        
        stats = pd.Series(region_dict).sort_values(ascending=False)
        stats = stats[stats > 0]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        top_n = stats.head(10)
        sns.barplot(x=top_n.values, y=top_n.index, ax=axes[0], palette='viridis')
        axes[0].set_title(f"Top Projections ({row['NeuronID']})")
        
        if len(stats) > 6:
            main = stats.head(6)
            other = pd.Series({'Others': stats.iloc[6:].sum()})
            pie_data = pd.concat([main, other])
        else:
            pie_data = stats
        axes[1].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
        axes[1].set_title("Distribution")
        plt.tight_layout()
        plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    """Example: Projection hierarchy analysis with organized output."""
    
    # Configuration
    ATLAS_PATH = r'D:\projectome_analysis\atlas\ARM_in_NMT_v2.1_sym.nii.gz'
    TABLE_PATH = r'D:\projectome_analysis\atlas\ARM_key_all.txt'
    ARM_KEY_PATH = TABLE_PATH
    SAMPLE_ID = '251637'
    
    # Load data
    atlas_nii = nib.load(ATLAS_PATH)
    atlas_data = atlas_nii.get_fdata()
    atlas_table = pd.read_csv(TABLE_PATH, delimiter='\t')
    
    # Initialize with hierarchy and organized output
    pop = PopulationRegionAnalysis(
        sample_id=SAMPLE_ID,
        atlas=atlas_data,
        atlas_table=atlas_table,
        arm_key_path=ARM_KEY_PATH,  # Load hierarchy for projections
        create_output_folder=True
    )
    
    # Process neurons
    pop.process(limit=5, level=6)
    
    # Add projection hierarchy columns
    # This adds: Proj_L1, Proj_L2, ..., Proj_L6, Proj_Highest
    pop.add_projection_hierarchy_columns(max_level=6)
    
    # Show example
    print("\n=== Example: Projection Hierarchy ===")
    row = pop.plot_dataframe.iloc[0]
    print(f"Neuron: {row['NeuronID']}")
    print(f"Terminal_Regions: {row['Terminal_Regions']}")
    print(f"Proj_L1 (coarsest): {row['Proj_L1']}")
    print(f"Proj_L3 (intermediate): {row['Proj_L3']}")
    print(f"Proj_L6 (finest): {row['Proj_L6']}")
    print(f"Proj_Highest: {row['Proj_Highest']}")
    
    # Save all results
    output_path = pop.save_all()
    print(f"\n[COMPLETE] Results saved to: {output_path}")
    
    # Folder structure:
    # output/
    # └── 251637_20250304_081245/
    #     ├── tables/
    #     │   ├── 251637_results.xlsx      (includes Proj_L1-6, Proj_Highest)
    #     │   └── 251637_soma_coordinates.xlsx
    #     └── reports/
    #         └── analysis_summary.txt
