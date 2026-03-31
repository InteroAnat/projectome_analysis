#!/usr/bin/env python3
"""
Brain Visualization Toolkit
===========================
Simple 3D visualization of brain meshes and neurons with mesh generation.

Based on vs_brain.py and step_by_step_insula_flatmap.py
Uses scatter plots for reliable display.

QUICK START:
    from brain_viz import BrainViz, RegionExtractor
    import pandas as pd
    
    # Initialize
    viz = BrainViz(nmt_dir=r"path/to/NMT")
    viz.load_brain_mesh()
    
    # Generate region meshes from ARM atlas
    extractor = RegionExtractor(nmt_dir=r"path/to/NMT")
    regions = extractor.extract_regions(
        atlas_path=r"path/to/ARM_atlas.nii.gz",
        key_path=r"path/to/ARM_key.txt",
        region_dict={
            'left_insula': ['CL_Ig', 'CL_Ial', 'CL_Iai'],
            'right_insula': ['CR_Ig', 'CR_Ial', 'CR_Iai']
        }
    )
    
    # Add generated meshes to viz
    viz.add_mesh_from_array('left_insula', regions['left_insula'], color='red')
    viz.add_mesh_from_array('right_insula', regions['right_insula'], color='blue')
    
    # Load neurons and plot
    df = pd.read_excel("neurons.xlsx")
    viz.load_neurons(df, type_col='Neuron_Type')
    viz.plot()

CUSTOMIZABLE ELEMENTS:
----------------------
Neuron Appearance:
- viz.neuron_colors: dict mapping types to colors
- viz.neuron_sizes: dict mapping types to sizes
- viz.neuron_alpha: transparency (default: 0.9)
- viz.neuron_edgecolors: edge color (default: 'black')
- viz.neuron_linewidth: edge width (default: 0.5)

Mesh Appearance:
- viz.mesh_colors: dict mapping mesh names to colors
- viz.mesh_sizes: dict mapping mesh names to point sizes
- viz.mesh_alphas: dict mapping mesh names to transparency
- viz.mesh_downsample: vertex skip factor (default: 1)

View Settings:
- viz.figsize: figure size (default: (12, 10))
- viz.elev: camera elevation (default: 30)
- viz.azim: camera azimuth (default: 60)
- viz.background_color: background color (default: 'white')
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from skimage import measure
from typing import Dict, List, Optional, Tuple, Union


# Default neuron type colors
NEURON_COLORS = {
    'PT': '#d62728',
    'CT': '#2ca02c',
    'ITs': '#9467bd',
    'ITc': '#e377c2',
    'ITi': '#17becf',
    'Unclassified': 'gray'
}


class BrainViz:
    """Brain mesh and neuron visualizer."""
    
    def __init__(self, nmt_dir: str, output_dir: str = "./output"):
        """
        Initialize visualizer.
        
        Args:
            nmt_dir: Path to NMT template directory
            output_dir: Directory for saving outputs
        """
        self.nmt_dir = nmt_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage: {name: {'verts': Nx3 array}}
        self.meshes = {}
        
        # Storage: {name: DataFrame with x, y, z, type columns}
        self.neurons = {}
        
        # Neuron appearance
        self.neuron_colors = NEURON_COLORS.copy()
        self.neuron_sizes = {}
        self.neuron_alpha = 0.9
        self.neuron_edgecolors = 'black'
        self.neuron_linewidth = 0.5
        
        # Mesh appearance
        self.mesh_colors = {}
        self.mesh_sizes = {}
        self.mesh_alphas = {}
        self.mesh_downsample = 1
        
        # View settings
        self.figsize = (12, 10)
        self.elev = 30
        self.azim = 60
        self.background_color = 'white'
        self.legend_markerscale = 2.0
    
    # ==================== MESH LOADING ====================
    
    def load_brain_mesh(self, step_size: int = 3, 
                       color: str = 'black',
                       size: float = 0.2,
                       alpha: float = 0.1):
        """
        Load brain surface mesh from NMT template.
        
        Args:
            step_size: Marching cubes step size (higher = sparser)
            color: Mesh point color
            size: Mesh point size
            alpha: Mesh transparency
        """
        seg_path = os.path.join(self.nmt_dir, "NMT_v2.1_sym_segmentation.nii.gz")
        print(f"Loading brain mesh from: {seg_path}")
        
        img = nib.load(seg_path)
        data = img.get_fdata()
        brain_mask = data > 0
        
        verts, _, _, _ = measure.marching_cubes(brain_mask, level=0.5, step_size=step_size)
        
        self.meshes['brain'] = {'verts': verts}
        self.mesh_colors['brain'] = color
        self.mesh_sizes['brain'] = size
        self.mesh_alphas['brain'] = alpha
        
        print(f"  Loaded: {len(verts)} vertices")
        return verts
    
    def add_mesh_from_array(self, name: str, verts: np.ndarray,
                           color: str = 'blue', size: float = 3, alpha: float = 0.05):
        """
        Add mesh from vertex array (e.g., from RegionExtractor).
        
        Args:
            name: Mesh name
            verts: Nx3 array of vertices
            color: Mesh color
            size: Point size
            alpha: Transparency
        """
        self.meshes[name] = {'verts': verts}
        self.mesh_colors[name] = color
        self.mesh_sizes[name] = size
        self.mesh_alphas[name] = alpha
        print(f"  Added mesh '{name}': {len(verts)} vertices")
        return verts
    
    def add_region(self, name: str, mask: np.ndarray,
                   color: str = 'blue', size: float = 3, alpha: float = 0.05):
        """
        Add region mesh from binary mask using marching cubes.
        
        Args:
            name: Region name
            mask: Binary 3D array
            color: Region color
            size: Point size
            alpha: Transparency
        """
        if mask.sum() < 100:
            print(f"  Skipping {name}: too few voxels ({mask.sum()})")
            return None
        
        try:
            verts, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, method='lorensen'
            )
            return self.add_mesh_from_array(name, verts, color, size, alpha)
        except Exception as e:
            print(f"  Failed to add '{name}': {e}")
            return None
    
    def add_mesh_from_file(self, name: str, pkl_path: str,
                          color: str = 'blue', size: float = 3, alpha: float = 0.05):
        """
        Load mesh from pickle file (e.g., from flatmap pipeline).
        
        Args:
            name: Mesh name
            pkl_path: Path to pickle file containing mesh object
            color: Mesh color
            size: Point size
            alpha: Transparency
        """
        import joblib
        try:
            mesh_data = joblib.load(pkl_path)
            if isinstance(mesh_data, dict) and 'mesh' in mesh_data:
                mesh = mesh_data['mesh']
            else:
                mesh = mesh_data
            
            verts = mesh.vertices if hasattr(mesh, 'vertices') else mesh['vertices']
            return self.add_mesh_from_array(name, verts, color, size, alpha)
        except Exception as e:
            print(f"  Failed to load '{name}': {e}")
            return None
    
    def remove_mesh(self, name: str):
        """Remove a mesh by name."""
        if name in self.meshes:
            del self.meshes[name]
            print(f"Removed mesh '{name}'")
    
    def list_meshes(self) -> List[str]:
        """List all loaded meshes."""
        return list(self.meshes.keys())
    
    # ==================== NEURON LOADING ====================
    
    def load_neurons(self, df: pd.DataFrame,
                    x: str = 'Soma_NII_X',
                    y: str = 'Soma_NII_Y',
                    z: str = 'Soma_NII_Z',
                    type_col: Optional[str] = 'Neuron_Type',
                    name: str = 'neurons'):
        """
        Load neurons from DataFrame.
        
        Args:
            df: DataFrame with neuron data
            x, y, z: Column names for coordinates
            type_col: Column name for neuron type (None if no types)
            name: Name for this neuron group
        """
        # Validate columns
        for col in [x, y, z]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
        
        # Build data dict
        data = {
            'x': df[x].values,
            'y': df[y].values,
            'z': df[z].values
        }
        
        if type_col and type_col in df.columns:
            data['type'] = df[type_col].values
        else:
            data['type'] = ['all'] * len(df)
        
        self.neurons[name] = pd.DataFrame(data)
        print(f"Loaded '{name}': {len(df)} neurons")
        print(f"  Types: {self.neurons[name]['type'].unique().tolist()}")
    
    def filter_neurons(self, group: str,
                      types: Optional[List[str]] = None,
                      x_range: Optional[Tuple[float, float]] = None,
                      y_range: Optional[Tuple[float, float]] = None,
                      z_range: Optional[Tuple[float, float]] = None,
                      new_name: str = 'filtered'):
        """
        Filter neurons and create new group.
        
        Args:
            group: Source group name
            types: List of types to keep
            x_range, y_range, z_range: Coordinate ranges
            new_name: Name for filtered group
        """
        if group not in self.neurons:
            raise ValueError(f"Unknown group: {group}")
        
        df = self.neurons[group].copy()
        
        if types:
            df = df[df['type'].isin(types)]
        if x_range:
            df = df[(df['x'] >= x_range[0]) & (df['x'] <= x_range[1])]
        if y_range:
            df = df[(df['y'] >= y_range[0]) & (df['y'] <= y_range[1])]
        if z_range:
            df = df[(df['z'] >= z_range[0]) & (df['z'] <= z_range[1])]
        
        self.neurons[new_name] = df
        print(f"Created '{new_name}': {len(df)} neurons")
        return df
    
    def list_neurons(self) -> List[str]:
        """List all neuron groups."""
        return list(self.neurons.keys())
    
    # ==================== PLOTTING ====================
    
    def set_view(self, elev: float = 30, azim: float = 60):
        """Set camera angle."""
        self.elev = elev
        self.azim = azim
    
    def plot(self, 
            meshes: Optional[List[str]] = None,
            neurons: Optional[List[str]] = None,
            title: str = "Brain Visualization",
            show_legend: bool = True,
            save_path: Optional[str] = None):
        """
        Create 3D plot.
        
        Args:
            meshes: List of mesh names to plot (None = all)
            neurons: List of neuron groups to plot (None = all)
            title: Plot title
            show_legend: Show legend
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=self.figsize, facecolor=self.background_color)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor(self.background_color)
        
        all_points = []
        
        # Plot meshes
        mesh_names = meshes if meshes else list(self.meshes.keys())
        for name in mesh_names:
            if name not in self.meshes:
                print(f"Warning: mesh '{name}' not found")
                continue
            
            mesh = self.meshes[name]
            verts = mesh['verts'][::self.mesh_downsample]
            
            color = self.mesh_colors.get(name, 'gray')
            size = self.mesh_sizes.get(name, 1)
            alpha = self.mesh_alphas.get(name, 0.1)
            
            ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                      c=color, s=size, alpha=alpha, label=name)
            
            all_points.append(verts)
        
        # Plot neurons
        neuron_names = neurons if neurons else list(self.neurons.keys())
        for name in neuron_names:
            if name not in self.neurons:
                print(f"Warning: neurons '{name}' not found")
                continue
            
            df = self.neurons[name]
            
            for ntype in df['type'].unique():
                subset = df[df['type'] == ntype]
                if len(subset) == 0:
                    continue
                
                color = self.neuron_colors.get(ntype, 'gray')
                size = self.neuron_sizes.get(ntype, 50)
                
                ax.scatter(subset['x'], subset['y'], subset['z'],
                          c=color, s=size, alpha=self.neuron_alpha,
                          edgecolors=self.neuron_edgecolors,
                          linewidth=self.neuron_linewidth,
                          label=f"{ntype} (n={len(subset)})")
                
                all_points.append(subset[['x', 'y', 'z']].values)
        
        # Set aspect ratio
        if all_points:
            all_coords = np.vstack(all_points)
            x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
            y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
            z_range = all_coords[:, 2].max() - all_coords[:, 2].min()
            ax.set_box_aspect((x_range, y_range, z_range))
        
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.view_init(elev=self.elev, azim=self.azim)
        
        if show_legend:
            legend = ax.legend(loc='upper left', fontsize=10, markerscale=self.legend_markerscale)
        
        plt.tight_layout()
        
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor=self.background_color)
            print(f"Saved: {full_path}")
        
        return fig, ax


# ==================== REGION EXTRACTOR ====================

class RegionExtractor:
    """
    Extract region meshes from atlas files.
    Based on step_by_step_insula_flatmap.py
    """
    
    def __init__(self, nmt_dir: str):
        """
        Initialize extractor.
        
        Args:
            nmt_dir: Path to NMT template directory
        """
        self.nmt_dir = nmt_dir
        self.template_data = None
        self._load_template()
    
    def _load_template(self):
        """Load NMT template for GM mask."""
        seg_path = os.path.join(self.nmt_dir, "NMT_v2.1_sym_segmentation.nii.gz")
        if os.path.exists(seg_path):
            self.template_data = nib.load(seg_path).get_fdata()
    
    def get_gm_mask(self) -> Optional[np.ndarray]:
        """Get grey matter mask (label = 2)."""
        if self.template_data is None:
            return None
        return (self.template_data == 2).astype(np.uint8)
    
    def load_atlas(self, atlas_path: str, key_path: str, level: int = 6) -> Tuple[np.ndarray, Dict]:
        """
        Load atlas and key file.
        
        Args:
            atlas_path: Path to atlas NIfTI
            key_path: Path to key TSV file
            level: Atlas level (for 5D atlases)
            
        Returns:
            (atlas_data, name_to_idx_dict)
        """
        # Load atlas
        img = nib.load(atlas_path)
        data = img.get_fdata()
        if data.ndim == 5:
            data = data[:, :, :, 0, level - 1]
        
        # Load key
        key_df = pd.read_csv(key_path, delimiter='\t')
        name_to_idx = dict(zip(key_df['Abbreviation'], key_df['Index']))
        
        print(f"Loaded atlas: {data.shape}")
        print(f"Loaded key: {len(name_to_idx)} regions")
        
        return data, name_to_idx
    
    def create_mask(self, atlas_data: np.ndarray, 
                   region_names: Union[str, List[str]],
                   name_to_idx: Dict[str, int]) -> Optional[np.ndarray]:
        """
        Create binary mask for region(s).
        
        Args:
            atlas_data: Atlas array
            region_names: Region name(s) from key
            name_to_idx: Mapping from names to indices
            
        Returns:
            Binary mask array
        """
        if isinstance(region_names, str):
            region_names = [region_names]
        
        indices = [name_to_idx[r] for r in region_names if r in name_to_idx]
        if not indices:
            print(f"  Warning: no valid regions found in {region_names}")
            return None
        
        mask = np.isin(atlas_data, indices).astype(np.uint8)
        return mask
    
    def extract_mesh(self, mask: np.ndarray, 
                    apply_gm_mask: bool = True,
                    min_voxels: int = 100) -> Optional[np.ndarray]:
        """
        Extract surface mesh from mask using marching cubes.
        
        Args:
            mask: Binary mask array
            apply_gm_mask: Whether to apply GM mask
            min_voxels: Minimum voxels required
            
        Returns:
            Vertices array (Nx3) or None
        """
        if apply_gm_mask:
            gm_mask = self.get_gm_mask()
            if gm_mask is not None:
                mask = mask & gm_mask
        
        if mask.sum() < min_voxels:
            print(f"  Too few voxels: {mask.sum()} (min: {min_voxels})")
            return None
        
        try:
            verts, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, method='lorensen'
            )
            print(f"  Extracted mesh: {len(verts)} vertices")
            return verts
        except Exception as e:
            print(f"  Failed to extract mesh: {e}")
            return None
    
    def extract_regions(self, 
                       atlas_path: str,
                       key_path: str,
                       region_dict: Dict[str, Union[str, List[str]]],
                       apply_gm_mask: bool = True,
                       level: int = 6) -> Dict[str, Optional[np.ndarray]]:
        """
        Extract multiple region meshes at once.
        
        Args:
            atlas_path: Path to atlas NIfTI
            key_path: Path to key TSV file
            region_dict: Dict of {output_name: [region_names]}
            apply_gm_mask: Whether to apply GM mask
            level: Atlas level
            
        Returns:
            Dict of {name: vertices_array or None}
        """
        print(f"\nExtracting regions from: {atlas_path}")
        
        # Load atlas
        atlas_data, name_to_idx = self.load_atlas(atlas_path, key_path, level)
        
        # Extract each region
        results = {}
        for name, region_names in region_dict.items():
            print(f"\nExtracting '{name}':")
            mask = self.create_mask(atlas_data, region_names, name_to_idx)
            if mask is not None:
                verts = self.extract_mesh(mask, apply_gm_mask)
                results[name] = verts
            else:
                results[name] = None
        
        return results
    
    def find_regions_by_name(self, key_path: str, 
                            search_term: str,
                            level: int = 6) -> pd.Series:
        """
        Find region abbreviations by searching full names.
        
        Args:
            key_path: Path to ARM key TSV
            search_term: String to search for (case-insensitive)
            level: Filter by Last_Level
            
        Returns:
            Series of Abbreviation matching the search
        """
        key_df = pd.read_csv(key_path, delimiter='\t')
        mask = (key_df['Full_Name'].str.contains(search_term, case=False) & 
                (key_df['Last_Level'] == level))
        return key_df.loc[mask, 'Abbreviation']


# ==================== UTILITY FUNCTIONS ====================

def load_pkl_mesh(pkl_path: str) -> Optional[np.ndarray]:
    """Load vertices from pickle file."""
    import joblib
    try:
        data = joblib.load(pkl_path)
        if isinstance(data, dict) and 'mesh' in data:
            return data['mesh'].vertices
        return data.vertices if hasattr(data, 'vertices') else None
    except Exception as e:
        print(f"Failed to load {pkl_path}: {e}")
        return None


# ==================== EXAMPLE ====================

if __name__ == "__main__":
    print(__doc__)
