"""
Monkey Projectome Analysis Module
=================================

Purpose:
    This script quantifies the projection patterns of reconstructed monkey neurons (SWC format)
    mapped onto the NMT v2.0 Atlas (CHARM + SARM composite). 

    It performs three main tasks:
    1. Geometry Mapping: Calculates how much axon length falls into each specific brain region.
    2. Classification: Categorizes neurons into IT (Intratelencephalic), PT (Pyramidal Tract), 
       or CT (Corticothalamic) based on projection targets.
    3. Population Statistics: Aggregates results for batch analysis and visualization.

Dependencies:
    - nibabel: For loading NIfTI (.nii.gz) atlas files.
    - pandas: For data handling and table lookups.
    - IONData & neuro_tracer: Custom labs libraries for SWC loading.

Key Logic:
    - Uses NMT Atlas ID ranges to determine broad anatomical groups (Brainstem vs Thalamus vs Cortex).
    - Handles 5D NIfTI files (X, Y, Z, Unused, Hierarchy Level).
    - Correctly maps Atlas Pixel IDs to Region Names using the 'Index' column.

Author: 
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from collections import defaultdict
#%%
# --- Custom Imports Setup ---
# Ensures the neuronVis library is accessible
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

import IONData 
import neuro_tracer as nt

# Initialize Data Provider
iondata = IONData.IONData()

# ==============================================================================
# 1. NEURON CLASSIFIER (Logic Engine)
# ==============================================================================
class NeuronClassifier:
    """
    Logic engine that categorizes neurons based on their terminal fields.
    
    It relies on the Integer ID ranges defined in the NMT Atlas structure:
    - Brainstem: IDs 461-618
    - Thalamus:  IDs 405-452
    - Telencephalon: IDs 1-373
    """
    def __init__(self, atlas_table):
        """
        Args:
            atlas_table (pd.DataFrame): The NMT lookup table containing 'Index', 
                                        'Abbreviation', and 'Full_Name'.
        """
        # Data Cleaning: Ensure 'Index' column is treated as Integers, not Strings.
        try:
            ids = pd.to_numeric(atlas_table['Index'], errors='coerce')
        except KeyError:
            # Fallback if the column is unnamed (usually 1st column)
            ids = pd.to_numeric(atlas_table.iloc[:, 0], errors='coerce')

        # Build Dictionary: {Abbreviation: ID} (e.g., {'VPL': 356})
        self.name_to_id = dict(zip(atlas_table['Abbreviation'], ids))
        
        # Build Dictionary: {Full_Name: ID} (e.g., {'Ventral Posterior': 356})
        if 'Full_Name' in atlas_table.columns:
             self.name_to_id.update(dict(zip(atlas_table['Full_Name'], ids)))

    def _get_broad_region(self, region_identifier):
        """
        Maps a specific region name (e.g., "VPL") to a broad parent group 
        (e.g., "Thalamus") using numerical ID ranges.
        """
        # 1. Convert Name -> ID
        rid = self.name_to_id.get(region_identifier)
        
        # 2. Safety Check: If lookup failed, maybe the input was already an ID number?
        if rid is None:
            try: rid = int(region_identifier)
            except: return 'Other'
        
        # 3. Force Integer Type for comparison
        try: rid = int(rid)
        except: return 'Other'

        # 4. Range Logic (Specific to NMT v2)
        if 461 <= rid <= 618: return 'Brainstem'
        if 405 <= rid <= 452: return 'Thalamus'
        if 1 <= rid <= 373: return 'Telencephalon'
        return 'Other'

    def classify_single_neuron(self, terminal_list):
        """
        Determines neuron class based on the hierarchy of projection targets.
        
        Priority Rule:
        1. PT (Pyramidal Tract): Projects to Brainstem (even if it also projects elsewhere).
        2. CT (Corticothalamic): Projects to Thalamus (but NOT Brainstem).
        3. IT (Intratelencephalic): Projects ONLY within Cortex/Striatum.
        """
        targets = set()
        for t_name in terminal_list:
            targets.add(self._get_broad_region(t_name))
            
        if 'Brainstem' in targets: return 'PT'
        if 'Thalamus' in targets: return 'CT'
        if 'Telencephalon' in targets and 'Thalamus' not in targets: return 'IT'
        return 'Unclassified'

# ==============================================================================
# 2. PER NEURON ANALYSIS (Math Engine)
# ==============================================================================
class region_analysis_per_neuron:
    """
    Performs geometric analysis on a single neuron trace (SWC).
    Calculates total axon length and length per brain region.
    """
    def __init__(self, neuron_tracer_obj, atlas_volume, atlas_table):
        """
        Args:
            neuron_tracer_obj: The loaded SWC object.
            atlas_volume (np.array): The 3D integer array of the brain (sliced to specific level).
            atlas_table (pd.DataFrame): The lookup table.
        """
        self.neuron = neuron_tracer_obj
        self.atlas = atlas_volume
        self.atlas_table = atlas_table
        
        # Critical Map: ID -> Name
        # We map row['Index'] (The Pixel Value) to row['Abbreviation'] (The Name)
        self.brain_region_map = {row['Index']: row['Abbreviation'] for _, row in self.atlas_table.iterrows()}
       
    def region_analysis(self):
        """Executes the calculation pipeline."""
        self.mapped_brain_region_lengths, self.neuron_total_length = self._calculate_neuronal_branch_length()
        self.soma_region, self.terminal_regions = self._soma_and_terminal_region()

    def _distance(self, p1, p2):
        """Euclidean distance between two SWC nodes."""
        return np.sqrt((p1.x_nii - p2.x_nii)**2 + (p1.y_nii - p2.y_nii)**2 + (p1.z_nii - p2.z_nii)**2)
    
    def _calculate_neuronal_branch_length(self):
        """
        Iterates through every segment of the neuron.
        Checks which Atlas Region ID the segment lies within.
        Sums length per ID.
        """
        brain_region_lengths = defaultdict(float)
        neuron_total_length = 0
        
        for branch in self.neuron.branches:
            for i in range(len(branch) - 1):
                current_node = branch[i]
                child_node = branch[i+1]
                edge_length = round(self._distance(current_node, child_node), 3)
                
                # Get coordinate in Atlas Space
                aa = tuple(np.round(np.array([current_node.x_nii, current_node.y_nii, current_node.z_nii])).astype(int).flatten())
                neuron_total_length += edge_length

                # Boundary Check
                if (0 <= aa[0] < self.atlas.shape[0] and
                    0 <= aa[1] < self.atlas.shape[1] and
                    0 <= aa[2] < self.atlas.shape[2]):

                    brain_region_id = int(self.atlas[aa])
                    if brain_region_id > 0: # Ignore background (0)
                        brain_region_lengths[brain_region_id] += edge_length
        
        # Convert Dictionary Keys from IDs (356) to Names (EGP)
        mapped = {}
        for index, length in brain_region_lengths.items():
            region_name = self.brain_region_map.get(index, f'Unknown_{index}')
            mapped[region_name] = length
            
        return mapped, neuron_total_length

    def _soma_and_terminal_region(self):
        """Identifies the specific region containing the Soma and Tips."""
        soma = self.neuron.root
        soma_pos = tuple(np.round(np.array([soma.x_nii, soma.y_nii, soma.z_nii])).astype(int).flatten())
        
        # Soma Lookup
        if (0 <= soma_pos[0] < self.atlas.shape[0] and 
            0 <= soma_pos[1] < self.atlas.shape[1] and 
            0 <= soma_pos[2] < self.atlas.shape[2]):
            s_id = int(self.atlas[soma_pos])
            soma_region = self.brain_region_map.get(s_id, f'Unknown_{s_id}')
        else:
            soma_region = "Out_of_Bounds"
        
        # Terminal Lookup
        terminal_regions = {}
        for node in self.neuron.terminal_nodes:
            node_pos = tuple(np.round(np.array([node.x_nii, node.y_nii, node.z_nii])).astype(int).flatten())
            if (0 <= node_pos[0] < self.atlas.shape[0] and 
                0 <= node_pos[1] < self.atlas.shape[1] and 
                0 <= node_pos[2] < self.atlas.shape[2]):
                t_id = int(self.atlas[node_pos])
                t_name = self.brain_region_map.get(t_id, f'Unknown_{t_id}')
                terminal_regions[t_name] = node_pos
            
        return soma_region, terminal_regions   

# ==============================================================================
# 3. POPULATION MANAGER (Orchestrator)
# ==============================================================================
class PopulationRegionAnalysis:
    """
    Manages the batch processing of multiple neurons.
    - Slices the 5D atlas.
    - Runs analysis per neuron.
    - Aggregates data into a master DataFrame.
    - Provides visualization methods.
    """
    def __init__(self, sample_id, atlas, atlas_table, nii_space='monkey'):
        self.sample_id = sample_id
        self.full_atlas = atlas # The raw NIfTI data (likely 5D)
        self.atlas_table = atlas_table
        self.nii_space = nii_space
        self.neuron_list = iondata.getNeuronListBySampleID(sample_id)
        self.classifier = NeuronClassifier(atlas_table)
        
        # Main results storage
        self.plot_dataframe = pd.DataFrame() 

    def process(self, limit=None, level=6, neuron_id=None):
    # 5D Slicing logic
        if self.full_atlas.ndim == 5:
            current_atlas = self.full_atlas[:, :, :, 0, level-1]
        else:
            current_atlas = self.full_atlas
            
        # --- FIX: EXACT ID MATCHING ---
        if neuron_id:
            # We assume neuron_id is like '001.swc'
            # We look for an exact string match in the database list
            target_list = [n for n in self.neuron_list if n['name'] == neuron_id]
            
            if not target_list:
                print(f"Error: Neuron '{neuron_id}' not found in sample {self.sample_id}.")
                print(f"Available (first 5): {[n['name'] for n in self.neuron_list[:5]]}")
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

            term_list = list(analysis.terminal_regions.keys())
            n_type = self.classifier.classify_single_neuron(term_list)

            row = {
                'SampleID': self.sample_id,
                'NeuronID': neuron.swc_filename,
                'Neuron_Type': n_type,
                'Soma_Region': analysis.soma_region,
                'Total_Length': analysis.neuron_total_length,
                'Terminal_Count': len(term_list),
                # --- FIX: Standardized Column Name ---
                'Terminal_Regions': term_list, 
                'Region_projection_length': analysis.mapped_brain_region_lengths 
            }
            data.append(row)

        self.plot_dataframe = pd.DataFrame(data)
    def get_region_matrix(self):
        """
        Transforms the 'Region_projection_length' column into a dense matrix.
        Rows: Neurons
        Cols: Brain Regions (Values = Axon Length)
        Useful for Heatmaps and Clustering.
        """
        if self.plot_dataframe.empty: return pd.DataFrame()
        
        dict_list = self.plot_dataframe['Region_projection_length'].tolist()
        matrix = pd.DataFrame(dict_list)
        matrix.fillna(0, inplace=True) # Zero length for non-projected regions
        
        # Add identifiers back
        matrix.insert(0, 'NeuronID', self.plot_dataframe['NeuronID'])
        matrix.insert(1, 'Neuron_Type', self.plot_dataframe['Neuron_Type'])
        
        return matrix

    # --- VISUALIZATION METHODS ---

    def plot_projection_by_soma(self, stat='median'):
        if self.plot_dataframe.empty: return
        df = self.plot_dataframe
        regions = df['Soma_Region'].unique()
        data = [df[df['Soma_Region'] == r]['Total_Length'] for r in regions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, labels=regions)
        ax.set_title(f'Projection Length by Soma Region ({stat})')
        plt.xticks(rotation=45)
        
        vals = [np.median(d) if stat=='median' else np.mean(d) for d in data]
        for i, val in enumerate(vals):
            ax.text(i+1, val, f"{val:.1f}", ha='center', va='bottom')
        plt.show()

    def plot_type_distribution(self):
        if self.plot_dataframe.empty: return
        self.plot_dataframe['Neuron_Type'].value_counts().plot(kind='bar', color='teal')
        plt.title("Neuron Types Distribution")
        plt.show()

    def plot_terminal_distribution(self):
        if self.plot_dataframe.empty: return
        exploded = self.plot_dataframe.explode('Terminal_Regions')
        counts = exploded['Terminal_Regions'].value_counts().head(20)
        
        plt.figure(figsize=(12, 6))
        counts.plot(kind='bar')
        plt.title("Top 20 Terminal Regions")
        plt.show()
        # ==========================================================================
    # NEW: SINGLE NEURON INSPECTOR
    # ==========================================================================
    def inspect_neuron(self, target_filename):
        """
        Args:
            target_filename (str): Exact filename, e.g. '001.swc'
        """
        if self.plot_dataframe.empty:
            print("Error: No data processed. Run process() first.")
            return

        # 1. Find the row (Exact match preferred)
        mask = self.plot_dataframe['NeuronID'] == target_filename
        
        if not mask.any():
            print(f"Error: '{target_filename}' not found in results.")
            return
        
        row = self.plot_dataframe[mask].iloc[0]
        
        # 2. Print Report
        print(f"\n{'='*40}")
        print(f"  REPORT: {row['NeuronID']}")
        print(f"{'='*40}")
        print(f"  • Type:      {row['Neuron_Type']}")
        print(f"  • Soma:      {row['Soma_Region']}")
        print(f"  • Length:    {row['Total_Length']:.3f}")
        # --- FIX: Accessing the correct column name 'Terminal_Regions' ---
        print(f"  • Targets:   {row['Terminal_Regions']}") 
        print("-" * 40)
        
        # 3. Plotting Logic (Same as before)
        region_dict = row['Region_projection_length']
        if not region_dict or sum(region_dict.values()) == 0: return

        stats = pd.Series(region_dict).sort_values(ascending=False)
        # Filter noise
        stats = stats[stats > 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar Plot
        top_n = stats.head(10)
        sns.barplot(x=top_n.values, y=top_n.index, ax=axes[0], palette='viridis')
        axes[0].set_title(f"Top Projections ({row['NeuronID']})")
        
        # Pie Plot
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
    # PATHS (Update to your combined SARM+CHARM atlas)
    atlas_path = r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz'
    table_path = r'D:\projectome_analysis\atlas\nmt_structures_labels.txt'
    
    # Load Files
    combined_atlas_nii = nib.load(atlas_path)
    atlas_data = combined_atlas_nii.get_fdata()
    global_id_df = pd.read_csv(table_path, delimiter='\t')

    # Initialize
    pop = PopulationRegionAnalysis('251637', atlas_data, global_id_df)
    
    # RUN Analysis (Batch of 10 neurons, Level 6 detail)
    pop.process(limit=10,level=3)
    
    # Generate Plots
    pop.plot_projection_by_soma()
    pop.plot_type_distribution()
    pop.plot_terminal_distribution()
    
    # Extract & Print Detailed Data
    detailed_matrix = pop.get_region_matrix()
    print("\n--- Detailed Regional Length Matrix (First 5 Rows) ---")
    print(detailed_matrix.head())
    
    pop.inspect_neuron('001.swc')

    # # Export
    # pop.plot_dataframe.to_csv("Summary_Results.csv", index=False)
    # detailed_matrix.to_csv("Detailed_Region_Lengths.csv", index=False)
# %%
