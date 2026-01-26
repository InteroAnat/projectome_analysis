"""
Monkey Projectome Region Ana
=================================
"""
"""
Key module for analysis of the projectome in the NMT space for region information.

update log:

v- 0.11: 19/01/2026: 
    some bug fixes for outlier plot, 
    now outlier plot can be controlled by a separate function but the outlier count will noted.
    
v - 0.12:
    1. potential updates: plot soma function
    2. move plotting functions to be static so that they can be applied externally.
    3. flat map display
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

# --- VISUALIZATION IMPORTS ---
from nilearn import plotting

# --- Custom Imports Setup ---
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

import IONData 
import neuro_tracer as nt

# Initialize Data Provider
iondata = IONData.IONData()

# ==============================================================================
# HELPER: OUTLIER INSPECTOR (Visual Debugger)
# ==============================================================================
def save_debug_snapshot(voxel_coords, neuron_name, template_img, point_type, folder="../resource/debug_outliers"):
    """
    Saves a snapshot to the resource folder in the parent directory.
    """
    if template_img is None: return
    
    # Create directory if it doesn't exist
    if not os.path.exists(folder): 
        try:
            os.makedirs(folder)
        except Exception as e:
            print(f"     [WARN] Could not create folder {folder}: {e}")
            return
    
    world_coords = nib.affines.apply_affine(template_img.affine, voxel_coords)
    
    try:
        display = plotting.plot_anat(
            template_img, 
            cut_coords=world_coords, 
            draw_cross=True, 
            title=f"Outlier {point_type}: {neuron_name} -> {voxel_coords}"
        )
        
        fname = f"{neuron_name}_{point_type}_ID0_{int(voxel_coords[0])}_{int(voxel_coords[1])}_{int(voxel_coords[2])}.png"
        full_path = os.path.join(folder, fname)
        
        display.savefig(full_path)
        display.close()
        print(f"     [DEBUG] Snapshot saved: {full_path}")
        
    except Exception as e:
        print(f"     [DEBUG] Failed to plot snapshot: {e}")

# ==============================================================================
# 1. NEURON CLASSIFIER
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

    def _get_broad_region(self, region_identifier):
        rid = self.name_to_id.get(region_identifier)
        if rid is None:
            try: rid = int(region_identifier)
            except: return 'Other'
        try: rid = int(rid)
        except: return 'Other'

        if 461 <= rid <= 618: return 'Brainstem'
        if 405 <= rid <= 452: return 'Thalamus'
        if 1 <= rid <= 373: return 'Telencephalon'
        return 'Other'

    def classify_single_neuron(self, terminal_list):
        targets = set()
        for t_name in terminal_list:
            targets.add(self._get_broad_region(t_name))
            
        if 'Brainstem' in targets: return 'PT'
        if 'Thalamus' in targets: return 'CT'
        if 'Telencephalon' in targets and 'Thalamus' not in targets: return 'IT'
        return 'Unclassified'

# ==============================================================================
# 2. PER NEURON ANALYSIS
# ==============================================================================
class region_analysis_per_neuron:
    def __init__(self, neuron_tracer_obj, atlas_volume, atlas_table):
        self.neuron = neuron_tracer_obj
        self.atlas = atlas_volume
        self.atlas_table = atlas_table
        # Fix: Use 'Index' column for keys
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

                if (0 <= aa[0] < self.atlas.shape[0] and
                    0 <= aa[1] < self.atlas.shape[1] and
                    0 <= aa[2] < self.atlas.shape[2]):

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
        
        if (0 <= soma_pos[0] < self.atlas.shape[0] and 
            0 <= soma_pos[1] < self.atlas.shape[1] and 
            0 <= soma_pos[2] < self.atlas.shape[2]):
            s_id = int(self.atlas[soma_pos])
            soma_region = self.brain_region_map.get(s_id, f'Unknown_{s_id}')
        else:
            soma_region = "Out_of_Bounds"
        
        terminal_regions = []
        for node in self.neuron.terminal_nodes:
            node_pos = tuple(np.round(np.array([node.x_nii, node.y_nii, node.z_nii])).astype(int).flatten())
            if (0 <= node_pos[0] < self.atlas.shape[0] and 
                0 <= node_pos[1] < self.atlas.shape[1] and 
                0 <= node_pos[2] < self.atlas.shape[2]):
                t_id = int(self.atlas[node_pos])
                t_name = self.brain_region_map.get(t_id, f'Unknown_{t_id}')
                
                terminal_regions.append({'region': t_name, 'coords': node_pos})
            
        return soma_region, terminal_regions   

# ==============================================================================
# 3. POPULATION MANAGER
# ==============================================================================
class PopulationRegionAnalysis:
    def __init__(self, sample_id, atlas, atlas_table, template_img=None, nii_space='monkey'):
        self.sample_id = sample_id
        self.full_atlas = atlas 
        self.atlas_table = atlas_table
        self.template_img = template_img 
        self.nii_space = nii_space
        self.neuron_list = iondata.getNeuronListBySampleID(sample_id)
        self.classifier = NeuronClassifier(atlas_table)
        self.plot_dataframe = pd.DataFrame() 
        self.neurons = {} 

    def process(self, limit=None, level=6, neuron_id=None):
        # 5D Slicing
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
            
            # --- OUTLIER DATA COLLECTION (No Plotting Here) ---
            neuron.outliers = []

            # 1. Check Soma
            if "Unknown" in analysis.soma_region:
                root = neuron.root
                coords = (int(root.x_nii), int(root.y_nii), int(root.z_nii))
                neuron.outliers.append({'type': 'Soma', 'region': analysis.soma_region, 'coords': coords})
            
            # 2. Check Terminals
            for item in analysis.terminal_regions:
                t_name = item['region']
                coords = item['coords']
                if "Unknown" in t_name:
                    neuron.outliers.append({'type': 'Terminal', 'region': t_name, 'coords': coords})

            # Store Neuron Object
            self.neurons[neuron.swc_filename] = neuron

            # --- CLASSIFICATION & STORAGE ---
            term_name_list = [item['region'] for item in analysis.terminal_regions]
            n_type = self.classifier.classify_single_neuron(term_name_list)

            row = {
                'SampleID': self.sample_id,
                'NeuronID': neuron.swc_filename,
                'Neuron_Type': n_type,
                'Soma_Region': analysis.soma_region,
                'Total_Length': analysis.neuron_total_length,
                'Terminal_Count': len(term_name_list),
                'Terminal_Regions': term_name_list, 
                'Region_projection_length': analysis.mapped_brain_region_lengths,
                'Outlier_Count': len(neuron.outliers),
                'Outlier_Details': neuron.outliers
            }
            data.append(row)

        self.plot_dataframe = pd.DataFrame(data)

    def export_outlier_snapshots(self, neuron_id, max_snapshots=3):
        """
        Generates images for outliers stored in a specific neuron object.
        Args:
            max_snapshots (int): Maximum number of images to generate per neuron (default 3).
        """
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
            # --- SAFETY CHECK ---
            if count >= max_snapshots:
                print(f"     [INFO] Max snapshots ({max_snapshots}) reached. Stopping.")
                break
            
            save_debug_snapshot(
                voxel_coords=error['coords'],
                neuron_name=neuron.swc_filename,
                template_img=self.template_img,
                point_type=error['type']
            )
            count += 1

    def get_region_matrix(self):
        if self.plot_dataframe.empty: return pd.DataFrame()
        dict_list = self.plot_dataframe['Region_projection_length'].tolist()
        matrix = pd.DataFrame(dict_list)
        matrix.fillna(0, inplace=True)
        matrix.insert(0, 'NeuronID', self.plot_dataframe['NeuronID'])
        matrix.insert(1, 'Neuron_Type', self.plot_dataframe['Neuron_Type'])
        return matrix

    def inspect_neuron(self, target_filename):
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
        print(f"  Length: {row['Total_Length']:.3f}")
        print(f"  Targets: {row['Terminal_Regions']}") 
        
        region_dict = row['Region_projection_length']
        if not region_dict or sum(region_dict.values()) == 0: return

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

    def plot_projection_by_soma(self, stat='median'):
        if self.plot_dataframe.empty: return
        df = self.plot_dataframe
        regions = df['Soma_Region'].unique()
        data = [df[df['Soma_Region'] == r]['Total_Length'] for r in regions]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, labels=regions)
        ax.set_title(f'Projection Length by Soma Region ({stat})')
        plt.xticks(rotation=45)
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
        
    def plot_soma_distribution(self):
        df = self.plot_dataframe
        if df.empty: return
        soma_counts = df['Soma_Region'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        soma_counts.plot(kind='bar', ax=ax)
        ax.set_title('Summary of Soma Distribution by Region')
        ax.set_ylabel('Number of Neurons')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        for i, height in enumerate(soma_counts):
            ax.text(i, height, str(height), ha='center', va='bottom')
        plt.show()



# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # 1. PATHS
    atlas_path = r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz'
    table_path = r'D:\projectome_analysis\atlas\nmt_structures_labels.txt'
    template_path = r'D:\projectome_analysis\atlas\NMT_v2.0_sym\NMT_v2.0_sym\NMT_v2.0_sym_SS.nii'
    
    # 2. LOAD
    combined_atlas_nii = nib.load(atlas_path)
    atlas_data = combined_atlas_nii.get_fdata()
    global_id_df = pd.read_csv(table_path, delimiter='\t')
    template_nii = nib.load(template_path)

    # 3. INIT
    pop = PopulationRegionAnalysis('251637', atlas_data, global_id_df, template_img=template_nii)
    
    # 4. RUN SINGLE NEURON TEST
    target_file = '001.swc'
    pop.process(neuron_id=target_file, level=6)
    
    # 5. CHECK OUTLIERS (On Demand)
    # pop.export_outlier_snapshots(target_file)
    pop.inspect_neuron(target_file)

    # 6. BATCH EXAMPLE
    # pop.process(limit=50, level=6)
    # bad_somas = pop.plot_dataframe[pop.plot_dataframe['Outlier_Count'] > 0]
    # if not bad_somas.empty:
    #     pop.export_outlier_snapshots(bad_somas.iloc[0]['NeuronID'])
    
