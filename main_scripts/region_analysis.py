#%%
import neuro_tracer as nt

import sys,copy,os,inspect

neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
sys.path.append(neurovis_path)

import IONData 
iondata = IONData.IONData()

from collections import defaultdict


from pathlib import Path
import matplotlib


import matplotlib.pyplot as plt


import numpy as np
import nibabel as nib  # parse the NII data
from collections import deque
import pandas as pd
from typing import Tuple
from io import StringIO

import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm

from scipy.cluster.hierarchy import dendrogram, linkage

import requests
from bs4 import BeautifulSoup
neuronlist=iondata.getNeuronListBySampleID('251637')
import nrrd
def plot_projection_by_soma(df, stat='median'):
    unique_regions = df['Soma_Region'].unique()
    data = [df[df['Soma_Region'] == region]['Projection_length'] for region in unique_regions]
    fig, ax = plt.subplots(figsize=(10, 6))
    bplot = ax.boxplot(data, labels=unique_regions)
    ax.set_title('Box Plot of Projection Length by Soma Region')
    ax.set_xlabel('Soma Regions')
    ax.set_ylabel('Projection Length')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add exact values (medians or means) above boxes
    if stat == 'median':
        values = [np.median(d) for d in data]
    elif stat == 'mean':
        values = [np.mean(d) for d in data]
    else:
        raise ValueError("stat must be 'median' or 'mean'")
    
    for i, val in enumerate(values):
        ax.text(i + 1, val, f"{val:.2f}", ha='center', va='bottom')

    plt.show()

def plot_soma_distribution(df):
    soma_counts = df['Soma_Region'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    soma_counts.plot(kind='bar', ax=ax)
    ax.set_title('Summary of Soma Distribution by Region')
    ax.set_xlabel('Soma Regions')
    ax.set_ylabel('Number of Neurons')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add exact counts above bars
    for i, height in enumerate(soma_counts):
        ax.text(i, height, str(height), ha='center', va='bottom')

    plt.show()


class region_analysis_per_neuron:
    def __init__(self, neuron_tracer_obj, atlas, atlas_table):
        self.neuron = neuron_tracer_obj
        self.exp_name = neuron_tracer_obj.exp_no
        self.atlas = atlas
        self.atlas_table = atlas_table
        self.brain_region_map = {index: row['Abbreviation'] for index, row in self.atlas_table.iterrows()}
       
    def region_analysis(self):

        self.mapped_brain_region_lengths, self.neuron_total_length = self._calculate_neuronal_branch_length()
        self.soma_region, self.terminal_regions = self._soma_and_terminal_region()
    def _distance(self,p1, p2,space='nii'):

        if space == 'nii':
            return np.sqrt((p1.x_nii - p2.x_nii)**2 + (p1.y_nii - p2.y_nii)**2 + (p1.z_nii - p2.z_nii)**2)        
        elif space == 'native':
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        else:
            raise ValueError(f"Invalid space: {space}")
    
    
    def _calculate_neuronal_branch_length(self):
            
            brain_region_lengths = defaultdict(float)
            neuron_total_length = 0
            
            for branch in self.neuron.branches:
                branch_length = 0
                current_node = branch[0]
                
                node_index = 0
                while node_index < len(branch) - 1:
                    current_node = branch[node_index]
                    child_id = node_index + 1
                    child_node = branch[child_id]
                    edge_length = round(self._distance(current_node, child_node),3)
                    aa = tuple(np.round(np.array([current_node.x_nii, current_node.y_nii, current_node.z_nii])).astype(int).flatten())
                    branch_length += edge_length

        
                    
                    if (0 <= aa[0] < self.atlas.shape[0] and
                        0 <= aa[1] < self.atlas.shape[1] and
                        0 <= aa[2] < self.atlas.shape[2]):

                        brain_region_id = int(self.atlas[aa])
                        brain_region_lengths[brain_region_id] += edge_length
                    else:
                        print(f"Warning: Point {aa},{[current_node.x_nii, current_node.y_nii, current_node.z_nii]} {self.neuron.swc_filename} node{current_node.id} is out of bounds of the atlas.")
                    node_index += 1
                neuron_total_length += branch_length
         
            mapped_brain_region_lengths ={}

            for index, length in brain_region_lengths.items():
               
                region = self.brain_region_map.get(index, 'Unknown')
                
                mapped_brain_region_lengths [region] = length
                
            return mapped_brain_region_lengths, neuron_total_length
    def _soma_and_terminal_region(self):
        soma=self.neuron.root
        soma_pos=tuple(np.round(np.array([soma.x_nii, soma.y_nii, soma.z_nii])).astype(int).flatten())
        soma_region = self.brain_region_map.get(int(self.atlas[soma_pos]), 'Unknown')
        
        terminal_regions = {}
        for node in self.neuron.terminal_nodes:
            node_pos=tuple(np.round(np.array([node.x_nii, node.y_nii, node.z_nii])).astype(int).flatten())
            node_region = int(self.atlas[node_pos])
            node_region = self.brain_region_map.get(node_region, 'Unknown')
            
            terminal_regions [node_region] = node_pos
        return soma_region, terminal_regions   

class PopulationRegionAnalysis:
    def __init__(self, sample_id, atlas, atlas_table, nii_space='monkey'):
        self.sample_id = sample_id
        self.atlas = atlas
        self.atlas_table = atlas_table
        self.nii_space = nii_space
        self.neuron_list = iondata.getNeuronListBySampleID(sample_id)
        self.neurons = {}
        self.soma_regions = []
        self.projection_lengths = []
        self.region_projection_lengths = []
        self.terminal_regions = []
        self.neuron_ids = []
        self.plot_dataframe=[]
    def process(self, limit=None):
        if limit is None:
            neurons_to_process = self.neuron_list
        else:
            neurons_to_process = self.neuron_list[:limit]

        for target_neuron in neurons_to_process:
            neuron = nt.neuro_tracer()
            neuron.process(target_neuron['sampleid'], target_neuron['name'], nii_space=self.nii_space)
            self.neurons[int(target_neuron['name'].split('.')[0])] = neuron

            analysis = region_analysis_per_neuron(neuron, self.atlas, self.atlas_table)
            analysis.region_analysis()

            self.soma_regions.append(analysis.soma_region)
            self.projection_lengths.append(analysis.neuron_total_length)
            self.region_projection_lengths.append(analysis.mapped_brain_region_lengths)
            self.terminal_regions.append(list(analysis.terminal_regions.keys()))
            self.sample_id = neuron.exp_no
            self.neuron_ids.append(neuron.swc_filename)
        self.__get_figure_data()
    def __get_figure_data(self):
        dataframe= pd.DataFrame({
            'SampleID':self.sample_id,
            'NeuronID': self.neuron_ids,
            'Soma_Region': self.soma_regions,
            'Projection_length': self.projection_lengths,
            'Region_projection_length': self.region_projection_lengths
        })
        self.plot_dataframe=dataframe

    def plot_projection_by_soma(self, stat='median'):
        df = self.plot_dataframe
        plot_projection_by_soma(df, stat)

    def plot_soma_distribution(self):
        df=self.plot_dataframe
        plot_soma_distribution(df)
    
if __name__ == '__main__':
    atlas,atlas_header=nrrd.read(r'D:\projectome_analysis\atlas\nmt_structure.nrrd')
    global_id_df = pd.read_csv(r'D:\projectome_analysis\atlas\NMT\tables_CHARM\CHARM_key_all.txt',delimiter='\t')

    pop = PopulationRegionAnalysis('251637', atlas, global_id_df)
    pop.process(limit=3)
    pop.plot_projection_by_soma(stat='median')  # or 'mean'
    pop.plot_soma_distribution()
    
#%%

# %%
