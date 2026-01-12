# %%
import sys,copy,os,inspect

# path definition
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
sys.path.append(neurovis_path)

# neuron_path = os.path.join(neurovis_path,"../resource/swc/192106/001.swc")
# sys.path.append(neuron_path)

import IONData
import pandas as pd
from io import StringIO

# branch is a chain of nodes
# class Branch:
# def __init__(self, root_node):
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Set before importing pyplot
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import numpy as np
import nibabel as nib  # parse the NII data
from collections import deque
import pandas as pd
from typing import Tuple
from io import StringIO

import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm


import requests
from bs4 import BeautifulSoup

from collections import defaultdict

from pathlib import Path

matplotlib.use('TkAgg')  # Set before importing pyplot




def distance(p1, p2,space='nii'):

    if space == 'nii':
        return np.sqrt((p1.x_nii - p2.x_nii)**2 + (p1.y_nii - p2.y_nii)**2 + (p1.z_nii - p2.z_nii)**2)        
    elif space == 'native':
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    else:
        raise ValueError(f"Invalid space: {space}")
class neuro_tracer:

    class Node:
        def __init__(self, swc_point):
            self.id = int(swc_point[0])
            self.type = int(swc_point[1])
            self.x = float(swc_point[2])
            self.y = float(swc_point[3])
            self.z = float(swc_point[4])
            self.radius = float(swc_point[5])
            self.parent = int(swc_point[6])
            self.children = []
            self.is_terminal = False
            self.order = 0
            self.processed_data =None
            self.output_dir = None
            
    class Branch:
        def __init__(self, branch):
            self.branch_nodes = branch
            self.branch_length = 0
            self.branch_order = branch[0].order 
    def __init__ (self):
        self.swc_filename= None
        self.exp_no= None
        self.nodes ={}
        self.root=[]
        self.branches=[]
        self.branches_in_id = []
        self.processed_data = None
        self.terminal_nodes = None
        
        self.reso = None
        self.nii_shape = None
        
# Check if the output directory exists, if not, create it
        
#         continue
# function to load swc or swc file path and extract the nodes
    def process(self,exp_no,swc_filename,nii_space='mouse',output_dir=None,swc=None):

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'processed_neurons', f'{exp_no}')
            self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self._loadSWC(exp_no,swc_filename,swc)
        self._acquire_nii_nodes(nii_space)
        self._find_children()
        self._mark_branch_terminals()
        self._construct_branches()
       
        
        
        
        self._export_file(output_dir)
        self._acquire_nii_nodes
        
        # self._calculate_branch_orders()
        # 
    def _loadSWC(self, exp_no, swc_filename, swc=None):
        # First check if we have a local file in the output directory
        local_swc_path = os.path.join(self.output_dir, swc_filename)
        
        # If no SWC data is provided directly
        if swc is None:
            # Check if file exists locally
            if os.path.exists(local_swc_path):
                print(f"Loading SWC from local file: {local_swc_path}")
                with open(local_swc_path, "r") as f:
                    swclines = f.read().split('\n')
            else:
                # Download from IONData if not found locally
                print("Fetching SWC data via iondata")
                iondata = IONData.IONData()
                swc_data = iondata.getNeuronByID(exp_no, swc_filename)
                swclines = swc_data.split('\n')
                
                # Save the downloaded SWC to output directory
                with open(local_swc_path, "w") as f:
                    f.write(swc_data)
                print(f"Saved SWC to: {local_swc_path}")
        else:
            # Handle direct SWC input (file path or string)
            if os.path.isfile(swc):
                print("Parse swc as file")
                with open(swc, "r") as f:
                    swclines = f.read().split('\n')
            else:
                print("Parse swc as string")
                swclines = swc.split('\n')
                
        # Process SWC lines
        for line in swclines:
            if len(line) == 0 or line[0] == '#' or line[0] == '\r':
                continue
            p = list(map(float, (line.split())))
            self.nodes[int(p[0])] = self.Node(p)
            
        self.swc_filename = swc_filename
        self.exp_no = exp_no
        
        # Sanity checks
        root_count = 0
        for node in self.nodes.values():
            if node.parent == -1:
                self.root = node
                root_count += 1
        if root_count != 1:
            print("Multiple root nodes found")
        if not self.root:
            raise ValueError("No root node found")

    
    def _acquire_nii_nodes(self,nii_space='mouse'):
        if nii_space =='mouse':
            self.reso =10
            self.nii_shape = (1320, 800, 1140)
        elif nii_space == 'monkey':
            self.reso = 250
            self.nii_shape = (256, 312, 200)
        
        l, w, h =self.nii_shape

        for node in self.nodes.values():
            
            node.x_nii = node.x / self.reso
            node.y_nii = node.y / self.reso
            node.z_nii = node.z / self.reso
            


    def _find_children(self): 
           
        for node in self.nodes.values():
                    if node.parent != -1 and node.parent in self.nodes:
                        parent = self.nodes[node.parent]
                        parent.children.append(node)
                       
    def _mark_branch_terminals(self):
       
        for node in self.nodes.values():
            if len(node.children) > 1:
                node.is_terminal = False 
            elif len(node.children) == 0:
                node.is_terminal = True
        terminal_nodes = []
        for node in self.nodes.values():
            if node.is_terminal:
                terminal_nodes.append(node)
        self.terminal_nodes = terminal_nodes
        print(f'there are {len(terminal_nodes)} terminal nodes')
    
  
        '''
        there are three cases:
        1. root node (soma), parent = -1 and children could be >1
        2. intermediate nodes, parent != -1 and children >1
        3. branching nodes, parent != -1 and children >1
        4. terminal nodes, parent != -1 and children =0
        
    
        '''
    def _construct_branches(self):
        
        print ("\nConstructing branches, assigning orders \n \n")
        
        def build_branch(current_node, parent_order, branch=None):
            if branch is None:
                branch = [current_node]
            
            current_order = parent_order
            current_node.order = current_order  # Use branch_order
            while len(current_node.children) == 1:
                current_node = current_node.children[0]
                current_node.order = current_order
                branch.append(current_node)
            
            if len(current_node.children) == 0:
                self.branches.append(branch[:])
                # if current_node.id < 40:
                #     # print(f"Terminal at {current_node.id}, order: {current_order}")
            if len(current_node.children) > 1:
                next_order = current_order + 1
                # if current_node.id < 40:
                #     # print(f"Branch point at {current_node.id}, next_order: {next_order}")
                #     # print(f'next node is  {current_node.children[0].id}')
                    
                if len(branch)>1: #key modification 
                        self.branches.append(branch[:])
                for child in current_node.children:
                    new_branch = [current_node,child]
                    build_branch(child, next_order, new_branch)
        
        def construct_id_branches(self):
            node_id_branches= []
            for branch in self.branches:
                branch_ids = '->'.join(str(int(node.id)) for node in branch)
                node_id_branches.append(branch_ids)
            return node_id_branches

    
        build_branch(self.root,0)
        print (f"\nFinished with {len(self.branches)} branches")
        self.branches_in_id = construct_id_branches(self)
        # print ('\n-----------------\n')
        # print(self.branches_in_id[:3])
        
    def _export_file(self, output_dir):
        """Export neuron data to CSV using pandas with round()"""
        self.processed_data = pd.DataFrame([
            {
                'id': int(node.id),
                'type': node.type,
                'x': round(node.x, 4),
                'y': round(node.y, 4),
                'z': round(node.z, 4),
                'x_nii': round(node.x_nii, 4),
                'y_nii': round(node.y_nii, 4),
                'z_nii': round(node.z_nii, 4),
                'radius': round(node.radius, 4),
                'parent': node.parent,
                'is_terminal': int(node.is_terminal),
                'order': node.order
            }
            for node in self.nodes.values()
        ])

        
        self.processed_data.to_csv(f'{output_dir}/{self.exp_no}_{os.path.splitext(self.swc_filename)[0]}.csv', index=False)
        
  

        # for file in os.listdir(self.output_dir):
        #         if file.endswith('.csv'):
        #             swc_csv = os.path.join(self.output_dir, file)
        #             swc_df = pd.read_csv(swc_csv)
        #             swc_nii = self.transform_to_nii(swc_df, reso=reso, nii_shape=nii_shape)
        #             self.nii_space_data =swc_nii.round(3)

        #             swc_nii.to_csv(os.path.join(nii_path, file.replace('.csv', '_transformed.csv')), index=False)
                    
                    
    def plot_neuron(self): 
        data=self.processed_data
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')  # Create 3D axes

        # Plot all nodes with branch order coloring
        scatter = ax.scatter(
            data['x'],
            data['y'],
            data['z'],
            c=data['order'],  # Color by branch order
            s=3,
            cmap='tab20',
            alpha=0.8
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Order')

        # Mark terminal nodes
        terminals = data[data['is_terminal'] == 1]
        ax.scatter(
            terminals['x'],
            terminals['y'],
            terminals['z'],
            s=20,
            marker='*',
            color='black',
            label='Terminal Nodes'
        )

        # Mark soma (root node)
        soma = data[data['parent'] == -1]
        ax.scatter(
            soma['x'],
            soma['y'],
            soma['z'],
            s=50,
            color='red',
            edgecolor='orange',  # Add contrasting border
            linewidth=1,       # Border thickness
            marker='X',        # Use X marker instead of default circle
            # label='Soma',
            # zorder=10  
            
        )

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Single Neuron Plot - {self.swc_filename}')
        ax.legend()

        # Display the plot
        plt.show()
        
   
    def calculate_branch_length(self, nii_pointID):
            if self.nii_space_data is None:
                raise ValueError("NIfTI space data not available. Run _generate_nii first.")
            
            brain_region_lengths = defaultdict(float)
            total_length = 0
            
            for branch in self.branches:
                branch_length = 0
                current_node = branch[0]
                
                node_index = 0
                while node_index < len(branch) - 1:
                    current_node = branch[node_index]
                    child_id = node_index + 1
                    child_node = branch[child_id]
                    edge_length = distance(current_node, child_node)
                    aa = tuple(np.round(np.array([current_node.x_nii, current_node.y_nii, current_node.z_nii])).astype(int).flatten())


                
                    if (0 <= aa[0] < nii_pointID.shape[0] and
                        0 <= aa[1] < nii_pointID.shape[1] and
                        0 <= aa[2] < nii_pointID.shape[2]):
                        brain_region_id = int(nii_pointID[aa])
                        branch_length += edge_length
                        brain_region_lengths[brain_region_id] += edge_length
                    else:
                        print(f"Warning: Point {aa} is out of bounds of the atlas.")
                    node_index += 1
                total_length += branch_length
            
            
            return brain_region_lengths, total_length
                
            
            
            

if __name__ == '__main__':
    test_neuron = neuro_tracer()
    test_neuron.process('251637','002.swc',nii_space='monkey')
    print(test_neuron.nodes[1].x_nii,test_neuron.nodes[1].x)
    print(test_neuron.nodes[1].y_nii,test_neuron.nodes[1].y)
    print(test_neuron.nodes[1].z_nii,test_neuron.nodes[1].z)
    test_neuron.plot_neuron()
   
    # test_neuron.plot_neuron()
    

# %%

# %%
