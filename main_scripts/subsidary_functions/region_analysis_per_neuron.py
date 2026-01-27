import numpy as np
from collections import defaultdict





class region_analysis_per_neuron:
    def __init__(self, neuron_tracer_obj,atlas,atlas_table):
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
                        print(f"Warning: Point {aa} is out of bounds of the atlas.")
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
