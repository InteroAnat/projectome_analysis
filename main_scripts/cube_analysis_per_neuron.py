import numpy as np
from collections import defaultdict

class cube_analysis_per_neuron:
    def __init__(self, neuron_tracer_obj, cubes):
        self.neuron = neuron_tracer_obj
        self.exp_name = neuron_tracer_obj.exp_no
        self.cubes = cubes        
    def _distance(self,p1, p2,space='nii'):

        if space == 'nii':
            return np.sqrt((p1.x_nii - p2.x_nii)**2 + (p1.y_nii - p2.y_nii)**2 + (p1.z_nii - p2.z_nii)**2)        
        elif space == 'native':
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        else:
            raise ValueError(f"Invalid space: {space}")
    
    
    def _calculate_neuronal_branch_length_in_cubes(self):
        """
    Calculate the length of neuronal branches within predefined cubes.
    
    Returns:
        cube_lengths (dict): Dictionary mapping cube indices to total branch lengths.
        neuron_total_length (float): Total length of all branches in the neuron.
        """
        cube_lengths = defaultdict(float)  # Accumulate lengths per cube
        neuron_total_length = 0  # Total length across all branches
        
        for branch in self.neuron.branches:
            branch_length = 0
            node_index = 0
            
            while node_index < len(branch) - 1:
                current_node = branch[node_index]
                child_node = branch[node_index + 1]
                edge_length = round(self._distance(current_node, child_node), 3)
                branch_length += edge_length
                
                # Get coordinates of the starting node in voxel space
                aa = tuple(np.round(np.array([current_node.x_nii, current_node.y_nii, current_node.z_nii])).astype(int).flatten())
                
                # Find which cube contains this point
                cube_found = False
                for cube_idx, cube in enumerate(self.cubes):
                    start = cube['start']
                    end = cube['end']
                    if (start[0] <= aa[0] < end[0] and
                        start[1] <= aa[1] < end[1] and
                        start[2] <= aa[2] < end[2]):
                        cube_lengths[cube_idx] += edge_length
                        break  # Exit loop once cube is found (assuming non-overlapping cubes)
                
                # if not cube_found:
                #     print(f"Warning: Point {aa} does not fall into any cube.")
                
                node_index += 1
            
            neuron_total_length += branch_length
        self.cube_lengths = cube_lengths
        self.neuron_total_length = neuron_total_length
        return cube_lengths, neuron_total_length