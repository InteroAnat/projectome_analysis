import numpy as np
from collections import defaultdict

class cube_analysis_per_neuron_terminal:
    def __init__(self, neuron_tracer_obj, cubes):
        """
        Initialize the cube analysis class for a neuron.

        Args:
            neuron_tracer_obj: Neuron object with branches and terminal_nodes() method.
            cubes: List of cube dictionaries, each with 'start' and 'end' coordinates.
        """
        self.neuron = neuron_tracer_obj
        self.exp_name = neuron_tracer_obj.exp_no
        self.cubes = cubes
        self.node_to_cube = {}  # Cache for node to cube mapping

    def _distance(self, p1, p2, space='nii'):
        """
        Calculate the Euclidean distance between two points.

        Args:
            p1, p2: Node objects with x_nii, y_nii, z_nii (or x, y, z) attributes.
            space: Coordinate space ('nii' or 'native').

        Returns:
            float: Distance between the points.
        """
        if space == 'nii':
            return np.sqrt((p1.x_nii - p2.x_nii)**2 + (p1.y_nii - p2.y_nii)**2 + (p1.z_nii - p2.z_nii)**2)
        elif space == 'native':
            return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        else:
            raise ValueError(f"Invalid space: {space}")

    def _find_cube(self, node):
        """
        Find the cube index containing the node's coordinates.

        Args:
            node: Node object with x_nii, y_nii, z_nii attributes.

        Returns:
            int or None: Index of the cube containing the node, or None if not found.
        """
        # Check cache first
        if node in self.node_to_cube:
            return self.node_to_cube[node]

        # Compute node coordinates
        aa = tuple(np.round(np.array([node.x_nii, node.y_nii, node.z_nii])).astype(int).flatten())

        # Search through cubes
        for cube_idx, cube in enumerate(self.cubes):
            start = cube['start']
            end = cube['end']
            if (start[0] <= aa[0] < end[0] and
                start[1] <= aa[1] < end[1] and
                start[2] <= aa[2] < end[2]):
                self.node_to_cube[node] = cube_idx
                return cube_idx

        # Node not in any cube
        self.node_to_cube[node] = None
        return None

    def _calculate_neuronal_branch_length_in_cubes(self):
        """
        Calculate terminal arbor lengths within cubes and total neuron length based on those arbors.

        Returns:
            tuple: (cube_lengths, neuron_total_length)
                - cube_lengths: defaultdict mapping cube indices to total terminal arbor lengths.
                - neuron_total_length: Total length of all terminal arbor edges.
        """
        cube_lengths = defaultdict(float)

        # Step 1: Get all terminal nodes
        terminal_nodes = self.neuron.terminal_nodes
        terminal_in_region_count = 0
        # Step 2: Calculate terminal arbor lengths within cubes
        for terminal_node in terminal_nodes:
            # Find the cube containing the terminal node
            terminal_cube = self._find_cube(terminal_node)
            if terminal_cube is None:  # Skip if terminal node is not in any cube
                continue
            terminal_in_region_count +=1
            # Trace back from terminal node through parents
            current_node = terminal_node
            while current_node.parent is not None:
                parent_node = self.neuron.nodes[current_node.parent]
                parent_cube = self._find_cube(parent_node)
                current_cube = self._find_cube(current_node)
                if parent_cube is not None:
                    edge_length = round(self._distance(parent_node, current_node), 3)

                    # Include edge length even if parent is not in the same cube
                    # as long as it is within the region's cube boundaries (any cube in the region)
                    # But if the parent node's cube is the same as the terminal node's cube, the cube lengths should be attributed to the current node's cube
                    if parent_cube == terminal_cube:
                        
                        cube_lengths[terminal_cube] += edge_length
                    else:
                        # include the edge length to the parent
                        cube_lengths[current_cube] += edge_length
                    current_node = parent_node  # Move to parent
                else:
                    break

        # Store results as instance variables
        self.cube_lengths = cube_lengths

        return cube_lengths,terminal_in_region_count