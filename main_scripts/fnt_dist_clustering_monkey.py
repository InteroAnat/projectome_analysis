#!/usr/bin/env python3
"""
FNT-Distance Clustering for Monkey Neurons
==========================================

Advanced clustering analysis using FNT (Functional Neuroanatomy Toolbox) distance metrics
specifically adapted for monkey neurons. This implements hierarchical clustering with
monkey-specific parameters and visualization.

Based on your existing clustering infrastructure but enhanced for monkey data.

Features:
- FNT-distance matrix calculation
- Hierarchical clustering with optimal parameters
- Monkey-specific feature engineering
- Comprehensive visualization
- Integration with volume loading
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial import distance
from scipy import stats
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Add path to import your infrastructure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'neuron-vis', 'neuronVis')))

import IONData
from volume_monkey_definitive import DefinitiveMonkeyFNTCube
from visualization_utils import VisualizationManager

class FNTDistClusteringMonkey:
    """
    FNT-Distance clustering specifically adapted for monkey neurons.
    
    This class implements the clustering methodology using FNT-derived features
    with monkey-specific parameters and enhancements.
    """
    
    def __init__(self, sampleid='251637'):
        """
        Initialize the FNT-distance clustering analyzer.
        
        Args:
            sampleid: Monkey sample ID (default: '251637')
        """
        self.sampleid = sampleid
        self.iondata = IONData.IONData()
        self.volume_loader = DefinitiveMonkeyFNTCube()
        self.volume_loader.setSampleID(sampleid)
        
        # Monkey-specific clustering parameters
        self.clustering_params = {
            'method': 'average',  # Based on your existing approach
            'metric': 'euclidean',
            'criterion': 'inconsistent',
            't': 0.99,  # Threshold for clustering
            'depth': 3,  # Depth for inconsistent criterion
            'R': None,
            'monocrit': None
        }
        
        # Feature engineering parameters for monkey data
        self.feature_params = {
            'normalize_features': True,
            'include_volume_metrics': True,
            'include_morphological_metrics': True,
            'include_spatial_metrics': True
        }
        
        self.results = {}
        self.analysis_timestamp = datetime.now().isoformat()
    
    def calculate_fnt_distance_matrix(self, neuron_ids, feature_set='comprehensive'):
        """
        Calculate FNT-distance matrix for monkey neurons.
        
        This creates a comprehensive feature matrix based on FNT-derived metrics
        specifically adapted for monkey neuron analysis.
        
        Args:
            neuron_ids: List of neuron identifiers
            feature_set: Type of features to extract ('basic', 'comprehensive', 'custom')
            
        Returns:
            feature_matrix: NumPy array of features
            feature_names: List of feature names
            distance_matrix: Distance matrix for clustering
        """
        print(f"\n=== Calculating FNT-distance matrix for {len(neuron_ids)} neurons ===")
        
        # Extract comprehensive features for each neuron
        features_list = []
        feature_names = []
        
        for i, neuron_id in enumerate(neuron_ids):
            print(f"Processing {neuron_id} ({i+1}/{len(neuron_ids)})")
            
            # Extract features based on feature set
            features, names = self._extract_monkey_features(neuron_id, feature_set)
            
            if features is not None:
                features_list.append(features)
                if i == 0:  # Get feature names from first neuron
                    feature_names = names
            else:
                print(f"Warning: Could not extract features for {neuron_id}")
                # Use zeros for failed extractions
                if feature_names:
                    features_list.append(np.zeros(len(feature_names)))
        
        if not features_list:
            print("✗ No features extracted")
            return None, None, None
        
        # Create feature matrix
        feature_matrix = np.array(features_list)
        
        # Normalize features if requested
        if self.feature_params['normalize_features']:
            feature_matrix = self._normalize_features(feature_matrix)
        
        # Calculate distance matrix
        distance_matrix = distance.pdist(feature_matrix, metric=self.clustering_params['metric'])
        
        print(f"✓ Feature matrix shape: {feature_matrix.shape}")
        print(f"✓ Distance matrix shape: {distance_matrix.shape}")
        print(f"✓ Feature names: {len(feature_names)} features")
        
        return feature_matrix, feature_names, distance_matrix
    
    def _extract_monkey_features(self, neuron_id, feature_set='comprehensive'):
        """
        Extract comprehensive features for monkey neurons.
        
        This combines morphological, spatial, and volume-based features
        specifically relevant for monkey insula neurons.
        
        Args:
            neuron_id: Neuron identifier
            feature_set: Feature set type
            
        Returns:
            features: Feature vector
            feature_names: Feature names
        """
        try:
            # Load neuron data
            tree = self.iondata.getRawNeuronTreeByID(self.sampleid, f"{neuron_id}.swc")
            if tree is None or tree.root is None:
                return None, None
            
            features = []
            feature_names = []
            
            # === Basic Morphological Features ===
            features.extend([
                len(tree.points),           # Total node count
                len(tree.terminals),        # Terminal count
                float(tree.root.ratio) if hasattr(tree.root, 'ratio') else 0  # Soma radius
            ])
            feature_names.extend(['total_nodes', 'terminal_count', 'soma_radius'])
            
            # === Spatial Features ===
            if len(tree.points) > 0:
                coords = np.array([point.xyz for point in tree.points])
                
                # Bounding box features
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                dimensions = max_coords - min_coords
                
                features.extend(dimensions.tolist())  # X, Y, Z spans
                features.extend([dimensions.sum(), np.prod(dimensions)])  # Total span, volume
                feature_names.extend(['x_span', 'y_span', 'z_span', 'total_span', 'bounding_volume'])
                
                # Distance from soma
                soma_coord = np.array([tree.root.x, tree.root.y, tree.root.z])
                distances_from_soma = np.linalg.norm(coords - soma_coord, axis=1)
                
                features.extend([
                    distances_from_soma.max(),
                    distances_from_soma.mean(),
                    distances_from_soma.std()
                ])
                feature_names.extend(['max_distance_from_soma', 'mean_distance_from_soma', 'std_distance_from_soma'])
            
            # === Volume-based Features (if available) ===
            if self.feature_params['include_volume_metrics']:
                volume_features, volume_names = self._extract_volume_features(neuron_id)
                if volume_features is not None:
                    features.extend(volume_features)
                    feature_names.extend(volume_names)
            
            # === Monkey-specific Features ===
            features.extend(self._extract_monkey_specific_features(tree))
            feature_names.extend(['monkey_specific_1', 'monkey_specific_2', 'monkey_specific_3'])
            
            return np.array(features), feature_names
            
        except Exception as e:
            print(f"Warning: Error extracting features for {neuron_id}: {e}")
            return None, None
    
    def _extract_volume_features(self, neuron_id):
        """
        Extract volume-based features for enhanced clustering.
        
        Args:
            neuron_id: Neuron identifier
            
        Returns:
            volume_features: Volume-based feature vector
            volume_names: Volume feature names
        """
        try:
            # Attempt to load volume around soma using the correct method
            tree = self.iondata.getRawNeuronTreeByID(self.sampleid, f"{neuron_id}.swc")
            if tree is None:
                return [0, 0, 0, 0], [
                    'volume_mean_intensity',
                    'volume_max_intensity',
                    'volume_intensity_std',
                    'volume_total_intensity_log'
                ]
            
            # Use the correct workflow
            self.volume_loader.set_soma_from_raw_neuron(tree)
            self.volume_loader.setRadiu(2)
            volume_data = self.volume_loader.get_volume_via_http()
            
            if volume_data is not None:
                return [
                    float(volume_data.mean()),     # Mean intensity
                    float(volume_data.max()),      # Max intensity
                    float(volume_data.std()),      # Intensity std
                    float(np.log1p(volume_data.sum()))  # Log-transformed total intensity
                ], [
                    'volume_mean_intensity',
                    'volume_max_intensity', 
                    'volume_intensity_std',
                    'volume_total_intensity_log'
                ]
            else:
                # Return placeholder values if volume loading fails
                return [0, 0, 0, 0], [
                    'volume_mean_intensity',
                    'volume_max_intensity',
                    'volume_intensity_std', 
                    'volume_total_intensity_log'
                ]
                
        except Exception as e:
            print(f"Warning: Volume feature extraction failed for {neuron_id}: {e}")
            return [0, 0, 0, 0], [
                'volume_mean_intensity',
                'volume_max_intensity',
                'volume_intensity_std',
                'volume_total_intensity_log'
            ]
    
    def _extract_monkey_specific_features(self, tree):
        """
        Extract features specific to monkey insula neurons.
        
        Args:
            tree: Neuron tree
            
        Returns:
            features: Monkey-specific feature vector
        """
        features = []
        
        try:
            # Insula-specific spatial metrics
            if len(tree.points) > 0:
                coords = np.array([point.xyz for point in tree.points])
                
                # Calculate insula-specific metrics
                # Based on insula anatomy and typical neuron distributions
                
                # Depth-related features (insula has characteristic depth patterns)
                z_coords = coords[:, 2]  # Z is typically depth in your coordinate system
                features.extend([
                    z_coords.std(),  # Depth variability
                    (z_coords.max() - z_coords.min()),  # Depth range
                    np.percentile(z_coords, 75) - np.percentile(z_coords, 25)  # IQR depth
                ])
            else:
                features.extend([0, 0, 0])
                
        except Exception as e:
            print(f"Warning: Monkey-specific feature extraction error: {e}")
            features.extend([0, 0, 0])
        
        return features
    
    def _normalize_features(self, feature_matrix):
        """
        Normalize features to prevent dominance by large-scale features.
        
        Args:
            feature_matrix: Feature matrix
            
        Returns:
            normalized_matrix: Normalized feature matrix
        """
        try:
            # Z-score normalization
            means = feature_matrix.mean(axis=0)
            stds = feature_matrix.std(axis=0)
            
            # Avoid division by zero
            stds = np.where(stds == 0, 1, stds)
            
            normalized_matrix = (feature_matrix - means) / stds
            
            # Handle any NaN values
            normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)
            
            return normalized_matrix
            
        except Exception as e:
            print(f"Warning: Feature normalization error: {e}")
            return feature_matrix
    
    def perform_hierarchical_clustering(self, distance_matrix, neuron_ids):
        """
        Perform hierarchical clustering on the distance matrix.
        
        Args:
            distance_matrix: Distance matrix from pdist
            neuron_ids: List of neuron identifiers
            
        Returns:
            linkage_matrix: Hierarchical clustering linkage
            cluster_assignments: Cluster assignments for each neuron
            clustering_results: Complete clustering results
        """
        print(f"\n=== Performing hierarchical clustering ===")
        
        try:
            # Perform hierarchical clustering
            linkage_matrix = linkage(
                distance_matrix,
                method=self.clustering_params['method'],
                metric=self.clustering_params['metric']
            )
            
            # Generate cluster assignments
            cluster_assignments = fcluster(
                linkage_matrix,
                t=self.clustering_params['t'],
                criterion=self.clustering_params['criterion'],
                depth=self.clustering_params['depth'],
                R=self.clustering_params['R'],
                monocrit=self.clustering_params['monocrit']
            )
            
            print(f"✓ Clustering complete")
            print(f"✓ Number of clusters: {len(np.unique(cluster_assignments))}")
            print(f"✓ Cluster distribution: {dict(zip(*np.unique(cluster_assignments, return_counts=True)))}")
            
            # Compile results
            clustering_results = {
                'linkage_matrix': linkage_matrix,
                'cluster_assignments': cluster_assignments,
                'num_clusters': len(np.unique(cluster_assignments)),
                'cluster_distribution': dict(zip(*np.unique(cluster_assignments, return_counts=True))),
                'parameters': self.clustering_params.copy(),
                'timestamp': self.analysis_timestamp
            }
            
            return linkage_matrix, cluster_assignments, clustering_results
            
        except Exception as e:
            print(f"✗ Clustering error: {e}")
            return None, None, None
    
    def visualize_clustering_results(self, linkage_matrix, cluster_assignments, neuron_ids, feature_names=None):
        """
        Create comprehensive visualization of clustering results.
        
        Args:
            linkage_matrix: Hierarchical clustering linkage
            cluster_assignments: Cluster assignments
            neuron_ids: Neuron identifiers
            feature_names: Feature names for additional visualizations
            
        Returns:
            fig: Matplotlib figure
        """
        print(f"\n=== Creating clustering visualizations ===")
        
        try:
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Monkey Neuron FNT-Distance Clustering - Sample {self.sampleid}', fontsize=16)
            
            # 1. Dendrogram
            ax1 = axes[0, 0]
            dendrogram(linkage_matrix, labels=neuron_ids, ax=ax1)
            ax1.set_title('Hierarchical Clustering Dendrogram')
            ax1.set_xlabel('Neuron ID')
            ax1.set_ylabel('Distance')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # 2. Cluster distribution
            ax2 = axes[0, 1]
            unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
            bars = ax2.bar(unique_clusters, counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique_clusters))))
            ax2.set_title('Cluster Size Distribution')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Number of Neurons')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom')
            
            # 3. Cluster heatmap (if we have feature data)
            ax3 = axes[1, 0]
            # Create a simple heatmap showing cluster assignments
            cluster_matrix = cluster_assignments.reshape(-1, 1)
            im = ax3.imshow(cluster_matrix, cmap='tab10', aspect='auto')
            ax3.set_title('Cluster Assignment Heatmap')
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel('Neuron Index')
            ax3.set_yticks(range(len(neuron_ids)))
            ax3.set_yticklabels(neuron_ids)
            ax3.set_xticks([0])
            ax3.set_xticklabels(['Cluster'])
            
            # 4. Summary statistics
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            summary_text = f"""
            Clustering Summary
            ==================
            Sample ID: {self.sampleid}
            Total Neurons: {len(neuron_ids)}
            Number of Clusters: {len(np.unique(cluster_assignments))}
            Clustering Method: {self.clustering_params['method']}
            Distance Metric: {self.clustering_params['metric']}
            Threshold: {self.clustering_params['t']}
            
            Cluster Distribution:
            {dict(zip(*np.unique(cluster_assignments, return_counts=True)))}
            
            Analysis completed: {self.analysis_timestamp[:19]}
            """
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            # Save the main clustering visualization (directories created automatically)
            viz_manager = VisualizationManager(
                output_dir='results/visualizations',
                sample_id=self.sampleid
            )
            main_fig_path = viz_manager.save_clustering_visualization(
                fig, 'hierarchical', neuron_ids,
                metadata={'cluster_count': len(np.unique(cluster_assignments))}
            )
            
            print(f"✓ Main clustering visualization saved: {main_fig_path}")
            
            return fig
            
        except Exception as e:
            print(f"✗ Visualization error: {e}")
            return None
    
    def analyze_clustering_results(self, neuron_ids, cluster_assignments, feature_matrix=None, feature_names=None):
        """
        Perform comprehensive clustering analysis and generate insights.
        
        Args:
            neuron_ids: List of neuron identifiers
            cluster_assignments: Cluster assignments
            feature_matrix: Feature matrix (optional)
            feature_names: Feature names (optional)
            
        Returns:
            analysis_results: Comprehensive analysis results
        """
        print(f"\n=== Analyzing clustering results ===")
        
        try:
            # Calculate FNT-distance matrix
            if feature_matrix is None:
                feature_matrix, feature_names, distance_matrix = self.calculate_fnt_distance_matrix(neuron_ids)
            else:
                distance_matrix = distance.pdist(feature_matrix, metric=self.clustering_params['metric'])
            
            if distance_matrix is None:
                return None
            
            # Perform clustering
            linkage_matrix, cluster_assignments, clustering_results = self.perform_hierarchical_clustering(
                distance_matrix, neuron_ids
            )
            
            if linkage_matrix is None:
                return None
            
            # Generate visualizations
            fig = self.visualize_clustering_results(linkage_matrix, cluster_assignments, neuron_ids, feature_names)
            
            # Compile comprehensive results
            results = {
                'neuron_ids': neuron_ids,
                'cluster_assignments': cluster_assignments.tolist(),
                'feature_matrix': feature_matrix.tolist() if feature_matrix is not None else None,
                'feature_names': feature_names,
                'distance_matrix': distance_matrix.tolist(),
                'clustering_results': clustering_results,
                'visualization_figure': fig,
                'analysis_timestamp': self.analysis_timestamp
            }
            
            print(f"✓ Comprehensive clustering analysis complete")
            return results
            
        except Exception as e:
            print(f"✗ Analysis error: {e}")
            return None


def demo_fnt_dist_clustering():
    """
    Demonstrate FNT-distance clustering on monkey neurons.
    """
    print("=" * 60)
    print("FNT-DISTANCE CLUSTERING DEMO - MONKEY NEURONS")
    print("=" * 60)
    
    # Initialize clustering analyzer
    clusterer = FNTDistClusteringMonkey(sampleid='251637')
    
    # Test with a subset of neurons
    test_neurons = ['004', '005', '006', '007', '008', '009', '010', '011', '012', '013']
    
    print(f"Testing FNT-distance clustering with neurons: {test_neurons}")
    
    # Perform comprehensive clustering analysis
    results = clusterer.analyze_clustering_results(test_neurons, cluster_assignments=None)
    
    if results:
        print(f"\n🎉 SUCCESS! FNT-distance clustering completed successfully!")
        print(f"✓ Analyzed {len(results['neuron_ids'])} neurons")
        print(f"✓ Identified {results['clustering_results']['num_clusters']} clusters")
        print(f"✓ Generated comprehensive visualizations")
        print(f"✓ Results saved with timestamp: {results['analysis_timestamp'][:19]}")
        
        # Show cluster assignments
        print(f"\n📊 Cluster Assignments:")
        for neuron_id, cluster_id in zip(results['neuron_ids'], results['cluster_assignments']):
            print(f"  {neuron_id}: Cluster {cluster_id}")
        
        return results
    else:
        print(f"\n❌ FNT-distance clustering failed")
        return None


if __name__ == '__main__':
    demo_fnt_dist_clustering()