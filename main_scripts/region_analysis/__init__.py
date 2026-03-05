"""
region_analysis - Monkey Projectome Region Analysis Package
Version: 4.3.0 (2026-03-08)

Multi-sheet export, projection strength, user CSV hierarchy,
laterality parser, 3 separate reports.
"""

from region_analysis.population import PopulationRegionAnalysis
from region_analysis.hierarchy import (
    RegionHierarchy,
    add_soma_hierarchy_column,
    add_projection_length_hierarchy,
    add_projection_hierarchy,
    extract_soma_level,
    hierarchy_summary,
    resolve_to_level,
)
from region_analysis.hierarchy_table import HierarchyTable
from region_analysis.classifier import NeuronClassifier
from region_analysis.neuron_analysis import RegionAnalysisPerNeuron
from region_analysis.output_manager import OutputManager
from region_analysis.laterality import LateralityParser, add_laterality_columns
from region_analysis.utils import load_processed_df, parse_terminal_regions

from region_analysis.plotting import (
    plot_soma_distribution_df,
    plot_type_distribution_df,
    plot_terminal_distribution_df,
    plot_projection_sites_count_df,
    plot_region_distribution,
    plot_laterality_summary_df,
    plot_neuron_projections,
)

__version__ = "4.3.0"

__all__ = [
    "PopulationRegionAnalysis",
    "RegionHierarchy",
    "HierarchyTable",
    "NeuronClassifier",
    "RegionAnalysisPerNeuron",
    "OutputManager",
    "LateralityParser",
    "add_soma_hierarchy_column",
    "add_projection_length_hierarchy",
    "extract_soma_level",
    "hierarchy_summary",
    "add_projection_hierarchy",
    "add_laterality_columns",
    "resolve_to_level",
    "plot_soma_distribution_df",
    "plot_type_distribution_df",
    "plot_terminal_distribution_df",
    "plot_projection_sites_count_df",
    "plot_region_distribution",
    "plot_laterality_summary_df",
    "plot_neuron_projections",
    "load_processed_df",
    "parse_terminal_regions",
]