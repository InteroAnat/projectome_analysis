"""
Hierarchy resolution and aggregation for brain region analysis.

Provides functions to resolve regions to different hierarchy levels
using both ARM key text files and CSV hierarchy tables.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import re
import os

import pandas as pd
import numpy as np

# Import hierarchy table for type hints
from . import hierarchy_table as ht


class RegionHierarchy:
    """
    Legacy ARM key parser for L1-L2 hierarchy.
    
    Parses ARM_key_all.txt format:
    Index\tAbbreviation\tFull_Name\tFirst_Level\tLast_Level
    """
    
    def __init__(self):
        self.level_map: Dict[str, Dict[int, str]] = {}
        self.index_to_abbr: Dict[int, str] = {}
        self.abbr_to_index: Dict[str, int] = {}
    
    @classmethod
    def from_file(cls, path: str) -> "RegionHierarchy":
        """Load hierarchy from ARM key file."""
        instance = cls()
        

        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Skip header if present
        start_idx = 0
        for i, line in enumerate(lines[:5]):
            if 'Index' in line or 'Abbreviation' in line:
                start_idx = i + 1
                break
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) >= 5:
                try:
                    idx = int(parts[0])
                    abbr = parts[1].strip()
                    full_name = parts[2].strip()
                    first_level = parts[3].strip()
                    last_level = parts[4].strip()
                    
                    # Store index mappings
                    instance.index_to_abbr[idx] = abbr
                    instance.abbr_to_index[abbr] = idx
                    
                    # Build level map (L1 = first_level, L2 = last_level)
                    instance.level_map[abbr] = {
                        1: first_level,
                        2: last_level,
                    }
                    
                except (ValueError, IndexError):
                    continue
        
        return instance
    
    def get_at_level(self, region: str, level: int) -> Optional[str]:
        """Get region at specified level (1 or 2 for ARM key)."""
        region = str(region).strip()
        
        # Strip prefix if present
        for prefix in ['CL_', 'CR_', 'SL_', 'SR_']:
            if region.startswith(prefix):
                region = region[len(prefix):]
                break
        
        if region in self.level_map:
            return self.level_map[region].get(level)
        
        return None


def _strip_prefix(name: str) -> str:
    """Strip CL_/CR_/SL_/SR_ prefix from region name."""
    if not name:
        return name
    name = str(name).strip()
    for prefix in ['CL_', 'CR_', 'SL_', 'SR_']:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def resolve_to_level(region: str, level: int, hierarchy: Dict = None,
                     hierarchy_table=None) -> Optional[str]:
    """
    Resolve a region to its name at specified hierarchy level.
    
    Args:
        region: Region name (e.g., "CL_ACC", "1001")
        level: Target hierarchy level (1-6)
        hierarchy: Legacy ARM hierarchy dict {region: {level: parent}}
        hierarchy_table: HierarchyTable or DualHierarchyTable instance
    
    Returns:
        Region name at target level, or None if not found
    """
    region = str(region).strip()
    
    # Try hierarchy table first (more reliable with index support)
    if hierarchy_table is not None:
        result = hierarchy_table.get_at_level(region, level)
        if result:
            return result
    
    # Fallback to legacy hierarchy
    if hierarchy:
        if isinstance(hierarchy, RegionHierarchy):
            return hierarchy.get_at_level(region, level)
        elif isinstance(hierarchy, dict):
            stripped = _strip_prefix(region)
            if stripped in hierarchy:
                return hierarchy[stripped].get(level)
    
    return None


def _aggregate_length_dict_to_level(length_dict: Dict[str, float],
                                    hierarchy,
                                    level: int,
                                    hierarchy_table=None,
                                    strip_prefixes: bool = False) -> Tuple[Dict[str, float], List[str]]:
    """
    Aggregate region lengths to target hierarchy level.
    
    Uses hierarchy_table for index-based lookup when available,
    falls back to name-based matching.
    
    Args:
        length_dict: {region_name_or_index: length}
        hierarchy: Legacy ARM hierarchy dict or RegionHierarchy
        level: Target level (1-6)
        hierarchy_table: HierarchyTable or DualHierarchyTable
        strip_prefixes: If True, remove CL_/CR_/SL_/SR_ prefixes from result
    
    Returns:
        Tuple of (aggregated_lengths, unmapped_regions)
    """
    aggregated = defaultdict(float)
    unmapped = []
    
    # Use hierarchy_table if available (more accurate with index support)
    if hierarchy_table is not None:
        return hierarchy_table.aggregate_to_level(length_dict, level, strip_prefixes=strip_prefixes)
    
    # Fallback to legacy name-based matching
    for region, length in length_dict.items():
        target = resolve_to_level(region, level, hierarchy, None)
        
        if target:
            if strip_prefixes:
                target = _strip_prefix(target)
            aggregated[target] += length
        else:
            # Try numeric index fallback
            try:
                idx = int(region)
                # If it's a number, we couldn't map it without table
                unmapped.append(region)
            except ValueError:
                # Strip prefix and try again
                base = _strip_prefix(region)
                if base != region:
                    target = resolve_to_level(base, level, hierarchy, None)
                    if target:
                        if strip_prefixes:
                            target = _strip_prefix(target)
                        aggregated[target] += length
                    else:
                        unmapped.append(region)
                else:
                    unmapped.append(region)
    
    return dict(aggregated), unmapped


def add_soma_hierarchy_column(df, hierarchy, soma_col: str = "Soma_Region",
                               max_level: int = 6,
                               hierarchy_table=None):
    """
    Add hierarchy columns for soma regions.
    
    Creates columns: Soma_Level_1, Soma_Level_2, ..., Soma_Level_{max_level}
    
    Args:
        df: DataFrame with soma region column
        hierarchy: RegionHierarchy or dict
        soma_col: Name of soma region column
        max_level: Maximum hierarchy level to create
        hierarchy_table: HierarchyTable for index-based lookup
    
    Returns:
        Modified DataFrame
    """
    import pandas as pd
    
    if soma_col not in df.columns:
        return df
    
    for level in range(1, max_level + 1):
        col_name = f"Soma_Level_{level}"
        
        results = []
        for soma_region in df[soma_col]:
            resolved = resolve_to_level(soma_region, level, hierarchy, hierarchy_table)
            results.append(resolved if resolved else None)
        
        df[col_name] = results
    
    return df


def extract_soma_level(df, level: int = 6) -> pd.Series:
    """
    Extract soma region data at specified hierarchy level.
    
    Args:
        df: DataFrame with hierarchy columns
        level: Hierarchy level to extract
    
    Returns:
        Series with region names at specified level
    """
    import pandas as pd
    
    col_name = f"Soma_Level_{level}"
    if col_name in df.columns:
        return df[col_name]
    
    # Fallback: use original soma region
    if "Soma_Region" in df.columns:
        return df["Soma_Region"]
    
    return pd.Series([None] * len(df), index=df.index)


def hierarchy_summary(df, max_level: int = 6) -> Any:
    """
    Create summary of hierarchy data.
    
    Args:
        df: DataFrame with hierarchy columns
        max_level: Maximum level to summarize
    
    Returns:
        Summary DataFrame
    """
    import pandas as pd
    
    rows = []
    
    for level in range(1, max_level + 1):
        col_name = f"Soma_Level_{level}"
        if col_name in df.columns:
            non_null = df[col_name].notna().sum()
            unique = df[col_name].nunique(dropna=True)
            rows.append({
                'Level': level,
                'Column': col_name,
                'Non_Null': non_null,
                'Unique_Regions': unique,
            })
    
    return pd.DataFrame(rows)


def add_projection_length_hierarchy(df, hierarchy: Dict, base_col: str,
                                    max_level: int = 6,
                                    min_level: int = 1,
                                    hierarchy_table=None) -> Any:
    """
    Add hierarchy columns for projection length data.
    
    Creates columns: Region_Projection_Length_L1, L2, ... up to max_level
    
    Args:
        df: DataFrame with projection data
        hierarchy: Legacy ARM hierarchy dict or RegionHierarchy
        base_col: Base column name (e.g., "Region_projection_length")
        max_level: Maximum hierarchy level to create
        min_level: Minimum hierarchy level to create (default 1)
        hierarchy_table: HierarchyTable or DualHierarchyTable for index-based lookup
    
    Returns:
        Modified DataFrame
    """
    import pandas as pd
    import ast
    
    def safe_literal_eval(val):
        """Safely evaluate string representation of dict/list."""
        if pd.isna(val) or val == '':
            return {}
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return {}
        return {}
    
    def aggregate_to_level_safe(length_dict, target_level):
        """Helper to aggregate with error handling."""
        if not length_dict:
            return {}
        agg, _ = _aggregate_length_dict_to_level(
            length_dict, hierarchy, target_level, hierarchy_table, strip_prefixes=False
        )
        return agg
    
    # Process each level
    for level in range(min_level, max_level + 1):
        col_name = f"Region_Projection_Length_L{level}"
        
        results = []
        for idx, row in df.iterrows():
            if base_col in row:
                length_dict = safe_literal_eval(row[base_col])
                agg = aggregate_to_level_safe(length_dict, level)
                results.append(agg)
            else:
                results.append({})
        
        df[col_name] = results
    
    return df


def add_projection_hierarchy(df, hierarchy, max_level: int = 6,
                              hierarchy_table=None):
    """
    Add projection hierarchy columns (terminal region-based).
    
    Similar to add_projection_length_hierarchy but for terminal regions.
    Creates columns: Terminal_Level_1, Terminal_Level_2, etc.
    
    Args:
        df: DataFrame with projection data
        hierarchy: Legacy hierarchy
        max_level: Maximum hierarchy level
        hierarchy_table: HierarchyTable for index-based lookup
    
    Returns:
        Modified DataFrame
    """
    import pandas as pd
    import ast
    
    def safe_literal_eval(val):
        if pd.isna(val) or val == '':
            return []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        return []
    
    terminal_col = "Terminal_Regions"
    if terminal_col not in df.columns:
        return df
    
    for level in range(1, max_level + 1):
        col_name = f"Terminal_Level_{level}"
        
        results = []
        for idx, row in df.iterrows():
            terminals = safe_literal_eval(row.get(terminal_col, []))
            
            resolved = []
            for term in terminals:
                r = resolve_to_level(term, level, hierarchy, hierarchy_table)
                if r and r not in resolved:
                    resolved.append(r)
            
            results.append(resolved)
        
        df[col_name] = results
    
    return df


def aggregate_projection_lengths_to_level(projection_lengths: Dict[str, Dict[str, float]],
                                          target_level: int,
                                          hierarchy: Dict = None,
                                          hierarchy_table=None) -> Dict[str, Dict[str, float]]:
    """
    Aggregate projection lengths for all neurons to target hierarchy level.
    
    Args:
        projection_lengths: {neuron_id: {region: length}}
        target_level: Target hierarchy level
        hierarchy: Legacy ARM hierarchy
        hierarchy_table: HierarchyTable for index-based lookup
    
    Returns:
        Aggregated dict: {neuron_id: {parent_region: total_length}}
    """
    result = {}
    
    for neuron_id, lengths in projection_lengths.items():
        agg, _ = _aggregate_length_dict_to_level(
            lengths, hierarchy, target_level, hierarchy_table
        )
        result[neuron_id] = agg
    
    return result


def get_region_at_level(region: str, level: int, 
                        hierarchy_table=None,
                        hierarchy: Dict = None) -> Optional[str]:
    """
    Get the parent region at specified hierarchy level.
    
    Convenience function that tries hierarchy_table first, then falls back
    to legacy hierarchy.
    
    Args:
        region: Region name or index
        level: Target level (1-6)
        hierarchy_table: HierarchyTable/DualHierarchyTable
        hierarchy: Legacy ARM hierarchy dict
    
    Returns:
        Parent region name or None
    """
    return resolve_to_level(region, level, hierarchy, hierarchy_table)


def build_level_summary(df, level: int = 6, value_col: str = "Projection_Length") -> Dict[str, float]:
    """
    Build summary statistics at specified hierarchy level.
    
    Args:
        df: DataFrame with hierarchy columns
        level: Hierarchy level to summarize
        value_col: Value column name in the dict
    
    Returns:
        Summary dict: {region: total_value}
    """
    import pandas as pd
    import ast
    
    col_name = f"Region_Projection_Length_L{level}"
    if col_name not in df.columns:
        return {}
    
    summary = defaultdict(float)
    
    for val in df[col_name]:
        if pd.isna(val):
            continue
        
        if isinstance(val, dict):
            length_dict = val
        elif isinstance(val, str):
            try:
                length_dict = ast.literal_eval(val)
            except:
                continue
        else:
            continue
        
        for region, length in length_dict.items():
            summary[region] += length
    
    return dict(summary)


def get_all_regions_at_level(df, level: int = 6) -> List[str]:
    """
    Get list of all unique regions at specified hierarchy level.
    
    Args:
        df: DataFrame with hierarchy columns
        level: Hierarchy level
    
    Returns:
        Sorted list of unique region names
    """
    import pandas as pd
    import ast
    
    col_name = f"Region_Projection_Length_L{level}"
    if col_name not in df.columns:
        return []
    
    regions = set()
    
    for val in df[col_name]:
        if pd.isna(val):
            continue
        
        if isinstance(val, dict):
            length_dict = val
        elif isinstance(val, str):
            try:
                length_dict = ast.literal_eval(val)
            except:
                continue
        else:
            continue
        
        regions.update(length_dict.keys())
    
    return sorted(regions)


def validate_hierarchy_columns(df, expected_levels: List[int] = None) -> Dict[str, Any]:
    """
    Validate that hierarchy columns exist and contain data.
    
    Args:
        df: DataFrame to validate
        expected_levels: List of expected level numbers (default [1-6])
    
    Returns:
        Validation report dict
    """
    if expected_levels is None:
        expected_levels = list(range(1, 7))
    
    report = {
        'missing_columns': [],
        'empty_columns': [],
        'valid_columns': [],
        'level_coverage': {}
    }
    
    for level in expected_levels:
        col_name = f"Region_Projection_Length_L{level}"
        
        if col_name not in df.columns:
            report['missing_columns'].append(col_name)
            continue
        
        # Check for non-empty values
        non_empty = df[col_name].notna().sum()
        has_data = non_empty > 0
        
        if has_data:
            report['valid_columns'].append(col_name)
            report['level_coverage'][level] = {
                'rows_with_data': int(non_empty),
                'total_rows': len(df)
            }
        else:
            report['empty_columns'].append(col_name)
    
    report['is_valid'] = len(report['valid_columns']) > 0
    return report
    report['is_valid'] = len(report['valid_columns']) > 0
    return report


def strip_prefixes_from_dict(length_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Strip CL_/CR_/SL_/SR_ prefixes from dictionary keys.
    Used when saving/displaying projection data (not for internal processing).
    
    Args:
        length_dict: Dictionary with region names as keys
    
    Returns:
        Dictionary with prefixes stripped from keys
    """
    result = {}
    for region, value in length_dict.items():
        clean_region = _strip_prefix(region)
        # Sum values if the same region appears multiple times after stripping
        result[clean_region] = result.get(clean_region, 0) + value
    return result


def strip_prefixes_from_projection_columns(df, levels: List[int] = None) -> Any:
    """
    Strip prefixes from projection hierarchy columns for display/saving.
    Creates new columns with '_clean' suffix.
    
    Args:
        df: DataFrame with projection hierarchy columns
        levels: List of levels to process (default [1-6])
    
    Returns:
        Modified DataFrame with additional clean columns
    """
    import pandas as pd
    
    if levels is None:
        levels = list(range(1, 7))
    
    for level in levels:
        col_name = f"Region_Projection_Length_L{level}"
        clean_col_name = f"Region_Projection_Length_L{level}_clean"
        
        if col_name not in df.columns:
            continue
        
        clean_values = []
        for val in df[col_name]:
            if pd.isna(val):
                clean_values.append({})
            elif isinstance(val, dict):
                clean_values.append(strip_prefixes_from_dict(val))
            else:
                clean_values.append({})
        
        df[clean_col_name] = clean_values
    
    return df
