"""
Hierarchy table parser for CHARM/SARM CSV files.
Supports v2 format with Level_X, Level_X_abbr, Level_X_index columns.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path


class HierarchyTable:
    """
    Parses and queries hierarchy CSV files (CHARM/SARM).
    
    Supports v2 format with columns:
    - Level_1, Level_2, ... Level_6 (full names)
    - Level_1_abbr, Level_2_abbr, ... (abbreviations with prefixes like CL_, SL_)
    - Level_1_index, Level_2_index, ... (numeric indices matching ARM_key_all.txt)
    
    Builds both:
    - region_paths: abbreviation -> list of region names at each level
    - index_paths: numeric index -> list of region names at each level
    """
    
    def __init__(self, df: pd.DataFrame = None, name: str = ""):
        self.df = df
        self.name = name
        self.region_paths: Dict[str, List[str]] = {}  # abbr -> path
        self.index_paths: Dict[int, List[str]] = {}   # index -> path
        self.max_level = 0
        
        if df is not None:
            self._parse_df()
    
    @staticmethod
    def _detect_level_columns(df: pd.DataFrame) -> Dict[int, str]:
        """Detect level columns (Level_1, L1, etc.) and return {level: column_name}."""
        level_cols = {}
        
        for col in df.columns:
            col_str = str(col)
            # Match patterns like "Level_1", "Level 1", "L1", "L_1"
            match = re.match(r'^[Ll](?:evel)?[_\s]*(\d+)$', col_str, re.IGNORECASE)
            if match:
                level = int(match.group(1))
                level_cols[level] = col_str
        
        return level_cols
    
    def _parse_df(self):
        """Parse DataFrame to build region and index paths."""
        df = self.df
        
        # Detect level columns
        level_cols = self._detect_level_columns(df)
        if not level_cols:
            return
        
        self.max_level = max(level_cols.keys())
        
        # Check for v2 format with _abbr columns
        has_abbr_cols = any('_abbr' in str(c) for c in df.columns)
        has_index_cols = any('_index' in str(c) for c in df.columns)
        

        
        # Build paths for each row
        for idx, row in df.iterrows():
            row_path = []
            
            for lv in sorted(level_cols.keys()):
                level_col = level_cols[lv]
                
                if has_abbr_cols:
                    # v2 format: use _abbr column for abbreviation
                    abbr_col = f"Level_{lv}_abbr"
                    if abbr_col in df.columns:
                        abbrev = str(row[abbr_col]).strip()
                        if abbrev and abbrev.lower() not in ('nan', 'none', ''):
                            row_path.append(abbrev)
                            continue
                
                # Fallback: use main column, strip prefix
                full_name = str(row[level_col]).strip()
                if full_name and full_name.lower() not in ('nan', 'none', ''):
                    abbrev = self._strip_prefix(full_name)
                    row_path.append(abbrev)
                else:
                    row_path.append(None)
            
            # Store by abbreviation (last non-None level)
            if row_path and row_path[-1]:
                self.region_paths[row_path[-1]] = row_path
            
            # Store by index if available - store ALL level indices for this row
            if has_index_cols:
                for lv in sorted(level_cols.keys()):
                    idx_col = f"Level_{lv}_index"
                    if idx_col in df.columns:
                        try:
                            region_idx = int(row[idx_col])
                            if region_idx > 0 and region_idx not in self.index_paths:
                                # Store path for this index - use the path up to this level
                                path_to_level = [p for p in row_path[:lv] if p is not None]
                                if path_to_level:
                                    self.index_paths[region_idx] = row_path
                        except (ValueError, TypeError):
                            continue
        

    
    @staticmethod
    def _strip_prefix(name: str) -> str:
        """Strip CL_/CR_/SL_/SR_ prefix."""
        if not name:
            return name
        name = str(name).strip()
        for prefix in ['CL_', 'CR_', 'SL_', 'SR_']:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    
    def _strip_prefix_from_path(self, path: List[str]) -> List[str]:
        """Strip prefixes from all entries in a path."""
        if not path:
            return path
        return [self._strip_prefix(p) if p else p for p in path]
    
    def get_path(self, region: str) -> Optional[List[str]]:
        """Get hierarchy path for a region (by name, abbreviation, or index)."""
        region = str(region).strip()
        
        # Try direct match
        if region in self.region_paths:
            return self.region_paths[region]
        
        # Try stripped version
        stripped = self._strip_prefix(region)
        if stripped in self.region_paths:
            return self.region_paths[stripped]
        
        # Try numeric index
        try:
            idx = int(region)
            if idx in self.index_paths:
                return self.index_paths[idx]
        except ValueError:
            pass
        
        return None
    
    def get_at_level(self, region: str, level: int) -> Optional[str]:
        """
        Get region name at specified level.
        
        Args:
            region: Region name, abbreviation, or index
            level: Target level (1-6)
        
        Returns:
            Region abbreviation at target level, or None
        """
        path = self.get_path(region)
        if path is None:
            return None
        
        # Handle Level_0 in CSV - if path has 7 elements (0-6), level 1 = index 1 (skip Level_0)
        if len(path) == 7 and self.max_level == 6:
            # Path includes Level_0, so adjust: level 1 = index 1
            level_idx = level
        else:
            # Standard: level 1 = index 0
            level_idx = level - 1
        
        if level_idx < 0 or level_idx >= len(path):
            return None
        
        return path[level_idx]
    
    def lookup_by_index(self, index: int, level: int = None) -> Optional[Union[str, List[str]]]:
        """
        Lookup region by numeric index.
        
        Args:
            index: Numeric region index (from ARM_key_all.txt)
            level: If specified, return region at this level; otherwise return full path
        
        Returns:
            Region name/path or None
        """
        if index not in self.index_paths:
            return None
        
        path = self.index_paths[index]
        if level is None:
            return path
        
        # Handle Level_0 in CSV - if path has 7 elements (0-6), level 1 = index 1 (skip Level_0)
        if len(path) == 7 and self.max_level == 6:
            level_idx = level
        else:
            level_idx = level - 1
        
        if level_idx < 0 or level_idx >= len(path):
            return None
        
        return path[level_idx]
    
    def aggregate_to_level(self, region_lengths: Dict[str, float], 
                           target_level: int) -> Dict[str, float]:
        """
        Aggregate region lengths to target hierarchy level using index-based lookup.
        
        Args:
            region_lengths: Dict of {region_name_or_index: length}
            target_level: Target hierarchy level (1-6)
        
        Returns:
            Aggregated dict of {parent_region: total_length}
        """
        result = {}
        unmapped = []
        
        for region_key, length in region_lengths.items():
            # Try direct lookup first
            target = self.get_at_level(region_key, target_level)
            
            if target is not None:
                result[target] = result.get(target, 0) + length
            else:
                # Try numeric index lookup
                try:
                    idx = int(region_key)
                    if idx in self.index_paths:
                        path = self.index_paths[idx]
                        level_idx = target_level - 1
                        if 0 <= level_idx < len(path) and path[level_idx]:
                            result[path[level_idx]] = result.get(path[level_idx], 0) + length
                        else:
                            unmapped.append(region_key)
                    else:
                        unmapped.append(region_key)
                except ValueError:
                    unmapped.append(region_key)
        
        return result, unmapped
    
    @classmethod
    def load_file(cls, path: Union[str, Path], name: str = "") -> "HierarchyTable":
        """Load hierarchy from CSV file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Hierarchy file not found: {path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(path, encoding='utf-8', errors='replace')
            
            if name == "":
                name = path.stem
            
            return cls(df, name=name)
            
        except Exception:
            raise


class DualHierarchyTable:
    """
    Combines cortical (CHARM) and subcortical (SARM) hierarchy tables.
    Routes lookups based on region prefix:
    - CL_*, CR_* -> cortical
    - SL_*, SR_* -> subcortical
    """
    
    def __init__(self, cortex: HierarchyTable = None, subcortex: HierarchyTable = None,
                 cortex_paths: List[Union[str, Path]] = None,
                 subcortical_paths: List[Union[str, Path]] = None):
        """
        Initialize dual hierarchy.
        
        Args:
            cortex: Pre-loaded HierarchyTable for cortex
            subcortex: Pre-loaded HierarchyTable for subcortex
            cortex_paths: Path(s) to cortical hierarchy CSV file(s)
            subcortical_paths: Path(s) to subcortical hierarchy CSV file(s)
        """
        # Load from paths if provided
        if cortex_paths:
            if isinstance(cortex_paths, (str, Path)):
                cortex_paths = [cortex_paths]
            cortex = HierarchyTable.load_file(cortex_paths[0], name="cortex")
        
        if subcortical_paths:
            if isinstance(subcortical_paths, (str, Path)):
                subcortical_paths = [subcortical_paths]
            subcortex = HierarchyTable.load_file(subcortical_paths[0], name="subcortex")
        
        self.cortex = cortex
        self.subcortex = subcortex
    
    def _get_table(self, region: str) -> Optional[HierarchyTable]:
        """Get appropriate table based on region prefix or index."""
        region = str(region).strip()
        
        if region.startswith(('CL_', 'CR_')):
            return self.cortex
        elif region.startswith(('SL_', 'SR_')):
            return self.subcortex
        else:
            # Try numeric index lookup
            try:
                idx = int(region)
                # Subcortical indices are typically >= 1000 (ARM convention)
                if idx >= 1000 and self.subcortex and idx in self.subcortex.index_paths:
                    return self.subcortex
                if idx < 1000 and self.cortex and idx in self.cortex.index_paths:
                    return self.cortex
                # Fallback: try both
                if self.subcortex and idx in self.subcortex.index_paths:
                    return self.subcortex
                if self.cortex and idx in self.cortex.index_paths:
                    return self.cortex
            except ValueError:
                pass
            
            # Try both tables by name
            if self.cortex and region in self.cortex.region_paths:
                return self.cortex
            if self.subcortex and region in self.subcortex.region_paths:
                return self.subcortex
            
            return self.cortex or self.subcortex
    
    def get_path(self, region: str) -> Optional[List[str]]:
        """Get hierarchy path for a region."""
        table = self._get_table(region)
        return table.get_path(region) if table else None
    
    def get_at_level(self, region: str, level: int) -> Optional[str]:
        """Get region at specified hierarchy level."""
        table = self._get_table(region)
        return table.get_at_level(region, level) if table else None
    
    def lookup_by_index(self, index: int, level: int = None) -> Optional[Union[str, List[str]]]:
        """Lookup by numeric index - try subcortex first (indices >= 1000), then cortex."""
        # Try subcortex first for high indices (>= 1000 typically subcortical)
        if self.subcortex:
            result = self.subcortex.lookup_by_index(index, level)
            if result:
                return result
        
        # Try cortex
        if self.cortex:
            result = self.cortex.lookup_by_index(index, level)
            if result:
                return result
        
        return None
    
    def aggregate_to_level(self, region_lengths: Dict[str, float], 
                           target_level: int,
                           strip_prefixes: bool = False) -> Tuple[Dict[str, float], List[str]]:
        """
        Aggregate region lengths to target level, routing to appropriate table.
        
        Args:
            region_lengths: Dict of {region_name_or_index: length}
            target_level: Target hierarchy level
            strip_prefixes: If True, remove CL_/CR_/SL_/SR_ prefixes from result keys
        
        Returns:
            Tuple of (aggregated_dict, unmapped_regions)
        """
        result = {}
        unmapped = []
        
        for region_key, length in region_lengths.items():
            # Get appropriate table
            table = self._get_table(region_key)
            
            if table is None:
                unmapped.append(region_key)
                continue
            
            # Try lookup
            target = table.get_at_level(region_key, target_level)
            
            if target is not None:
                # Strip prefix if requested
                if strip_prefixes:
                    target = HierarchyTable._strip_prefix(target)
                result[target] = result.get(target, 0) + length
            else:
                # Try numeric index
                try:
                    idx = int(region_key)
                    target = self.lookup_by_index(idx, target_level)
                    if target:
                        # Strip prefix if requested
                        if strip_prefixes:
                            target = HierarchyTable._strip_prefix(target)
                        result[target] = result.get(target, 0) + length
                    else:
                        unmapped.append(region_key)
                except ValueError:
                    unmapped.append(region_key)
        
        return result, unmapped
    
    @classmethod
    def load(cls, cortex_path: Union[str, Path] = None, 
             subcortex_path: Union[str, Path] = None) -> "DualHierarchyTable":
        """Load both hierarchy tables."""
        cortex = None
        subcortex = None
        
        if cortex_path:
            cortex = HierarchyTable.load_file(cortex_path, name="cortex")
        
        if subcortex_path:
            subcortex = HierarchyTable.load_file(subcortex_path, name="subcortex")
        
        return cls(cortex, subcortex)


def load_dual_hierarchy_from_paths(atlas_dir: Union[str, Path] = None) -> Optional[DualHierarchyTable]:
    """
    Load CHARM/SARM hierarchy tables from atlas directory.
    
    Args:
        atlas_dir: Path to atlas directory containing CHARM_key_table_v2.csv and SARM_key_table_v2.csv
    
    Returns:
        DualHierarchyTable or None if files not found
    """
    if atlas_dir is None:
        return None
    
    atlas_dir = Path(atlas_dir)
    
    # Try v2 files first, then legacy files
    cortex_files = [
        atlas_dir / "CHARM_key_table_v2.csv",
        atlas_dir / "CHARM" / "CHARM_key_table_v2.csv",
        atlas_dir / "NMT_v2.1_sym" / "tables_ARM" / "ARM_key_table.csv",
    ]
    
    subcortex_files = [
        atlas_dir / "SARM_key_table_v2.csv",
        atlas_dir / "SARM" / "SARM_key_table_v2.csv",
    ]
    
    cortex_path = None
    for f in cortex_files:
        if f.exists():
            cortex_path = f
            break
    
    subcortex_path = None
    for f in subcortex_files:
        if f.exists():
            subcortex_path = f
            break
    
    if cortex_path or subcortex_path:

        return DualHierarchyTable.load(cortex_path, subcortex_path)
    
    return None
