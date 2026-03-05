"""
hierarchy_table.py - User-provided explicit hierarchy tables.

Loads CSV/Excel files with columns like:
    Level 3 | Level 4 | Level 5 | Level 6

Cell format: '{index}: {full_name} ({abbreviation})'
Example:     '3: anterior_cingulate_cortex (ACC)'

Supports:
    - Multiple files (cortex, subcortex, brainstem, etc.)
    - Flexible column naming (Level 3, L3, Level_3)
    - Laterality prefix stripping (CL_, CR_, SL_, SR_)
    - Multi-strategy matching (exact -> full_name -> normalized)
    - Conflict detection across files
    - Match diagnostics

Usage:
    from region_analysis.hierarchy_table import HierarchyTable

    table = HierarchyTable.from_files(
        'cortex_hierarchy.csv',
        'subcortex_hierarchy.csv',
    )
    table.aggregate_to_level('CL_area_24a', 3)  # -> 'ACC'
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ======================================================================
LATERALITY_PREFIXES = ("CL_", "CR_", "SL_", "SR_")

# Regex for cell format: '3: anterior_cingulate_cortex (ACC)'
_CELL_PATTERN = re.compile(r"^(\d+)\s*:\s*(.+?)\s*\(([^)]+)\)\s*$")

# Regex patterns for detecting level columns
_LEVEL_PATTERNS = [
    re.compile(r"^Level[\s_]*(\d+)$", re.IGNORECASE),
    re.compile(r"^L(\d+)$", re.IGNORECASE),
    re.compile(r"^Lv[\s_]*(\d+)$", re.IGNORECASE),
    re.compile(r"^Hierarchy[\s_]*(\d+)$", re.IGNORECASE),
    re.compile(r"^Level_(\d+)$", re.IGNORECASE),  # For v2 files: Level_0, Level_1, etc.
]


def _normalize(name: str) -> str:
    """Lowercase, strip underscores/spaces/hyphens/apostrophes."""
    return (
        name.lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
        .replace("'", "")
        .replace("'", "")
    )


def _strip_prefix(name: str) -> str:
    """Remove CL_/CR_/SL_/SR_ prefix."""
    if pd.isna(name):
        return ""
    s = str(name)
    for prefix in LATERALITY_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s


def _parse_cell(cell_text: str) -> Optional[Tuple[int, str, str]]:
    """
    Parse '3: anterior_cingulate_cortex (ACC)'
    -> (3, 'anterior_cingulate_cortex', 'ACC')

    Returns None if cell doesn't match expected format.
    """
    if not cell_text or cell_text.strip() == "" or cell_text.strip().lower() == "nan":
        return None
    m = _CELL_PATTERN.match(cell_text.strip())
    if m:
        return int(m.group(1)), m.group(2).strip(), m.group(3).strip()
    return None


def _detect_level_columns(df: pd.DataFrame) -> Dict[int, str]:
    """
    Detect which DataFrame columns correspond to which hierarchy levels.

    Returns {level_int: column_name}, e.g. {3: 'Level 3', 4: 'Level 4', ...}
    """
    result = {}
    for col in df.columns:
        col_str = str(col).strip()
        for pattern in _LEVEL_PATTERNS:
            m = pattern.match(col_str)
            if m:
                level = int(m.group(1))
                result[level] = col
                break
    return result


# ======================================================================
class HierarchyTable:
    """
    Explicit hierarchy from user-provided CSV/Excel files.

    For each region abbreviation, stores the full path from coarsest
    to finest level. Handles laterality prefix stripping and flexible
    matching when looking up atlas region names.
    """

    def __init__(self):
        # abbreviation -> {level: abbreviation_at_that_level, ...}
        self.region_paths: Dict[str, Dict[int, str]] = {}

        # abbreviation -> metadata
        self.region_info: Dict[str, dict] = {}

        # Match indexes (rebuilt after each load)
        self._abbrev_index: Dict[str, str] = {}
        self._full_name_index: Dict[str, str] = {}
        self._normalized_index: Dict[str, str] = {}

        # Tracking
        self._loaded_files: List[str] = []
        self._conflicts: List[str] = []
        self._min_level: int = 99
        self._max_level: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_files(cls, *paths: str) -> "HierarchyTable":
        """Load hierarchy from one or more CSV/Excel/TSV files."""
        table = cls()
        for p in paths:
            table.load_file(p)
        table._print_summary()
        return table

    def load_file(self, path: str):
        """Parse one file and add regions to the lookup."""
        path_str = str(path)
        print(f"[HIERARCHY TABLE] Loading: {path_str}")

        # Read file
        if path_str.endswith(".xlsx"):
            df = pd.read_excel(path_str)
        elif path_str.endswith(".csv"):
            df = pd.read_csv(path_str)
        else:
            # Try tab-delimited
            df = pd.read_csv(path_str, sep="\t")

        # Detect level columns
        level_cols = _detect_level_columns(df)
        if not level_cols:
            raise ValueError(
                f"No level columns found in {path_str}. "
                f"Expected 'Level 3', 'L3', etc. Found: {list(df.columns)}"
            )

        levels = sorted(level_cols.keys())
        self._min_level = min(self._min_level, min(levels))
        self._max_level = max(self._max_level, max(levels))
        print(f"    Detected levels: {levels}, Rows: {len(df)}")

        # Check for v2 format (separate _abbr columns)
        abbr_cols = {}
        index_cols = {}
        for lv in levels:
            abbr_col = f"{level_cols[lv]}_abbr"
            index_col = f"{level_cols[lv]}_index"
            if abbr_col in df.columns:
                abbr_cols[lv] = abbr_col
            if index_col in df.columns:
                index_cols[lv] = index_col
        
        is_v2_format = len(abbr_cols) > 0
        if is_v2_format:
            print(f"    Detected v2 format (separate abbr columns)")

        n_parsed = 0
        for _, row in df.iterrows():
            row_path: Dict[int, str] = {}
            row_info: Dict[int, Tuple[int, str, str]] = {}

            for lv in levels:
                if is_v2_format and lv in abbr_cols:
                    # v2 format: use plain abbreviation from _abbr column
                    abbrev = str(row[abbr_cols[lv]]).strip()
                    full_name = str(row[level_cols[lv]]).strip()
                    idx = int(row[index_cols[lv]]) if lv in index_cols and pd.notna(row[index_cols[lv]]) else 0
                    if abbrev and abbrev.lower() != 'nan':
                        row_path[lv] = abbrev
                        row_info[lv] = (idx, full_name, abbrev)
                else:
                    # v1 format: parse "1: name (abbr)" style
                    cell = str(row[level_cols[lv]]).strip()
                    parsed = _parse_cell(cell)
                    if parsed:
                        idx, full_name, abbrev = parsed
                        row_path[lv] = abbrev
                        row_info[lv] = (idx, full_name, abbrev)

            if not row_path:
                continue
            n_parsed += 1

            seen_abbrevs: Set[str] = set()
            for lv in levels:
                if lv not in row_path:
                    continue
                abbrev = row_path[lv]
                if abbrev in seen_abbrevs:
                    continue
                seen_abbrevs.add(abbrev)

                finest_lv = lv
                for check_lv in levels:
                    if check_lv > lv and row_path.get(check_lv) == abbrev:
                        finest_lv = check_lv

                sub_path = {
                    cl: row_path[cl]
                    for cl in levels
                    if cl in row_path and cl <= finest_lv
                }
                self._register(abbrev, sub_path)

                if abbrev not in self.region_info and lv in row_info:
                    self.region_info[abbrev] = {
                        "full_name": row_info[lv][1],
                        "csv_index": row_info[lv][0],
                        "native_level": lv,
                    }

        self._rebuild_indexes()
        self._loaded_files.append(path_str)
        print(f"    Parsed: {n_parsed} rows -> {len(self.region_paths)} total regions")

    def _register(self, abbrev: str, path: Dict[int, str]):
        if abbrev in self.region_paths:
            existing = self.region_paths[abbrev]
            for lv, val in path.items():
                if lv in existing and existing[lv] != val:
                    self._conflicts.append(
                        f"'{abbrev}' L{lv}: existing='{existing[lv]}' vs new='{val}'"
                    )
            existing.update(path)
        else:
            self.region_paths[abbrev] = path.copy()

    def _rebuild_indexes(self):
        self._abbrev_index.clear()
        self._full_name_index.clear()
        self._normalized_index.clear()
        for abbrev in self.region_paths:
            self._abbrev_index[abbrev] = abbrev
            norm = _normalize(abbrev)
            if norm not in self._normalized_index:
                self._normalized_index[norm] = abbrev
            info = self.region_info.get(abbrev)
            if info and info.get("full_name"):
                fn = info["full_name"]
                if fn not in self._full_name_index:
                    self._full_name_index[fn] = abbrev
                norm_fn = _normalize(fn)
                if norm_fn not in self._normalized_index:
                    self._normalized_index[norm_fn] = abbrev

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def aggregate_to_level(
        self, atlas_region: str, target_level: int
    ) -> Optional[str]:
        """
        Given an atlas region name (e.g. 'CL_area_24a'), return the
        ancestor abbreviation at *target_level*.

        Steps:
            1. Try exact match (for v2 files where abbreviations have prefixes)
            2. Strip laterality prefix -> 'area_24a'
            3. Match against CSV entries
            4. Walk up the stored path to target_level

        Returns None if not found or level not covered.
        """
        if pd.isna(atlas_region):
            return None
        region_str = str(atlas_region)
        # Try exact match first (v2 format)
        if region_str in self.region_paths:
            path = self.region_paths[region_str]
            return path.get(target_level)
        # Fall back to stripped matching (v1 format)
        base = _strip_prefix(region_str)
        key = self._find_match(base)
        if key is None:
            return None
        path = self.region_paths.get(key, {})
        return path.get(target_level)

    def get_path(self, atlas_region: str) -> Optional[Dict[int, str]]:
        """
        Return the full path dict for an atlas region.

        Example::

            get_path('CL_area_24a')
            -> {3: 'ACC', 4: 'area_24', 5: 'area_24a/b', 6: 'area_24a'}
        """
        if pd.isna(atlas_region):
            return None
        region_str = str(atlas_region)
        # Try exact match first (v2 format)
        if region_str in self.region_paths:
            return self.region_paths[region_str].copy()
        # Fall back to stripped matching (v1 format)
        base = _strip_prefix(region_str)
        key = self._find_match(base)
        if key is None:
            return None
        return self.region_paths.get(key, {}).copy()

    def _find_match(self, base_name: str) -> Optional[str]:
        """
        Multi-strategy matching:
            1. Exact abbreviation match
            2. Exact full_name match
            3. Normalized match (lowercase, no underscores/spaces)
        """
        if base_name in self._abbrev_index:
            return self._abbrev_index[base_name]
        if base_name in self._full_name_index:
            return self._full_name_index[base_name]
        norm = _normalize(base_name)
        if norm and norm in self._normalized_index:
            return self._normalized_index[norm]
        return None

    def has_region(self, atlas_region: str) -> bool:
        """Check if an atlas region can be found in this table."""
        region_str = str(atlas_region)
        # Try exact match first (for v2 files where abbreviations have prefixes)
        if region_str in self.region_paths:
            return True
        # Fall back to stripped matching
        base = _strip_prefix(region_str)
        return self._find_match(base) is not None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def match_report(self, atlas_abbreviations: list) -> pd.DataFrame:
        """
        Show how atlas regions match against the hierarchy table.

        Args:
            atlas_abbreviations: list of atlas region names (with prefixes)

        Returns:
            DataFrame with columns: Atlas_Region, Base_Name, Matched, CSV_Key, Path
        """
        rows = []
        for atlas_name in atlas_abbreviations:
            base = _strip_prefix(str(atlas_name))
            key = self._find_match(base)
            path = self.region_paths.get(key, {}) if key else {}
            rows.append(
                {
                    "Atlas_Region": atlas_name,
                    "Base_Name": base,
                    "Matched": key is not None,
                    "CSV_Key": key,
                    "Path": str(path) if path else "",
                }
            )
        df = pd.DataFrame(rows)
        n = len(df)
        n_m = df["Matched"].sum()
        print(f"\n[MATCH REPORT] {n_m}/{n} matched ({n_m / max(n, 1) * 100:.1f}%)")
        if n_m < n:
            unmatched = df[~df["Matched"]]["Base_Name"].tolist()
            print(f"  Unmatched ({n - n_m}):")
            for name in unmatched[:20]:
                print(f"    - {name}")
            if len(unmatched) > 20:
                print(f"    ... and {len(unmatched) - 20} more")
        return df

    def debug_region(self, atlas_region: str):
        base = _strip_prefix(str(atlas_region))
        print(f"\n[DEBUG] Atlas region: {atlas_region}")
        print(f"  Base (prefix stripped): {base}")
        key = self._find_match(base)
        if key is None:
            print(f"  NOT FOUND in hierarchy table")
            norm = _normalize(base)
            print(f"  Tried: exact='{base}', normalized='{norm}'")
            suggestions = self._find_similar(base, 5)
            if suggestions:
                print(f"  Similar entries:")
                for s, _ in suggestions:
                    print(f"    - {s}")
            return
        path = self.region_paths.get(key, {})
        info = self.region_info.get(key, {})
        print(f"  Matched: {key}")
        print(f"  Full name: {info.get('full_name', '?')}")
        print(f"  Path:")
        for lv in sorted(path.keys()):
            print(f"    L{lv}: {path[lv]}")

    def _find_similar(
        self, name: str, max_results: int = 5
    ) -> List[Tuple[str, float]]:
        norm = _normalize(name)
        candidates = []
        for abbrev in self.region_paths:
            norm_a = _normalize(abbrev)
            if norm in norm_a or norm_a in norm:
                candidates.append((abbrev, 1.0))
            elif len(set(norm) & set(norm_a)) / max(len(set(norm)), 1) > 0.5:
                candidates.append((abbrev, 0.5))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:max_results]

    def _print_summary(self):
        print(f"\n[HIERARCHY TABLE] Summary:")
        print(f"  Files: {len(self._loaded_files)}")
        for f in self._loaded_files:
            print(f"    - {f}")
        print(f"  Regions: {len(self.region_paths)}")
        print(f"  Levels: L{self._min_level} to L{self._max_level}")
        if self._conflicts:
            print(f"\n  CONFLICTS ({len(self._conflicts)}):")
            for c in self._conflicts[:10]:
                print(f"    {c}")
        else:
            print(f"  No conflicts")

    @property
    def levels(self) -> List[int]:
        all_lv: Set[int] = set()
        for path in self.region_paths.values():
            all_lv.update(path.keys())
        return sorted(all_lv)

    @property
    def n_regions(self) -> int:
        return len(self.region_paths)

    def get_regions_at_level(self, level: int) -> List[str]:
        regions: Set[str] = set()
        for path in self.region_paths.values():
            if level in path:
                regions.add(path[level])
        return sorted(regions)


# ======================================================================
# DUAL HIERARCHY TABLE - Separate cortex vs subcortical hierarchies
# ======================================================================

class DualHierarchyTable:
    """
    Manages separate hierarchy tables for cortical (CL_/CR_) and 
    subcortical (SL_/SR_) regions to avoid conflicts.
    
    Routes regions automatically based on their prefix:
        - CL_, CR_ -> cortex_table
        - SL_, SR_ -> subcortical_table
        - No prefix -> tries both (cortex first, then subcortical)
    
    Usage:
        dual = DualHierarchyTable(
            cortex_paths=['cortex_hierarchy.csv'],
            subcortical_paths=['subcortex_hierarchy.csv', 'thalamus_hierarchy.csv']
        )
        result = dual.aggregate_to_level('CL_area_24a', 3)  # Uses cortex table
        result = dual.aggregate_to_level('SL_VPL', 3)       # Uses subcortical table
    """
    
    CORTICAL_PREFIXES = ("CL_", "CR_")
    SUBCORTICAL_PREFIXES = ("SL_", "SR_")
    
    def __init__(
        self,
        cortex_paths: Optional[List[str]] = None,
        subcortical_paths: Optional[List[str]] = None,
    ):
        """
        Initialize dual hierarchy system.
        
        Args:
            cortex_paths: List of CSV/Excel files for cortical regions
            subcortical_paths: List of CSV/Excel files for subcortical regions
        """
        self.cortex_table: Optional[HierarchyTable] = None
        self.subcortical_table: Optional[HierarchyTable] = None
        
        if cortex_paths:
            print("\n" + "="*60)
            print("LOADING CORTEX HIERARCHY")
            print("="*60)
            self.cortex_table = HierarchyTable.from_files(*cortex_paths)
        
        if subcortical_paths:
            print("\n" + "="*60)
            print("LOADING SUBCORTICAL HIERARCHY")
            print("="*60)
            self.subcortical_table = HierarchyTable.from_files(*subcortical_paths)
        
        self._print_summary()
    
    def _get_region_type(self, region: str) -> str:
        """Determine if region is cortical, subcortical, or unknown."""
        if pd.isna(region):
            return "unknown"
        s = str(region)
        if any(s.startswith(p) for p in self.CORTICAL_PREFIXES):
            return "cortical"
        if any(s.startswith(p) for p in self.SUBCORTICAL_PREFIXES):
            return "subcortical"
        return "unknown"
    
    def _get_table_for_region(self, region: str) -> Optional[HierarchyTable]:
        """Get the appropriate hierarchy table for a region."""
        rtype = self._get_region_type(region)
        if rtype == "cortical":
            return self.cortex_table
        elif rtype == "subcortical":
            return self.subcortical_table
        else:
            # No prefix - try cortex first, then subcortical
            if self.cortex_table:
                return self.cortex_table
            return self.subcortical_table
    
    def aggregate_to_level(self, region: str, target_level: int) -> Optional[str]:
        """
        Resolve region to target level using the appropriate hierarchy.
        
        Args:
            region: Atlas region name (e.g., 'CL_area_24a', 'SL_VPL')
            target_level: Target hierarchy level
            
        Returns:
            Resolved region name or None
        """
        table = self._get_table_for_region(region)
        if table is None:
            return None
        return table.aggregate_to_level(region, target_level)
    
    def get_path(self, region: str) -> Optional[Dict[int, str]]:
        """Get full hierarchy path for a region."""
        table = self._get_table_for_region(region)
        if table is None:
            return None
        return table.get_path(region)
    
    def has_region(self, region: str) -> bool:
        """Check if region exists in appropriate hierarchy."""
        table = self._get_table_for_region(region)
        if table is None:
            return False
        return table.has_region(region)
    
    def match_report(self, atlas_abbreviations: list) -> pd.DataFrame:
        """
        Generate match report for all regions, separated by type.
        
        Args:
            atlas_abbreviations: List of atlas region names
            
        Returns:
            Combined DataFrame with Region_Type column
        """
        # Split by region type
        cortical = []
        subcortical = []
        unknown = []
        
        for region in atlas_abbreviations:
            rtype = self._get_region_type(region)
            if rtype == "cortical":
                cortical.append(region)
            elif rtype == "subcortical":
                subcortical.append(region)
            else:
                unknown.append(region)
        
        print(f"\n[DUAL HIERARCHY] Region distribution:")
        print(f"  Cortical: {len(cortical)}")
        print(f"  Subcortical: {len(subcortical)}")
        print(f"  Unknown type: {len(unknown)}")
        
        # Get reports from each table
        all_reports = []
        
        if cortical and self.cortex_table:
            print(f"\n[CORTICAL REGIONS]")
            report = self.cortex_table.match_report(cortical)
            report["Region_Type"] = "cortical"
            all_reports.append(report)
        
        if subcortical and self.subcortical_table:
            print(f"\n[SUBCORTICAL REGIONS]")
            report = self.subcortical_table.match_report(subcortical)
            report["Region_Type"] = "subcortical"
            all_reports.append(report)
        
        if unknown:
            print(f"\n[UNKNOWN TYPE REGIONS]")
            # Try both tables for unknown regions
            unknown_reported = set()
            if self.cortex_table:
                report = self.cortex_table.match_report(unknown)
                matched = report[report["Matched"]]["Atlas_Region"].tolist()
                unknown_reported.update(matched)
                report["Region_Type"] = "unknown"
                all_reports.append(report)
            
            remaining = [r for r in unknown if r not in unknown_reported]
            if remaining and self.subcortical_table:
                report = self.subcortical_table.match_report(remaining)
                report["Region_Type"] = "unknown"
                all_reports.append(report)
        
        if all_reports:
            return pd.concat(all_reports, ignore_index=True)
        return pd.DataFrame()
    
    def debug_region(self, region: str):
        """Debug a specific region through the appropriate hierarchy."""
        rtype = self._get_region_type(region)
        table = self._get_table_for_region(region)
        
        print(f"\n{'='*60}")
        print(f"DUAL HIERARCHY DEBUG: {region}")
        print(f"Region type: {rtype}")
        print(f"{'='*60}")
        
        if table is None:
            print(f"[ERROR] No hierarchy table available for {rtype} regions")
            return
        
        table.debug_region(region)
        
        # Also show what the other table would return (for comparison)
        other_table = None
        if rtype == "cortical" and self.subcortical_table:
            other_table = self.subcortical_table
            other_name = "subcortical"
        elif rtype == "subcortical" and self.cortex_table:
            other_table = self.cortex_table
            other_name = "cortical"
        
        if other_table:
            base = _strip_prefix(region)
            if other_table.has_region(region) or other_table._find_match(base):
                print(f"\n[NOTE] This region also exists in {other_name} hierarchy!")
                print(f"       This may indicate a naming conflict.")
    
    def _print_summary(self):
        """Print summary of both hierarchies."""
        print(f"\n{'='*60}")
        print("DUAL HIERARCHY TABLE SUMMARY")
        print(f"{'='*60}")
        
        if self.cortex_table:
            print(f"\n[CORTEX]")
            print(f"  Regions: {self.cortex_table.n_regions}")
            print(f"  Levels: L{self.cortex_table._min_level} to L{self.cortex_table._max_level}")
            if self.cortex_table._conflicts:
                print(f"  Conflicts: {len(self.cortex_table._conflicts)}")
        else:
            print(f"\n[CORTEX] Not loaded")
        
        if self.subcortical_table:
            print(f"\n[SUBCORTICAL]")
            print(f"  Regions: {self.subcortical_table.n_regions}")
            print(f"  Levels: L{self.subcortical_table._min_level} to L{self.subcortical_table._max_level}")
            if self.subcortical_table._conflicts:
                print(f"  Conflicts: {len(self.subcortical_table._conflicts)}")
        else:
            print(f"\n[SUBCORTICAL] Not loaded")
    
    @property
    def n_regions(self) -> int:
        """Total regions across both hierarchies."""
        n = 0
        if self.cortex_table:
            n += self.cortex_table.n_regions
        if self.subcortical_table:
            n += self.subcortical_table.n_regions
        return n
