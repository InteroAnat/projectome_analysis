"""
hierarchy.py - Hierarchical brain region management.

Dual-source resolution: HierarchyTable (CSV, priority) + RegionHierarchy (ARM key, fallback).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from region_analysis.utils import parse_terminal_regions


# ======================================================================
# UNIFIED RESOLVER
# ======================================================================
def resolve_to_level(
    region: str,
    target_level: int,
    arm_hierarchy: "Optional[RegionHierarchy]" = None,
    hierarchy_table: "Optional[object]" = None,
) -> Optional[str]:
    """Resolve region to target level. CSV table first, ARM key fallback."""
    if hierarchy_table is not None:
        result = hierarchy_table.aggregate_to_level(region, target_level)
        if result is not None:
            return result
    if arm_hierarchy is not None:
        result = arm_hierarchy.aggregate_to_level(region, target_level)
        if result is not None:
            return result
    return None


# ======================================================================
# RegionHierarchy (ARM key tree)
# ======================================================================
class RegionHierarchy:
    def __init__(self, arm_key_df: pd.DataFrame):
        self.arm_key_df = arm_key_df.copy()
        self.parent_map: Dict[str, Optional[str]] = {}
        self.children_map: Dict[str, List[str]] = {}
        self.level_map: Dict[str, int] = {}
        self.name_map: Dict[str, str] = {}
        self._build_hierarchy()

    @classmethod
    def from_file(cls, arm_key_path: str) -> "RegionHierarchy":
        if arm_key_path.endswith(".xlsx"):
            df = pd.read_excel(arm_key_path)
        elif arm_key_path.endswith(".csv"):
            df = pd.read_csv(arm_key_path)
        else:
            df = pd.read_csv(arm_key_path, sep="\t")
        required = {"Index", "Abbreviation", "First_Level"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"arm_key missing columns: {missing}")
        print(f"    Loaded arm_key: {len(df)} regions, levels 1-{int(df['First_Level'].max())}")
        return cls(df)

    def _build_hierarchy(self):
        df_sorted = self.arm_key_df.sort_values("Index")
        stack: Dict[int, str] = {}
        for _, row in df_sorted.iterrows():
            abbrev = str(row["Abbreviation"])
            level = int(row["First_Level"])
            self.level_map[abbrev] = level
            self.name_map[abbrev] = row.get("Full_Name", abbrev)
            if level > 1 and (level - 1) in stack:
                parent = stack[level - 1]
                self.parent_map[abbrev] = parent
                self.children_map.setdefault(parent, []).append(abbrev)
            else:
                self.parent_map[abbrev] = None
            stack[level] = abbrev
            for lv in [k for k in stack if k > level]:
                del stack[lv]

    def get_parent(self, region: str) -> Optional[str]:
        return self.parent_map.get(region)

    def get_children(self, region: str) -> List[str]:
        return self.children_map.get(region, [])

    def get_level(self, region: str) -> int:
        return self.level_map.get(region, -1)

    def get_ancestors(self, region: str) -> List[str]:
        anc, cur = [], self.get_parent(region)
        while cur:
            anc.append(cur)
            cur = self.get_parent(cur)
        return anc

    def get_descendants(self, region: str) -> List[str]:
        desc = []
        for ch in self.get_children(region):
            desc.append(ch)
            desc.extend(self.get_descendants(ch))
        return desc

    def aggregate_to_level(self, region: str, target_level: int) -> Optional[str]:
        if pd.isna(region):
            return None
        region_str = str(region)
        cl = self.get_level(region_str)
        if cl == -1:
            return None
        if cl == target_level:
            return region_str
        if cl < target_level:
            return None
        cur = region_str
        while cur is not None:
            cur_lv = self.get_level(cur)
            if cur_lv == target_level:
                return cur
            if cur_lv < target_level:
                return None
            cur = self.get_parent(cur)
        return None

    def get_full_path(self, region: str) -> List[str]:
        if pd.isna(region):
            return []
        return [region] + self.get_ancestors(region)

    def get_regions_at_level(self, level: int) -> List[str]:
        return [r for r, lv in self.level_map.items() if lv == level]

    def get_ancestor_at_each_level(
        self, region: str, max_level: int = 6
    ) -> Dict[str, str]:
        if pd.isna(region):
            return {}
        result = {}
        for lv in range(1, max_level + 1):
            anc = self.aggregate_to_level(str(region), lv)
            if anc is not None:
                result[f"L{lv}"] = anc
        return result

    def debug_region(self, region: str):
        lv = self.get_level(region)
        print(f"\n[DEBUG ARM] Region: {region}")
        print(f"  Level: {lv}")
        if lv == -1:
            print("  NOT FOUND in ARM key!")
            return
        print(f"  Parent: {self.get_parent(region)}")
        print(f"  Children: {self.get_children(region)}")
        chain = [region] + self.get_ancestors(region)
        levels = [f"{r}(L{self.get_level(r)})" for r in chain]
        print(f"  Chain: {' -> '.join(levels)}")
        ancestors = self.get_ancestor_at_each_level(region)
        print(f"  Ancestor map: {ancestors}")

    def print_tree(self, root: str = None, indent: int = 0, max_depth: int = 6):
        if indent > max_depth:
            return
        if root is None:
            for r in sorted(r for r, p in self.parent_map.items() if p is None):
                self.print_tree(r, 0, max_depth)
            return
        prefix = "  " * indent + ("|- " if indent else "")
        print(f"{prefix}{root} (L{self.get_level(root)})")
        for ch in self.get_children(root):
            self.print_tree(ch, indent + 1, max_depth)


# ======================================================================
# SOMA HIERARCHY — condensed dict column
# ======================================================================
def _build_hierarchy_dict(
    region: str,
    hierarchy: Optional[RegionHierarchy],
    max_level: int,
    hierarchy_table=None,
) -> dict:
    if pd.isna(region):
        return {}
    result = {}
    for lv in range(1, max_level + 1):
        anc = resolve_to_level(
            str(region), lv,
            arm_hierarchy=hierarchy,
            hierarchy_table=hierarchy_table,
        )
        if anc is not None:
            result[f"L{lv}"] = anc
    return result


def add_soma_hierarchy_column(
    df: pd.DataFrame,
    hierarchy: Optional[RegionHierarchy],
    region_col: str = "Soma_Region",
    max_level: int = 6,
    hierarchy_table=None,
) -> pd.DataFrame:
    df = df.copy()
    df["Soma_Region_Hierarchy"] = df[region_col].apply(
        lambda r: _build_hierarchy_dict(r, hierarchy, max_level, hierarchy_table)
    )
    unmapped = df[df["Soma_Region_Hierarchy"].apply(lambda d: len(d) == 0)]
    if len(unmapped) > 0:
        regions = unmapped[region_col].unique()
        print(
            f"[HIERARCHY] WARNING: {len(unmapped)} neurons unmappable soma: "
            f"{list(regions)[:10]}"
        )
    return df


def extract_soma_level(
    df: pd.DataFrame, level: int, col: str = "Soma_Region_Hierarchy"
) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    key = f"L{level}"
    return df[col].apply(lambda d: d.get(key) if isinstance(d, dict) else None)


# ======================================================================
# PROJECTION-LENGTH HIERARCHY
# ======================================================================
def _aggregate_length_dict_to_level(
    length_dict: dict,
    hierarchy: Optional[RegionHierarchy],
    level: int,
    hierarchy_table=None,
) -> Tuple[dict, List[str]]:
    if not isinstance(length_dict, dict) or not length_dict:
        return {}, []
    agg = defaultdict(float)
    unmapped_regions = []
    unmapped_total = 0.0
    for region, length in length_dict.items():
        mapped = resolve_to_level(
            str(region), level,
            arm_hierarchy=hierarchy,
            hierarchy_table=hierarchy_table,
        )
        if mapped is not None:
            agg[mapped] += length
        else:
            unmapped_total += length
            unmapped_regions.append(str(region))
    if unmapped_total > 0:
        agg["_Unmapped"] += unmapped_total
    return dict(agg), unmapped_regions


def add_projection_length_hierarchy(
    df: pd.DataFrame,
    hierarchy: Optional[RegionHierarchy],
    length_col: str = "Region_projection_length",
    max_level: int = 6,
    min_level: int = 1,
    hierarchy_table=None,
) -> pd.DataFrame:
    """
    Create ``Region_Projection_Length_L{min_level}`` … ``L{max_level}`` + ``_finest``.
    
    Args:
        df: Input DataFrame
        hierarchy: RegionHierarchy instance
        length_col: Column containing projection length dictionaries
        max_level: Maximum hierarchy level (default: 6)
        min_level: Minimum hierarchy level (default: 1)
        hierarchy_table: Optional HierarchyTable for CSV-based lookups
    """
    df = df.copy()
    if length_col not in df.columns:
        print(f"[WARN] Column '{length_col}' not found.")
        return df

    for lv in range(min_level, max_level + 1):
        col_name = f"Region_Projection_Length_L{lv}"
        results = df[length_col].apply(
            lambda d, _lv=lv: _aggregate_length_dict_to_level(
                d, hierarchy, _lv, hierarchy_table
            )
        )
        df[col_name] = results.apply(lambda x: x[0])
        n_unmapped = results.apply(lambda x: len(x[1])).sum()
        if n_unmapped > 0:
            all_un: Set[str] = set()
            for ulist in results.apply(lambda x: x[1]):
                all_un.update(ulist)
            print(
                f"[HIERARCHY L{lv}] {n_unmapped} entries unmapped. "
                f"Examples: {sorted(all_un)[:5]}"
            )

    df = df.rename(columns={length_col: "Region_Projection_Length_finest"})
    return df


# ======================================================================
# HIERARCHY SUMMARY
# ======================================================================
def hierarchy_summary(df: pd.DataFrame, max_level: int = 6) -> pd.DataFrame:
    rows = []
    for lv in range(1, max_level + 1):
        data = extract_soma_level(df, lv)
        valid = data.dropna()
        counts = valid.value_counts() if not valid.empty else pd.Series(dtype=int)
        rows.append(
            {
                "Level": lv,
                "N_Regions": len(counts),
                "N_Neurons": len(valid),
                "N_Unmapped": len(data) - len(valid),
                "Mean_per_Region": round(counts.mean(), 1) if len(counts) else 0,
                "Top_Region": counts.index[0] if len(counts) else None,
            }
        )
    return pd.DataFrame(rows)


# ======================================================================
# TERMINAL-SITE HIERARCHY (lists)
# ======================================================================
def add_projection_hierarchy(
    df: pd.DataFrame,
    hierarchy: Optional[RegionHierarchy],
    max_level: int = 6,
    hierarchy_table=None,
) -> pd.DataFrame:
    df = df.copy()
    print(f"[PROJECTION HIERARCHY] Adding Proj_L1 to Proj_L{max_level}...")
    for lv in range(1, max_level + 1):
        df[f"Proj_L{lv}"] = df["Terminal_Regions"].apply(
            lambda r, _lv=lv: [
                resolve_to_level(str(x), _lv, hierarchy, hierarchy_table)
                for x in (
                    r if isinstance(r, (list, tuple)) else parse_terminal_regions(r)
                )
            ]
        )
    df["Proj_Highest"] = df["Terminal_Regions"].apply(
        lambda r: list(r) if isinstance(r, (list, tuple)) else parse_terminal_regions(r)
    )
    total = df["Terminal_Count"].sum()
    print(f"    Processed {total} projections across {len(df)} neurons")
    return df


# ======================================================================
# DEPRECATED
# ======================================================================
def add_hierarchy_levels_to_df(
    df: pd.DataFrame,
    hierarchy: RegionHierarchy,
    region_col: str = "Soma_Region",
    max_level: int = 6,
) -> pd.DataFrame:
    import warnings

    warnings.warn(
        "add_hierarchy_levels_to_df is deprecated; use add_soma_hierarchy_column",
        DeprecationWarning,
        stacklevel=2,
    )
    df = df.copy()
    for lv in range(1, max_level + 1):
        df[f"Region_L{lv}"] = df[region_col].apply(
            lambda r, _lv=lv: hierarchy.aggregate_to_level(r, _lv)
        )
    return df