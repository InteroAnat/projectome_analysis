"""
population.py - Population-level batch neuron analysis.

save_all() produces:
    tables/{sample_id}_results.xlsx  (7+ sheets)
    reports/analysis_summary.txt
    reports/terminal_report.txt
    reports/projection_sites_report.txt
    plots/*.png  (all plots)
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from region_analysis.output_manager import OutputManager
from region_analysis.hierarchy import (
    RegionHierarchy,
    add_soma_hierarchy_column,
    add_projection_length_hierarchy,
    add_projection_hierarchy,
    extract_soma_level,
    resolve_to_level,
)
from region_analysis.hierarchy_table import DualHierarchyTable
from region_analysis.classifier import NeuronClassifier
from region_analysis.neuron_analysis import RegionAnalysisPerNeuron
from region_analysis.laterality import LateralityParser, add_laterality_columns
from region_analysis.utils import (
    parse_terminal_regions, save_debug_snapshot, is_known_terminal_target,
    terminal_target_region_counts,
    load_processed_df, normalize_region_dataframe, parse_projection_lengths,
    projection_column_names,
    neuron_metadata,
)
from region_analysis.plotting import (
    plot_soma_distribution_df,
    plot_type_distribution_df,
    plot_terminal_distribution_df,
    plot_projection_sites_count_df,
    plot_region_distribution,
    plot_region_distribution_stacked,
    plot_laterality_summary_df,
    plot_neuron_projections,
)


_NEUROVIS_PATH = os.path.abspath(r"D:\projectome_analysis\neuron-vis\neuronVis")
if _NEUROVIS_PATH not in sys.path:
    sys.path.append(_NEUROVIS_PATH)

import IONData  # noqa: E402
import neuro_tracer as nt  # noqa: E402

_iondata = IONData.IONData()


class PopulationRegionAnalysis:

    def __init__(
        self,
        sample_id: str,
        atlas: np.ndarray,
        atlas_table: pd.DataFrame,
        template_img=None,
        nii_space: str = "monkey",
        arm_key_path: str = None,
        hierarchy_csv: str = None,
        cortex_hierarchy_csv: str = None,
        subcortical_hierarchy_csv: str = None,
        auto_hierarchy: bool = True,
        auto_laterality: bool = True,
        output_base: str = None,
        create_output_folder: bool = False,
        show_plots: bool = True,
        neuron_list: Optional[List[dict]] = None,
        swc_source_dir: str = None,
        hemisphere_reference=None,
    ):
        """
        Args:
            sample_id: Sample ID
            atlas: Atlas volume
            atlas_table: Region lookup table
            template_img: Template for debug snapshots
            nii_space: Coordinate space
            arm_key_path: Path to ARM key file (fallback L1-L2)
            hierarchy_csv: Single CSV for all regions (legacy, use cortex/subcortical instead)
            cortex_hierarchy_csv: CSV file(s) for cortical regions (CL_/CR_)
            subcortical_hierarchy_csv: CSV file(s) for subcortical regions (SL_/SR_)
            auto_hierarchy: Auto-add hierarchy columns
            auto_laterality: Auto-add laterality columns
            output_base: Root for output folders
            create_output_folder: Create timestamped output tree
            show_plots: Whether to display plots
        """
        self.sample_id = sample_id
        self.full_atlas = atlas
        self.atlas_table = atlas_table
        self.template_img = template_img
        self.nii_space = nii_space
        self.neuron_list = (list(neuron_list) if neuron_list is not None
                            else _iondata.getNeuronListBySampleID(sample_id))
        self.swc_source_dir = Path(swc_source_dir) if swc_source_dir is not None else None
        self.load_failures = []
        self.hemisphere_reference = hemisphere_reference
        self.classifier = NeuronClassifier(atlas_table)
        self.plot_dataframe = pd.DataFrame()
        self.neurons: dict = {}

        self.auto_hierarchy = auto_hierarchy
        self.auto_laterality = auto_laterality
        self.hierarchy: Optional[RegionHierarchy] = None
        self.hierarchy_table = None  # Legacy single table
        self.dual_hierarchy: Optional[DualHierarchyTable] = None  # New dual system
        self._arm_key_path = arm_key_path
        self.show_plots = show_plots

        self.output: Optional[OutputManager] = None
        if create_output_folder:
            self.output = OutputManager(
                base_path=output_base,
                sample_id=sample_id,
                create_timestamp=True,
            )

        # Load ARM key (fallback for L1-L2)
        if arm_key_path:
            self._load_hierarchy(arm_key_path)
        elif auto_hierarchy:
            self._detect_and_load_hierarchy()

        # Load user CSV hierarchy (PRIORITY for L3-L6)
        # Use dual hierarchy system if cortex/subcortical paths provided
        if cortex_hierarchy_csv or subcortical_hierarchy_csv:
            self._load_dual_hierarchy(cortex_hierarchy_csv, subcortical_hierarchy_csv)
        elif hierarchy_csv:
            # Legacy single-file loading
            self._load_hierarchy_csv(hierarchy_csv)

    # ==================================================================
    # HIERARCHY LOADING
    # ==================================================================
    def _detect_and_load_hierarchy(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "..", "atlas", "ARM_key_all.txt"),
            os.path.join(script_dir, "atlas", "ARM_key_all.txt"),
            r"D:\projectome_analysis\atlas\ARM_key_all.txt",
            r"D:\projectome_analysis\neuron_tables\arm_key.xlsx",
            r"D:\projectome_analysis\neuron_tables\ARM_key_all.txt",
            "../atlas/ARM_key_all.txt",
        ]
        for path in candidates:
            if os.path.exists(path):
                self._load_hierarchy(path)
                return

    def _load_hierarchy(self, arm_key_path: str):
        try:
            self.hierarchy = RegionHierarchy.from_file(arm_key_path)
            self._arm_key_path = arm_key_path
        except Exception:
            self.hierarchy = None

    def _load_hierarchy_csv(self, path):
        """
        Load user-provided hierarchy CSV/Excel file(s).
        
        Args:
            path: Single file path (str) or list of paths.
                  CSV takes PRIORITY over ARM key for all levels it covers.
        """
        from region_analysis.hierarchy_table import HierarchyTable

        paths = [path] if isinstance(path, (str, Path)) else list(path)
        if not paths:
            raise ValueError("At least one hierarchy file is required")
        self.hierarchy_table = HierarchyTable.from_files(*paths)

    def load_hierarchy_csv(self, path):
        """
        Load hierarchy CSV after initialization.
        Can be called multiple times — regions accumulate.
        
        Args:
            path: Single file path (str) or list of paths.
        """
        from region_analysis.hierarchy_table import HierarchyTable
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        if not paths:
            raise ValueError("At least one hierarchy file is required")
        loaded = HierarchyTable.from_files(*paths)
        if self.hierarchy_table is not None:
            loaded = HierarchyTable(pd.concat([self.hierarchy_table.df, loaded.df], ignore_index=True))
        self.hierarchy_table = loaded

    def _load_dual_hierarchy(self, cortex_paths=None, subcortical_paths=None):
        """Load separate hierarchy tables for cortex and subcortical regions."""
        def normalize_paths(paths):
            if paths is None:
                return []
            if isinstance(paths, str):
                return [paths]
            return list(paths)
        
        cortex_valid = [p for p in normalize_paths(cortex_paths) if os.path.exists(p)]
        subcortical_valid = [p for p in normalize_paths(subcortical_paths) if os.path.exists(p)]
        
        if cortex_valid or subcortical_valid:
            self.dual_hierarchy = DualHierarchyTable(
                cortex_paths=cortex_valid if cortex_valid else None,
                subcortical_paths=subcortical_valid if subcortical_valid else None,
            )

    # ==================================================================
    # PROCESSING
    # ==================================================================
    def process(
        self,
        limit: int = None,
        level: int = 6,
        neuron_id: Union[str, List[str]] = None,
        add_hierarchy: bool = None,
        add_laterality: bool = None,
    ):
        """
        Process neurons and build the analysis dataframe.
        
        Args:
            limit: Maximum number of neurons to process (from full list)
            level: Atlas hierarchy level (1-6, default: 6)
            neuron_id: Single neuron ID (str like '001.swc') or list of IDs
                       to process specific neurons
            add_hierarchy: Override auto_hierarchy setting
            add_laterality: Override auto_laterality setting
        """
        if isinstance(level, (bool, np.bool_)) or not isinstance(level, (int, np.integer)) or not 1 <= level <= 6:
            raise ValueError("Atlas level must be an integer between 1 and 6")
        if self.full_atlas.ndim not in (3, 5):
            raise ValueError("Atlas must be a selected 3D level or a 5D hierarchy")
        if self.full_atlas.ndim == 5 and (self.full_atlas.shape[3] != 1 or level > self.full_atlas.shape[4]):
            raise ValueError("Atlas hierarchy geometry does not contain the requested level")
        if neuron_id is not None and not isinstance(neuron_id, str):
            selected_ids = list(neuron_id)
            if len(selected_ids) != len(set(selected_ids)):
                raise ValueError("Requested neuron IDs must be unique")
            neuron_id = selected_ids

        current_atlas = (
            self.full_atlas[:, :, :, 0, level - 1]
            if self.full_atlas.ndim == 5
            else self.full_atlas
        )

        # Build target list based on neuron_id parameter
        if neuron_id is not None:
            # Normalize to list
            if isinstance(neuron_id, str):
                neuron_ids = [neuron_id]
            else:
                neuron_ids = list(neuron_id)
            
            # Find matching neurons
            target_list = []
            not_found = []
            for nid in neuron_ids:
                matches = [n for n in self.neuron_list if n["name"] == nid]
                if matches:
                    target_list.extend(matches)
                else:
                    not_found.append(nid)
            
            if not_found:
                print(f"[WARN] Neurons not found: {not_found}")
            
            if not target_list:
                print(f"[ERROR] No matching neurons found for: {neuron_ids}")
                return
        elif limit:
            target_list = self.neuron_list[:limit]
        else:
            target_list = self.neuron_list

        identities = []
        for entry in target_list:
            sample = str(entry.get("sampleid", "")).strip()
            name = entry.get("name")
            if sample != str(self.sample_id).strip():
                raise ValueError("Selected neuron entry belongs to another sample or lacks sampleid")
            if not isinstance(name, str) or not name.strip() or Path(name).name != name:
                raise ValueError("Selected neuron entry requires a nonempty basename")
            identities.append((sample, name))
        if len(identities) != len(set(identities)):
            raise ValueError("Selected full neuron identities must be unique")

        print(f"Processing {len(target_list)} neurons at Level {level}...")

        data = []
        self.load_failures = []
        for entry in target_list:
            print(f"  -> {entry['name']}")
            neuron = nt.neuro_tracer()
            try:
                source = None
                if self.swc_source_dir is not None:
                    source_path = self.swc_source_dir / entry["name"]
                    if not source_path.is_file():
                        raise FileNotFoundError(f"Declared local SWC source missing: {source_path}")
                    source = str(source_path)
                neuron.process(
                    entry["sampleid"], entry["name"], nii_space=self.nii_space,
                    swc=source,
                    output_dir=(str(self.output.output_dir / "processed_neurons")
                                if self.swc_source_dir is not None and self.output else None),
                )
            except Exception as exc:
                self.load_failures.append({"SampleID": entry.get("sampleid", self.sample_id),
                                           "NeuronID": entry.get("name"), "error": str(exc)})
                print(f"     Load failed for {entry.get('sampleid', self.sample_id)}/{entry.get('name', 'Unknown')}: {exc}")
                continue

            analysis = RegionAnalysisPerNeuron(neuron, current_atlas, self.atlas_table)
            analysis.run()

            neuron.outliers = []
            if "Unknown" in analysis.soma_region or analysis.soma_region == "Out_of_Bounds":
                root = neuron.root
                neuron.outliers.append(
                    {
                        "type": "Soma",
                        "region": analysis.soma_region,
                        "coords": (
                            int(root.x_nii),
                            int(root.y_nii),
                            int(root.z_nii),
                        ),
                    }
                )
            for item in analysis.terminal_regions:
                if "Unknown" in item["region"] or item["region"] == "Out_of_Bounds":
                    neuron.outliers.append(
                        {
                            "type": "Terminal",
                            "region": item["region"],
                            "coords": item["coords"],
                        }
                    )
            self.neurons[neuron.swc_filename] = neuron

            seen, term_unique = set(), []
            for item in analysis.terminal_regions:
                r = item["region"]
                if r not in seen:
                    seen.add(r)
                    term_unique.append(r)

            n_type = self.classifier.classify_single_neuron(
                term_unique, analysis.soma_region
            )
            root = neuron.root

            data.append(
                {
                    "SampleID": self.sample_id,
                    "NeuronID": neuron.swc_filename,
                    "Neuron_Type": n_type,
                    "Soma_Region": analysis.soma_region,
                    "Soma_NII_X": round(root.x_nii, 4),
                    "Soma_NII_Y": round(root.y_nii, 4),
                    "Soma_NII_Z": round(root.z_nii, 4),
                    "Soma_Phys_X": round(root.x, 4),
                    "Soma_Phys_Y": round(root.y, 4),
                    "Soma_Phys_Z": round(root.z, 4),
                    "Total_Length": analysis.neuron_total_length,
                    "Length_Unit": "voxel",
                    "Terminal_Count": len(term_unique),
                    "Terminal_Regions": term_unique,
                    "Region_projection_length": analysis.mapped_brain_region_lengths,
                    "Outlier_Count": len(neuron.outliers),
                    "Outlier_Details": neuron.outliers,
                }
            )

        self.plot_dataframe = pd.DataFrame(data)
        if self.plot_dataframe.empty:
            return

        do_hier = add_hierarchy if add_hierarchy is not None else self.auto_hierarchy
        hier_available = self.hierarchy is not None or self.hierarchy_table is not None or self.dual_hierarchy is not None
        if do_hier and hier_available:
            self._apply_hierarchy_columns(max_level=6, projection_min_level=6)

        do_lat = add_laterality if add_laterality is not None else self.auto_laterality
        if do_lat:
            self._apply_laterality_columns()

    # ==================================================================
    # HIERARCHY HELPERS
    # ==================================================================
    def _apply_hierarchy_columns(self, max_level: int = 6, projection_min_level: int = 1):
        """
        Apply hierarchy columns with configurable levels for soma and projections.
        
        Uses dual hierarchy system if available, otherwise falls back to
        single hierarchy table or ARM key.
        
        Args:
            max_level: Max hierarchy level for soma regions (default: 6, covers L1-L6)
            projection_min_level: Min hierarchy level for projections (default: 1, covers L1-L6)
        """
        # Determine which hierarchy source to use
        if self.dual_hierarchy is not None:
            hierarchy_source = self.dual_hierarchy
        elif self.hierarchy_table is not None:
            hierarchy_source = self.hierarchy_table
        elif self.hierarchy is not None:
            hierarchy_source = None  # Will use ARM key in functions
        else:
            return

        self.plot_dataframe = add_soma_hierarchy_column(
            self.plot_dataframe, self.hierarchy, max_level=max_level,
            hierarchy_table=hierarchy_source,
        )
        self.plot_dataframe = add_projection_hierarchy(
            self.plot_dataframe, self.hierarchy, max_level=max_level,
            hierarchy_table=hierarchy_source,
        )
        
        # Projection hierarchy: projection_min_level to max_level
        self.plot_dataframe = add_projection_length_hierarchy(
            self.plot_dataframe,
            self.hierarchy,
            "Region_projection_length",
            max_level,
            min_level=projection_min_level,
            hierarchy_table=hierarchy_source,
        )
        
        # Verify all projection columns exist
        self._ensure_projection_columns_exist(projection_min_level, max_level)

    def _ensure_projection_columns_exist(self, min_level: int, max_level: int):
        """Ensure projection length columns exist, create empty dicts if missing."""
        for lv in range(min_level, max_level + 1):
            col_name = f"Region_Projection_Length_L{lv}"
            if col_name not in self.plot_dataframe.columns:
                self.plot_dataframe[col_name] = self.plot_dataframe.apply(lambda _: {}, axis=1)

    def add_hierarchy_columns(
        self, arm_key_path: str = None, max_level: int = 6
    ) -> pd.DataFrame:
        if self.plot_dataframe.empty:
            print("Error: No data.")
            return None
        if arm_key_path and (
            self._arm_key_path != arm_key_path or self.hierarchy is None
        ):
            self._load_hierarchy(arm_key_path)
        if self.hierarchy is None and self.hierarchy_table is None and self.dual_hierarchy is None:
            self._detect_and_load_hierarchy()
        if self.hierarchy is None and self.hierarchy_table is None and self.dual_hierarchy is None:
            raise FileNotFoundError("No hierarchy source found.")
        self._apply_hierarchy_columns(max_level, projection_min_level=1)
        return self.plot_dataframe

    def add_projection_hierarchy_columns(
        self, max_level: int = 6, min_level: int = 6, arm_key_path: str = None
    ) -> pd.DataFrame:
        """
        Add projection hierarchy columns for specified level range.
        
        Args:
            max_level: Maximum hierarchy level (default: 6 - finest)
            min_level: Minimum hierarchy level (default: 6 - only finest level)
            arm_key_path: Optional path to ARM key file
        """
        if self.plot_dataframe.empty:
            print("[ERROR] No data.")
            return None
        if arm_key_path and self.hierarchy is None:
            self._load_hierarchy(arm_key_path)
        if self.hierarchy is None and self.hierarchy_table is None and self.dual_hierarchy is None:
            print("[ERROR] No hierarchy source.")
            return None
        
        self.plot_dataframe = add_projection_hierarchy(
            self.plot_dataframe,
            self.hierarchy,
            max_level,
            hierarchy_table=self.dual_hierarchy or self.hierarchy_table,
        )
        
        # Also ensure projection length columns exist
        self.plot_dataframe = add_projection_length_hierarchy(
            self.plot_dataframe,
            self.hierarchy,
            "Region_projection_length",
            max_level,
            min_level=min_level,
            hierarchy_table=self.dual_hierarchy or self.hierarchy_table,
        )
        
        self._ensure_projection_columns_exist(min_level, max_level)
        return self.plot_dataframe

    # ==================================================================
    # LATERALITY
    # ==================================================================
    def _apply_laterality_columns(self):
        length_col = None
        for c in ("Region_Projection_Length_finest", "Region_projection_length"):
            if c in self.plot_dataframe.columns:
                length_col = c
                break
        reference = getattr(self, "hemisphere_reference", None)
        reference_sides = None
        if reference is not None:
            sides = []
            for _, row in self.plot_dataframe.iterrows():
                neuron = self.neurons.get(row["NeuronID"])
                xyz = ([neuron.root.x_nii, neuron.root.y_nii, neuron.root.z_nii]
                       if neuron is not None else [row["Soma_NII_" + a] for a in "XYZ"])
                sides.append(reference.sample(xyz))
            reference_sides = pd.Series(sides, index=self.plot_dataframe.index)
        self.plot_dataframe = add_laterality_columns(
            self.plot_dataframe,
            soma_col="Soma_Region",
            terminal_col="Terminal_Regions",
            length_col=length_col,
            soma_side_reference=reference_sides,
            soma_side_reference_source=reference.source if reference is not None else None,
        )
        if hasattr(self, "classifier"):
            self.plot_dataframe["Neuron_Type"] = self.plot_dataframe.apply(
                lambda row: self.classifier.classify_single_neuron(
                    row["Terminal_Regions"], row["Soma_Region"], soma_side=row["Soma_Side"]), axis=1)

    def apply_soma_labels(self, mapping: dict, source: str):
        """Apply reviewed labels and recompute dependent views together.

        This edits this analysis instance only. It does not promote annotations
        or modify source workbooks. Conflicts with the reference remain explicit.
        """
        if not source or not str(source).strip():
            raise ValueError("Refined soma labels require source provenance")
        if "Soma_Region_Auto" not in self.plot_dataframe:
            self.plot_dataframe["Soma_Region_Auto"] = self.plot_dataframe.Soma_Region
        labels = self.plot_dataframe.NeuronID.map(mapping)
        use = labels.notna()
        self.plot_dataframe.loc[use, "Soma_Region"] = labels.loc[use]
        self.plot_dataframe.loc[use, "Soma_Label_Source"] = source
        self._apply_hierarchy_columns(max_level=6, projection_min_level=1)
        self._apply_laterality_columns()
        return self.plot_dataframe

    def add_laterality(self) -> pd.DataFrame:
        if self.plot_dataframe.empty:
            print("[ERROR] No data.")
            return None
        self._apply_laterality_columns()
        return self.plot_dataframe

    # ==================================================================
    # LOAD PRE-SAVED
    # ==================================================================
    def load_processed_dataframe(self, df_or_path) -> pd.DataFrame:
        if isinstance(df_or_path, pd.DataFrame):
            loaded = normalize_region_dataframe(df_or_path)
        elif isinstance(df_or_path, (str, os.PathLike)):
            loaded = load_processed_df(df_or_path)
        else:
            raise TypeError("Expected DataFrame or file path")
        self.plot_dataframe = loaded
        print(f"Loaded {len(self.plot_dataframe)} neurons")
        return self.plot_dataframe

    # ==================================================================
    # DERIVED VIEWS
    # ==================================================================
    def get_summary_df(self) -> pd.DataFrame:
        """Sheet 1: Scalar-only summary."""
        cols = [
            "NeuronUID", "SampleID", "NeuronID", "Neuron_Type", "Soma_Region", "Soma_Side",
            "Soma_Side_Method", "Soma_Side_Conflict", "Soma_Side_Reference_Source",
            "Soma_Side_Name", "Soma_Side_Reference", "Soma_Region_Auto", "Soma_Label_Source",
            "Soma_NII_X", "Soma_NII_Y", "Soma_NII_Z",
            "Soma_Phys_X", "Soma_Phys_Y", "Soma_Phys_Z",
            "Total_Length", "Length_Unit", "Terminal_Count",
            "N_Ipsilateral", "N_Contralateral", "N_Laterality_Unknown",
            "Total_Ipsilateral_Length", "Total_Contralateral_Length", "Total_Unknown_Laterality_Length",
            "Laterality_Index", "Outlier_Count",
        ]
        available = [c for c in cols if c in self.plot_dataframe.columns]
        return self.plot_dataframe[available].copy()

    def get_soma_hierarchy_df(self) -> pd.DataFrame:
        """Sheet 2: Hierarchy dict expanded to L1–L6."""
        identity = [column for column in ("NeuronUID", "SampleID", "NeuronID", "Soma_Region")
                    if column in self.plot_dataframe]
        neuron_metadata(self.plot_dataframe)
        df = self.plot_dataframe[identity].copy()
        for lv in range(1, 7):
            col = f"Soma_Level_{lv}"
            if col in self.plot_dataframe:
                df[f"L{lv}"] = self.plot_dataframe[col]
        if "Soma_Region_Hierarchy" in self.plot_dataframe.columns:
            hier_exp = self.plot_dataframe["Soma_Region_Hierarchy"].apply(
                lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series(dtype=str)
            )
            for lv in range(1, 7):
                key = f"L{lv}"
                if key not in hier_exp.columns:
                    hier_exp[key] = None
            hier_exp = hier_exp[[f"L{lv}" for lv in range(1, 7)]]
            for key in hier_exp:
                if key in df:
                    conflicts = df[key].notna() & hier_exp[key].notna() & df[key].ne(hier_exp[key])
                    if conflicts.any():
                        raise ValueError(f"Conflicting soma hierarchy values for {key}")
                    df[key] = df[key].fillna(hier_exp[key])
                else:
                    df[key] = hier_exp[key]
        return df

    def get_projection_matrix(self, level: str = "finest") -> pd.DataFrame:
        """Neuron x region matrix in the existing atlas-voxel length unit."""
        if level == "finest":
            candidates = [
                "Region_Projection_Length_finest",
                "Region_projection_length",
            ]
        else:
            # Handle both string and int level
            level_int = int(level) if isinstance(level, str) and level.isdigit() else level
            candidates = [f"Region_Projection_Length_L{level_int}"]

        length_col = None
        for c in candidates:
            if c in self.plot_dataframe.columns:
                length_col = c
                break
        
        if length_col is None:
            available_proj_cols = [
                c for c in self.plot_dataframe.columns 
                if 'Projection_Length' in c or 'projection_length' in c
            ]
            print(f"[WARN] No projection-length column for level='{level}'")
            print(f"  Available projection columns: {available_proj_cols}")
            return pd.DataFrame()

        # Convert column to list of dicts, handling non-dict values
        # Keep prefixes internally for correct laterality classification
        data_list = []
        for val in self.plot_dataframe[length_col]:
            data_list.append(parse_projection_lengths(val))
        
        matrix = pd.DataFrame(data_list)
        matrix.fillna(0, inplace=True)
        
        if matrix.empty:
            print(f"[WARN] Empty matrix for level='{level}'")
            return pd.DataFrame()
        
        regular = sorted([c for c in matrix.columns if c != "_Unmapped"])
        ordered = regular + (["_Unmapped"] if "_Unmapped" in matrix.columns else [])
        matrix = matrix[ordered]
        metadata = neuron_metadata(self.plot_dataframe)
        matrix = pd.concat([metadata, matrix.reset_index(drop=True)], axis=1)

        if "_Unmapped" in matrix.columns:
            total_u = matrix["_Unmapped"].sum()
            total_a = matrix.drop(columns=metadata.columns).sum().sum()
            if total_u > 0:
                pct = total_u / total_a * 100 if total_a > 0 else 0
                print(
                    f"[MATRIX L{level}] _Unmapped: {total_u:.1f} atlas voxels ({pct:.1f}%)"
                )
        
        return matrix

    def get_projection_strength(self, level: str = "finest") -> pd.DataFrame:
        """Sheet 4: log10(length + 1)."""
        matrix = self.get_projection_matrix(level)
        if matrix.empty:
            return matrix
        numeric_cols = [column for column in matrix if column not in ("NeuronUID", "SampleID", "NeuronID", "Neuron_Type")]
        strength = matrix.copy()
        strength[numeric_cols] = np.log10(matrix[numeric_cols] + 1).round(4)
        return strength

    def get_projection_matrix_split(self, level: str = "finest") -> tuple:
        """
        Split projection matrix into ipsilateral and contralateral DataFrames.
        
        Uses original data with prefixes for correct laterality classification,
        then strips prefixes for clean output.
        
        Args:
            level: Hierarchy level ("finest" or 1-6)
            
        Returns:
            Tuple of (ipsi_df, contra_df) with prefixes stripped.
        """
        from region_analysis.laterality import LateralityParser
        
        # Get source column with prefixes (for correct laterality classification)
        if level == "finest":
            candidates = ["Region_Projection_Length_finest", "Region_projection_length"]
        else:
            level_int = int(level) if isinstance(level, str) and level.isdigit() else level
            candidates = [f"Region_Projection_Length_L{level_int}"]
        
        source_col = None
        for c in candidates:
            if c in self.plot_dataframe.columns:
                source_col = c
                break
        
        if source_col is None:
            return pd.DataFrame(), pd.DataFrame()
        
        # Ensure laterality columns exist
        if "Soma_Side" not in self.plot_dataframe.columns:
            self._apply_laterality_columns()
        
        metadata = neuron_metadata(self.plot_dataframe)
        lengths = self.plot_dataframe[source_col].apply(parse_projection_lengths)
        atlas_regions = getattr(self, "atlas_table", pd.DataFrame()).get("Abbreviation", [])
        column_names = projection_column_names(
            {region for mapping in lengths for region in mapping}, atlas_regions)

        # Build rows with clean (stripped) column names
        ipsi_data = []
        contra_data = []
        
        for position, (_, row) in enumerate(self.plot_dataframe.iterrows()):
            soma_side = row.get("Soma_Side", "Unknown")
            length_dict = lengths.iloc[position]
            
            ipsi_row = {}
            contra_row = {}
            
            # Fill in values with stripped column names
            for region, length in length_dict.items():
                if length == 0:
                    continue
                lat = LateralityParser.classify_with_soma_side(soma_side, region)
                clean_region = column_names[region]
                
                if lat == "Ipsilateral":
                    ipsi_row[clean_region] = ipsi_row.get(clean_region, 0) + length
                elif lat == "Contralateral":
                    contra_row[clean_region] = contra_row.get(clean_region, 0) + length
            
            ipsi_data.append(ipsi_row)
            contra_data.append(contra_row)
        
        # Create DataFrames
        ipsi_df = pd.concat([metadata, pd.DataFrame(ipsi_data).fillna(0)], axis=1)
        contra_df = pd.concat([metadata, pd.DataFrame(contra_data).fillna(0)], axis=1)
        
        return ipsi_df, contra_df

    def get_projection_strength_split(self, level: str = "finest") -> tuple:
        """Log-transform the final split lengths, after anatomical aggregation."""
        results = []
        for lengths in self.get_projection_matrix_split(level):
            strength = lengths.copy()
            targets = [column for column in strength if column not in ("NeuronUID", "SampleID", "NeuronID", "Neuron_Type")]
            strength[targets] = np.log10(strength[targets] + 1).round(4)
            results.append(strength)
        return tuple(results)

    def get_terminal_sites_df(self) -> pd.DataFrame:
        """Sheet 5: Long-format terminals."""
        rows = []
        for _, nr in self.plot_dataframe.iterrows():
            nid = nr["NeuronID"]
            identity = {column: nr[column] for column in ("NeuronUID", "SampleID") if column in nr}
            ntype = nr.get("Neuron_Type", "")
            soma = nr.get("Soma_Region", "")
            lat_info = nr.get("Terminal_Laterality", None)
            if isinstance(lat_info, list) and lat_info:
                for term in lat_info:
                    rows.append(
                        {
                            **identity,
                            "NeuronID": nid,
                            "Neuron_Type": ntype,
                            "Terminal_Region": term.get("region", ""),
                            "Side": term.get("side", ""),
                            "Laterality": term.get("laterality", ""),
                        }
                    )
            else:
                terminals = nr.get("Terminal_Regions", [])
                if not isinstance(terminals, (list, tuple)):
                    terminals = parse_terminal_regions(terminals)
                for t_region in terminals:
                    rows.append(
                        {
                            **identity,
                            "NeuronID": nid,
                            "Neuron_Type": ntype,
                            "Terminal_Region": t_region,
                            "Side": LateralityParser.get_side(t_region),
                            "Laterality": LateralityParser.classify_with_soma_side(nr.get("Soma_Side", LateralityParser.get_side(soma)), t_region),
                        }
                    )
        df = pd.DataFrame(rows)
        hierarchy_source = self.dual_hierarchy or self.hierarchy_table
        if (self.hierarchy or hierarchy_source) and not df.empty:
            for lv in [1, 3, 6]:
                df[f"Terminal_L{lv}"] = df["Terminal_Region"].apply(
                    lambda r, _lv=lv: resolve_to_level(
                        str(r), _lv, self.hierarchy, hierarchy_source
                    )
                )
        return df

    def get_laterality_df(self) -> pd.DataFrame:
        """Sheet 6: Per-neuron laterality scalars."""
        scalar_cols = [
            "NeuronUID", "SampleID", "NeuronID", "Neuron_Type", "Soma_Region", "Soma_Side",
            "Soma_Side_Method", "Soma_Side_Conflict", "Soma_Side_Name", "Soma_Side_Reference",
            "N_Ipsilateral", "N_Contralateral", "N_Laterality_Unknown",
            "Total_Ipsilateral_Length", "Total_Contralateral_Length",
            "Total_Unknown_Laterality_Length", "Laterality_Index",
        ]
        available = [c for c in scalar_cols if c in self.plot_dataframe.columns]
        df = self.plot_dataframe[available].copy()
        for list_col in ("Ipsilateral_Regions", "Contralateral_Regions"):
            if list_col in self.plot_dataframe.columns:
                df[list_col] = self.plot_dataframe[list_col].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else ""
                )
        return df

    def get_outlier_df(self) -> pd.DataFrame:
        """Sheet 7: Long-format outliers."""
        rows = []
        for _, nr in self.plot_dataframe.iterrows():
            details = nr.get("Outlier_Details", [])
            identity = {column: nr[column] for column in ("NeuronUID", "SampleID") if column in nr}
            if not isinstance(details, list):
                if isinstance(details, str):
                    try:
                        import ast
                        details = ast.literal_eval(details)
                    except Exception:
                        continue
                else:
                    continue
            for o in details:
                coords = o.get("coords", (None, None, None))
                if isinstance(coords, (list, tuple)) and len(coords) >= 3:
                    vx, vy, vz = coords[0], coords[1], coords[2]
                else:
                    vx, vy, vz = None, None, None
                rows.append(
                    {
                        **identity,
                        "NeuronID": nr["NeuronID"],
                        "Neuron_Type": nr.get("Neuron_Type", ""),
                        "Outlier_Type": o.get("type", ""),
                        "Region": o.get("region", ""),
                        "Voxel_X": vx,
                        "Voxel_Y": vy,
                        "Voxel_Z": vz,
                    }
                )
        return pd.DataFrame(rows)

    # ==================================================================
    # DIAGNOSTICS
    # ==================================================================
    def diagnose_projection_regions(self, level: str = "finest"):
        """
        Diagnose which projection regions cannot be resolved to hierarchy levels.
        
        Args:
            level: Hierarchy level to check ("finest" or 1-6)
        """
        print(f"\n{'='*60}")
        print(f"PROJECTION REGION DIAGNOSTICS (level={level})")
        print(f"{'='*60}")
        
        # Get all unique regions from projections
        all_regions = set()
        length_col = f"Region_Projection_Length_L{level}" if level != "finest" else ("Region_Projection_Length_finest" if "Region_Projection_Length_finest" in self.plot_dataframe else "Region_projection_length")
        
        if length_col not in self.plot_dataframe.columns:
            print(f"[ERROR] Column {length_col} not found")
            print(f"Available: {[c for c in self.plot_dataframe.columns if 'Projection' in c]}")
            return
        
        for proj_dict in self.plot_dataframe[length_col]:
            if isinstance(proj_dict, dict):
                all_regions.update(proj_dict.keys())
        
        print(f"\nTotal unique regions in projections: {len(all_regions)}")
        
        # Check resolution for each region
        unmapped = []
        region_type_counts = {"Cortical": 0, "Subcortical": 0, "Unknown": 0}
        
        from region_analysis.laterality import LateralityParser
        
        for region in sorted(all_regions):
            rtype = LateralityParser.get_region_type(region)
            region_type_counts[rtype] += 1
            
            # Try to resolve at each level
            can_resolve = False
            for lv in range(1, 7):
                result = resolve_to_level(
                    region, lv, self.hierarchy, 
                    self.dual_hierarchy or self.hierarchy_table
                )
                if result is not None:
                    can_resolve = True
                    break
            
            if not can_resolve:
                unmapped.append((region, rtype))
        
        print(f"\nRegion type distribution:")
        for k, v in region_type_counts.items():
            print(f"  {k}: {v}")
        
        if unmapped:
            print(f"\n[WARNING] {len(unmapped)} regions CANNOT be resolved:")
            print("\nTop 20 unmapped regions:")
            for region, rtype in unmapped[:20]:
                print(f"  - {region} ({rtype})")
            
            # Show sample debug for first unmapped region
            if unmapped:
                print(f"\nDetailed debug for first unmapped region:")
                self.debug_region_resolution(unmapped[0][0])
        else:
            print(f"\n[SUCCESS] All regions can be resolved!")
        
        print(f"{'='*60}\n")
        return unmapped

    def check_projection_column_names(self, level: str = "finest"):
        """
        Check what region names are in a projection column.
        Useful for verifying prefix stripping is working.
        """
        col_name = f"Region_Projection_Length_L{level}" if level != "finest" else ("Region_Projection_Length_finest" if "Region_Projection_Length_finest" in self.plot_dataframe else "Region_projection_length")
        
        if col_name not in self.plot_dataframe.columns:
            print(f"[ERROR] Column {col_name} not found")
            return
        
        print(f"\n{'='*60}")
        print(f"COLUMN NAME CHECK: {col_name}")
        print(f"{'='*60}")
        
        # Collect all unique keys across all dictionaries
        all_keys = set()
        prefixed = []
        clean = []
        
        for proj_dict in self.plot_dataframe[col_name]:
            if isinstance(proj_dict, dict):
                for key in proj_dict.keys():
                    all_keys.add(key)
        
        # Categorize
        for key in sorted(all_keys):
            if key.startswith(('CL_', 'CR_', 'SL_', 'SR_')):
                prefixed.append(key)
            else:
                clean.append(key)
        
        print(f"\nTotal unique regions: {len(all_keys)}")
        print(f"  With prefixes (CL_/CR_/SL_/SR_): {len(prefixed)}")
        print(f"  Clean names: {len(clean)}")
        
        if prefixed:
            print(f"\nPrefixed regions (should be cleaned):")
            for r in prefixed[:20]:
                print(f"  - {r}")
        
        if clean:
            print(f"\nClean regions (sample):")
            for r in clean[:20]:
                print(f"  - {r}")
        
        print(f"{'='*60}\n")

    def test_specific_regions(self, regions: list = None):
        """
        Test hierarchy resolution for specific problematic regions.
        
        Args:
            regions: List of region names to test (default: SL_HF, SR_HF, SL_DMId)
        """
        if regions is None:
            regions = ['SL_HF', 'SR_HF', 'SL_DMId']
        
        print(f"\n{'='*60}")
        print(f"TESTING SPECIFIC REGIONS")
        print(f"{'='*60}")
        
        hierarchy_source = self.dual_hierarchy or self.hierarchy_table
        
        for region in regions:
            print(f"\n{region}:")
            for lv in [1, 2, 3, 4, 5, 6]:
                result = resolve_to_level(region, lv, self.hierarchy, hierarchy_source)
                status = f"-> {result}" if result else "FAILED"
                print(f"  L{lv}: {status}")
        
        print(f"{'='*60}\n")

    # ==================================================================
    # REGION MATRIX (backward compat)
    # ==================================================================
    def get_region_matrix(self) -> pd.DataFrame:
        return self.get_projection_matrix("finest")

    # ==================================================================
    # INSPECT
    # ==================================================================
    def inspect_neuron(self, target_filename: str):
        if self.plot_dataframe.empty:
            print("Error: No data.")
            return
        mask = self.plot_dataframe["NeuronID"] == target_filename
        if not mask.any():
            print(f"Error: '{target_filename}' not found.")
            return
        row = self.plot_dataframe[mask].iloc[0]

        print(f"\n{'='*50}")
        print(f"REPORT: {row['NeuronID']}")
        print(f"{'='*50}")
        print(f"  Type: {row['Neuron_Type']} | Soma: {row['Soma_Region']}")

        if "Soma_Region_Hierarchy" in self.plot_dataframe.columns:
            hier = row["Soma_Region_Hierarchy"]
            if isinstance(hier, dict) and hier:
                parts = [f"{k}: {v}" for k, v in sorted(hier.items())]
                print(f"  Hierarchy: {' -> '.join(parts)}")

        if "Soma_Side" in self.plot_dataframe.columns:
            print(
                f"  Side: {row.get('Soma_Side', '?')} | "
                f"Ipsi: {row.get('N_Ipsilateral', '?')} | "
                f"Contra: {row.get('N_Contralateral', '?')} | "
                f"LI: {row.get('Laterality_Index', '?')}"
            )

        for tag in ("NII", "Phys"):
            x = row.get(f"Soma_{tag}_X", "?")
            y = row.get(f"Soma_{tag}_Y", "?")
            z = row.get(f"Soma_{tag}_Z", "?")
            print(f"  Soma {tag}: ({x}, {y}, {z})")

        print(f"  Length: {row['Total_Length']:.3f}")
        print(f"  Terminals ({row['Terminal_Count']}): {row['Terminal_Regions']}")
        print(f"  Outliers: {row['Outlier_Count']}")

        length_col = None
        for c in ("Region_Projection_Length_finest", "Region_projection_length"):
            if c in self.plot_dataframe.columns:
                length_col = c
                break
        if length_col and isinstance(row[length_col], dict):
            top5 = sorted(
                row[length_col].items(), key=lambda x: x[1], reverse=True
            )[:5]
            if top5:
                print(f"\n  Top projections (atlas voxels -> log10 strength):")
                for region, length in top5:
                    strength = np.log10(length + 1)
                    print(f"    {region}: {length:.2f} atlas voxels -> {strength:.3f}")

        if length_col:
            save = (
                str(self.output.get_plot_path(f"inspect_{target_filename}"))
                if self.output
                else None
            )
            plot_neuron_projections(row[length_col], row["NeuronID"], save_path=save, show=self.show_plots)

    # ==================================================================
    # DEBUG HIERARCHY
    # ==================================================================
    def debug_hierarchy(self, region: str):
        """Report actual hierarchy paths and resolution using supported APIs."""
        source = self.dual_hierarchy or self.hierarchy_table
        print(f"HIERARCHY DEBUG: {region}")
        if source:
            print(f"CSV path: {source.get_path(region)}")
        for level in range(1, 7):
            print(f"  L{level}: {resolve_to_level(region, level, self.hierarchy, source)}")

    def debug_region_resolution(self, region: str):
        """Report explicit domain routing without hypothetical helper methods."""
        print(f"Region {region}: side={LateralityParser.get_side(region)}, type={LateralityParser.get_region_type(region)}")
        if self.dual_hierarchy:
            table = self.dual_hierarchy._get_table(region)
            print(f"Selected table: {table.name if table else None}")
        self.debug_hierarchy(region)

    # ==================================================================
    # OUTLIER EXPORT
    # ==================================================================
    def export_outlier_snapshots(self, neuron_id: str, max_snapshots: int = 3):
        if neuron_id not in self.neurons:
            print(f"Neuron {neuron_id} not loaded.")
            return
        neuron = self.neurons[neuron_id]
        if not neuron.outliers:
            print(f"No outliers for {neuron_id}.")
            return
        folder = (
            str(self.output.debug_dir) if self.output else "../resource/debug_outliers"
        )
        print(f"Exporting outliers for {neuron_id} (limit {max_snapshots})...")
        for i, err in enumerate(neuron.outliers):
            if i >= max_snapshots:
                break
            save_debug_snapshot(
                err["coords"],
                neuron.swc_filename,
                self.template_img,
                err["type"],
                folder,
            )

    # ==================================================================
    # PLOTTING — delegates to plotting.py, auto-saves when output exists
    # ==================================================================
    def plot_soma_distribution(self, **kw):
        s = (
            str(self.output.get_plot_path("soma_distribution"))
            if self.output
            else None
        )
        show = kw.pop("show", self.show_plots)
        return plot_soma_distribution_df(self.plot_dataframe, save_path=s, show=show, **kw)

    def plot_type_distribution(self, **kw):
        s = (
            str(self.output.get_plot_path("type_distribution"))
            if self.output
            else None
        )
        show = kw.pop("show", self.show_plots)
        return plot_type_distribution_df(self.plot_dataframe, save_path=s, show=show, **kw)

    def plot_terminal_distribution(self, **kw):
        s = (
            str(self.output.get_plot_path("terminal_distribution"))
            if self.output
            else None
        )
        r = (
            str(self.output.get_report_path("terminal_report"))
            if self.output
            else None
        )
        show = kw.pop("show", self.show_plots)
        return plot_terminal_distribution_df(
            self.plot_dataframe, save_path=s, save_report_path=r, show=show, **kw
        )

    def plot_projection_sites_count(self, **kw):
        s = (
            str(self.output.get_plot_path("projection_sites_count"))
            if self.output
            else None
        )
        r = (
            str(self.output.get_report_path("projection_sites_report"))
            if self.output
            else None
        )
        show = kw.pop("show", self.show_plots)
        return plot_projection_sites_count_df(
            self.plot_dataframe, save_path=s, save_report_path=r, show=show, **kw
        )

    def plot_region_dist(self, level: int, **kw):
        s = (
            str(self.output.get_plot_path(f"region_dist_L{level}"))
            if self.output
            else None
        )
        show = kw.pop("show", self.show_plots)
        return plot_region_distribution(
            self.plot_dataframe, level, save_path=s, show=show, **kw
        )

    def plot_region_dist_stacked(self, levels: list = None, **kw):
        """
        Plot stacked regional distribution for multiple hierarchy levels.

        Args:
            levels: List of levels to plot (default: [1, 2, 3, 4, 5, 6])
            **kw: Additional arguments passed to plot_region_distribution_stacked
        """
        s = (
            str(self.output.get_plot_path("region_dist_stacked"))
            if self.output
            else None
        )
        show = kw.pop("show", self.show_plots)
        return plot_region_distribution_stacked(
            self.plot_dataframe, levels=levels, save_path=s, show=show, **kw
        )

    def plot_laterality_summary(self, **kw):
        s = (
            str(self.output.get_plot_path("laterality_summary"))
            if self.output
            else None
        )
        show = kw.pop("show", self.show_plots)
        return plot_laterality_summary_df(self.plot_dataframe, save_path=s, show=show, **kw)

    def plot_projection_by_soma(self, stat: str = "median", show: bool = None):
        if self.plot_dataframe.empty:
            return None
        if show is None:
            show = self.show_plots
        df = self.plot_dataframe
        regions = df["Soma_Region"].unique()
        data = [df.loc[df["Soma_Region"] == r, "Total_Length"] for r in regions]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(data, labels=regions)
        ax.set_title(f"Projection Length by Soma Region ({stat})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if self.output:
            p = self.output.get_plot_path("projection_by_soma")
            fig.savefig(p, dpi=300, bbox_inches="tight")
            print(f"[SAVED] {p}")
        if show:
            plt.show()
        return fig

    # ==================================================================
    # SAVE
    # ==================================================================
    def save_plot(self, fig, name: str):
        if self.output is None:
            plt.show()
            return
        p = self.output.get_plot_path(name)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {p}")
        plt.close(fig)

    def save_report(self, content: str, name: str):
        if self.output is None:
            print(content)
            return
        p = self.output.get_report_path(name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[SAVED] {p}")

    def save_all(
        self,
        include_strength_levels: list = None,
        generate_plots: bool = True,
    ) -> Optional[str]:
        """
        Save everything:
            tables/  → multi-sheet Excel (includes laterality columns)
            reports/ → 3 text reports
            plots/   → all plots as PNG

        Args:
            include_strength_levels: Extra hierarchy levels (e.g. [1, 3, 6]).
                                     Will generate Projection_Length_L{n} and
                                     Projection_Strength_L{n} sheets WITH ipsi/contra
                                     columns (e.g., Projection_Strength_L6_IPSI).
            generate_plots: If True, generate and save all plots.
        """
        if self.plot_dataframe.empty:
            print("[ERROR] No data. Run process() first.")
            return None

        if self.output is None:
            print("[WARN] No output manager. Set create_output_folder=True.")
            fallback = f"{self.sample_id}_results.xlsx"
            self._write_workbook(fallback, include_strength_levels)
            return None

        print("\n" + "=" * 60)
        print(f"SAVING ALL RESULTS: {self.sample_id}")
        print("=" * 60)

        # Ensure all required hierarchy columns exist before saving
        if include_strength_levels:
            self._ensure_hierarchy_columns_for_levels(include_strength_levels)

        # 1. TABLES (includes laterality columns integrated)
        path = self.output.get_table_path(f"{self.sample_id}_results")
        self._write_workbook(str(path), include_strength_levels)

        # 2. REPORTS (3 separate files)
        print("\n[REPORTS]")
        self._save_analysis_summary_report()
        self._save_terminal_report()
        self._save_projection_sites_report()

        # 3. PLOTS
        if generate_plots:
            print("\n[PLOTS]")
            self._generate_all_plots()

        self.output.print_summary()
        return str(self.output.output_dir)

    def _ensure_hierarchy_columns_for_levels(self, levels: list):
        """
        Ensure hierarchy columns exist for all requested levels.
        Creates missing levels on-demand when save_all() requests them.
        
        Args:
            levels: List of levels to ensure exist (e.g., [3, 6] for L3 and L6)
        """
        if not levels:
            return
        
        missing_levels = []
        for lv in levels:
            col_name = f"Region_Projection_Length_L{lv}"
            if col_name not in self.plot_dataframe.columns:
                missing_levels.append(lv)
        
        if missing_levels:
            print(f"[HIERARCHY] Creating missing projection columns: {missing_levels}")
            if self.hierarchy is not None or self.hierarchy_table is not None or self.dual_hierarchy is not None:
                # Only create the missing levels (not all L1-L6)
                min_missing = min(missing_levels)
                max_missing = max(missing_levels)
                print(f"[HIERARCHY] Generating levels {min_missing}-{max_missing}")
                self._apply_hierarchy_columns(max_level=6, projection_min_level=min_missing)
            else:
                print("[WARN] No hierarchy source available. Creating empty columns.")
                self._ensure_projection_columns_exist(min(missing_levels), max(missing_levels))


    def _write_workbook(
        self,
        path: str,
        include_strength_levels: list = None,
    ):
        print(f"\n[EXCEL] Writing: {path}")
        
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Sheet 1: Summary
            summary = self.get_summary_df()
            summary.to_excel(writer, sheet_name="Summary", index=False)
            print(
                f"  [Sheet] Summary: {len(summary)} rows x {len(summary.columns)} cols"
            )

            # Sheet 2: Soma Hierarchy
            soma_hier = self.get_soma_hierarchy_df()
            soma_hier.to_excel(writer, sheet_name="Soma_Hierarchy", index=False)
            print(
                f"  [Sheet] Soma_Hierarchy: {len(soma_hier)} rows x {len(soma_hier.columns)} cols"
            )

            # Sheet 3: Terminal Sites
            terminals = self.get_terminal_sites_df()
            terminals.to_excel(writer, sheet_name="Terminal_Sites", index=False)
            print(f"  [Sheet] Terminal_Sites: {len(terminals)} rows")

            # Sheet 4: Laterality (scalar summary)
            laterality = self.get_laterality_df()
            if len(laterality.columns) > 2:
                laterality.to_excel(
                    writer, sheet_name="Laterality", index=False
                )
                print(
                    f"  [Sheet] Laterality: {len(laterality)} rows x {len(laterality.columns)} cols"
                )
            else:
                print("  [Sheet] Laterality: SKIPPED (insufficient columns)")

            # Sheet 5: Outliers
            outliers = self.get_outlier_df()
            outliers.to_excel(writer, sheet_name="Outliers", index=False)
            n_o = len(outliers)
            print(
                f"  [Sheet] Outliers: {n_o} events"
                + (" (clean!)" if n_o == 0 else "")
            )

            # Projection sheets with laterality splits
            # Preserve absolute atlas labels and unknown-side lengths for
            # reconstruction and cross-sheet QC; splits alone lose that data.
            all_lengths = self.get_projection_matrix("finest")
            if not all_lengths.empty:
                all_lengths.to_excel(writer, sheet_name="Projection_Length_all", index=False)
            # Sheet 6-7: Projection Length finest (ipsi/contra split)
            ipsi_len, contra_len = self.get_projection_matrix_split("finest")
            if not ipsi_len.empty:
                ipsi_len.to_excel(writer, sheet_name="Projection_Length_ipsi", index=False)
                print(f"  [Sheet] Projection_Length_ipsi: {len(ipsi_len)} neurons x {len(ipsi_len.columns)-2} regions")
            if not contra_len.empty:
                contra_len.to_excel(writer, sheet_name="Projection_Length_contra", index=False)
                print(f"  [Sheet] Projection_Length_contra: {len(contra_len)} neurons x {len(contra_len.columns)-2} regions")

            # Sheet 8-9: Projection Strength finest (ipsi/contra split)
            ipsi_str, contra_str = self.get_projection_strength_split("finest")
            if not ipsi_str.empty:
                ipsi_str.to_excel(writer, sheet_name="Projection_Strength_ipsi", index=False)
                print(f"  [Sheet] Projection_Strength_ipsi: {len(ipsi_str)} neurons x {len(ipsi_str.columns)-2} regions")
            if not contra_str.empty:
                contra_str.to_excel(writer, sheet_name="Projection_Strength_contra", index=False)
                print(f"  [Sheet] Projection_Strength_contra: {len(contra_str)} neurons x {len(contra_str.columns)-2} regions")

            # Extra sheets for specified hierarchy levels
            if include_strength_levels:
                for lv in include_strength_levels:
                    # Length split
                    ipsi_len_lv, contra_len_lv = self.get_projection_matrix_split(lv)
                    if not ipsi_len_lv.empty:
                        ipsi_len_lv.to_excel(
                            writer,
                            sheet_name=f"Projection_Length_L{lv}_ipsi",
                            index=False,
                        )
                        print(f"  [Sheet] Projection_Length_L{lv}_ipsi: {len(ipsi_len_lv)} neurons x {len(ipsi_len_lv.columns)-2} regions")
                    if not contra_len_lv.empty:
                        contra_len_lv.to_excel(
                            writer,
                            sheet_name=f"Projection_Length_L{lv}_contra",
                            index=False,
                        )
                        print(f"  [Sheet] Projection_Length_L{lv}_contra: {len(contra_len_lv)} neurons x {len(contra_len_lv.columns)-2} regions")
                    
                    # Strength split
                    ipsi_str_lv, contra_str_lv = self.get_projection_strength_split(lv)
                    if not ipsi_str_lv.empty:
                        ipsi_str_lv.to_excel(
                            writer,
                            sheet_name=f"Projection_Strength_L{lv}_ipsi",
                            index=False,
                        )
                        print(f"  [Sheet] Projection_Strength_L{lv}_ipsi: {len(ipsi_str_lv)} neurons x {len(ipsi_str_lv.columns)-2} regions")
                    if not contra_str_lv.empty:
                        contra_str_lv.to_excel(
                            writer,
                            sheet_name=f"Projection_Strength_L{lv}_contra",
                            index=False,
                        )
                        print(f"  [Sheet] Projection_Strength_L{lv}_contra: {len(contra_str_lv)} neurons x {len(contra_str_lv.columns)-2} regions")

        print(f"\n[SAVED] {path}")

    # ==================================================================
    # 3 SEPARATE REPORTS
    # ==================================================================
    def _save_analysis_summary_report(self):
        """Report 1/3: Overall analysis summary."""
        df = self.plot_dataframe
        n = len(df)

        lines = [
            "=" * 60,
            f"REGION ANALYSIS REPORT - {self.sample_id}",
            "=" * 60,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
            "--- OVERVIEW ---",
            f"Total neurons: {n}",
        ]

        if "Neuron_Type" in df.columns:
            lines.append("\n--- Neuron Types ---")
            for t, c in df["Neuron_Type"].value_counts().items():
                lines.append(f"  {t}: {c} ({c / n * 100:.1f}%)")

        lines.append("\n--- Soma Regions (Top 10) ---")
        for r, c in df["Soma_Region"].value_counts().head(10).items():
            lines.append(f"  {r}: {c}")

        if "Soma_Side" in df.columns:
            lines.append("\n--- Laterality ---")
            for side, c in df["Soma_Side"].value_counts().items():
                lines.append(f"  Soma side {side}: {c}")
            if "N_Ipsilateral" in df.columns:
                lines.append(
                    f"  Total ipsi terminal regions:   {int(df['N_Ipsilateral'].sum())}"
                )
                lines.append(
                    f"  Total contra terminal regions: {int(df['N_Contralateral'].sum())}"
                )
            if "Laterality_Index" in df.columns:
                valid = df["Laterality_Index"].dropna()
                if not valid.empty:
                    lines.append(
                        f"  Laterality Index - mean: {valid.mean():.3f}, "
                        f"median: {valid.median():.3f}"
                    )

        length_col = None
        for c in ("Region_Projection_Length_finest", "Region_projection_length"):
            if c in df.columns:
                length_col = c
                break
        if length_col:
            all_lengths = []
            for d in df[length_col]:
                if isinstance(d, dict):
                    all_lengths.extend(d.values())
            if all_lengths:
                arr = np.array(all_lengths)
                log_arr = np.log10(arr + 1)
                lines.append("\n--- Projection Strength (Population) ---")
                lines.append(
                    f"  Raw length: mean={arr.mean():.2f}, "
                    f"median={np.median(arr):.2f}, max={arr.max():.2f} atlas voxels"
                )
                lines.append(
                    f"  Log strength: mean={log_arr.mean():.3f}, "
                    f"median={np.median(log_arr):.3f}, max={log_arr.max():.3f}"
                )

        lines += [
            "",
            "--- Morphology ---",
            f"Total length - Mean: {df['Total_Length'].mean():.2f}, "
            f"Std: {df['Total_Length'].std():.2f} atlas voxels",
            f"Unique terminal-region count - Mean: {df['Terminal_Count'].mean():.2f}, "
            f"Max: {int(df['Terminal_Count'].max())}",
        ]

        n_out_neurons = int((df["Outlier_Count"] > 0).sum())
        total_outliers = int(df["Outlier_Count"].sum())
        lines += [
            "",
            "--- Outliers ---",
            f"Neurons with outliers: {n_out_neurons} ({n_out_neurons / n * 100:.1f}%)",
            f"Total outlier events: {total_outliers}",
        ]

        # Add hierarchy column status
        lines.append("\n--- Hierarchy Columns Status ---")
        for lv in range(1, 7):
            col_name = f"Region_Projection_Length_L{lv}"
            status = "OK" if col_name in df.columns else "MISSING"
            lines.append(f"  L{lv}: {status}")

        lines += [
            "",
            "--- Excel Workbook Contents ---",
            "  Sheet 1: Summary                    (scalar overview)",
            "  Sheet 2: Soma_Hierarchy             (L1-L6 expanded)",
            "  Sheet 3: Terminal_Sites             (one row per unique terminal region)",
            "  Sheet 4: Laterality                 (ipsi/contra scalars)",
            "  Sheet 5: Outliers                   (one row per outlier)",
            "",
            "  Projection Length Sheets (split by laterality):",
            "    Projection_Length_ipsi          (ipsilateral regions only)",
            "    Projection_Length_contra        (contralateral regions only)",
            "    Projection_Length_L{n}_ipsi     (level n, ipsilateral)",
            "    Projection_Length_L{n}_contra   (level n, contralateral)",
            "",
            "  Projection Strength Sheets (split by laterality):",
            "    Projection_Strength_ipsi        (ipsilateral regions, log10)",
            "    Projection_Strength_contra      (contralateral regions, log10)",
            "    Projection_Strength_L{n}_ipsi   (level n, ipsilateral, log10)",
            "    Projection_Strength_L{n}_contra (level n, contralateral, log10)",
            "",
            "=" * 60,
        ]

        report = "\n".join(lines)
        self.save_report(report, "analysis_summary")
        print(report)

    def _save_terminal_report(self):
        """Report 2/3: Terminal distribution statistics."""
        df = self.plot_dataframe
        if df.empty or "Terminal_Regions" not in df.columns:
            return

        df_temp = df.copy()
        df_temp["Terminal_Regions"] = df_temp["Terminal_Regions"].apply(
            parse_terminal_regions
        )
        df_temp["_known"] = df_temp["Terminal_Regions"].apply(
            lambda x: sum(is_known_terminal_target(r) for r in x)
        )
        df_temp["_unk"] = df_temp["Terminal_Regions"].apply(
            lambda x: sum(not is_known_terminal_target(r) for r in x)
        )
        total_known = int(df_temp["_known"].sum())
        total_unknown = int(df_temp["_unk"].sum())
        total_sites = total_known + total_unknown
        n_with_unk = int((df_temp["_unk"] > 0).sum())
        n_only_known = int(((df_temp["_unk"] == 0) & (df_temp["_known"] > 0)).sum())
        n_no_targets = int(df_temp["Terminal_Regions"].map(len).eq(0).sum())
        total_n = len(df_temp)

        counts = terminal_target_region_counts(df_temp["Terminal_Regions"]).head(30)

        lines = [
            "=" * 60,
            "TERMINAL REGION DISTRIBUTION REPORT",
            "=" * 60,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Sample: {self.sample_id}",
            "",
            "--- Known vs Unknown Statistics ---",
            "  Unknown includes explicit unknown, outside, unmapped, absent and invalid targets.",
            f"  Total projection sites: {total_sites}",
            f"  Known sites: {total_known} ({total_known / max(total_sites, 1) * 100:.1f}%)",
            f"  Unknown sites: {total_unknown} ({total_unknown / max(total_sites, 1) * 100:.1f}%)",
            f"  Neurons with unknown regions: {n_with_unk} ({n_with_unk / total_n * 100:.1f}%)",
            f"  Neurons with only known regions: {n_only_known} ({n_only_known / total_n * 100:.1f}%)",
            f"  Neurons with no target entries: {n_no_targets} ({n_no_targets / total_n * 100:.1f}%)",
            "",
            "--- Terminal Region Distribution (Top 30) ---",
            f"  Total unique regions: {len(counts)}",
            f"  Total entries (known only): {int(counts.sum())}",
            "",
        ]
        for i, (region, cnt) in enumerate(counts.items(), 1):
            pct = cnt / max(counts.sum(), 1) * 100
            lines.append(f"    {i:2d}. {region}: {cnt} neurons ({pct:.1f}%)")
        lines += ["", "=" * 60]

        report = "\n".join(lines)
        self.save_report(report, "terminal_report")

    def _save_projection_sites_report(self):
        """Report 3/3: Projection sites count + outlier statistics."""
        df = self.plot_dataframe
        if df.empty or "Terminal_Regions" not in df.columns:
            return

        df_temp = df.copy()
        df_temp["Terminal_Regions"] = df_temp["Terminal_Regions"].apply(
            parse_terminal_regions
        )
        df_temp["_psc"] = df_temp["Terminal_Regions"].apply(
            lambda x: sum(is_known_terminal_target(r) for r in x)
        )
        df_temp["_usc"] = df_temp["Terminal_Regions"].apply(
            lambda x: sum(not is_known_terminal_target(r) for r in x)
        )
        if "Outlier_Count" not in df_temp.columns:
            df_temp["Outlier_Count"] = 0

        total_n = len(df_temp)
        n_wu = int((df_temp["_usc"] > 0).sum())
        n_wo = int((df_temp["Outlier_Count"] > 0).sum())

        lines = [
            "=" * 60,
            "PROJECTION SITES STATISTICS (KNOWN REGIONS ONLY)",
            "=" * 60,
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Sample: {self.sample_id}",
            "",
            f"Total neurons analyzed: {total_n}",
            "",
            "--- Known Projection Sites (Excluding Unknown) ---",
            f"  Total known sites: {int(df_temp['_psc'].sum())}",
            f"  Mean known sites per neuron: {df_temp['_psc'].mean():.2f}",
            f"  Median: {df_temp['_psc'].median():.1f}",
            f"  Min: {int(df_temp['_psc'].min())}",
            f"  Max: {int(df_temp['_psc'].max())}",
            f"  Std: {df_temp['_psc'].std():.2f}",
            "",
            "  Distribution of known sites per neuron:",
        ]
        known_dist = df_temp["_psc"].value_counts().sort_index()
        for sites, count in known_dist.head(15).items():
            lines.append(
                f"    {int(sites)} site(s): {int(count)} neurons "
                f"({count / total_n * 100:.1f}%)"
            )

        lines += [
            "",
            "--- Unknown Projection Sites (Excluded from Plot) ---",
            "  Unknown includes explicit unknown, outside, unmapped, absent and invalid targets.",
            f"  Neurons with unknown sites: {n_wu} ({n_wu / total_n * 100:.1f}%)",
            f"  Total unknown sites excluded: {int(df_temp['_usc'].sum())}",
            f"  Mean unknown sites per neuron: {df_temp['_usc'].mean():.2f}",
            "",
            "--- Outlier Statistics ---",
            f"  Neurons with outliers: {n_wo} ({n_wo / total_n * 100:.1f}%)",
            f"  Total outliers: {int(df_temp['Outlier_Count'].sum())}",
            f"  Mean outliers per neuron: {df_temp['Outlier_Count'].mean():.2f}",
            "",
            "  Distribution of outlier count per neuron:",
        ]
        outlier_dist = df_temp["Outlier_Count"].value_counts().sort_index()
        for outliers, count in outlier_dist.head(10).items():
            lines.append(
                f"    {int(outliers)} outlier(s): {int(count)} neurons "
                f"({count / total_n * 100:.1f}%)"
            )
        lines += ["", "=" * 60]

        report = "\n".join(lines)
        self.save_report(report, "projection_sites_report")

    # ==================================================================
    # GENERATE ALL PLOTS
    # ==================================================================
    def _generate_all_plots(self):
        """Generate and save all standard plots."""
        df = self.plot_dataframe
        if df.empty:
            return

        print("  Generating plots...")

        # 1. Type distribution
        try:
            s = str(self.output.get_plot_path("type_distribution"))
            plot_type_distribution_df(df, save_path=s, show=self.show_plots)
        except Exception as e:
            print(f"  [WARN] type_distribution failed: {e}")

        # 2. Soma distribution
        try:
            s = str(self.output.get_plot_path("soma_distribution"))
            plot_soma_distribution_df(df, save_path=s, show=self.show_plots)
        except Exception as e:
            print(f"  [WARN] soma_distribution failed: {e}")

        # 3. Terminal distribution
        try:
            s = str(self.output.get_plot_path("terminal_distribution"))
            plot_terminal_distribution_df(df, save_path=s, show=self.show_plots)
        except Exception as e:
            print(f"  [WARN] terminal_distribution failed: {e}")

        # 4. Projection sites count
        try:
            s = str(self.output.get_plot_path("projection_sites_count"))
            plot_projection_sites_count_df(df, save_path=s, show=self.show_plots)
        except Exception as e:
            print(f"  [WARN] projection_sites_count failed: {e}")

        # 5. Laterality summary
        if "N_Ipsilateral" in df.columns:
            try:
                s = str(self.output.get_plot_path("laterality_summary"))
                plot_laterality_summary_df(df, save_path=s, show=self.show_plots)
            except Exception as e:
                print(f"  [WARN] laterality_summary failed: {e}")

        # 6. Stacked region distribution (all levels in one plot)
        if "Soma_Region_Hierarchy" in df.columns:
            try:
                s = str(self.output.get_plot_path("region_dist_stacked"))
                plot_region_distribution_stacked(df, save_path=s, show=self.show_plots)
            except Exception as e:
                print(f"  [WARN] region_dist_stacked failed: {e}")

        # 7. Projection by soma
        if len(df["Soma_Region"].unique()) > 1:
            try:
                regions = df["Soma_Region"].unique()
                data = [
                    df.loc[df["Soma_Region"] == r, "Total_Length"] for r in regions
                ]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.boxplot(data, labels=regions)
                ax.set_title("Projection Length by Soma Region")
                plt.xticks(rotation=45)
                plt.tight_layout()
                p = self.output.get_plot_path("projection_by_soma")
                fig.savefig(p, dpi=300, bbox_inches="tight")
                print(f"  [SAVED] {p}")
                if self.show_plots:
                    plt.show()
                plt.close(fig)
            except Exception as e:
                print(f"  [WARN] projection_by_soma failed: {e}")

        print("  All plots generated.")

    # ==================================================================
    # UTILITY METHODS
    # ==================================================================
    def get_available_hierarchy_levels(self) -> list:
        """Return list of hierarchy levels with projection columns available."""
        available = []
        for lv in range(1, 7):
            col_name = f"Region_Projection_Length_L{lv}"
            if col_name in self.plot_dataframe.columns:
                available.append(lv)
        return available

    def print_hierarchy_status(self):
        """Print detailed status of hierarchy columns."""
        print("\n" + "=" * 50)
        print("HIERARCHY COLUMNS STATUS")
        print("=" * 50)
        
        # Soma hierarchy
        if "Soma_Region_Hierarchy" in self.plot_dataframe.columns:
            print("\nSoma_Region_Hierarchy: OK")
            # Check what levels are populated
            sample = self.plot_dataframe["Soma_Region_Hierarchy"].iloc[0]
            if isinstance(sample, dict):
                print(f"  Levels present: {list(sample.keys())}")
        else:
            print("\nSoma_Region_Hierarchy: MISSING")
        
        # Projection length columns
        print("\nProjection Length Columns:")
        for lv in range(1, 7):
            col_name = f"Region_Projection_Length_L{lv}"
            if col_name in self.plot_dataframe.columns:
                # Count non-empty dicts
                non_empty = self.plot_dataframe[col_name].apply(
                    lambda x: len(x) if isinstance(x, dict) else 0
                )
                total_regions = non_empty.sum()
                neurons_with_data = (non_empty > 0).sum()
                print(f"  L{lv}: OK ({neurons_with_data} neurons, {total_regions} total regions)")
            else:
                print(f"  L{lv}: MISSING")
        
        # Raw projection length
        if "Region_projection_length" in self.plot_dataframe.columns:
            print("\nRegion_projection_length (raw): OK")
        else:
            print("\nRegion_projection_length (raw): MISSING")
        
        print("=" * 50)

    def list_available_neurons(self) -> List[str]:
        """Return list of all available neuron IDs for this sample."""
        return [n["name"] for n in self.neuron_list]

    def get_neuron_count(self) -> dict:
        """
        Return neuron counts.
        
        Returns:
            dict with keys:
                - 'total': Total neurons available in sample
                - 'processed': Successfully processed neurons in current dataframe
                - 'completion_rate': Percentage of processed neurons over total available on server
        """
        total = len(self.neuron_list)
        processed = len(self.plot_dataframe)
        success_rate = (processed / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'processed': processed,
            'completion_rate': round(success_rate, 2)
        }
    
    def get_processed_count(self) -> int:
        """Return number of successfully processed neurons."""
        return len(self.plot_dataframe)
    
    def get_total_count(self) -> int:
        """Return total number of neurons available for this sample."""
        return len(self.neuron_list)
