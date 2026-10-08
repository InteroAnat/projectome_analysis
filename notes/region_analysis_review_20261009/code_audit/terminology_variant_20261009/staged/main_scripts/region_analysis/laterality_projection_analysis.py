"""
laterality_projection_analysis.py - Laterality-based projection analysis.

Generates separate tables for ipsilateral and contralateral projections,
including retained reconstruction lengths in the source unit and display
values log10(length + 1). Hierarchy exports use PopulationRegionAnalysis.
Legacy endpoint labels come from leaves of all SWC compartments; they do
not verify biological terminal sites or boutons.

Can be run standalone or as part of the analysis pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse

import numpy as np
import pandas as pd

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from region_analysis.laterality import LateralityParser
from region_analysis.utils import (load_processed_df, normalize_region_dataframe,
    projection_column_names, neuron_metadata, terminal_target_status)


# ==============================================================================
# LATERALITY PROJECTION ANALYZER
# ==============================================================================

class LateralityProjectionAnalyzer:
    """
    Analyze projections split by laterality (ipsilateral vs contralateral).
    
    Generates separate tables for each side with:
    - Retained reconstruction lengths in the declared input unit
    - Display values log10(retained regional length + 1), source-unit dependent
    - Soma hemisphere information
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with neuron data including laterality columns
        """
        self.df = normalize_region_dataframe(df)
        self._validate_columns()
        if "SampleID" not in self.df and "NeuronUID" not in self.df:
            raise ValueError("Standalone analysis requires NeuronUID or SampleID plus NeuronID")
        if self.df.NeuronID.isna().any() or not self.df.NeuronID.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("NeuronID must be present literal text")
        if "NeuronUID" in self.df:
            if not self.df.NeuronUID.map(lambda v: isinstance(v, str) and "::" in v and bool(v.split("::", 1)[0].strip())).all():
                raise ValueError("NeuronUID must encode SampleID::NeuronID")
            uid_samples = self.df.NeuronUID.str.split("::", n=1).str[0]
            uid_neurons = self.df.NeuronUID.str.split("::", n=1).str[1]
            if not uid_neurons.eq(self.df.NeuronID).all():
                raise ValueError("NeuronUID disagrees with NeuronID")
            if "SampleID" not in self.df:
                self.df["SampleID"] = uid_samples
        if self.df.SampleID.isna().any():
            raise ValueError("SampleID must be present")
        self.df["SampleID"] = self.df.SampleID.map(
            lambda v: str(int(v)) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) and float(v).is_integer() else str(v).strip())
        if self.df.SampleID.eq("").any() or self.df.SampleID.str.contains("::", regex=False).any():
            raise ValueError("SampleID must be present")
        expected = self.df.SampleID + "::" + self.df.NeuronID
        if "NeuronUID" in self.df and not self.df.NeuronUID.eq(expected).all():
            raise ValueError("NeuronUID disagrees with SampleID/NeuronID")
        self.df["NeuronUID"] = expected
        neuron_metadata(self.df)  # Shared missing/duplicate exact-identity guard.
        if "Length_Unit" not in self.df:
            self.df["Length_Unit"] = "unspecified source units"
        else:
            self.df["Length_Unit"] = self.df.Length_Unit.fillna("unspecified source units")
            if not self.df.Length_Unit.map(lambda value: isinstance(value, str)).all():
                raise ValueError("Length_Unit must be literal text")
            self.df["Length_Unit"] = self.df.Length_Unit.str.strip().replace("", "unspecified source units")
        if self.df.Length_Unit.nunique() > 1:
            raise ValueError("Cannot pool projection lengths with mixed source units")
        self.unknown_laterality = pd.DataFrame()
        if "Soma_Side" not in self.df:
            self.df["Soma_Side"] = self.df.Soma_Region.apply(LateralityParser.get_side)
        
    def _validate_columns(self):
        """Check required columns exist."""
        required = ["NeuronID", "Soma_Region"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for laterality columns
        has_laterality = "Soma_Side" in self.df.columns
        has_terminal_lat = "Terminal_Laterality" in self.df.columns
        
        print(f"[LATERALITY ANALYSIS] Available columns:")
        print(f"  Soma_Side: {has_laterality}")
        print(f"  Terminal_Laterality: {has_terminal_lat}")
        print(f"  Projection columns: {[c for c in self.df.columns if 'projection' in c.lower()]}")
        
        if not has_laterality:
            print("[WARN] No Soma_Side column found. Run add_laterality_columns first.")
    
    def analyze(
        self,
        length_col: str = "Region_Projection_Length_finest",
        levels: List[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate ipsilateral and contralateral projection tables.
        
        Classify absolute target labels using the resolved Soma_Side. If only
        precomputed dictionaries exist, recover their nonoverlapping absolute
        labels and reclassify them; stale ipsi/contra placement is not trusted.
        Unknown-side/target lengths are returned separately, never imputed zero.
        
        Args:
            length_col: Column containing projection length dictionaries (fallback)
            levels: Unsupported here; use the population hierarchy workflow.
            
        Returns:
            Ipsi/contra tables plus exact-identity unknown_laterality records.
            Side tables are conditional on available side and target labels;
            they do not turn excluded/unresolved observations into zeros.
        """
        if levels:
            raise ValueError("Standalone hierarchy export is unavailable; use PopulationRegionAnalysis with explicit hierarchy inputs")
        
        # Check for pre-computed laterality columns
        has_precomputed = (
            "Ipsilateral_Projection_Length" in self.df.columns and
            "Contralateral_Projection_Length" in self.df.columns
        )
        
        has_absolute = any(column in self.df for column in (length_col, "Region_projection_length", "Region_Projection_Length_finest"))
        if has_precomputed and not has_absolute:
            return self._analyze_precomputed()
        else:
            print("[LATERALITY ANALYSIS] Using fallback classification")
            return self._analyze_fallback(length_col)
    
    def _analyze_precomputed(self) -> Dict[str, pd.DataFrame]:
        """Recover only absolute, nonoverlapping target maps and reclassify."""
        recovered = []
        columns = ["Ipsilateral_Projection_Length", "Contralateral_Projection_Length"]
        if "Unknown_Laterality_Projection_Length" in self.df:
            columns.append("Unknown_Laterality_Projection_Length")
        for _, row in self.df.iterrows():
            union = {}
            for column in columns:
                for target, length in row[column].items():
                    if target in union:
                        raise ValueError("Precomputed laterality dictionaries overlap in absolute targets")
                    if not target.startswith(("CL_", "CR_", "SL_", "SR_")) and terminal_target_status(target) == "known":
                        raise ValueError("Precomputed lengths require absolute prefixed anatomical targets")
                    union[target] = length
            recovered.append(union)
        self.df["Region_projection_length"] = recovered
        return self._analyze_fallback("Region_projection_length")

    def _analyze_fallback(self, length_col: str) -> Dict[str, pd.DataFrame]:
        """Fallback: classify projections using LateralityParser."""
        # Determine which length column to use
        if length_col not in self.df.columns:
            for col in ["Region_projection_length", "Region_Projection_Length_finest"]:
                if col in self.df.columns:
                    length_col = col
                    break
        
        if length_col not in self.df.columns:
            raise ValueError("No recoverable absolute projection-length source found")
        
        print(f"[LATERALITY ANALYSIS] Using column: {length_col}")
        print(f"[LATERALITY ANALYSIS] Processing {len(self.df)} neurons")
        
        ipsi_data = []
        contra_data = []
        n_unknown_soma = 0
        n_empty_projections = 0
        unknown_records = []
        
        for position, (_, row) in enumerate(self.df.iterrows()):
            neuron_id = row["NeuronID"]
            soma_region = row.get("Soma_Region", "")
            soma_side = row.get("Soma_Side", "")
            soma_side = soma_side if isinstance(soma_side, str) and soma_side in ("L", "R") else "Unknown"
            neuron_type = row.get("Neuron_Type", "")
            identity = {key: row[key] for key in ("SampleID", "NeuronUID", "Length_Unit")}
            proj_lengths = row[length_col]
            for region, length in proj_lengths.items():
                if terminal_target_status(region) != "known" or LateralityParser.classify_with_soma_side(soma_side, region) == "Unknown":
                    unknown_records.append({**identity, "NeuronID": neuron_id,
                        "Neuron_Type": neuron_type, "Soma_Region": soma_region,
                        "Soma_Side": soma_side, "Target_Region": region, "Length": length})
            
            # Debug first few rows
            if position < 3:
                print(f"  [DEBUG] Neuron {neuron_id}: soma='{soma_region}', side='{soma_side}'")
            
            # Check if soma side is known
            if soma_side not in ("L", "R"):
                n_unknown_soma += 1
                if position < 3:
                    print(f"    -> Unknown soma side")
                continue
            
            # Get projections
            proj_lengths = row.get(length_col, {})
            if not isinstance(proj_lengths, dict) or not proj_lengths:
                n_empty_projections += 1
                if position < 3:
                    print(f"    -> No projections data")
                continue
            
            if position < 3:
                print(f"    -> {len(proj_lengths)} projection regions")
            
            # Get terminal laterality info if available
            term_lat = row.get("Terminal_Laterality", [])
            
            # Split by laterality
            ipsi_regions = {}
            contra_regions = {}
            
            for region, length in proj_lengths.items():
                lat = (LateralityParser.classify_with_soma_side(soma_side, region)
                       if terminal_target_status(region) == "known" else "Unknown")
                
                if position < 3 and len(ipsi_regions) == 0 and len(contra_regions) == 0:
                    print(f"    -> Sample region '{region}': laterality='{lat}'")
                
                if lat == "Ipsilateral":
                    ipsi_regions[region] = length
                elif lat == "Contralateral":
                    contra_regions[region] = length
            
            if position < 3:
                print(f"    -> Ipsi: {len(ipsi_regions)}, Contra: {len(contra_regions)}")
            
            if ipsi_regions:
                ipsi_data.append({
                    **identity,
                    "NeuronID": neuron_id,
                    "Neuron_Type": neuron_type,
                    "Soma_Region": soma_region,
                    "Soma_Side": soma_side,
                    "Projections": ipsi_regions,
                    "Total_Length": sum(ipsi_regions.values()),
                    "N_Regions": len(ipsi_regions),
                })
            
            if contra_regions:
                contra_data.append({
                    **identity,
                    "NeuronID": neuron_id,
                    "Neuron_Type": neuron_type,
                    "Soma_Region": soma_region,
                    "Soma_Side": soma_side,
                    "Projections": contra_regions,
                    "Total_Length": sum(contra_regions.values()),
                    "N_Regions": len(contra_regions),
                })
        
        print(f"[LATERALITY ANALYSIS] Results:")
        print(f"  Total neurons: {len(self.df)}")
        print(f"  Unknown soma side: {n_unknown_soma}")
        print(f"  Empty projections: {n_empty_projections}")
        print(f"  With ipsilateral projections: {len(ipsi_data)}")
        print(f"  With contralateral projections: {len(contra_data)}")
        
        ipsi_df = pd.DataFrame(ipsi_data) if ipsi_data else pd.DataFrame()
        contra_df = pd.DataFrame(contra_data) if contra_data else pd.DataFrame()
        
        ipsi_result = self._expand_projections(ipsi_df, "ipsilateral")
        contra_result = self._expand_projections(contra_df, "contralateral")
        self.unknown_laterality = pd.DataFrame(unknown_records)
        
        return {
            "ipsilateral": ipsi_result,
            "contralateral": contra_result,
            "unknown_laterality": self.unknown_laterality.copy(),
        }
    
    def _get_region_laterality(
        self,
        region: str,
        soma_region: str,
        term_lat: List[Dict],
    ) -> str:
        """
        Determine if region is ipsilateral or contralateral.
        
        First checks pre-computed Terminal_Laterality, then falls back
        to parsing region name.
        
        Returns: "Ipsilateral", "Contralateral", or "Unknown"
        """
        # Try pre-computed laterality
        if isinstance(term_lat, list):
            for item in term_lat:
                if item.get("region") == region:
                    lat = item.get("laterality", "Unknown")
                    # Normalize to capitalized format
                    if lat and lat.lower() == "ipsilateral":
                        return "Ipsilateral"
                    elif lat and lat.lower() == "contralateral":
                        return "Contralateral"
                    return "Unknown"
        
        # Fallback: parse from region name
        return LateralityParser.classify(soma_region, region)
    
    def _expand_projections(
        self,
        df: pd.DataFrame,
        side: str,
    ) -> pd.DataFrame:
        """
        Expand projection dictionaries to separate columns.
        
        Preserve exact neuron metadata and source units; remove hemisphere
        prefixes only through the shared collision-safe anatomical namespace.
        Display strength is log10(final aggregated regional length + 1).
        """
        from region_analysis.laterality import LateralityParser
        
        if df.empty:
            return df
        
        # Get all unique regions across all neurons (cleaned names)
        all_regions_raw = set()
        for proj in df["Projections"]:
            if isinstance(proj, dict):
                all_regions_raw.update(proj.keys())
        
        # Use all input targets so homonym names agree across the two sides.
        input_regions = set(all_regions_raw)
        for column in ("Region_projection_length", "Region_Projection_Length_finest",
                       "Ipsilateral_Projection_Length", "Contralateral_Projection_Length"):
            if column in self.df:
                for mapping in self.df[column]:
                    input_regions.update(mapping)
        names = projection_column_names(input_regions)
        all_regions_clean = sorted({names[region] for region in all_regions_raw})
        
        # Build result
        result_rows = []
        for _, row in df.iterrows():
            new_row = {
                **{key: row[key] for key in ("SampleID", "NeuronUID", "Length_Unit")},
                "NeuronID": row["NeuronID"],
                "Neuron_Type": row.get("Neuron_Type", ""),
                "Soma_Region": row["Soma_Region"],
                "Soma_Side": row.get("Soma_Side", ""),
                f"{side}_total_length": row.get("Total_Length", 0),
                f"{side}_n_regions": row.get("N_Regions", 0),
            }
            
            projections = row.get("Projections", {})
            
            # Add length and strength for each cleaned region name
            # Sum lengths if multiple raw regions map to same cleaned name
            clean_to_length = {}
            for raw_region, length in projections.items():
                clean_name = names[raw_region]
                clean_to_length[clean_name] = clean_to_length.get(clean_name, 0) + length
            
            for region in all_regions_clean:
                length = clean_to_length.get(region, 0)
                strength = np.log10(length + 1) if length > 0 else 0
                
                new_row[f"{region}_length"] = length
                new_row[f"{region}_strength"] = round(strength, 4)
            
            result_rows.append(new_row)
        
        return pd.DataFrame(result_rows)
    
    def save_excel(
        self,
        output_path: str,
        results: Dict[str, pd.DataFrame] = None,
    ) -> str:
        """
        Save results to Excel with separate sheets for lengths and strengths.
        
        Sheets:
            - Ipsilateral_Length: Projection lengths for ipsilateral regions
            - Ipsilateral_Strength: log10(length + 1) for ipsilateral regions
            - Contralateral_Length: Projection lengths for contralateral regions
            - Contralateral_Strength: log10(length + 1) for contralateral regions
            - Unknown_Laterality: Unresolved-side/target lengths with exact identities
            - Summary: Descriptive statistics in unchanged source units
        
        Args:
            output_path: Path to output Excel file
            results: Results dict from analyze() (runs if not provided)
            
        Returns:
            Path to saved file
        """
        if results is None:
            results = self.analyze()
        if Path(output_path).exists():
            raise FileExistsError("Use a fresh laterality output file; existing tables are protected")
        
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Split into length and strength DataFrames
            for side in ["ipsilateral", "contralateral"]:
                df = results.get(side)
                if df is None or df.empty:
                    # Empty sheets
                    pd.DataFrame({"Note": [f"No {side} projections found"]}).to_excel(
                        writer, sheet_name=f"{side.title()}_Length", index=False
                    )
                    pd.DataFrame({"Note": [f"No {side} projections found"]}).to_excel(
                        writer, sheet_name=f"{side.title()}_Strength", index=False
                    )
                    continue
                
                # Separate length and strength columns
                meta_cols = ["NeuronUID", "SampleID", "NeuronID", "Neuron_Type", "Length_Unit", "Soma_Region", "Soma_Side",
                            f"{side}_total_length", f"{side}_n_regions"]
                
                # Length columns: end with '_length' but not '_strength'
                length_cols = [c for c in df.columns 
                              if c.endswith("_length") and c not in meta_cols]
                # Strength columns: end with '_strength'
                strength_cols = [c for c in df.columns if c.endswith("_strength")]
                
                # Create length DataFrame (metadata + length columns)
                length_df = df[meta_cols + length_cols].copy()
                
                # Create strength DataFrame (metadata + strength columns)
                # Rename strength columns to remove '_strength' suffix for clarity
                strength_df = df[meta_cols + strength_cols].copy()
                
                # Write sheets
                length_df.to_excel(
                    writer, sheet_name=f"{side.title()}_Length", index=False
                )
                print(f"  [Sheet] {side.title()}_Length: {len(length_df)} neurons x {len(length_cols)} regions")
                
                strength_df.to_excel(
                    writer, sheet_name=f"{side.title()}_Strength", index=False
                )
                print(f"  [Sheet] {side.title()}_Strength: {len(strength_df)} neurons x {len(strength_cols)} regions")
            
            # Summary sheet
            summary = self._create_summary(results)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            unknown = results.get("unknown_laterality", self.unknown_laterality)
            if unknown is not None and not unknown.empty:
                unknown.to_excel(writer, sheet_name="Unknown_Laterality", index=False)
        
        print(f"\n[SAVED] {output_path}")
        return output_path
    
    def _create_summary(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create summary statistics."""
        rows = []
        
        for side in ["ipsilateral", "contralateral"]:
            df = results.get(side)
            if df is None or df.empty:
                rows.append({
                    "Side": side,
                    "N_Neurons": 0,
                    "Mean_Regions": 0,
                    "Mean_Total_Length": 0,
                })
                continue
            
            side_key = side[:4]  # "ipsi" or "contra"
            units = df.Length_Unit.dropna().unique()
            if len(units) != 1:
                raise ValueError("Cannot pool projection lengths with mixed source units")
            rows.append({
                "Side": side,
                "N_Neurons": len(df),
                "Mean_Regions": df[f"{side}_n_regions"].mean(),
                "Mean_Total_Length": df[f"{side}_total_length"].mean(),
                "Length_Unit": units[0],
            })
        
        return pd.DataFrame(rows)


# ==============================================================================
# STANDALONE ENTRY POINT
# ==============================================================================

def main():
    """Run laterality projection analysis from command line."""
    parser = argparse.ArgumentParser(
        description="Split retained reconstruction lengths by soma-relative side; display values are log10(length + 1) in the declared source unit"
    )
    parser.add_argument(
        "input",
        help="Excel/CSV with exact neuron identities (NeuronUID or SampleID+NeuronID), soma labels and absolute target-length dictionaries",
    )
    parser.add_argument(
        "-o", "--output",
        help="Fresh output Excel path; existing files are protected",
        default=None,
    )
    parser.add_argument(
        "--length-col",
        help="Retained reconstruction-length dictionary column; source units are preserved and unresolved-side lengths are reported separately",
        default="Region_Projection_Length_finest",
    )
    parser.add_argument(
        "--levels",
        help="Reserved hierarchy request; use PopulationRegionAnalysis for hierarchy exports",
        default="",
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_processed_df(args.input)
    
    print(f"[LOADED] {len(df)} neurons from {args.input}")
    
    # Determine output path
    if args.output is None:
        base = Path(args.input).stem
        args.output = f"{base}_laterality_projections.xlsx"
    
    # Parse levels
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    
    # Run analysis
    analyzer = LateralityProjectionAnalyzer(df)
    results = analyzer.analyze(length_col=args.length_col, levels=levels)
    
    # Save
    analyzer.save_excel(args.output, results)
    
    print("\n[COMPLETE]")


if __name__ == "__main__":
    main()
