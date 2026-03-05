"""
laterality_projection_analysis.py - Laterality-based projection analysis.

Generates separate tables for ipsilateral and contralateral projections,
including lengths and strengths at multiple hierarchy levels.

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
sys.path.insert(0, str(SCRIPT_DIR))

from region_analysis.laterality import LateralityParser


# ==============================================================================
# LATERALITY PROJECTION ANALYZER
# ==============================================================================

class LateralityProjectionAnalyzer:
    """
    Analyze projections split by laterality (ipsilateral vs contralateral).
    
    Generates separate tables for each side with:
    - Projection lengths (mm) per region
    - Projection strengths (log10) per region
    - Soma hemisphere information
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with neuron data including laterality columns
        """
        self.df = df.copy()
        self._validate_columns()
        
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
        
        Uses pre-computed Ipsilateral_Projection_Length and 
        Contralateral_Projection_Length columns if available.
        
        Args:
            length_col: Column containing projection length dictionaries (fallback)
            levels: Hierarchy levels to include (default: [3, 4, 5, 6])
            
        Returns:
            Dictionary with 'ipsilateral' and 'contralateral' DataFrames
        """
        if levels is None:
            levels = [3, 4, 5, 6]
        
        # Check for pre-computed laterality columns
        has_precomputed = (
            "Ipsilateral_Projection_Length" in self.df.columns and
            "Contralateral_Projection_Length" in self.df.columns
        )
        
        if has_precomputed:
            print("[LATERALITY ANALYSIS] Using pre-computed ipsi/contra columns")
            return self._analyze_precomputed()
        else:
            print("[LATERALITY ANALYSIS] Using fallback classification")
            return self._analyze_fallback(length_col)
    
    def _analyze_precomputed(self) -> Dict[str, pd.DataFrame]:
        """Analyze using pre-computed Ipsilateral_Projection_Length etc."""
        ipsi_data = []
        contra_data = []
        
        for _, row in self.df.iterrows():
            neuron_id = row["NeuronID"]
            soma_region = row.get("Soma_Region", "")
            soma_side = row.get("Soma_Side", "")
            neuron_type = row.get("Neuron_Type", "")
            
            # Get pre-computed projections
            ipsi_proj = row.get("Ipsilateral_Projection_Length", {})
            contra_proj = row.get("Contralateral_Projection_Length", {})
            
            if isinstance(ipsi_proj, dict) and ipsi_proj:
                ipsi_data.append({
                    "NeuronID": neuron_id,
                    "Neuron_Type": neuron_type,
                    "Soma_Region": soma_region,
                    "Soma_Side": soma_side,
                    "Projections": ipsi_proj,
                    "Total_Length": sum(ipsi_proj.values()),
                    "N_Regions": len(ipsi_proj),
                })
            
            if isinstance(contra_proj, dict) and contra_proj:
                contra_data.append({
                    "NeuronID": neuron_id,
                    "Neuron_Type": neuron_type,
                    "Soma_Region": soma_region,
                    "Soma_Side": soma_side,
                    "Projections": contra_proj,
                    "Total_Length": sum(contra_proj.values()),
                    "N_Regions": len(contra_proj),
                })
        
        print(f"[LATERALITY ANALYSIS] Results:")
        print(f"  Total neurons: {len(self.df)}")
        print(f"  With ipsilateral projections: {len(ipsi_data)}")
        print(f"  With contralateral projections: {len(contra_data)}")
        
        # Create DataFrames
        ipsi_df = pd.DataFrame(ipsi_data) if ipsi_data else pd.DataFrame()
        contra_df = pd.DataFrame(contra_data) if contra_data else pd.DataFrame()
        
        # Expand projection dictionaries to columns
        ipsi_result = self._expand_projections(ipsi_df, "ipsilateral")
        contra_result = self._expand_projections(contra_df, "contralateral")
        
        return {
            "ipsilateral": ipsi_result,
            "contralateral": contra_result,
        }
    
    def _analyze_fallback(self, length_col: str) -> Dict[str, pd.DataFrame]:
        """Fallback: classify projections using LateralityParser."""
        # Determine which length column to use
        if length_col not in self.df.columns:
            for col in ["Region_projection_length", "Region_Projection_Length_finest"]:
                if col in self.df.columns:
                    length_col = col
                    break
        
        if length_col not in self.df.columns:
            print(f"[ERROR] No projection length column found")
            return {}
        
        print(f"[LATERALITY ANALYSIS] Using column: {length_col}")
        print(f"[LATERALITY ANALYSIS] Processing {len(self.df)} neurons")
        
        ipsi_data = []
        contra_data = []
        n_unknown_soma = 0
        n_empty_projections = 0
        
        for idx, row in self.df.iterrows():
            neuron_id = row["NeuronID"]
            soma_region = row.get("Soma_Region", "")
            soma_side = row.get("Soma_Side", "")
            neuron_type = row.get("Neuron_Type", "")
            
            # Debug first few rows
            if idx < 3:
                print(f"  [DEBUG] Neuron {neuron_id}: soma='{soma_region}', side='{soma_side}'")
            
            # Check if soma side is known
            if not soma_side or soma_side == "Unknown":
                n_unknown_soma += 1
                if idx < 3:
                    print(f"    -> Unknown soma side")
                continue
            
            # Get projections
            proj_lengths = row.get(length_col, {})
            if not isinstance(proj_lengths, dict) or not proj_lengths:
                n_empty_projections += 1
                if idx < 3:
                    print(f"    -> No projections data")
                continue
            
            if idx < 3:
                print(f"    -> {len(proj_lengths)} projection regions")
            
            # Get terminal laterality info if available
            term_lat = row.get("Terminal_Laterality", [])
            
            # Split by laterality
            ipsi_regions = {}
            contra_regions = {}
            
            for region, length in proj_lengths.items():
                lat = self._get_region_laterality(region, soma_region, term_lat)
                
                if idx < 3 and len(ipsi_regions) == 0 and len(contra_regions) == 0:
                    print(f"    -> Sample region '{region}': laterality='{lat}'")
                
                if lat == "Ipsilateral":
                    ipsi_regions[region] = length
                elif lat == "Contralateral":
                    contra_regions[region] = length
            
            if idx < 3:
                print(f"    -> Ipsi: {len(ipsi_regions)}, Contra: {len(contra_regions)}")
            
            if ipsi_regions:
                ipsi_data.append({
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
        
        return {
            "ipsilateral": ipsi_result,
            "contralateral": contra_result,
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
        
        Creates columns for:
        - Length per region (with CL_/CR_/SL_/SR_ prefixes removed)
        - Strength (log10) per region
        """
        from region_analysis.laterality import LateralityParser
        
        if df.empty:
            return df
        
        # Get all unique regions across all neurons (cleaned names)
        all_regions_raw = set()
        for proj in df["Projections"]:
            if isinstance(proj, dict):
                all_regions_raw.update(proj.keys())
        
        # Clean region names by removing prefixes
        all_regions_clean = sorted(set(
            LateralityParser.get_base_name(r) for r in all_regions_raw
        ))
        
        # Build result
        result_rows = []
        for _, row in df.iterrows():
            new_row = {
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
                clean_name = LateralityParser.get_base_name(raw_region)
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
            - Ipsilateral_Strength: log10(strength) for ipsilateral regions
            - Contralateral_Length: Projection lengths for contralateral regions
            - Contralateral_Strength: log10(strength) for contralateral regions
            - Summary: Statistics for both sides
        
        Args:
            output_path: Path to output Excel file
            results: Results dict from analyze() (runs if not provided)
            
        Returns:
            Path to saved file
        """
        if results is None:
            results = self.analyze()
        
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
                meta_cols = ["NeuronID", "Neuron_Type", "Soma_Region", "Soma_Side",
                            f"{side}_total_length", f"{side}_n_regions"]
                
                # Length columns: end with '_length' but not '_strength'
                length_cols = [c for c in df.columns 
                              if c.endswith("_length") and not c.endswith("_strength")]
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
            rows.append({
                "Side": side,
                "N_Neurons": len(df),
                "Mean_Regions": df[f"{side}_n_regions"].mean(),
                "Mean_Total_Length": df[f"{side}_total_length"].mean(),
            })
        
        return pd.DataFrame(rows)


# ==============================================================================
# STANDALONE ENTRY POINT
# ==============================================================================

def main():
    """Run laterality projection analysis from command line."""
    parser = argparse.ArgumentParser(
        description="Laterality-based projection analysis"
    )
    parser.add_argument(
        "input",
        help="Input Excel or CSV file with neuron data",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output Excel file path",
        default=None,
    )
    parser.add_argument(
        "--length-col",
        help="Column name for projection lengths",
        default="Region_Projection_Length_finest",
    )
    parser.add_argument(
        "--levels",
        help="Hierarchy levels to include (comma-separated)",
        default="3,4,5,6",
    )
    
    args = parser.parse_args()
    
    # Load data
    if args.input.endswith(".xlsx"):
        df = pd.read_excel(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        print("[ERROR] Input must be .xlsx or .csv")
        sys.exit(1)
    
    print(f"[LOADED] {len(df)} neurons from {args.input}")
    
    # Determine output path
    if args.output is None:
        base = Path(args.input).stem
        args.output = f"{base}_laterality_projections.xlsx"
    
    # Parse levels
    levels = [int(x) for x in args.levels.split(",")]
    
    # Run analysis
    analyzer = LateralityProjectionAnalyzer(df)
    results = analyzer.analyze(length_col=args.length_col, levels=levels)
    
    # Save
    analyzer.save_excel(args.output, results)
    
    print("\n[COMPLETE]")


if __name__ == "__main__":
    main()
