"""
laterality.py - Ipsilateral / contralateral projection parsing.

Prefix logic:
    CL_ → Cortical Left      CR_ → Cortical Right
    SL_ → Subcortical Left    SR_ → Subcortical Right
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd


class LateralityParser:
    LEFT_PREFIXES = ("CL_", "SL_")
    RIGHT_PREFIXES = ("CR_", "SR_")
    ALL_PREFIXES = LEFT_PREFIXES + RIGHT_PREFIXES

    @staticmethod
    def get_side(region_name: str) -> str:
        if pd.isna(region_name):
            return "Unknown"
        s = str(region_name)
        if any(s.startswith(p) for p in LateralityParser.LEFT_PREFIXES):
            return "L"
        if any(s.startswith(p) for p in LateralityParser.RIGHT_PREFIXES):
            return "R"
        return "Unknown"

    @staticmethod
    def get_base_name(region_name: str) -> str:
        if pd.isna(region_name):
            return str(region_name)
        s = str(region_name)
        for prefix in LateralityParser.ALL_PREFIXES:
            if s.startswith(prefix):
                return s[len(prefix):]
        return s

    @staticmethod
    def get_region_type(region_name: str) -> str:
        if pd.isna(region_name):
            return "Unknown"
        s = str(region_name)
        if s.startswith("CL_") or s.startswith("CR_"):
            return "Cortical"
        if s.startswith("SL_") or s.startswith("SR_"):
            return "Subcortical"
        return "Unknown"

    @staticmethod
    def classify(soma_region: str, terminal_region: str) -> str:
        soma_side = LateralityParser.get_side(soma_region)
        term_side = LateralityParser.get_side(terminal_region)
        if soma_side == "Unknown" or term_side == "Unknown":
            return "Unknown"
        return "Ipsilateral" if soma_side == term_side else "Contralateral"

    @staticmethod
    def parse_terminal_list(
        soma_region: str, terminal_regions: list
    ) -> List[Dict[str, str]]:
        if not isinstance(terminal_regions, (list, tuple)):
            terminal_regions = []
        results = []
        for r in terminal_regions:
            results.append(
                {
                    "region": r,
                    "side": LateralityParser.get_side(r),
                    "laterality": LateralityParser.classify(soma_region, r),
                }
            )
        return results

    @staticmethod
    def split_projection_lengths(
        soma_region: str, projection_dict: dict
    ) -> Tuple[dict, dict, dict]:
        if not isinstance(projection_dict, dict):
            return {}, {}, {}
        ipsi, contra, unk = {}, {}, {}
        for region, length in projection_dict.items():
            lat = LateralityParser.classify(soma_region, region)
            if lat == "Ipsilateral":
                ipsi[region] = length
            elif lat == "Contralateral":
                contra[region] = length
            else:
                unk[region] = length
        return ipsi, contra, unk


def add_laterality_columns(
    df: pd.DataFrame,
    soma_col: str = "Soma_Region",
    terminal_col: str = "Terminal_Regions",
    length_col: str = None,
) -> pd.DataFrame:
    df = df.copy()
    P = LateralityParser

    if length_col is None:
        for candidate in ("Region_Projection_Length_finest", "Region_projection_length"):
            if candidate in df.columns:
                length_col = candidate
                break

    df["Soma_Side"] = df[soma_col].apply(P.get_side)

    df["Terminal_Laterality"] = df.apply(
        lambda row: P.parse_terminal_list(row[soma_col], row[terminal_col]),
        axis=1,
    )

    df["N_Ipsilateral"] = df["Terminal_Laterality"].apply(
        lambda x: sum(1 for t in x if t["laterality"] == "Ipsilateral")
    )
    df["N_Contralateral"] = df["Terminal_Laterality"].apply(
        lambda x: sum(1 for t in x if t["laterality"] == "Contralateral")
    )
    df["N_Laterality_Unknown"] = df["Terminal_Laterality"].apply(
        lambda x: sum(1 for t in x if t["laterality"] == "Unknown")
    )

    df["Ipsilateral_Regions"] = df["Terminal_Laterality"].apply(
        lambda x: list(
            dict.fromkeys(t["region"] for t in x if t["laterality"] == "Ipsilateral")
        )
    )
    df["Contralateral_Regions"] = df["Terminal_Laterality"].apply(
        lambda x: list(
            dict.fromkeys(t["region"] for t in x if t["laterality"] == "Contralateral")
        )
    )

    if length_col and length_col in df.columns:
        splits = df.apply(
            lambda row: P.split_projection_lengths(row[soma_col], row[length_col]),
            axis=1,
        )
        df["Ipsilateral_Projection_Length"] = splits.apply(lambda x: x[0])
        df["Contralateral_Projection_Length"] = splits.apply(lambda x: x[1])
        df["Unknown_Laterality_Projection_Length"] = splits.apply(lambda x: x[2])

        df["Total_Ipsilateral_Length"] = df["Ipsilateral_Projection_Length"].apply(
            lambda d: sum(d.values()) if d else 0.0
        )
        df["Total_Contralateral_Length"] = df["Contralateral_Projection_Length"].apply(
            lambda d: sum(d.values()) if d else 0.0
        )
        df["Total_Unknown_Laterality_Length"] = df[
            "Unknown_Laterality_Projection_Length"
        ].apply(lambda d: sum(d.values()) if d else 0.0)

        def _lat_idx(row):
            denom = row["Total_Ipsilateral_Length"] + row["Total_Contralateral_Length"]
            if denom == 0:
                return None
            return round(row["Total_Contralateral_Length"] / denom, 4)

        df["Laterality_Index"] = df.apply(_lat_idx, axis=1)
    else:
        print(
            "[LATERALITY] No projection-length column — length columns skipped."
        )

    n = len(df)
    n_ipsi = (df["N_Ipsilateral"] > 0).sum()
    n_contra = (df["N_Contralateral"] > 0).sum()
    print(
        f"[LATERALITY] {n} neurons: "
        f"{n_ipsi} with ipsi targets, {n_contra} with contra targets"
    )
    return df