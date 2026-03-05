"""
classifier.py - Neuron type classification (PT / CT / ITs / ITc / ITi).

Uses ARM atlas index ranges for NMT v2.1.
"""

import pandas as pd


class NeuronClassifier:
    def __init__(self, atlas_table: pd.DataFrame):
        try:
            ids = pd.to_numeric(atlas_table["Index"], errors="coerce")
        except KeyError:
            ids = pd.to_numeric(atlas_table.iloc[:, 0], errors="coerce")
        self.name_to_id = dict(zip(atlas_table["Abbreviation"], ids))
        if "Full_Name" in atlas_table.columns:
            self.name_to_id.update(dict(zip(atlas_table["Full_Name"], ids)))

    def _get_detailed_category(self, region_identifier: str) -> str:
        rid = self.name_to_id.get(region_identifier)
        if rid is None:
            return "Other"
        rid = int(rid)
        if (1169 <= rid <= 1325) or (1669 <= rid <= 1825):
            return "PT_Target"
        if (1083 <= rid <= 1107) or (1583 <= rid <= 1607):
            return "PT_Target"
        if (1111 <= rid <= 1168) or (1611 <= rid <= 1668):
            return "Thalamus"
        if (1051 <= rid <= 1061) or (1551 <= rid <= 1561):
            return "Striatum"
        if 1 <= rid <= 500:
            return "Cortex_L"
        if 501 <= rid <= 1000:
            return "Cortex_R"
        return "Other"

    def classify_single_neuron(self, terminal_list: list, soma_region: str) -> str:
        soma_side = "Unknown"
        if "CL_" in soma_region or "SL_" in soma_region:
            soma_side = "L"
        elif "CR_" in soma_region or "SR_" in soma_region:
            soma_side = "R"
        is_PT = is_CT = has_striatum = has_contra = has_ipsi = False
        for t_name in terminal_list:
            cat = self._get_detailed_category(t_name)
            if cat == "PT_Target":
                is_PT = True
            elif cat == "Thalamus":
                is_CT = True
            elif cat == "Striatum":
                has_striatum = True
            elif cat == "Cortex_L":
                if soma_side == "L":
                    has_ipsi = True
                elif soma_side == "R":
                    has_contra = True
            elif cat == "Cortex_R":
                if soma_side == "R":
                    has_ipsi = True
                elif soma_side == "L":
                    has_contra = True
        if is_PT:
            return "PT"
        if is_CT:
            return "CT"
        if has_striatum:
            return "ITs"
        if has_contra:
            return "ITc"
        if has_ipsi:
            return "ITi"
        return "Unclassified"