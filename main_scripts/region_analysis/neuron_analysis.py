"""
neuron_analysis.py - Per-neuron region & projection length analysis.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from decimal import Decimal, InvalidOperation


class RegionAnalysisPerNeuron:
    def __init__(self, neuron_tracer_obj, atlas_volume: np.ndarray, atlas_table: pd.DataFrame):
        self.neuron = neuron_tracer_obj
        self.atlas = atlas_volume
        self.atlas_table = atlas_table
        if atlas_volume.ndim != 3:
            raise ValueError("Per-neuron analysis requires a selected 3D atlas level")
        if not {"Index", "Abbreviation"}.issubset(atlas_table.columns) or atlas_table.empty:
            raise ValueError("Atlas key requires nonempty Index and Abbreviation columns")
        indices = []
        for raw in atlas_table["Index"]:
            try:
                number = Decimal(str(raw))
            except InvalidOperation as exc:
                raise ValueError("Atlas label indices must be finite nonnegative integers") from exc
            if (isinstance(raw, (bool, np.bool_)) or not number.is_finite()
                    or number != number.to_integral_value() or number < 0):
                raise ValueError("Atlas label indices must be finite nonnegative integers")
            indices.append(int(number))
        if len(indices) != len(set(indices)):
            raise ValueError("Atlas label indices must be unique")
        labels = atlas_table["Abbreviation"].tolist()
        if any(not isinstance(label, str) or not label.strip() or label != label.strip() for label in labels):
            raise ValueError("Atlas abbreviations must be nonblank literal text without surrounding whitespace")
        if len(labels) != len(set(labels)):
            raise ValueError("Atlas abbreviations must be unique to prevent regional length overwrite")
        self.brain_region_map = dict(zip(indices, labels))
        self.mapped_brain_region_lengths: dict = {}
        self.neuron_total_length: float = 0.0
        self.soma_region: str = ""
        self.terminal_regions: list = []

    def run(self):
        all_region_lengths, self.neuron_total_length = (
            self._calculate_neuronal_branch_length()
        )
        self.soma_region, self.terminal_regions = self._soma_and_terminal_region()
        # Retain all branch length in regions containing a legacy graph leaf.
        terminal_region_names = {t["region"] for t in self.terminal_regions}
        self.mapped_brain_region_lengths = {
            region: length
            for region, length in all_region_lengths.items()
            if region in terminal_region_names
        }

    region_analysis = run

    def get_terminal_regions_set(self) -> set:
        """Return set of region names that contain terminals."""
        return {t["region"] for t in self.terminal_regions}

    @staticmethod
    def _distance(p1, p2) -> float:
        return float(
            np.sqrt(
                (p1.x_nii - p2.x_nii) ** 2
                + (p1.y_nii - p2.y_nii) ** 2
                + (p1.z_nii - p2.z_nii) ** 2
            )
        )

    def _calculate_neuronal_branch_length(self):
        brain_region_lengths = defaultdict(float)
        total_length = 0.0
        for branch in self.neuron.branches:
            for i in range(len(branch) - 1):
                cur, nxt = branch[i], branch[i + 1]
                edge = round(self._distance(cur, nxt), 3)
                total_length += edge
                vox = tuple(
                    np.round([cur.x_nii, cur.y_nii, cur.z_nii]).astype(int).flatten()
                )
                if all(0 <= vox[d] < self.atlas.shape[d] for d in range(3)):
                    rid = int(self.atlas[vox])
                    if rid > 0:
                        brain_region_lengths[rid] += edge
        mapped = {
            self.brain_region_map.get(idx, f"Unknown_{idx}"): length
            for idx, length in brain_region_lengths.items()
        }
        return mapped, total_length

    def _soma_and_terminal_region(self):
        soma = self.neuron.root
        soma_pos = tuple(
            np.round([soma.x_nii, soma.y_nii, soma.z_nii]).astype(int).flatten()
        )
        if all(0 <= soma_pos[d] < self.atlas.shape[d] for d in range(3)):
            sid = int(self.atlas[soma_pos])
            soma_region = self.brain_region_map.get(sid, f"Unknown_{sid}")
        else:
            soma_region = "Out_of_Bounds"
        terminal_regions = []
        for node in self.neuron.terminal_nodes:
            pos = tuple(
                np.round([node.x_nii, node.y_nii, node.z_nii]).astype(int).flatten()
            )
            if all(0 <= pos[d] < self.atlas.shape[d] for d in range(3)):
                tid = int(self.atlas[pos])
                t_name = self.brain_region_map.get(tid, f"Unknown_{tid}")
                terminal_regions.append({"region": t_name, "coords": pos})
            else:
                # Preserve missing-label endpoints in counts and QC instead
                # of silently dropping their targets from the neuron table.
                terminal_regions.append({"region": "Out_of_Bounds", "coords": pos})
        return soma_region, terminal_regions
