"""Focused regressions for human/atlas conflict retention and soma counts."""
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
from map_projections_by_soma_origin import evidence_categories, soma_count_map, source_title


class OriginEvidenceRegression(unittest.TestCase):
    def test_human_evidence_survives_neighbor_and_unknown_assignment(self):
        data = pd.DataFrame({'ARMIndex': ['549', '0', '42'], 'ARMFullName': ['CR_precentral_operular_area', 'Atlas background', 'CL_lateral_agranular_insula_area'], 'Henry_coarse_INS_visual_evidence': ['True', 'True', 'True']})
        self.assertEqual(list(evidence_categories(data)), ['HenryPrCOConflict', 'HenryUnassigned', 'atlasAndHenry'])

    def test_unknown_without_human_evidence_stays_candidate(self):
        data = pd.DataFrame({'ARMIndex': ['0', '542'], 'ARMFullName': ['Atlas background', 'CR_lateral_agranular_insula_area'], 'Henry_coarse_INS_visual_evidence': ['False', 'False']})
        self.assertEqual(list(evidence_categories(data)), ['unassignedCandidate', 'atlasOnly'])

    def test_new_human_conflict_requires_explicit_review(self):
        data = pd.DataFrame({'ARMIndex': ['99'], 'ARMFullName': ['unreviewed_parcel'], 'Henry_coarse_INS_visual_evidence': ['True']})
        with self.assertRaisesRegex(ValueError, 'unexpected parcel'):
            evidence_categories(data)

    def test_coincident_somata_count_neurons_not_occupied_voxels(self):
        counts = soma_count_map([[1, 1, 1], [1, 1, 1], [0, 0, 0]], (2, 2, 2))
        self.assertEqual(counts[1, 1, 1], 2)
        self.assertEqual(int(counts.sum()), 3)

    def test_no_silent_rounding_or_negative_index_wrap(self):
        for voxels in ([[.5, 1, 1]], [[-1, 1, 1]], [[2, 1, 1]]):
            with self.assertRaises(ValueError):
                soma_count_map(voxels, (2, 2, 2))

    def test_title_states_assignment_and_source_direction(self):
        title = source_title('CL_granular_insula', 229, 'L')
        self.assertIn('somata assigned to', title)
        self.assertIn('CL granular insula (Left; ARM level 6 index 229)', title)


if __name__ == '__main__':
    unittest.main()
