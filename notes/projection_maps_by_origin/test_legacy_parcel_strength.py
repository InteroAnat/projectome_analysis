"""Regressions for the legacy definition and animal-balanced aggregation."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
from map_legacy_strength_to_arm_parcels import retained_lengths, summarize_by_animal


class LegacyStrengthRegression(unittest.TestCase):
    def test_all_compartment_gate_and_proximal_assignment_are_preserved(self):
        atlas = np.ones((4, 1, 1, 6), dtype=int)
        atlas[2:, ...] = 2
        rows = [(1, 1, 0, 0, 0, 1, -1), (2, 3, 1, 0, 0, 1, 1),
                (3, 3, 2, 0, 0, 1, 2), (4, 3, 3, 0, 0, 1, 3)]
        lengths, qc = retained_lengths(rows, atlas, [1, 1, 1])
        # Dendritic leaf activates label 2; proximal lengths in label 1 are gated out.
        self.assertEqual(lengths, [{2: 1.0}] * 6)
        self.assertEqual(qc['AllCompartmentLengthVoxel'], 3)

    def test_leaf_gate_is_recalculated_at_each_actual_level(self):
        atlas = np.ones((3, 1, 1, 6), dtype=int)
        atlas[2, 0, 0, 5] = 2
        rows = [(1, 1, 0, 0, 0, 1, -1), (2, 2, 1, 0, 0, 1, 1), (3, 2, 2, 0, 0, 1, 2)]
        lengths, _ = retained_lengths(rows, atlas, [1, 1, 1])
        self.assertEqual(lengths[:5], [{1: 2.0}] * 5)
        self.assertEqual(lengths[5], {})

    def test_rounding_and_outside_leaf_do_not_invent_a_target(self):
        atlas = np.ones((2, 1, 1, 6), dtype=int)
        rows = [(1, 1, 0, 0, 0, 1, -1), (2, 2, 2.123456, 0, 0, 1, 1)]
        lengths, qc = retained_lengths(rows, atlas, [1, 1, 1])
        self.assertEqual(lengths, [{}] * 6)
        self.assertEqual(qc['AllCompartmentLengthVoxel'], 2.123)
        self.assertEqual(qc['OutsideReferenceLeafCount'], 1)

    def test_equal_animal_means_include_selected_zero_not_absent_animals(self):
        frame = pd.DataFrame({'SourceARMIndex':['42'] * 4, 'AnimalID':['A', 'A', 'A', 'B'],
            'Level':[6] * 4, 'TargetID':['target'] * 4, 'NeuronUID':['1','2','3','4'],
            'RetainedLengthVoxel':[0., 0., 0., 99.], 'LegacyStrength':[0., 0., 0., 2.]})
        animal, group = summarize_by_animal(frame)
        self.assertEqual(len(animal), 2)
        self.assertEqual(group.MeanLegacyStrength.iloc[0], 1)
        self.assertEqual(group.MeanRetainedLengthVoxel.iloc[0], 49.5)
        self.assertEqual(group.ContributingAnimalCount.iloc[0], 2)


if __name__ == '__main__':
    unittest.main()
