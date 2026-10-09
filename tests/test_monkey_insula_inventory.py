"""Regression contracts for legacy claims, exact identities and missing evidence."""
import unittest
import pandas as pd

from group_analysis.data_progress.insula_inventory import build_monkey_view


class MonkeyInventoryTests(unittest.TestCase):
    def inputs(self):
        overview = pd.DataFrame([dict(monkey_id="797", fmost_id="252790",
            overview_injection_sites="vAIC L", overview_analysis_target_n=92,
            overview_data_status="Pending", overview_source_cells="Monkey Data!B9:F9")])
        legacy = pd.DataFrame([dict(monkey_id="797", fmost_id="252790",
            injection_sites="vAIC L", ion_n_traced=0, insula_corrected_n=0,
            insula_in_combined_n=0, insula_distribution_combined="—",
            step1_auto_insula_n=0, five_um_local="No")])
        samples = pd.DataFrame([dict(sample="252790", n_listed_reconstructions="3",
            n_atlas_INS="1", n_potential_INS_candidate_lower_bound="2",
            potential_count_status="complete_in_snapshot", copied5um_status="not_copied_at_scoped_root",
            copied5um_evidence_scope="checked local root only", copied5um_slice_evidence_date="")])
        neurons = pd.DataFrame([dict(sample="252790", uid=f"252790::{i}",
            excluded="False", atlas_INS=str(i == 1), potential_INS_candidate=str(i <= 2)) for i in range(1, 4)])
        selected = pd.DataFrame([dict(sample="252790", uid="252790::1", registry_animal="797",
            Henry_coarse_INS_visual_evidence="False", coarse_review_state="candidate-only")])
        return [overview, legacy, samples, neurons, selected, {"252790": None}]

    def test_stale_zero_claim_does_not_hide_current_neurons(self):
        output, differences = build_monkey_view(*self.inputs())
        self.assertEqual(output.iloc[0].legacy_reconstruction_claim_n, 0)
        self.assertEqual(output.iloc[0].snapshot_listed_reconstructions_n, 3)
        self.assertIn("listed_reconstructions", set(differences.field))

    def test_missing_step1_workbook_is_not_observed_zero(self):
        output, _ = build_monkey_view(*self.inputs())
        self.assertTrue(pd.isna(output.iloc[0].legacy_step1_atlas_INS_n))
        self.assertEqual(output.iloc[0].legacy_step1_status, "unavailable_workbook_not_zero")

    def test_spatial_candidate_overlap_is_explicit(self):
        output, _ = build_monkey_view(*self.inputs())
        self.assertEqual(output.iloc[0].snapshot_atlas_INS_n, 1)
        self.assertEqual(output.iloc[0].spatial_candidate_lower_bound_n, 2)
        self.assertEqual(output.iloc[0].spatial_candidate_not_atlas_INS_n, 1)

    def test_absent_manual_evidence_is_missing(self):
        output, _ = build_monkey_view(*self.inputs())
        self.assertTrue(pd.isna(output.iloc[0].henry_visual_INS_in_selected_n))
        self.assertEqual(output.iloc[0].dataset_availability_elsewhere, "unknown")
        self.assertFalse(output.iloc[0].anatomical_acceptance)

    def test_exact_sample_conflict_is_not_an_alias_merge(self):
        inputs = self.inputs()
        inputs[1].loc[0, "fmost_id"] = "252985"
        with self.assertRaisesRegex(ValueError, "Conflicting fMOST identity"):
            build_monkey_view(*inputs)

    def test_duplicate_identity_cannot_inflate_counts(self):
        inputs = self.inputs()
        inputs[3] = pd.concat([inputs[3], inputs[3].iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate exact identity"):
            build_monkey_view(*inputs)

    def test_snapshot_total_requires_neuron_level_agreement(self):
        inputs = self.inputs()
        inputs[2].loc[0, "n_atlas_INS"] = "2"
        with self.assertRaisesRegex(ValueError, "snapshot count disagrees"):
            build_monkey_view(*inputs)

    def test_selected_monkey_conflict_is_rejected(self):
        inputs = self.inputs()
        inputs[4].loc[0, "registry_animal"] = "948"
        with self.assertRaisesRegex(ValueError, "animal identity disagrees"):
            build_monkey_view(*inputs)


if __name__ == "__main__":
    unittest.main()
