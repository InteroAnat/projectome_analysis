"""Scientific screen and snapshot provenance contracts; no network/image access."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_analysis/scripts"))
import audit_insula_inventory as inventory


BOXES = [("IAL", 10.0, 12.0, 40.0, 42.0, 20.0, 22.0)]


class ScreenTests(unittest.TestCase):
    def test_fold_both_sides_and_phys_mm_padding(self):
        left = inventory.spatial_screen((21000, 41000, 21000), BOXES)
        right = inventory.spatial_screen((43000, 41000, 21000), BOXES)
        self.assertEqual(left["bbox_hits"], "IAL")
        self.assertEqual(right["bbox_hits"], "IAL")
        self.assertEqual(left["folded_x_mm"], 11)
        self.assertEqual(left["coordinate_side_candidate"], "L")
        self.assertEqual(right["coordinate_side_candidate"], "R")
        # 2 mm is 8 NII voxels, not 2 voxels or 2 um.
        self.assertEqual(inventory.spatial_screen((18000, 44000, 24000), BOXES)["bbox_hits"], "IAL")
        self.assertEqual(inventory.spatial_screen((17999, 44000, 24000), BOXES)["bbox_hits"], "")

    def test_unknown_zero_suffix_missing_region_and_no_promotion(self):
        explicit = inventory.classify_neuron("CL_Unknown_0", (21000, 41000, 21000), BOXES, {"IAL"})
        self.assertEqual(explicit["portal_region_base"], "UNKNOWN")
        self.assertIn("atlas_explicit_unknown", explicit["screen_reasons"])
        self.assertTrue(explicit["potential_INS_candidate"])
        self.assertEqual(explicit["anatomical_membership"], "not_assessed")
        self.assertEqual(explicit["layer_identity"], "not_assessed")
        self.assertEqual(explicit["VEN_identity"], "not_assessed")
        blank = inventory.classify_neuron("", None, BOXES, {"IAL"})
        self.assertIn("missing_region_metadata", blank["screen_reasons"])
        self.assertNotIn("atlas_explicit_unknown", blank["screen_reasons"])
        self.assertIsNone(blank["potential_INS_candidate"])
        subcortical = inventory.classify_neuron("SL_Pi_999", None, BOXES, {"PI"})
        self.assertFalse(subcortical["atlas_INS"])
        self.assertEqual(subcortical["portal_region_base"], "SL_PI")
        g = inventory.classify_neuron("G_48", (0, 0, 0), BOXES, {"IAL"})
        self.assertFalse(g["potential_INS_candidate"])
        self.assertTrue(g["atlasINS_or_G_or_foldedBox_candidate"])

    def test_missing_or_nonfinite_coordinates_are_unassessed(self):
        for xyz in (None, (None, 41000, 21000), (float("nan"), 41000, 21000)):
            with self.subTest(xyz=xyz):
                result = inventory.spatial_screen(xyz, BOXES)
                self.assertEqual(result["coordinate_status"], "missing")
                self.assertEqual(result["bbox_status"], "missing_coordinates")
        self.assertEqual(inventory.base_label("CR_area_7op_124"), "AREA_7OP")

    def test_adjacent_or_excluded_never_accepted(self):
        for label in ("G_48", "PrCO_49", "Cl_304", "F4_85"):
            row = inventory.classify_neuron(label, (21000, 41000, 21000), BOXES, {"IAL"})
            self.assertFalse(row["atlas_INS"])
            self.assertEqual(row["anatomical_membership"], "not_assessed")
        excluded = inventory.classify_neuron("Ial_42", (21000, 41000, 21000), BOXES, {"IAL"}, excluded=True)
        self.assertTrue(excluded["atlas_INS"])
        self.assertFalse(excluded["potential_INS_candidate"])

    def test_exact_sample_channel_and_neuron_names(self):
        issues = []
        rows = [{"sampleid": "250432", "name": "001.swc"},
                {"sampleid": "250432", "name": "1.swc"},
                {"sampleid": "250432ch2", "name": "001.swc"},
                {"sampleid": "250432", "name": "duplicate.swc"},
                {"sampleid": "250432", "name": "duplicate.swc"}]
        exact = inventory.exact_records(rows, "250432", issues, "neurons")
        self.assertEqual(set(exact), {"001.swc", "1.swc"})
        self.assertEqual({i["issue"] for i in issues}, {"sample_identity_mismatch", "ambiguous_duplicate_identity"})
        second = inventory.exact_records(rows, "250432ch2", [], "neurons")
        self.assertEqual(set(second), {"001.swc"})

    def test_stale_tracker_vs_dated_verified_series_and_exact_channel(self):
        series = {"252714": {"status": "verified_copied_series", "checked_at": "2026-10-02",
                              "channel": "CH1", "relative_path": "252714-CH1_resample/resample_5um"}}
        observation = {"checked_at": "2026-10-08T23:18:43+08:00", "root": "scoped_root",
                       "directories": ["252714-CH1_resample"]}
        row = inventory.reconcile_series("252714", {"5_micron_data_copied": "No"}, series, observation)
        self.assertEqual(row["copied5um_status"], "verified_copied_series")
        self.assertEqual(row["copied5um_discrepancy"], "tracker_no_vs_verified_series")
        self.assertEqual(row["copied5um_evidence_date"], "2026-10-02")
        self.assertEqual(row["copied5um_directory_observation_date"], observation["checked_at"])
        channel = inventory.reconcile_series("252714ch2", {}, series, observation)
        self.assertEqual(channel["copied5um_status"], "not_copied_at_scoped_root")
        self.assertEqual(channel["copied5um_directory_observation_status"], "absent_at_scoped_root")
        self.assertEqual(channel["copied5um_evidence_date"], observation["checked_at"])


class SnapshotTests(unittest.TestCase):
    def test_snapshot_timestamp_order_handles_timezone_offsets(self):
        self.assertGreater(inventory.timestamp_order("2026-10-08T15:00:01+00:00"),
                           inventory.timestamp_order("2026-10-08T22:59:59+08:00"))

    def test_later_failure_replaces_old_success_without_zero_and_hash_drift_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            older, newer = root / "older", root / "newer"
            older.mkdir()
            newer.mkdir()
            path = older / "a_neurons.json"
            path.write_text(json.dumps([{"sampleid": "a", "name": "001.swc"}]), encoding="utf-8")
            success = {"sample": "a", "kind": "neurons", "http_status": 200,
                       "file": "older/a_neurons.json", "sha256": inventory.sha256(path), "checked_at": "2026-10-02"}
            (older / "manifest.json").write_text(json.dumps({"records": [success]}), encoding="utf-8")
            failure = {"sample": "a", "kind": "neurons", "http_status": 500, "checked_at": "2026-10-08"}
            (newer / "manifest.json").write_text(json.dumps({"records": [failure]}), encoding="utf-8")
            result = inventory.load_snapshots(root, [older, newer], [])
            self.assertEqual(result[("a", "neurons")]["status"], "request_failed")
            self.assertIsNone(result[("a", "neurons")]["data"])
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                inventory.load_snapshots(root, [older], [])

    def test_all_sample_build_keeps_missing_counts_and_separate_injection_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def csv_file(relative, rows):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                inventory.write_csv(path, rows)
            csv_file("group_analysis/portal_audit_20260926/portal_macaque_samples.csv", [
                {"fMOST_id": "a", "project_id": "Monkey", "injection_structure": "Ial_42"},
                {"fMOST_id": "bch2", "project_id": "Monkey", "injection_structure": ""}])
            csv_file("group_analysis/portal_audit_20260928/portal_refresh_vs_20260926.csv", [
                {"sample": "a", "http_status": "200", "n_portal_now": "2", "n_insula_now": "1", "n_prco_now": "0", "status": "unchanged"}])
            csv_file("group_analysis/docs/dataset_status_manifest.csv", [
                {"fmost_id": "a", "animal": "936", "injection_sites": "planned FIDP", "5_micron_data_copied": "No"}])
            csv_file("group_analysis/visual_review_20261002/manifest/identity_manifest.csv", [
                {"SampleID": "a", "NeuronID": "001.swc", "portal_region": "Ial_42", "nmt_frame": ""}])
            csv_file("group_analysis/staging_20260926/reference/251637_subregion_bboxes_folded.csv", [
                {"sub_region": "IAL", "X_lo_q005": 40, "X_hi_q995": 48,
                 "Y_lo_q005": 160, "Y_hi_q995": 168, "Z_lo_q005": 80, "Z_hi_q995": 88,
                 "x_frame": "Soma_NII_X_folded", "midline_nii_x": 128, "nii_voxel_mm": 0.25}])
            for relative, content in {
                "group_analysis/scripts/insula_label_set.py": "# fixture\n",
                "group_analysis/scripts/cohort.py": "# fixture\n",
                "main_scripts/region_labels.py": "# fixture\n",
                "atlas/ARM_key_all.txt": "Full_Name\tAbbreviation\nInsular cortex\tIal\n",
                "notes/bulk_visual_review_20261002/server_inventory.md": "Date: 2026-10-02\n"}.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "manifest.json").write_text(json.dumps({"records": [
                {"sample": "a", "kind": "neurons", "http_status": 503, "checked_at": "2026-10-08"}]}), encoding="utf-8")
            output = root / inventory.DEFAULT_OUTPUT
            inventory.build_inventory(root, output, [snapshot])
            rows = {r["sample"]: r for r in inventory.read_csv(output / "sample_inventory.csv")}
            self.assertEqual(set(rows), {"a", "bch2"})
            self.assertEqual(rows["a"]["n_listed_reconstructions"], "")
            self.assertEqual(rows["a"]["n_atlas_INS"], "")
            self.assertEqual(rows["a"]["neuron_list_status"], "request_failed")
            self.assertEqual(rows["a"]["tracker_injection_claim"], "planned FIDP")
            self.assertEqual(rows["a"]["portal_injection_structure"], "Ial_42")
            self.assertEqual(rows["a"]["anatomical_acceptance"], "not_assessed")
            self.assertEqual(rows["bch2"]["n_potential_INS_candidate"], "")
            self.assertEqual(rows["bch2"]["copied5um_status"], "not_in_saved_source_inventory")
            with self.assertRaisesRegex(ValueError, "Output must stay"):
                inventory.build_inventory(root, root / "canonical", [snapshot])

    def test_capture_bounded_and_failed_requests_persist_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "metadata"
            response = mock.Mock(status_code=200)
            response.json.return_value = [{"project_id": "Monkey", "fMOST_id": "252714"},
                                          {"project_id": "Monkey", "fMOST_id": "252714ch2"},
                                          {"project_id": "Mouse", "fMOST_id": "mouse"}]
            with mock.patch("requests.get", return_value=response) as get:
                inventory.capture_live(root, directory, max_requests=2, timeout=1)
            self.assertEqual(get.call_count, 2)
            self.assertTrue(all("swc/" not in c.args[0] and "/cube/" not in c.args[0] for c in get.call_args_list))
            records = json.loads((directory / "manifest.json").read_text())["records"]
            self.assertEqual([r["kind"] for r in records], ["sample_info", "neurons"])
            with self.assertRaises(ValueError):
                inventory.capture_live(root, root / "invalid", max_requests=201)


if __name__ == "__main__":
    unittest.main()
