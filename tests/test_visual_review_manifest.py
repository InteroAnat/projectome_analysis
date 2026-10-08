"""Selection, coverage, review-table and orthogonal-aspect contracts.

Tests write temporary directories only. They do not touch cohort workbooks
or the persisted review crops.
"""

from pathlib import Path
import json
import sys
import tempfile
import hashlib
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_analysis" / "scripts"))
import visual_review_20261002 as review


INSULA = {"IAL", "IG", "IA/ID"}
BOXES = [("IAL", 40.0, 70.0, 200.0, 220.0, 70.0, 95.0)]


class SelectionTests(unittest.TestCase):
    def reasons(self, portal="", step1=None, staging="", in_staging=False, nii=None,
                in_portal=True, in_step1=False):
        return review.classify_identity(
            portal, step1, staging, in_staging, nii, BOXES, INSULA, in_portal, in_step1)

    def test_blank_portal_region_is_not_unknown(self):
        self.assertEqual(review.portal_base_label(""), "")
        self.assertEqual(review.portal_base_label(None), "")
        self.assertEqual(self.reasons(portal="", in_portal=True), [])

    def test_validated_blank_source_is_separate_from_atlas_unknown(self):
        self.assertEqual(review.review_category(["metadata_unresolved_raw_source"]), "metadata_unresolved")
        self.assertEqual(review.review_category(["metadata_unresolved_raw_source", "portal_explicit_unknown"]), "Unknown")
        self.assertEqual(review.validated_unresolved_sources(None), {})

    def test_activated_extension_survives_default_regeneration_and_fails_on_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "validation.json"
            report.write_text("validation snapshot", encoding="utf-8")
            inputs = root / "manifest/selection_inputs.json"
            inputs.parent.mkdir()
            inputs.write_text(json.dumps({"validated_unresolved_sources": {
                "report_path": str(report), "report_sha256": review.sha256_file(report)}}), encoding="utf-8")
            self.assertEqual(review.retained_unresolved_report(root, None), report)
            report.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                review.retained_unresolved_report(root, None)
            explicit = root / "new_validation.json"
            self.assertEqual(review.retained_unresolved_report(root, explicit), explicit)

    def test_live_regional_lock_prevents_manifest_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with review.exclusive_batch(root / "manifest/regional_context.lock"):
                with mock.patch.object(review, "_build_manifest") as replacement:
                    with self.assertRaises(OSError):
                        review.build_manifest(root)
                    replacement.assert_not_called()
            with mock.patch.object(review, "_build_manifest", return_value="built") as replacement:
                self.assertEqual(review.build_manifest(root), "built")
                replacement.assert_called_once_with(root, None)

    def test_unresolved_source_requires_unchanged_own_sample_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "251730" / "226.swc"
            raw.parent.mkdir(parents=True)
            payload = b"1 1 10 20 30 1 -1\n2 2 11 20 30 1 1\n"
            raw.write_bytes(payload)
            record = {"uid": "251730|226.swc", "url": "http://10.10.31.31/swc/newswc/Monkey/251730/swc_raw/226.swc",
                      "status": "structurally_valid", "graph_validated": True, "metadata_status": "unresolved",
                      "local_path": "raw/251730/226.swc", "bytes": len(payload), "nodes": 2,
                      "sha256": hashlib.sha256(payload).hexdigest(), "raw_root_xyz": [10, 20, 30]}
            report = root / "validation.json"
            def save(state="terminal_complete"):
                report.write_text(json.dumps({"state": state, "records": [record]}), encoding="utf-8")
            with mock.patch.object(review, "PROJECT_ROOT", root):
                save()
                self.assertEqual(set(review.validated_unresolved_sources(report)), {record["uid"]})
                save("running")
                with self.assertRaisesRegex(ValueError, "not terminal"):
                    review.validated_unresolved_sources(report)
                save()
                record["url"] = record["url"].replace("251730", "252790")
                save()
                with self.assertRaisesRegex(ValueError, "route"):
                    review.validated_unresolved_sources(report)
                record["url"] = record["url"].replace("252790", "251730")
                save()
                raw.write_bytes(payload.replace(b"1 -1", b"1 2"))
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    review.validated_unresolved_sources(report)
                record["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
                record["bytes"] = raw.stat().st_size
                save()
                with self.assertRaisesRegex(ValueError, "exactly one root"):
                    review.validated_unresolved_sources(report)

    def test_explicit_labels_and_numeric_suffixes(self):
        self.assertEqual(review.review_category(self.reasons("Ial_42")), "INS")
        self.assertEqual(review.review_category(self.reasons("G_7")), "G")
        self.assertEqual(review.review_category(self.reasons("PrCO_3")), "PrCO")
        self.assertEqual(review.review_category(self.reasons("area_44_76")), "adjacent")
        self.assertEqual(review.review_category(self.reasons("claustrum_9")), "adjacent")
        self.assertIn("portal_explicit_unknown", self.reasons("Unknown_0"))

    def test_step1_empty_is_unknown_and_coordinate_screen_is_separate(self):
        reasons = self.reasons(portal="area_46_10", step1="", in_step1=True, in_portal=True)
        self.assertIn("step1_unknown", reasons)
        self.assertEqual(review.review_category(reasons), "Unknown")
        screened = self.reasons(portal="area_46_10", nii=(78.0, 210.0, 80.0))
        self.assertEqual(review.review_category(screened), "possible_insula")
        outside = self.reasons(portal="area_46_10", nii=(10.0, 10.0, 10.0))
        self.assertEqual(outside, [])

    def test_staging_keeper_and_missing_step1_are_retained(self):
        reasons = self.reasons(portal="area_46_10", staging="G", in_staging=True, in_step1=False)
        self.assertIn("staging_keeper", reasons)
        self.assertIn("staging_G", reasons)
        self.assertIn("missing_from_step1", reasons)
        self.assertEqual(review.review_category(reasons), "G")

    def test_exact_project_id_ignores_channel_suffix_records(self):
        records = [
            {"fMOST_id": "250432ch2", "project_id": "other"},
            {"fMOST_id": "250432", "project_id": "Monkey"},
        ]
        self.assertEqual(review.exact_project_id(records, "250432"), "Monkey")
        with self.assertRaises(ValueError):
            review.exact_project_id(records[:1], "250432")

    def test_folded_bbox_uses_midline_distance(self):
        # Folded X is |nii_x - 128|. 178 folds to 50 and lands in the box; 178 raw would not.
        self.assertEqual(review.folded_bbox_hits(178.0, 210.0, 80.0, BOXES), ["IAL"])
        self.assertEqual(review.folded_bbox_hits(10.0, 210.0, 80.0, BOXES), [])

    def test_blank_portal_text_is_absent_not_nan(self):
        self.assertEqual(review.display_label(float("nan")), "absent")
        self.assertEqual(review.display_label(""), "absent")
        self.assertEqual(review.display_label("IA/ID"), "IA/ID")

    def test_source_status_does_not_treat_pending_samples_as_available(self):
        self.assertEqual(review.image_source_status("252385"), "available")
        self.assertEqual(review.image_source_status("252790"), "source_pending")
        self.assertEqual(review.canon_neuron_id("25"), "025.swc")
        self.assertEqual(review.canon_neuron_id("025.swc"), "025.swc")

    def test_historical_area_44_keeps_its_anatomical_number(self):
        self.assertEqual(review.step1_reasons("area_44", set()), ["step1_adjacent"])
        self.assertEqual(review.step1_reasons("CL_area_44", set()), ["step1_adjacent"])
        self.assertEqual(review.step1_reasons("CR_area_44", set()), ["step1_adjacent"])
        self.assertEqual(review.historical_base_label("CL_area_44"), "AREA_44")
        self.assertEqual(review.portal_base_label("area_44"), "AREA")
        self.assertEqual(review.review_category(self.reasons("area_44_76")), "adjacent")
        self.assertEqual(review.label_reasons("step1", "area_44_76", set(), portal=False), [])
        self.assertEqual(review.step1_reasons("Unknown_0", set()), ["step1_unknown"])
        self.assertEqual(review.step1_reasons("G", set()), ["step1_G"])

    def test_staging_origin_keeps_the_workbook_source(self):
        row = {
            "SampleID": "252385", "NeuronID": "050.swc",
            "Soma_Region_Source": "coord_inferred_from_251637_padded_distant",
            "NeuronUID": "252385|050", "_source_row": 12,
        }
        origin = review.staging_origin(row, "book.xlsx")
        self.assertEqual(origin["region_source"], "coord_inferred_from_251637_padded_distant")
        self.assertEqual(origin["neuron_uid"], "252385|050")
        self.assertEqual(origin["row_uid"], "book.xlsx|252385|050.swc|row12")

    def test_coordinate_screen_stays_unpadded(self):
        self.assertEqual(review.COORDINATE_SCREEN_PAD_VOX, 0)
        # |57.9 - 128| = 70.1, just outside the unpadded box and inside an 8-voxel pad.
        self.assertEqual(review.folded_bbox_hits(57.9, 210, 80, BOXES), [])
        self.assertEqual(review.folded_bbox_hits(57.9, 210, 80, BOXES, pad_vox=8), ["IAL"])
        self.assertEqual(self.reasons(portal="area_46_10", nii=(57.9, 210.0, 80.0)), [])

    def test_missing_neighbors_are_not_marked_ok(self):
        status, coverage = review.production_from_coverage(
            True, True, [], soma_missing=[[212, 88, 155]], soma_complete=False, context_complete=True)
        self.assertEqual((status, coverage), ("partial", "partial_neighbor"))
        status, coverage = review.production_from_coverage(
            True, True, [], soma_missing=[], soma_complete=False, context_complete=True)
        self.assertEqual((status, coverage), ("partial", "partial_neighbor"))
        status, coverage = review.production_from_coverage(
            True, True, [], soma_missing=[], soma_complete=True, context_complete=True)
        self.assertEqual((status, coverage), ("ok", "complete"))
        status, coverage = review.production_from_coverage(
            True, False, ["soma: Soma's central high-resolution cube is unavailable"])
        self.assertEqual((status, coverage), ("partial", "center_unavailable"))
        status, coverage = review.production_from_coverage(
            False, False, ["swc: raw SWC unavailable after 3 requests"])
        self.assertEqual((status, coverage), ("error", "swc_unavailable"))

    def test_raw_swc_routes_stay_on_the_own_sample(self):
        url, caches, processed = review.documented_swc_routes(
            "252385", "d001.swc", "Monkey", Path("out"))
        self.assertIn("/252385/swc_raw/d001.swc", url.replace("\\", "/"))
        self.assertTrue(all(path.parent.name == "252385" and path.parent.parent.name == "swc_raw" for path in caches))
        self.assertNotIn("swc_raw", processed.as_posix())
        self.assertIn("252385", processed.as_posix())


class TraceSlabTests(unittest.TestCase):
    def test_off_plane_edges_stay_out_of_the_trace_slab(self):
        # Index equals micrometres. Displayed section is z=10. The five-section
        # slab is 8:13. An edge at 8.2-8.8 is outside that one section and inside
        # the slab. An edge at 25-28 is outside the slab and must not be drawn.
        rows = [
            (1, 2, 0, 0, 10.2, 1, -1),
            (2, 2, 0, 0, 10.8, 1, 1),
            (3, 2, 0, 0, 8.2, 1, -1),
            (4, 2, 0, 0, 8.8, 1, 3),
            (5, 2, 0, 0, 25.0, 1, -1),
            (6, 2, 0, 0, 28.0, 1, 5),
        ]
        shape = (30, 20, 20)
        single = review._segments_in_view(rows, [0, 0, 0], [1, 1, 1], shape, z_bounds=(10, 11))
        slab = review.trace_slab(rows, [0, 0, 0], [1, 1, 1], shape, 10.4)
        self.assertEqual(slab["z_start"], 8)
        self.assertEqual(slab["z_stop_exclusive"], 13)
        self.assertAlmostEqual(slab["nominal_depth_um"], 5.0)
        self.assertEqual(len(single), 1)
        self.assertEqual(slab["edges_in_slab"], 2)
        self.assertEqual(slab["edges_outside_slab"], 1)
        self.assertEqual(len(slab["segments"]), 2)

    def test_highres_context_grid_is_bounded(self):
        cubes = review.highres_context_cubes([40200.2, 31689.6, 19036.2])
        self.assertGreater(len(cubes), 1)
        self.assertLess(len(cubes), 400)
        one = review.highres_context_cubes([100, 100, 100], field_um=100, depth_um=30)
        self.assertEqual(len(one), 1)
        canvas, report = review.derive_highres_context(
            [50, 50, 6], lambda x, y, z: __import__("numpy").ones((90, 360, 360), dtype="uint16"),
            field_um=80, depth_um=12, factor=2, max_cubes=10)
        self.assertEqual(report["status"], "derived_highres")
        self.assertGreater(int(canvas.max()), 0)
        blocked, report = review.derive_highres_context(
            [40200.2, 31689.6, 19036.2], lambda *idx: None, max_cubes=10)
        self.assertIsNone(blocked)
        self.assertEqual(report["status"], "not_run")


class ReviewTableTests(unittest.TestCase):
    def test_oversized_source_errors_survive_csv_and_xlsx_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._manifest(root, [("252383", "094.swc")])
            folder = root / "252383" / "094"
            folder.mkdir(parents=True)
            errors = [{"cube_xyz": [index, 83, 102], "reason": "dated HTTP 404 " * 20}
                      for index in range(160)]
            (folder / "provenance.json").write_text(json.dumps({
                "derived_context": {"source_errors": errors}}), encoding="utf-8")
            review.write_correction_table(root)
            import pandas as pd
            csv = pd.read_csv(root / "tables/correction_table.csv", dtype=str, keep_default_na=False)
            book = pd.read_excel(root / "tables/correction_table.xlsx", dtype=str, keep_default_na=False)
            value = csv.loc[0, "context_source_errors"]
            self.assertEqual(value, book.loc[0, "context_source_errors"])
            detail = json.loads(value)
            payload = (root / detail["detail_file"]).read_bytes()
            self.assertEqual(json.loads(payload), errors)
            self.assertEqual(detail["items"], len(errors))
            self.assertEqual(detail["sha256"], __import__("hashlib").sha256(payload).hexdigest())
            self.assertEqual(review.table_source_errors([], root), "[]")
            small = errors[:1]
            self.assertEqual(json.loads(review.table_source_errors(small, root)), small)

    def test_oversized_human_note_fails_before_review_tables_are_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._manifest(root, [("252383", "094.swc")])
            review.write_correction_table(root)
            tables = root / "tables"
            import pandas as pd
            frame = pd.read_csv(tables / "correction_table.csv", dtype=str, keep_default_na=False)
            frame.loc[0, "human_notes"] = "x" * 32768
            frame.to_csv(tables / "correction_table.csv", index=False)
            previous = {name: (tables / name).read_bytes()
                        for name in ("correction_table.csv", "correction_table.xlsx")}
            with self.assertRaisesRegex(RuntimeError, "Excel cell limit exceeded"):
                review.write_correction_table(root)
            for name, payload in previous.items():
                self.assertEqual((tables / name).read_bytes(), payload)

    def _manifest(self, root: Path, rows: list[tuple[str, str]]) -> None:
        manifest = root / "manifest"
        manifest.mkdir(parents=True, exist_ok=True)
        lines = ["SampleID,NeuronID,UID,project_id,image_source_status,review_category"]
        for sample, neuron in rows:
            lines.append(f"{sample},{neuron},{sample}|{neuron},Monkey,available,INS")
        (manifest / "identity_manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_regeneration_preserves_either_table_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._manifest(root, [("252385", "001.swc"), ("252385", "002.swc")])
            tables = root / "tables"
            tables.mkdir()
            csv_path = tables / "correction_table.csv"
            xlsx_path = tables / "correction_table.xlsx"
            csv_path.write_text(
                "UID,human_region,human_notes\n252385|001.swc,IG,\n", encoding="utf-8")
            import pandas as pd
            pd.DataFrame([{
                "UID": "252385|001.swc", "human_region": "", "human_notes": "keep me",
            }]).to_excel(xlsx_path, index=False)
            review.write_correction_table(root)
            saved = pd.read_csv(csv_path, dtype=str).fillna("")
            first = saved[saved["UID"] == "252385|001.swc"].iloc[0]
            second = saved[saved["UID"] == "252385|002.swc"].iloc[0]
            self.assertEqual(first["human_region"], "IG")
            self.assertEqual(first["human_notes"], "keep me")
            self.assertEqual(second["human_region"], "")
            self.assertEqual(second["human_notes"], "")
            review.write_correction_table(root)
            again = pd.read_csv(csv_path, dtype=str).fillna("")
            kept = again[again["UID"] == "252385|001.swc"].iloc[0]
            self.assertEqual(kept["human_region"], "IG")
            self.assertEqual(kept["human_notes"], "keep me")
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace(",IG,", ",IG,", 1), encoding="utf-8")
            book = pd.read_excel(xlsx_path, dtype=str).fillna("")
            book.loc[book["UID"] == "252385|001.swc", "human_region"] = "G"
            before_csv = csv_path.read_bytes()
            before_xlsx = xlsx_path.read_bytes()
            book.to_excel(xlsx_path, index=False)
            conflict_xlsx = xlsx_path.read_bytes()
            with self.assertRaises(RuntimeError):
                review.write_correction_table(root)
            self.assertEqual(csv_path.read_bytes(), before_csv)
            self.assertEqual(xlsx_path.read_bytes(), conflict_xlsx)

    def test_verify_does_not_rewrite_reviewer_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest"
            manifest.mkdir()
            (manifest / "canonical_hashes.json").write_text(json.dumps({"canon": "abc"}), encoding="utf-8")
            (manifest / "identity_manifest.csv").write_text(
                "SampleID,NeuronID,UID,project_id,image_source_status,review_category\n"
                "250432,001.swc,250432|001.swc,Monkey,source_pending,adjacent\n",
                encoding="utf-8")
            tables = root / "tables"
            tables.mkdir()
            header = ["SampleID", "NeuronID", "UID", "image_source_status", "production_status",
                      "review_category", "coverage_status", "human_region"]
            header.extend(review.PANEL_FILES)
            row = ["250432", "001.swc", "250432|001.swc", "source_pending", "source_pending",
                   "adjacent", "", "IG"]
            row.extend([""] * len(review.PANEL_FILES))
            import pandas as pd
            frame = pd.DataFrame([dict(zip(header, row))])
            frame.to_csv(tables / "correction_table.csv", index=False)
            frame.to_excel(tables / "correction_table.xlsx", index=False)
            before_csv = (tables / "correction_table.csv").read_bytes()
            before_xlsx = (tables / "correction_table.xlsx").read_bytes()
            with mock.patch.object(review, "canonical_hashes", return_value={"canon": "abc"}):
                review.verify_outputs(root)
            self.assertEqual((tables / "correction_table.csv").read_bytes(), before_csv)
            self.assertEqual((tables / "correction_table.xlsx").read_bytes(), before_xlsx)
            frame.loc[0, "human_region"] = "G"
            frame.to_excel(tables / "correction_table.xlsx", index=False)
            conflict_csv = (tables / "correction_table.csv").read_bytes()
            conflict_xlsx = (tables / "correction_table.xlsx").read_bytes()
            with mock.patch.object(review, "canonical_hashes", return_value={"canon": "abc"}):
                with self.assertRaises(SystemExit):
                    review.verify_outputs(root)
            self.assertEqual((tables / "correction_table.csv").read_bytes(), conflict_csv)
            self.assertEqual((tables / "correction_table.xlsx").read_bytes(), conflict_xlsx)

    def test_duplicate_rows_conflict_and_orphans_are_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tables = root / "tables"
            tables.mkdir()
            duplicate = tables / "correction_table.csv"
            duplicate.write_text(
                "UID,human_region,human_notes\n"
                "252385|003.swc,IAL,first decision\n"
                "252385|003.swc,IG,second decision\n",
                encoding="utf-8")
            with self.assertRaises(RuntimeError):
                review.load_human_annotations(duplicate)
            self.assertIn("first decision", duplicate.read_text(encoding="utf-8"))
            self._manifest(root, [("252385", "003.swc")])
            duplicate.write_text(
                "UID,human_region,human_notes\n252385|004.swc,IAL,reviewed identity removed by manifest\n",
                encoding="utf-8")
            review.write_correction_table(root)
            import pandas as pd
            main = pd.read_csv(tables / "correction_table.csv", dtype=str).fillna("")
            archive = pd.read_csv(tables / "orphaned_annotations.csv", dtype=str).fillna("")
            self.assertNotIn("252385|004.swc", set(main["UID"]))
            self.assertIn("252385|004.swc", set(archive["UID"]))
            self.assertEqual(archive.iloc[0]["human_region"], "IAL")
            review.write_correction_table(root)
            again = pd.read_csv(tables / "orphaned_annotations.csv", dtype=str).fillna("")
            self.assertIn("252385|004.swc", set(again["UID"]))
            self.assertEqual(again.iloc[0]["human_notes"], "reviewed identity removed by manifest")


class OrthoGeometryTests(unittest.TestCase):
    def test_xz_aspect_uses_physical_spacing(self):
        specs = review.ortho_plane_specs((270, 1080, 1080), (100.2, 200.4, 80.5), (0.65, 0.65, 3.0))
        xz = specs["XZ"]
        self.assertAlmostEqual(xz["aspect"], 3.0 / 0.65)
        self.assertAlmostEqual(xz["width_um"], 702.0)
        self.assertAlmostEqual(xz["height_um"], 810.0)
        self.assertEqual(xz["marker"], (100, 80))
        self.assertAlmostEqual(specs["XY"]["aspect"], 1.0)
        displayed = (270 * xz["aspect"]) / 1080
        self.assertAlmostEqual(displayed, xz["height_um"] / xz["width_um"])
        self.assertGreater(displayed, 1.0)

    def test_ortho_axes_keep_the_physical_aspect(self):
        import matplotlib.pyplot as plt
        import numpy as np
        volume = np.zeros((4, 5, 6), dtype=np.uint16)
        volume[1, 2, 3] = 800
        fig, axes = review.build_ortho_figure(volume, (3.2, 2.4, 1.2), (0.65, 0.65, 3.0))
        try:
            self.assertAlmostEqual(float(axes[0].get_aspect()), 1.0)
            self.assertAlmostEqual(float(axes[1].get_aspect()), 3.0 / 0.65)
            self.assertAlmostEqual(float(axes[2].get_aspect()), 3.0 / 0.65)
            self.assertEqual(len(axes[1].collections), 1)
        finally:
            plt.close(fig)


class CanonicalBaselineTests(unittest.TestCase):
    def test_baseline_is_kept_and_a_mismatch_does_not_replace_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cohort.txt"
            source.write_text("v1", encoding="utf-8")
            out = root / "review"
            review.guard_canonical_baseline(out, files=(source,))
            baseline = out / "manifest" / "canonical_hashes.json"
            first = baseline.read_bytes()
            review.guard_canonical_baseline(out, files=(source,))
            self.assertEqual(baseline.read_bytes(), first)
            source.write_text("v2", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                review.guard_canonical_baseline(out, files=(source,))
            self.assertEqual(baseline.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
