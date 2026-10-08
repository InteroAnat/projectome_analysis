"""Independent endpoint readback, including corruption beyond recorded hashes."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_endpoint_maps import Fixture, trace

spec = importlib.util.spec_from_file_location("endpoint_reviewer", ROOT / "group_analysis/scripts/review_endpoint_run.py")
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


def replace_artifact(fixture, name, content):
    path = fixture.output / name
    path.write_text(content, encoding="utf-8")
    provenance = fixture.output / "run_provenance.json"
    record = json.loads(provenance.read_text(encoding="utf-8"))
    record["artifacts"][name]["sha256"] = reviewer.sha256(path)
    provenance.write_text(json.dumps(record), encoding="utf-8")


def mutate_csv(fixture, name, mutate):
    rows = reviewer.csv_rows(fixture.output / name)
    mutate(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    replace_artifact(fixture, name, stream.getvalue())


class EndpointReviewTests(unittest.TestCase):
    def test_blank_original_source_label_preserved_and_bound_for_human_ins(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), animal="936", subregion="HumanINS_R")
            fixture.add("1 1 0 0 0 1 -1\n", animal="936", subregion="HumanINS_R")
            for entry in fixture.records:
                entry["OriginalSourceLabel"] = ""
                entry["AnatomyStatus"] = "existing_human_coarse_INS_fine_parcel_separate"
            fixture.run()
            report = reviewer.review(fixture.output, fixture.root / "readback")
            self.assertEqual(report["selected_neurons"], 2)
            self.assertEqual(report["computable_neurons"], 1)
            for name in ("input_manifest.csv", "per_neuron_qc.csv", "per_neuron_target_counts.csv"):
                for row in reviewer.csv_rows(fixture.output / name):
                    self.assertEqual(row["OriginalSourceLabel"], "")
                    if "source_metadata_json" in row:
                        self.assertEqual(json.loads(row["source_metadata_json"])["OriginalSourceLabel"], "")
            name = "leaf_records.jsonl"
            leaves = [json.loads(line) for line in (fixture.output / name).read_text().splitlines()]
            for leaf in leaves:
                self.assertEqual(leaf["OriginalSourceLabel"], "")
                self.assertEqual(leaf["source_metadata"]["OriginalSourceLabel"], "")
            record = json.loads((fixture.output / "run_provenance.json").read_text())
            self.assertEqual(record["unassessed_neurons"][0]["OriginalSourceLabel"], "")
            for nested in (False, True):
                changed = json.loads(json.dumps(leaves))
                if nested:
                    changed[0]["source_metadata"]["OriginalSourceLabel"] = "invented_INS"
                else:
                    changed[0]["OriginalSourceLabel"] = "invented_INS"
                replace_artifact(fixture, name, "".join(json.dumps(leaf) + "\n" for leaf in changed))
                with self.assertRaisesRegex(ValueError, "metadata differs"):
                    reviewer.review(fixture.output, fixture.root / "corrupted_readback")
                self.assertFalse((fixture.root / "corrupted_readback").exists())
            replace_artifact(fixture, name, "".join(json.dumps(leaf) + "\n" for leaf in leaves))
            for name in ("per_neuron_qc.csv", "per_neuron_target_counts.csv"):
                original = (fixture.output / name).read_text()
                mutate_csv(fixture, name, lambda rows: rows[0].update(OriginalSourceLabel="invented_INS"))
                with self.assertRaisesRegex(ValueError, "metadata differs"):
                    reviewer.review(fixture.output, fixture.root / "corrupted_readback")
                replace_artifact(fixture, name, original)
            manifest = reviewer.csv_rows(fixture.output / "input_manifest.csv")
            manifest[0]["OriginalSourceLabel"] = None
            with self.assertRaisesRegex(ValueError, "Nonstring manifest"):
                reviewer.source_expectations(manifest, (4, 2, 2), np.diag([2, 2, 2, 1]))

    def test_nullable_integral_decimal_indices_and_invalid_decimal_values(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0), (4, 0, 0)]))
            fixture.run()
            for name in ("per_neuron_target_counts.csv", "animal_region_summaries.csv"):
                mutate_csv(fixture, name, lambda rows: [row.update(target_index=str(float(row["target_index"])))
                                                     for row in rows if row["target_index"]])
            reviewer.review(fixture.output, fixture.root / "readback")
        for value in ("NaN", "Infinity", "-Infinity", "1.5", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reviewer.integer(value, "test target")

    def test_group_counts_availability_and_animal_lists_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), animal="a")
            fixture.add("1 1 0 0 0 1 -1\n", animal="b")
            fixture.run()
            path = fixture.output / "run_provenance.json"
            baseline = json.loads(path.read_text())
            mutations = [("n_selected", 999), ("n_computable", 999), ("n_animals", 999),
                         ("map_available", False), ("contributing_animals", ["a", "a"]),
                         ("unavailable_animals", []), ("unavailable_animals", ["invented"])]
            for field, value in mutations:
                with self.subTest(field=field, value=value):
                    record = json.loads(json.dumps(baseline))
                    record["group_maps"][0][field] = value
                    path.write_text(json.dumps(record))
                    with self.assertRaises(ValueError):
                        reviewer.review(fixture.output, fixture.root / "readback")
                    self.assertFalse((fixture.root / "readback").exists())
            for index, value in ((0, False), (1, True)):
                record = json.loads(json.dumps(baseline))
                record["animal_maps"][index]["map_available"] = value
                path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "Map availability differs"):
                    reviewer.review(fixture.output, fixture.root / "readback")

    def test_duplicate_missing_group_regions_and_run_bindings_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), subregion="one")
            fixture.add(trace([(2, 0, 0)]), subregion="two")
            fixture.run()
            path = fixture.output / "run_provenance.json"
            baseline = json.loads(path.read_text())
            for values in ([baseline["group_maps"][0]] * 2, baseline["group_maps"][:1], baseline["group_maps"] * 2):
                path.write_text(json.dumps({**baseline, "group_maps": values}))
                with self.assertRaisesRegex(ValueError, "source-region coverage differs"):
                    reviewer.review(fixture.output, fixture.root / "readback")
            for field, value in (("source_code_sha256", {}), ("artifacts", {}),
                                 ("scientific_status", "accepted"), ("biological_innervation_state", "present")):
                path.write_text(json.dumps({**baseline, field: value}))
                with self.assertRaises(ValueError):
                    reviewer.review(fixture.output, fixture.root / "readback")

    def test_saved_regional_summary_metadata_and_unavailable_values_after_rehash(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), subregion="one")
            fixture.add("1 1 0 0 0 1 -1\n", subregion="two")
            fixture.run()
            name = "animal_region_summaries.csv"
            baseline = (fixture.output / name).read_text()
            for available, field, value in ((True, "target_full_name", "fabricated"), (True, "map_available", "False"),
                                             (False, "endpoint_count", "1"), (False, "map_available", "True"),
                                             (False, "target_index", "1")):
                with self.subTest(available=available, field=field):
                    replace_artifact(fixture, name, baseline)
                    def mutate(rows):
                        row = next(row for row in rows if (row["Subregion"] == "one") == available)
                        row[field] = value
                    mutate_csv(fixture, name, mutate)
                    with self.assertRaises(ValueError):
                        reviewer.review(fixture.output, fixture.root / "readback")

    def test_unavailable_groups_cannot_have_map_fields_or_files(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add("1 1 0 0 0 1 -1\n")
            fixture.run()
            path = fixture.output / "run_provenance.json"
            baseline = json.loads(path.read_text())
            for scope in ("animal_maps", "group_maps"):
                with self.subTest(scope=scope):
                    record = json.loads(json.dumps(baseline))
                    record[scope][0]["count_path"] = "fabricated.nii.gz"
                    path.write_text(json.dumps(record))
                    with self.assertRaisesRegex(ValueError, "Unavailable group has map fields"):
                        reviewer.review(fixture.output, fixture.root / "readback")
            path.write_text(json.dumps(baseline))
            nib.save(nib.load(fixture.reference), fixture.output / "group_maps/fabricated.nii.gz")
            with self.assertRaisesRegex(ValueError, "Unexpected or missing saved map files"):
                reviewer.review(fixture.output, fixture.root / "readback")

    def test_leaf_original_metadata_source_contract_and_metrics_after_rehash(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.run()
            name = "leaf_records.jsonl"
            baseline = json.loads((fixture.output / name).read_text())
            edits = [("source_metadata", {**baseline["source_metadata"], "ExtraSourceStatus": "invented"}),
                     ("OriginalSourceLabel", "invented"), ("SWCPath", "invented"),
                     ("ReferenceSHA256", "0" * 64), ("source_path", "invented"),
                     ("coordinate_frame", "nifti_world_mm"), ("index_scale_um", [2, 2, 2]),
                     ("leaf_class", "unresolved_leaf"), ("terminal_branch_length_mm", 999),
                     ("terminal_edge_type_transition", False), ("coincident_leaf_count", 999)]
            for field, value in edits:
                with self.subTest(field=field):
                    leaf = {**baseline, field: value}
                    replace_artifact(fixture, name, json.dumps(leaf) + "\n")
                    with self.assertRaises((ValueError, AssertionError)):
                        reviewer.review(fixture.output, fixture.root / "readback")

    def test_per_neuron_target_metadata_labels_volumes_after_rehash(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.run()
            name = "per_neuron_target_counts.csv"
            baseline = (fixture.output / name).read_text()
            for field, value in (("AnimalID", "other"), ("OriginalSourceLabel", "other"),
                                 ("source_metadata_json", "{}"), ("target_abbreviation", "CR_fake"),
                                 ("target_full_name", "fake"), ("target_volume_mm3", "999")):
                with self.subTest(field=field):
                    replace_artifact(fixture, name, baseline)
                    mutate_csv(fixture, name, lambda rows: rows[0].update({field: value}))
                    with self.assertRaises((ValueError, AssertionError)):
                        reviewer.review(fixture.output, fixture.root / "readback")

    def test_qc_accounting_and_unresolved_unassessed_lists_after_rehash(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add("1 1 0 0 0 1 -1\n2 0 1 0 0 1 1\n")
            fixture.run()
            name = "per_neuron_qc.csv"
            baseline_csv = (fixture.output / name).read_text()
            for field, value in (("candidate_status", "candidate_endpoints_present"), ("root_node_id", "99"),
                                 ("occupied_candidate_voxel_count", "1"), ("leaf_class_counts_json", "{}"),
                                 ("unknown_target_counts_by_level_json", '{"1":1}'),
                                 ("in_brain_mask_candidate_count", "1"), ("soma_anchor_node_id", "2")):
                with self.subTest(field=field):
                    replace_artifact(fixture, name, baseline_csv)
                    mutate_csv(fixture, name, lambda rows: rows[0].update({field: value}))
                    with self.assertRaises(ValueError):
                        reviewer.review(fixture.output, fixture.root / "readback")
            replace_artifact(fixture, name, baseline_csv)
            path = fixture.output / "run_provenance.json"
            baseline = json.loads(path.read_text())
            for field, value in (("accounting", {}), ("unassessed_neurons", []), ("unresolved_compartment_neurons", [])):
                with self.subTest(field=field):
                    record = {**baseline, field: value}
                    path.write_text(json.dumps(record))
                    with self.assertRaises(ValueError):
                        reviewer.review(fixture.output, fixture.root / "readback")

    def test_recorded_reference_geometry_and_units_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.run()
            path = fixture.output / "run_provenance.json"
            baseline = json.loads(path.read_text())
            for field, value in (("shape", [9, 2, 2]), ("affine_mm", np.eye(4).tolist()),
                                 ("voxel_volume_mm3", 999), ("reference_sha256", "0" * 64),
                                 ("map_value_units", {}), ("hemisphere_label_contingency", {})):
                with self.subTest(field=field):
                    path.write_text(json.dumps({**baseline, field: value}))
                    with self.assertRaises(ValueError):
                        reviewer.review(fixture.output, fixture.root / "readback")

    def test_all_input_and_saved_map_coded_mm_geometry_even_after_rehash(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.run()
            record_path = fixture.output / "run_provenance.json"
            baseline = json.loads(record_path.read_text())
            for key in ("reference", "atlas", "hemisphere_mask", "brain_mask"):
                path = Path(baseline["inputs"][key]["path"])
                original = path.read_bytes()
                for mutation in ("units", "uncoded", "conflicting"):
                    with self.subTest(key=key, mutation=mutation):
                        path.write_bytes(original)
                        image = nib.load(path)
                        changed = nib.Nifti1Image(np.asanyarray(image.dataobj).copy(), image.affine, image.header.copy())
                        if mutation == "units":
                            changed.header.set_xyzt_units("meter")
                        elif mutation == "uncoded":
                            changed.set_qform(None, code=0)
                            changed.set_sform(None, code=0)
                        else:
                            changed.set_qform(np.diag([3., 3., 3., 1.]), code=2)
                        nib.save(changed, path)
                        record = json.loads(json.dumps(baseline))
                        record["inputs"][key]["sha256"] = reviewer.sha256(path)
                        record_path.write_text(json.dumps(record))
                        with self.assertRaises(ValueError):
                            reviewer.review(fixture.output, fixture.root / "readback")
                path.write_bytes(original)
            record_path.write_text(json.dumps(baseline))
            entry = baseline["animal_maps"][0]
            path = fixture.output / entry["count_path"]
            image = nib.load(path)
            changed = nib.Nifti1Image(image.get_fdata(), image.affine, image.header.copy())
            changed.set_qform(np.diag([3., 3., 3., 1.]), code=2)
            nib.save(changed, path)
            entry["count_sha256"] = reviewer.sha256(path)
            record_path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError, "coded transform differs"):
                reviewer.review(fixture.output, fixture.root / "readback")

    def test_source_bom_and_independent_graph_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add("\ufeff" + trace([(1, 0, 0)]))
            fixture.run()
            reviewer.review(fixture.output, fixture.root / "readback")
            manifest = reviewer.csv_rows(fixture.output / "input_manifest.csv")
            entry = manifest[0]
            source = Path(entry["SWCPath"])
            invalid_sources = ["0 1 0 0 0 1 -1\n", "1 -2 0 0 0 1 -1\n", "1 1 0 0 0 -1 -1\n",
                               "1 1 0 0 0 NaN -1\n", "1 1 0 0 0 1 -1\n2 2 1 0 0 1 3\n3 2 2 0 0 1 2\n",
                               "1 1 0 0 0 1 -1\n2 2 1 0 0 1 99\n", "1 1 0 0 0 1 -1\n2 2 1 0 0 1 2\n"]
            for text in invalid_sources:
                with self.subTest(text=text):
                    source.write_text(text)
                    entry["SWCSHA256"] = reviewer.sha256(source)
                    with self.assertRaises(ValueError):
                        reviewer.source_expectations(manifest, (4, 2, 2), np.diag([2, 2, 2, 1]))

    def test_graph_branches_proximal_coincident_custom_types_and_soma_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add("1 1 0 0 0 1 -1\n2 2 0 0 0 1 1\n3 0 1 0 0 1 2\n"
                        "4 2 1 0 0 1 3\n5 2 1 0 0 1 3\n6 3 2 0 0 1 1\n7 1 2 0 0 1 6\n")
            fixture.add("1 0 0 0 0 1 -1\n2 2 0 0 0 1 1\n3 3 0 0 0 1 2\n")
            fixture.add("1 1 0 0 0 1 -1\n2 2 99 0 0 1 1\n", animal="outside")
            fixture.run(brain_mask=None)
            report = reviewer.review(fixture.output, fixture.root / "readback")
            self.assertEqual(report["selected_neurons"], 3)
            self.assertEqual(report["computable_neurons"], 2)
            self.assertEqual(report["candidate_axonal_leaves"], 3)
            self.assertEqual(report["outside_reference_candidates"], 1)

    def test_legacy_explicit_underscore_side_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            text = fixture.atlas_key.read_text().replace("CL_", "L_").replace("CR_", "R_")
            fixture.atlas_key.write_text(text)
            fixture.run()
            report = reviewer.review(fixture.output, fixture.root / "readback")
            self.assertEqual(report["status"], "passed")

    def test_optimized_python_rejects_fabricated_group_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.run()
            path = fixture.output / "run_provenance.json"
            record = json.loads(path.read_text())
            record["group_maps"][0]["n_computable"] = 999
            path.write_text(json.dumps(record))
            result = subprocess.run([sys.executable, "-B", "-O", str(ROOT / "group_analysis/scripts/review_endpoint_run.py"),
                                     "--run", str(fixture.output), "--output", str(fixture.root / "optimized_readback")],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Group conditional denominator differs", result.stderr)
            self.assertFalse((fixture.root / "optimized_readback").exists())

    def test_saved_sources_all_leaves_maps_tables_and_unassessed_group(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1.1, 0, 0), (1.2, 0, 0)]) + "4 3 2 0 0 1 1\n")
            fixture.add("1 1 0 0 0 1 -1\n", subregion="unassessed")
            fixture.run()
            report = reviewer.review(fixture.output, fixture.root / "readback")
            self.assertEqual(report["selected_neurons"], 2)
            self.assertEqual(report["computable_neurons"], 1)
            self.assertEqual(report["full_graph_leaves"], 4)
            self.assertEqual(report["candidate_axonal_leaves"], 2)
            self.assertEqual(report["table_checks"]["unavailable_groups"], 1)
            self.assertEqual(len(report["map_files"]), 6)
            self.assertEqual(report["full_saved_voxels_read"], 6 * 4 * 2 * 2)
            with self.assertRaises(FileExistsError):
                reviewer.review(fixture.output, fixture.root / "readback")

    def test_fractional_occupancy_and_equal_animal_weighting(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), animal="a")
            for _ in range(2):
                fixture.add(trace([(1, 0, 0), (1.1, 0, 0)]), animal="b")
            fixture.add(trace([(2, 0, 0)]), animal="b")
            fixture.run()
            report = reviewer.review(fixture.output, fixture.root / "readback")
            self.assertEqual(report["computable_neurons"], 4)
            self.assertEqual(len(report["map_files"]), 9)

    def test_exact_faces_outside_and_mask_free_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(np.nextafter(.5, -np.inf), 0, 0), (.5, 0, 0), (3.5, 0, 0)]))
            fixture.run(brain_mask=None)
            report = reviewer.review(fixture.output, fixture.root / "readback")
            self.assertEqual(report["candidate_axonal_leaves"], 3)
            self.assertEqual(report["outside_reference_candidates"], 1)

    def test_changed_leaf_voxel_rejected_even_after_artifact_hash_updated(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.run()
            name = "leaf_records.jsonl"
            leaf = json.loads((fixture.output / name).read_text())
            leaf["voxel"] = [0, 0, 0]
            replace_artifact(fixture, name, json.dumps(leaf) + "\n")
            with self.assertRaisesRegex(ValueError, "Leaf voxel mismatch"):
                reviewer.review(fixture.output, fixture.root / "readback")
            self.assertFalse((fixture.root / "readback").exists())

    def test_changed_target_label_rejected_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.run()
            name = "leaf_records.jsonl"
            leaf = json.loads((fixture.output / name).read_text())
            leaf["atlas_levels"][0]["index"] = 101
            replace_artifact(fixture, name, json.dumps(leaf) + "\n")
            with self.assertRaisesRegex(ValueError, "atlas-level lookup differs"):
                reviewer.review(fixture.output, fixture.root / "readback")

    def test_wrong_group_map_rejected_even_after_map_hash_updated(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), animal="a")
            fixture.add(trace([(1, 0, 0), (1.1, 0, 0), (1.2, 0, 0)]), animal="b")
            fixture.run()
            provenance = fixture.output / "run_provenance.json"
            record = json.loads(provenance.read_text())
            entry = record["group_maps"][0]
            path = fixture.output / entry["count_path"]
            image = nib.load(path)
            values = image.get_fdata()
            values[1, 0, 0] = 2.5
            modified = nib.Nifti1Image(values, image.affine, image.header)
            nib.save(modified, path)
            entry["count_sha256"] = reviewer.sha256(path)
            provenance.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(AssertionError):
                reviewer.review(fixture.output, fixture.root / "readback")


if __name__ == "__main__":
    unittest.main()
