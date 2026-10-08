"""Shared quotas, dated failures, resumability, and intensity-independent coverage."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_analysis" / "scripts"))
import fmost_context_resources as resources
import regional_context_batch_20261003 as batch
import visual_review_20261002 as review


class ContextBudgetTests(unittest.TestCase):
    def toolkit(self, root, sample="252790"):
        return SimpleNamespace(sample_id=sample,
            cache_http_dir=str(root / "cache" / "cubes" / sample / "high_res_http"),
            _block_errors={}, _download_http_block=mock.Mock(return_value=None))

    def test_cache_and_request_admission_stop_before_network_or_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            toolkit = self.toolkit(root)
            for kwargs, expected in (({"cache_bytes": 1}, "shared cache-byte budget"),
                                     ({"max_requests": 1}, "request-attempt budget")):
                source = resources.BoundedCubeSource(root, min_free_bytes=1, **kwargs)
                with mock.patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
                    self.assertIsNone(source(toolkit, 1, 2, 3))
                self.assertIn(expected, toolkit._block_errors[(1, 2, 3)])
                toolkit._download_http_block.assert_not_called()
                self.assertEqual(source.requests_upper_bound, 0)

    def test_disk_margin_and_runtime_stop_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            toolkit = self.toolkit(root)
            source = resources.BoundedCubeSource(root, min_free_bytes=1)
            with mock.patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
                self.assertIsNone(source(toolkit, 1, 2, 3))
            self.assertIn("free-space margin", toolkit._block_errors[(1, 2, 3)])
            with mock.patch.object(source, "runtime_expired", return_value=True):
                self.assertIsNone(source(toolkit, 1, 2, 3))
            self.assertIn("runtime budget", toolkit._block_errors[(1, 2, 3)])
            toolkit._download_http_block.assert_not_called()

    def test_cached_zero_intensity_is_usable_even_when_network_budget_is_full(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            toolkit = self.toolkit(root)
            path = Path(toolkit.cache_http_dir) / "3" / "1_2_3.tif"
            path.parent.mkdir(parents=True)
            tifffile.imwrite(path, np.zeros((90, 360, 360), dtype=np.uint16), compression="zlib")
            before = path.read_bytes()
            source = resources.BoundedCubeSource(root, cache_bytes=1, max_requests=1, min_free_bytes=1)
            data = source(toolkit, 1, 2, 3)
            self.assertEqual(data.shape, (90, 360, 360))
            self.assertEqual(int(data.max()), 0)
            self.assertEqual(source.cached_hits, 1)
            toolkit._download_http_block.assert_not_called()
            self.assertEqual(path.read_bytes(), before)

    def test_404_observation_is_dated_sample_specific_and_expires(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            toolkit = self.toolkit(root)
            toolkit._block_errors[(1, 2, 3)] = "HTTP Error 404: Not Found"
            source = resources.BoundedCubeSource(root, min_free_bytes=1, failure_ttl_seconds=100)
            with mock.patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                    mock.patch.object(resources.time, "time", return_value=1000):
                source(toolkit, 1, 2, 3)
            self.assertEqual(toolkit._download_http_block.call_count, 1)
            saved = json.loads(source.failure_path.read_text())
            self.assertEqual(saved["252790|1|2|3"]["observed_unix"], 1000)
            with mock.patch.object(resources.time, "time", return_value=1050):
                source(toolkit, 1, 2, 3)
            self.assertEqual(toolkit._download_http_block.call_count, 1)
            self.assertIn("does not establish anatomical absence", toolkit._block_errors[(1, 2, 3)])
            other = self.toolkit(root, "250432")
            with mock.patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                    mock.patch.object(resources.time, "time", return_value=1050):
                source(other, 1, 2, 3)
            self.assertEqual(other._download_http_block.call_count, 1)
            with mock.patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                    mock.patch.object(resources.time, "time", return_value=1101):
                source(toolkit, 1, 2, 3)
            self.assertEqual(toolkit._download_http_block.call_count, 2)

    def test_replacement_accounts_for_actual_cache_growth(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            toolkit = self.toolkit(root)
            path = Path(toolkit.cache_http_dir) / "3" / "1_2_3.tif"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"bad")
            source = resources.BoundedCubeSource(root, min_free_bytes=1)
            def replacement(*index):
                path.write_bytes(b"longer validated replacement")
                return np.ones((1, 1, 1), dtype=np.uint16)
            toolkit._download_http_block.side_effect = replacement
            with mock.patch.object(resources.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)):
                source(toolkit, 1, 2, 3)
            self.assertEqual(source.present_bytes, path.stat().st_size)
            self.assertEqual(source.requests_upper_bound, resources.HTTP_ATTEMPTS)


class ContextBatchTests(unittest.TestCase):
    def test_more_insula_candidates_are_prioritized_without_dropping_other_categories(self):
        import pandas as pd
        rows = pd.DataFrame([
            {"SampleID": sample, "NeuronID": neuron, "UID": f"{sample}|{neuron}", "review_category": category}
            for sample, neuron, category in (
                ("250432", "001.swc", "INS"),
                ("250432", "002.swc", "Unknown"),
                ("252790", "011.swc", "INS"),
                ("252790", "010.swc", "INS"),
                ("251730", "226.swc", "metadata_unresolved"),
                ("252790", "012.swc", "adjacent"),
            )
        ])
        with mock.patch.object(review, "load_manifest", return_value=rows):
            selected = batch.selected_rows(Path("unused"))
        self.assertEqual(selected["UID"].tolist(), [
            "252790|010.swc", "252790|011.swc", "250432|001.swc",
            "250432|002.swc", "251730|226.swc", "252790|012.swc"])
        self.assertEqual(set(selected["UID"]), set(rows["UID"]))
        self.assertEqual(selected["_sample_insula_count"].tolist(), [2, 2, 1, 1, 0, 2])

    def test_resume_retries_resource_stops_but_can_reuse_observed_404_gaps(self):
        from fmost_derived_context import derive_highres_context
        from fmost_image_geometry import native_affine
        native = [234.2, 234.2, 271]
        data, report = derive_highres_context(native,
            lambda x,y,z: None if [x,y,z] == [0,0,0] else np.ones((90,360,360),dtype=np.uint16),
            field_um=24, depth_um=12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "252790" / "001"
            folder.mkdir(parents=True)
            (root / "manifest").mkdir()
            swc = root / "swc_raw" / "252790" / "001.swc"
            swc.parent.mkdir(parents=True)
            swc.write_text("1 1 234.2 234.2 271 1 -1\n")
            report.update(UID="252790|001.swc",swc_sha256=review.sha256_file(swc),
                          source_errors=[{"reason":"HTTP Error 404: Not Found"}])
            path = root / "manifest" / "derived_context_pilot_252790_001.json"
            path.write_text(json.dumps(report))
            image = nib.Nifti1Image(data.transpose(2,1,0),native_affine(report["origin_xyz_um"],report["derived_spacing_xyz_um"]))
            image.header.set_xyzt_units("micron")
            nib.save(image,str(folder / "context.nii.gz"))
            acquisition={"sample_id":"252790","sampling":report["sampling"]}
            (folder / "context.nii.gz.json").write_text(json.dumps({"acquisition":acquisition}))
            for name in ["context_unmarked.png","context_marked.png","context_trace.png","context_stack.png","context_ortho.png"]:
                (folder / name).write_bytes(b"panel presence fixture")
            self.assertIsNotNone(batch.reusable_context(root,"252790","001.swc",minimum_field=1))
            self.assertIsNone(batch.reusable_context(root,"252790","001.swc"))
            report["source_errors"]=[{"reason":"Resource admission stopped: request-attempt budget"}]
            path.write_text(json.dumps(report))
            self.assertIsNone(batch.reusable_context(root,"252790","001.swc",minimum_field=1))
            report["source_errors"]=[{"reason":"HTTP Error 404: Not Found"}]
            report["unattempted"]=[[0,0,1]]
            path.write_text(json.dumps(report))
            self.assertIsNone(batch.reusable_context(root,"252790","001.swc",minimum_field=1))

    def test_z_boundary_retains_requested_field_with_same_working_memory(self):
        field, plan = batch.requested_field([40000, 31000, 270])
        interior_field, interior = batch.requested_field([40000, 31000, 330])
        self.assertEqual(field, 4000)
        self.assertEqual(interior_field, 4000)
        self.assertGreater(plan["cube_count"], 400)
        self.assertLessEqual(plan["cube_count"], batch.FIELD_MAX_CUBES)
        self.assertEqual(plan["cube_count"], 2 * interior["cube_count"])
        self.assertEqual(plan["shape_zyx"], interior["shape_zyx"])
        self.assertEqual(plan["working_bytes_estimate"], interior["working_bytes_estimate"])
        self.assertLessEqual(plan["working_bytes_estimate"], plan["max_working_bytes"])
        self.assertLessEqual(plan["source_bytes_estimate"], batch.FIELD_MAX_SOURCE_BYTES)
        self.assertEqual(plan["derived_spacing_xyz_um"], [5.2, 5.2, 3.0])
        self.assertNotIn("fallback_reason", plan)

    def test_larger_field_still_obeys_memory_and_source_admission(self):
        field, plan = batch.requested_field([40000, 31000, 270], preferred=8000)
        self.assertEqual(field, 2800)
        self.assertIn("fallback_reason", plan)
        self.assertLessEqual(plan["working_bytes_estimate"], plan["max_working_bytes"])
        self.assertLessEqual(plan["source_bytes_estimate"], batch.FIELD_MAX_SOURCE_BYTES)

    def test_sampler_retains_full_field_across_two_z_blocks(self):
        from fmost_derived_context import derive_highres_context
        cube = np.ones((90, 360, 360), dtype=np.uint16)
        data, report = derive_highres_context(
            [40000, 31000, 270], lambda *index: cube,
            max_cubes=batch.FIELD_MAX_CUBES,
            max_source_bytes=batch.FIELD_MAX_SOURCE_BYTES)
        self.assertEqual(report["requested_field_um_xyz"], [4000, 4000, 30])
        self.assertEqual(report["status"], "derived_highres")
        self.assertGreater(len(report["loaded"]), 400)
        self.assertEqual(len(report["loaded"]), report["cube_count"])
        self.assertEqual(data.shape, tuple(report["shape_zyx"]))
        self.assertTrue(np.all(data == 1))
        self.assertTrue(report["root_pixel_covered"])

    def test_zero_intensity_and_missing_source_have_different_coverage_values(self):
        report = {"shape_zyx": [1, 1, 2], "source_first_index_xyz": [359, 0, 0],
                  "source_stride_xyz": [1, 1, 1], "loaded": [[0, 0, 0]],
                  "derived_spacing_xyz_um": [.65, .65, 3], "origin_xyz_um": [359 * .65, 0, 0]}
        mask = resources.coverage_mask(report)
        np.testing.assert_array_equal(mask, [[[1, 0]]])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            review.write_derived_coverage(folder, report, report["origin_xyz_um"])
            image = nib.load(str(folder / "context_coverage.nii.gz"))
            np.testing.assert_array_equal(np.asanyarray(image.dataobj).ravel(), [1, 0])
            self.assertEqual(image.header.get_xyzt_units()[0], "micron")

    def test_budget_stop_keeps_every_uid_in_run_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest").mkdir()
            frame = pd.DataFrame([
                {"SampleID": "252790", "NeuronID": "001.swc", "UID": "252790|001.swc", "review_category": "Unknown", "project_id": "Monkey"},
                {"SampleID": "252790", "NeuronID": "002.swc", "UID": "252790|002.swc", "review_category": "INS", "project_id": "Monkey"}])
            frame.to_csv(root / "manifest" / "identity_manifest.csv", index=False)
            with mock.patch.object(review, "guard_canonical_baseline"), \
                    mock.patch.object(review, "write_gallery"), \
                    mock.patch.object(resources.BoundedCubeSource, "runtime_expired", return_value=True), \
                    mock.patch.object(review, "fetch_raw_swc") as fetch:
                result = batch.run(root, ["252790"], min_free_bytes=1)
            self.assertEqual(result["state"], "terminal_budget_stop")
            self.assertEqual(list(result["identities"]), ["252790|002.swc", "252790|001.swc"])
            self.assertTrue(all(row["state"] == "not_attempted" for row in result["identities"].values()))
            fetch.assert_not_called()
            self.assertEqual(len(json.loads((root / "manifest" / "regional_context_run_20261003.json").read_text())["identities"]), 2)
            first_record = Path(result["terminal_record"])
            original = first_record.read_bytes()
            self.assertEqual(json.loads(original), result)
            with mock.patch.object(review, "guard_canonical_baseline"), \
                    mock.patch.object(review, "write_gallery", side_effect=OSError("gallery refresh failed")), \
                    mock.patch.object(resources.BoundedCubeSource, "runtime_expired", return_value=True), \
                    mock.patch.object(review, "fetch_raw_swc") as second_fetch:
                with self.assertRaisesRegex(OSError, "gallery refresh failed"):
                    batch.run(root, ["252790"], min_free_bytes=1)
            second_fetch.assert_not_called()
            latest = json.loads((root / "manifest" / "regional_context_run_20261003.json").read_text())
            self.assertEqual(latest["state"], "terminal_budget_stop")
            self.assertNotEqual(latest["terminal_record"], str(first_record))
            self.assertEqual(json.loads(Path(latest["terminal_record"]).read_text()), latest)
            self.assertEqual(first_record.read_bytes(), original)
            self.assertEqual(len(list((root / "manifest" / "regional_runs").glob("*.json"))), 2)

    def test_failed_replacement_preserves_existing_derived_report_and_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "252790" / "014"
            folder.mkdir(parents=True)
            (root / "manifest").mkdir()
            nii = folder / "context.nii.gz"
            nii.write_bytes(b"existing validated derived pixels")
            provenance = {"derived_context": {"status": "derived_partial"}, "UID": "252790|014.swc"}
            (folder / "provenance.json").write_text(json.dumps(provenance))
            previous = root / "manifest" / "derived_context_pilot_252790_014.json"
            previous.write_text('{"status":"derived_partial","previous":"validated"}')
            before = previous.read_bytes()
            frame = pd.DataFrame([{"SampleID":"252790","NeuronID":"014.swc","project_id":"Monkey"}])
            rows = review.parse_swc("1 1 100 100 100 1 -1\n")
            failed = {"status":"context_unavailable","reason":"central source unavailable","loaded":[]}
            with mock.patch.object(review,"load_manifest",return_value=frame), \
                    mock.patch.object(review,"fetch_raw_swc",return_value=(rows,"hash","url","raw")), \
                    mock.patch.object(review,"derive_highres_context",return_value=(None,failed)) as derive:
                review.pilot_derived_context("252790","014.swc",root,
                    max_cubes=batch.FIELD_MAX_CUBES, max_source_bytes=batch.FIELD_MAX_SOURCE_BYTES)
            self.assertEqual(derive.call_args.kwargs["max_cubes"], batch.FIELD_MAX_CUBES)
            self.assertEqual(derive.call_args.kwargs["max_source_bytes"], batch.FIELD_MAX_SOURCE_BYTES)
            self.assertEqual(previous.read_bytes(),before)
            self.assertEqual(nii.read_bytes(),b"existing validated derived pixels")
            self.assertEqual(json.loads((folder / "provenance.json").read_text()),provenance)
            self.assertTrue((root / "manifest" / "derived_context_attempt_252790_014.json").is_file())

    def test_changed_raw_swc_cannot_relabel_an_existing_soma_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "252790" / "014"
            folder.mkdir(parents=True)
            (folder / "soma.nii.gz").write_bytes(b"old raw SWC soma crop")
            path = folder / "provenance.json"
            path.write_text(json.dumps({"swc_sha256":"old hash", "native_soma_x_um":100}))
            before = path.read_bytes()
            frame = pd.DataFrame([{"SampleID":"252790","NeuronID":"014.swc","project_id":"Monkey"}])
            rows = review.parse_swc("1 1 200 100 100 1 -1\n")
            with mock.patch.object(review,"load_manifest",return_value=frame), \
                    mock.patch.object(review,"fetch_raw_swc",return_value=(rows,"new hash","url","raw")), \
                    mock.patch.object(review,"derive_highres_context") as derive:
                with self.assertRaisesRegex(RuntimeError,"lineage"):
                    review.pilot_derived_context("252790","014.swc",root)
            derive.assert_not_called()
            self.assertEqual(path.read_bytes(),before)

    def test_batch_lock_prevents_a_second_process(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "regional.lock"
            code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                    "from pathlib import Path;from regional_context_batch_20261003 import exclusive_batch;"
                    "\ntry:\n with exclusive_batch(Path(sys.argv[2])): print('acquired')"
                    "\nexcept OSError: print('held')")
            directory = str(Path(batch.__file__).parent)
            with batch.exclusive_batch(lock):
                child = subprocess.run([sys.executable,"-B","-c",code,directory,str(lock)],
                                       capture_output=True,text=True,timeout=20)
            self.assertEqual(child.returncode,0,child.stderr)
            self.assertEqual(child.stdout.strip(),"held")
            with batch.exclusive_batch(lock):
                pass  # OS ownership was released; the retained file does not block resume.


if __name__ == "__main__":
    unittest.main()
