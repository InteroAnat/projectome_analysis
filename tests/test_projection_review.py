"""Fail-closed saved-map review, including real builder output and python -O."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
import build_projection_maps as builder
import review_projection_run as reviewer


class SavedProjectionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference.nii.gz"
        image = nib.Nifti1Image(np.zeros((4, 4, 4)), np.diag([2., 2., 2., 1.]))
        image.header.set_xyzt_units("mm")
        nib.save(image, self.reference)
        self.mask = self.root / "mask.nii.gz"
        data = np.zeros((4, 4, 4))
        data[:2] = 1
        image = nib.Nifti1Image(data, image.affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, self.mask)
        self.rows = []
        # Region IAL has unequal per-animal neuron counts: A mean 2 mm,
        # B mean 6 mm. Equal-animal mean is 4 mm, pooled mean is 14/3 mm.
        for animal, number, region, endpoint in [("A", "001", "IAL", 1),
                                                ("B", "002", "IAL", 3),
                                                ("B", "003", "IAL", 3),
                                                ("A", "004", "IDD5", 2)]:
            source = self.root / f"{number}.swc"
            source.write_text(f"1 1 0 0 0 1 -1\n2 2 {endpoint} 0 0 1 1\n", encoding="utf-8")
            self.rows.append(dict(AnimalID=animal, SampleID=f"00{animal}", NeuronID=source.name,
                                  Subregion=region, SWCPath=str(source), SWCSHA256=builder.sha256(source),
                                  ReferenceSHA256=builder.sha256(self.reference),
                                  CoordinateFrame="atlas_index_um", IndexScaleUm="1;1;1",
                                  AnatomyStatus="synthetic", RegistrationStatus="synthetic"))
        self.manifest = self.root / "manifest.csv"
        pd.DataFrame(self.rows).to_csv(self.manifest, index=False)
        self.run_dir = self.root / "run"
        self.output = self.root / "review"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, mask=True):
        with redirect_stdout(io.StringIO()):
            return builder.build(self.manifest, self.reference, self.run_dir,
                                 brain_mask=self.mask if mask else None)

    def save_record(self, record):
        (self.run_dir / "run_provenance.json").write_text(json.dumps(record), encoding="utf-8")

    def read_metrics(self):
        return pd.read_csv(self.run_dir / "per_neuron_measurements.csv",
                           dtype={"SampleID": str, "AnimalID": str, "NeuronID": str})

    def reject(self, pattern=None):
        with self.assertRaisesRegex((ValueError, AssertionError), pattern or ".*"):
            reviewer.review(self.run_dir, self.output)
        self.assertFalse(self.output.exists())

    def replace_map(self, record, entry, kind, *, data=None, affine=None, units="mm"):
        path = self.run_dir / entry[f"{kind}_path"]
        old = nib.load(path)
        image = nib.Nifti1Image(old.get_fdata() if data is None else data,
                              old.affine if affine is None else affine)
        image.header.set_xyzt_units(units)
        nib.save(image, path)
        entry[f"{kind}_sha256"] = builder.sha256(path)
        self.save_record(record)

    def test_masked_builder_run_and_review_provenance(self):
        run = self.build()
        hashes = {p: builder.sha256(p) for p in self.run_dir.rglob("*") if p.is_file()}
        report = reviewer.review(self.run_dir, self.output)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["neurons"], 4)
        self.assertEqual(len(report["all_map_files"]), 10)
        self.assertEqual(report["brain_mask_coverage_status"], "assessed")
        self.assertGreater(report["outside_brain_mask_in_FOV_length_mm"], 0)
        self.assertEqual(report["reviewer"]["version"], reviewer.REVIEWER_VERSION)
        self.assertEqual(report["reviewer"]["sha256"], builder.sha256(reviewer.__file__))
        self.assertEqual(report["per_neuron_measurements_sha256"], builder.sha256(self.run_dir / "per_neuron_measurements.csv"))
        self.assertEqual(hashes, {p: builder.sha256(p) for p in hashes})
        ial = next(entry for entry in run["group_maps"] if entry["Subregion"] == "IAL")
        self.assertAlmostEqual(nib.load(self.run_dir / ial["length_path"]).get_fdata().sum(), 4)
        self.assertEqual(json.loads((self.output / "projection_run_readback.json").read_text())["neurons"], 4)

    def test_mask_free_run_reports_coverage_unassessed_and_render_fails_early(self):
        self.build(mask=False)
        with self.assertRaisesRegex(ValueError, "Rendering requires a brain mask"):
            reviewer.review(self.run_dir, self.output, render=True)
        self.assertFalse(self.output.exists())
        report = reviewer.review(self.run_dir, self.output)
        self.assertEqual(report["status"], "passed")
        self.assertIsNone(report["outside_brain_mask_in_FOV_length_mm"])
        self.assertIn("not assessed", report["brain_mask_coverage_status"])

    def test_existing_destination_is_preserved(self):
        self.build()
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("original")
        with self.assertRaises(FileExistsError):
            reviewer.review(self.run_dir, self.output)
        self.assertEqual(sentinel.read_text(), "original")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_duplicate_and_replaced_metrics_identities_fail(self):
        self.build()
        original = self.read_metrics()
        for change in ("duplicate", "missing"):
            with self.subTest(change=change):
                frame = original.copy()
                frame.loc[1, ["SampleID", "NeuronID"]] = (frame.loc[0, ["SampleID", "NeuronID"]].tolist()
                                                          if change == "duplicate" else ["00A", "999.swc"])
                frame.to_csv(self.run_dir / "per_neuron_measurements.csv", index=False)
                self.reject("composite neuron identities")

    def test_metrics_metadata_and_required_columns_fail(self):
        self.build()
        original = self.read_metrics()
        for column in ("AnimalID", "Subregion", "SWCSHA256", "AnatomyStatus", "RegistrationStatus", "coordinate_frame"):
            with self.subTest(column=column):
                frame = original.copy()
                frame.loc[0, column] = "corrupt"
                frame.to_csv(self.run_dir / "per_neuron_measurements.csv", index=False)
                self.reject("differs from manifest")
        original.drop(columns=["in_reference_length_mm"]).to_csv(self.run_dir / "per_neuron_measurements.csv", index=False)
        self.reject("Missing required per-neuron")

    def test_exact_group_region_coverage_rejects_duplicate_and_extra(self):
        original = self.build()
        for mode in ("duplicate", "extra", "missing"):
            with self.subTest(mode=mode):
                record = copy.deepcopy(original)
                if mode == "duplicate":
                    record["group_maps"][1] = copy.deepcopy(record["group_maps"][0])
                elif mode == "extra":
                    record["group_maps"][0]["Subregion"] = "OTHER"
                else:
                    record["group_maps"].pop()
                self.save_record(record)
                self.reject("group regions")

    def test_duplicate_animal_groups_fail(self):
        record = self.build()
        record["animal_maps"][1] = copy.deepcopy(record["animal_maps"][0])
        self.save_record(record)
        self.reject("animal/subregion")

    def test_missing_code_provenance_and_manifest_hash_fail(self):
        original = self.build()
        record = copy.deepcopy(original)
        record["source_code_sha256"].pop("swc_validation.py")
        self.save_record(record)
        self.reject("source code provenance")
        record = copy.deepcopy(original)
        record["manifest_sha256"] = "0" * 64
        self.save_record(record)
        self.reject("manifest hash mismatch")

    def test_source_and_reference_hashes_fail(self):
        original = self.build()
        source = Path(self.rows[0]["SWCPath"])
        original_bytes = source.read_bytes()
        source.write_bytes(original_bytes + b"\n")
        self.reject("SWC source hash mismatch")
        source.write_bytes(original_bytes)
        record = copy.deepcopy(original)
        record["reference_sha256"] = "0" * 64
        self.save_record(record)
        self.reject("Manifest reference hash mismatch")

    def test_map_hash_geometry_pixels_and_density_fail(self):
        original = self.build()
        entry = original["animal_maps"][0]
        path = self.run_dir / entry["length_path"]
        raw = path.read_bytes()
        cases = ("hash", "geometry", "negative", "nonfinite", "density")
        density_path = self.run_dir / entry["density_path"]
        density_raw = density_path.read_bytes()
        for case in cases:
            with self.subTest(case=case):
                path.write_bytes(raw)
                density_path.write_bytes(density_raw)
                record = copy.deepcopy(original)
                entry = record["animal_maps"][0]
                if case == "hash":
                    entry["length_sha256"] = "0" * 64
                    self.save_record(record)
                elif case == "geometry":
                    affine = nib.load(path).affine.copy()
                    affine[0, 3] += 1
                    self.replace_map(record, entry, "length", affine=affine)
                elif case in {"negative", "nonfinite"}:
                    data = nib.load(path).get_fdata()
                    data[0, 0, 0] = -1 if case == "negative" else np.nan
                    self.replace_map(record, entry, "length", data=data)
                else:
                    data = nib.load(density_path).get_fdata() * 2
                    self.replace_map(record, entry, "density", data=data)
                self.reject()

    def test_pooled_neuron_group_map_is_not_equal_animal_mean(self):
        record = self.build()
        entry = next(entry for entry in record["group_maps"] if entry["Subregion"] == "IAL")
        animals = [e for e in record["animal_maps"] if e["Subregion"] == "IAL"]
        pooled = sum(nib.load(self.run_dir / e["length_path"]).get_fdata() * e["selected_neurons"]
                     for e in animals) / 3
        self.replace_map(record, entry, "length", data=pooled)
        self.replace_map(record, entry, "density", data=pooled / 8)
        self.reject()

    def test_length_accounting_and_mask_coverage_fail(self):
        self.build()
        original = self.read_metrics()
        for column in ("selected_axon_length_mm", "in_brain_mask_length_mm"):
            with self.subTest(column=column):
                frame = original.copy()
                frame.loc[0, column] += 1
                frame.to_csv(self.run_dir / "per_neuron_measurements.csv", index=False)
                self.reject()
        original.drop(columns=["in_brain_mask_length_mm"]).to_csv(self.run_dir / "per_neuron_measurements.csv", index=False)
        self.reject("missing coverage")

    def test_python_optimized_process_still_rejects_corrupt_manifest_hash(self):
        record = self.build(mask=False)
        record["manifest_sha256"] = "0" * 64
        self.save_record(record)
        result = subprocess.run([sys.executable, "-B", "-O", str(Path(reviewer.__file__)),
                                 "--run", str(self.run_dir), "--output", str(self.output)],
                                cwd=ROOT, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Preserved manifest hash mismatch", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
