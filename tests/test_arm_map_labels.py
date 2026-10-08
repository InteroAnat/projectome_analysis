"""Fail closed when scientific map labels disagree with their bound ARM key."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import pandas as pd
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
from render_projection_slices import arm_source_labels, render


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ARMMapLabelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.atlas = self.root / "ARM.nii.gz"
        self.atlas.write_bytes(b"atlas-hash-binding-fixture")
        self.key = self.root / "ARM_key.txt"
        pd.DataFrame({"Index": [101, 102], "Abbreviation": ["CL_IA", "SL_Pi"],
                      "Full_Name": ["CL_agranular_insular_cortex", "SL_pineal_gland"],
                      "First_Level": [6, 6], "Last_Level": [6, 6]}).to_csv(self.key, sep="\t", index=False)
        self.labels = pd.DataFrame([
            {"Subregion": "ARM6_101_L", "ARMLevel": "6", "ARMIndex": "101",
             "ARMAbbreviation": "CL_IA", "ARMFullName": "CL_agranular_insular_cortex",
             "Hemisphere": "L", "DisplayLabel": "CL_agranular_insular_cortex (Left)",
             "AtlasSHA256": sha(self.atlas), "AtlasKeySHA256": sha(self.key),
             "AtlasPath": str(self.atlas), "AtlasKeyPath": str(self.key), "SourceARMStatus": "mapped"},
            {"Subregion": "ARM6_102_L", "ARMLevel": "6", "ARMIndex": "102",
             "ARMAbbreviation": "SL_Pi", "ARMFullName": "SL_pineal_gland",
             "Hemisphere": "L", "DisplayLabel": "SL_pineal_gland (Left)",
             "AtlasSHA256": sha(self.atlas), "AtlasKeySHA256": sha(self.key),
             "AtlasPath": str(self.atlas), "AtlasKeyPath": str(self.key), "SourceARMStatus": "mapped"}])
        self.manifest = self.labels.loc[[0, 1, 0]].reset_index(drop=True)
        self.label_path, self.manifest_path = self.root / "labels.csv", self.root / "manifest.csv"

    def check(self):
        self.labels.to_csv(self.label_path, index=False)
        self.manifest.to_csv(self.manifest_path, index=False)
        return arm_source_labels(self.label_path, self.manifest_path)

    def test_repeated_neurons_join_exact_full_names_and_domains(self):
        names, receipt = self.check()
        self.assertEqual(names["ARM6_101_L"], "CL_agranular_insular_cortex (Left)")
        self.assertEqual(names["ARM6_102_L"], "SL_pineal_gland (Left)")
        self.assertEqual(receipt["atlas_key_sha256"], sha(self.key))

    def test_display_name_disagreement_and_missing_group_fail(self):
        for defect in ("wrong_name", "missing_group", "duplicate_group"):
            with self.subTest(defect=defect):
                original = self.labels.copy()
                if defect == "wrong_name":
                    self.labels.loc[0, "DisplayLabel"] = "HumanINS"
                elif defect == "missing_group":
                    self.labels = self.labels.iloc[:1].copy()
                else:
                    self.labels = pd.concat([self.labels, self.labels.iloc[:1]], ignore_index=True)
                with self.assertRaises(ValueError):
                    self.check()
                self.labels = original

    def test_both_tables_cannot_relabel_a_real_arm_index(self):
        self.labels.loc[0, "ARMFullName"] = "SL_pineal_gland"
        self.labels.loc[0, "DisplayLabel"] = "SL_pineal_gland (Left)"
        self.manifest.loc[self.manifest.ARMIndex.eq("101"), "ARMFullName"] = "SL_pineal_gland"
        self.manifest.loc[self.manifest.ARMIndex.eq("101"), "DisplayLabel"] = "SL_pineal_gland (Left)"
        with self.assertRaisesRegex(ValueError, "actual ARM key"):
            self.check()

    def test_changed_key_hash_is_rejected(self):
        with self.key.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Changed ARM"):
            self.check()

    def test_background_has_explicit_unresolved_name_and_no_invented_parcel(self):
        row = self.labels.iloc[0].copy()
        row.update({"Subregion": "ARM6_0_L", "ARMIndex": "0", "ARMAbbreviation": "",
                    "ARMFullName": "Atlas background", "SourceARMStatus": "zero_unassigned",
                    "DisplayLabel": "Atlas background (Left)"})
        self.labels = pd.DataFrame([row])
        self.manifest = self.labels.copy()
        names, _ = self.check()
        self.assertEqual(names, {"ARM6_0_L": "Atlas background (Left)"})

    def test_conflicting_name_within_run_and_other_atlas_level_fail(self):
        self.manifest.loc[2, "ARMFullName"] = "contradictory"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.check()

    def test_render_separates_mapped_parcel_from_unresolved_location_qc(self):
        background = self.labels.iloc[0].copy()
        background.update({"Subregion": "ARM6_0_L", "ARMIndex": "0", "ARMAbbreviation": "",
                           "ARMFullName": "Atlas background", "SourceARMStatus": "zero_unassigned",
                           "DisplayLabel": "Atlas background (Left)"})
        self.labels = pd.concat([self.labels.iloc[:1], pd.DataFrame([background])], ignore_index=True)
        self.manifest = self.labels.copy()
        self.check()
        affine = np.diag([0.25, 0.25, 0.25, 1.0])
        reference, mask = self.root / "reference.nii.gz", self.root / "mask.nii.gz"
        nib.save(nib.Nifti1Image(np.arange(27, dtype=np.float32).reshape(3, 3, 3), affine), reference)
        nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), affine), mask)
        maps = []
        for index, row in self.labels.iterrows():
            path = self.root / f"map_{index}.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.float32), affine), path)
            maps.append({"AnimalID": "1", "Subregion": row.Subregion, "n_selected": 1,
                         "n_computable": 1, "map_available": True,
                         "density_path": path.name, "density_sha256": sha(path)})
        inputs = {name: {"path": str(path), "sha256": sha(path)} for name, path in
                  (("reference", reference), ("brain_mask", mask), ("atlas", self.atlas),
                   ("atlas_key", self.key), ("manifest", self.manifest_path))}
        provenance = self.root / "run_provenance.json"
        provenance.write_text(json.dumps({"status": "software_verified_candidate_endpoints",
            "inputs": inputs, "affine_mm": affine.tolist(), "shape": [3, 3, 3],
            "space_entity": "NMTtest", "animal_maps": maps, "group_maps": []}), encoding="utf-8")
        readback = self.root / "review.json"
        readback.write_text(json.dumps({"status": "passed", "run_provenance_sha256": sha(provenance)}), encoding="utf-8")
        result = render(self.root, readback, self.root / "figures", metric="endpoint-density",
                        cut_policy="fixed", slice_voxels=[1, 1, 1], label_map=self.label_path)
        self.assertEqual(len(result["figures"]), 1)
        self.assertEqual(result["figures"][0]["source_location_status"], "mapped")
        self.assertFalse(result["include_unresolved_qc"])
        result = render(self.root, readback, self.root / "optional_qc_figures", metric="endpoint-density",
                        cut_policy="fixed", slice_voxels=[1, 1, 1], label_map=self.label_path,
                        include_unresolved_qc=True)
        self.assertEqual(len(result["figures"]), 2)
        by_status = {item["source_location_status"]: item for item in result["figures"]}
        self.assertEqual(by_status["mapped"]["source_groups"], ["ARM6_101_L"])
        self.assertEqual(by_status["unresolved"]["source_groups"], ["ARM6_0_L"])
        self.assertIn("unresolvedQC", by_status["unresolved"]["file"])
        for item in maps:
            self.assertEqual(sha(self.root / item["density_path"]), item["density_sha256"])
        self.manifest.loc[2, "ARMFullName"] = self.manifest.loc[0, "ARMFullName"]
        self.labels["ARMLevel"] = "3"
        self.manifest["ARMLevel"] = "3"
        with self.assertRaisesRegex(ValueError, "level 6"):
            self.check()


if __name__ == "__main__":
    unittest.main()
