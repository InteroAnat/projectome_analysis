"""Actual small ARM outputs verify end-branch selection and conditional means."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
spec = importlib.util.spec_from_file_location("end_branch_builder", ROOT / "group_analysis/scripts/build_axon_end_branch_maps.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def save_image(path, data):
    image = nib.Nifti1Image(data, np.eye(4))
    image.header.set_xyzt_units("mm")
    image.set_sform(np.eye(4), 2)
    image.set_qform(np.eye(4), 2)
    nib.save(image, path)


class EndBranchWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        base = cls.base
        shape = (6, 2, 2)
        cls.reference = base / "reference.nii.gz"
        save_image(cls.reference, np.zeros(shape, dtype=np.float32))
        data = np.ones((*shape, 1, 6), dtype=np.int16)
        data[3:] = 501
        cls.atlas = base / "ARM.nii.gz"
        save_image(cls.atlas, data)
        cls.key = base / "ARM_key.txt"
        pd.DataFrame({"Index": [1, 501], "Abbreviation": ["CL_A", "CR_A"],
            "Full_Name": ["CL_official_left", "CR_official_right"],
            "First_Level": [1, 1], "Last_Level": [6, 6]}).to_csv(cls.key, sep="\t", index=False)
        traces = {
            "a": ("A", "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 2 2 0 0 1 2\n4 2 3 0 0 1 3\n5 2 2 1 0 1 3\n"),
            "b": ("A", "1 1 0 0 0 1 -1\n2 2 4 0 0 1 1\n"),
            "c": ("A", "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 3 2 0 0 1 2\n"),
            "d": ("B", "1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n"),
            "e": ("C", "1 1 0 0 0 1 -1\n2 3 1 0 0 1 1\n"),
        }
        rows = []
        for name, (animal, text) in traces.items():
            source = base / f"{name}.swc"
            source.write_text(text, encoding="utf-8")
            rows.append({"AnimalID": animal, "SampleID": animal, "NeuronID": source.name,
                "Subregion": "ARM6_1_L", "SWCPath": str(source), "SWCSHA256": builder.sha256(source),
                "ReferenceSHA256": builder.sha256(cls.reference), "CoordinateFrame": "nifti_world_mm",
                "IndexScaleUm": "", "AnatomyStatus": "synthetic", "RegistrationStatus": "synthetic",
                "ARMLevel": "6", "ARMIndex": "1", "ARMAbbreviation": "CL_A", "ARMFullName": "CL_official_left",
                "Hemisphere": "L", "SourceARMStatus": "mapped"})
        cls.manifest = base / "ledger.csv"
        pd.DataFrame(rows).to_csv(cls.manifest, index=False)
        cls.output = base / "result"
        with redirect_stdout(io.StringIO()):
            cls.receipt = builder.build(cls.manifest, cls.reference, cls.atlas, cls.key, cls.output, input_root=base)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_distal_chains_exclude_upstream_shaft_and_six_levels_conserve(self):
        self.assertEqual(self.receipt["eligible_neurons"], 3)
        self.assertEqual(self.receipt["original_axon_ends"], 4)
        for level in range(1, 7):
            table = pd.read_csv(self.output / f"ARM_L{level}_axon_end_branch_length_mm.csv").set_index("NeuronID")
            columns = [c for c in table if c.startswith(f"L{level}:")]
            self.assertAlmostEqual(table.loc["a.swc", columns].sum(), 2)
            self.assertAlmostEqual(table.loc["b.swc", columns].sum(), 4)
            self.assertTrue(table.loc[["c.swc", "e.swc"], columns].isna().all().all())

    def test_equal_available_animal_mean_excludes_unassessed_animal(self):
        entry = self.receipt["group_maps"][0]
        self.assertEqual(entry["selected_neurons"], 5)
        self.assertEqual(entry["eligible_neurons"], 3)
        self.assertEqual(entry["contributing_animals"], ["A", "B"])
        data = nib.load(self.output / entry["length_path"]).get_fdata()
        self.assertAlmostEqual(data.sum(), 2)
        self.assertEqual(len(list((self.output / "animal_maps").glob("*.nii.gz"))), 2)
        self.assertEqual(len(list((self.output / "group_maps").glob("*.nii.gz"))), 1)

    def test_qc_identity_sources_and_units_bound(self):
        qc = pd.read_csv(self.output / "per_neuron_measurements.csv").set_index("NeuronID")
        self.assertTrue(qc.loc[["c.swc", "e.swc"], "total_length_mm"].isna().all())
        self.assertFalse(self.receipt["scientific_anatomical_acceptance"])
        for name, expected in self.receipt["artifacts"].items():
            self.assertEqual(builder.sha256(self.output / name), expected)
        self.assertEqual(self.receipt["length_value_units"], "reference-template mm per end-eligible neuron")
        with pd.ExcelFile(self.output / "arm_axon_end_branch_hierarchy_tables.xlsx") as book:
            self.assertEqual(book.sheet_names,
                             ["Neuron_QC", "Official_ARM_Targets"] + [f"L{level}_EndBranch_mm" for level in range(1, 7)])

    def test_existing_output_preserved(self):
        before = builder.sha256(self.output / "run_provenance.json")
        with self.assertRaises(FileExistsError):
            builder.build(Path("missing"), Path("missing"), Path("missing"), Path("missing"), self.output)
        self.assertEqual(builder.sha256(self.output / "run_provenance.json"), before)

    def test_fabricated_source_name_rejected_before_output(self):
        altered = pd.read_csv(self.manifest, dtype=str, keep_default_na=False)
        altered.loc[0, "ARMFullName"] = "invented"
        bad = self.base / "bad.csv"
        altered.to_csv(bad, index=False)
        with self.assertRaisesRegex(ValueError, "official ARM"):
            builder.build(bad, self.reference, self.atlas, self.key, self.base / "bad-output", input_root=self.base)
        self.assertFalse((self.base / "bad-output").exists())


if __name__ == "__main__":
    unittest.main()
