"""Independent read-only checks of the pinned ARM label/display contract.

These checks validate resource correspondence, not neuron registration or anatomy.
No production modules are imported and no source data are changed.
"""
from pathlib import Path
import csv
import hashlib
import re
import unittest

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
KEY = ROOT / "atlas/ARM_key_all.txt"
ATLAS = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
REFERENCE = ROOT / "atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz"


class ARMLabelContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with KEY.open(encoding="utf-8-sig", newline="") as stream:
            cls.rows = list(csv.DictReader(stream, delimiter="\t"))
        cls.key = {int(row["Index"]): row for row in cls.rows}
        cls.image = nib.load(ATLAS)
        cls.data = np.asanyarray(cls.image.dataobj)

    def test_exact_resources_and_grid(self):
        for path, expected in (
            (KEY, "be343eacb2418f2493f740f4039adf068c110a3cb71465645cedadcf2866c972"),
            (ATLAS, "8f7f1dec9fe6ccd1b6c8fce45e3b2cbc7d542ce4a4256500ee562e7e25ddbac7"),
            (REFERENCE, "9e37a94c4b9e5865aabb9fd3b51dcc3b2cb16f3daed39c967e68acf16cf92bee"),
        ):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        reference = nib.load(REFERENCE)
        self.assertEqual(self.image.shape, (256, 312, 200, 1, 6))
        self.assertEqual(reference.shape, self.image.shape[:3])
        self.assertEqual(self.image.header.get_xyzt_units()[0], "mm")
        self.assertEqual(reference.header.get_xyzt_units()[0], "mm")
        np.testing.assert_array_equal(self.image.affine, reference.affine)
        self.assertEqual(int(self.image.header["qform_code"]), 5)
        self.assertEqual(int(self.image.header["sform_code"]), 5)

    def test_unique_exact_names_and_safe_independent_ids(self):
        self.assertEqual(len(self.rows), 1124)
        for field in ("Index", "Abbreviation", "Full_Name"):
            self.assertEqual(len({row[field] for row in self.rows}), 1124)
        tokens = []
        for row in self.rows:
            self.assertEqual(row["Abbreviation"][:3], row["Full_Name"][:3])
            self.assertIn(row["Full_Name"][:3], ("CL_", "CR_", "SL_", "SR_"))
            for level in range(int(row["First_Level"]), int(row["Last_Level"]) + 1):
                token = f"ARM_L{level}_idx{int(row['Index']):04d}"
                self.assertRegex(token, r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
                tokens.append(token.casefold())
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertEqual(self.key[229]["Full_Name"], "CL_granular_insula")
        self.assertEqual(self.key[729]["Full_Name"], "CR_granular_insula")
        self.assertEqual(self.key[49]["Full_Name"], "CL_precentral_operular_area")
        # Real key names contain slashes: a display name is not a filename ID.
        name = self.key[123]["Full_Name"]
        self.assertIsNone(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name))
        self.assertEqual(name.replace("_", " "),
                         "CL rostral inferior parietal lobule area 7b (PFG/PF)")

    def test_actual_level_six_has_lossless_key_correspondence(self):
        observed = set(map(int, np.unique(self.data[:, :, :, 0, 5]))) - {0}
        self.assertEqual(len(observed), 684)
        self.assertTrue(observed <= set(self.key))
        for index in observed:
            row = self.key[index]
            self.assertLessEqual(int(row["First_Level"]), 6)
            self.assertGreaterEqual(int(row["Last_Level"]), 6)

    def test_actual_lower_level_metadata_conflicts_remain_visible(self):
        conflicts = []
        for level in range(1, 7):
            ids, counts = np.unique(self.data[:, :, :, 0, level - 1], return_counts=True)
            for index, count in zip(ids, counts):
                if index == 0:
                    continue
                row = self.key[int(index)]
                if not int(row["First_Level"]) <= level <= int(row["Last_Level"]):
                    conflicts.append((level, int(index), int(count)))
        self.assertEqual(conflicts, [(4, 45, 1641), (4, 545, 1641),
                                     (5, 45, 1641), (5, 545, 1641)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
