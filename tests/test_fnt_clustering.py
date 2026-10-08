"""Integration regressions for FNT identity, geometry and explicit diagnostics."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_scripts"))
import fnt_dist_clustering as clustering


class FntClusteringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = ["007.swc", "002.swc", "001.swc", "009.swc",
                      "003.swc", "008.swc", "006.swc", "004.swc"]
        self.joined = self.root / "example_joined.fnt"
        self.joined.write_text("".join(f"0 Neuron {n}\n" for n in self.names))
        n = len(self.names)
        values = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                values[i, j] = values[j, i] = (1 if i // 4 == j // 4 else 10) + (i + j) * .01
        self.values = values
        self.dist = self.root / "example_dist.txt"
        pd.DataFrame([(i, j, values[i, j] if i != j else .2)
                      for i in range(n) for j in range(i, n)],
                     columns=["I", "J", "Score"]).to_csv(self.dist, sep="\t", index=False)
        self.types = self.root / "annotations.csv"
        frame = pd.DataFrame({"NeuronID": self.names, "Neuron_Type": ["ITi"] * 4 + ["PT"] * 4,
                              "SampleID": ["A", "B"] * 4})
        frame.iloc[::-1].to_csv(self.types, index=False)
        self.matrix = pd.DataFrame(values, index=self.names, columns=self.names)

    def test_joined_identity_order_beats_annotation_or_folder_order(self):
        raw, types, meta, aligned = clustering.load_data(
            self.dist, self.types, fnt_folder=self.root / "irrelevant", return_metadata=True)
        self.assertEqual(raw.index.tolist(), self.names)
        self.assertEqual(aligned.NeuronID.tolist(), self.names)
        np.testing.assert_array_equal(raw.to_numpy(), self.values)
        self.assertEqual(meta["identity_source"], "joined FNT Neuron markers in file order")
        self.assertEqual(types["007.swc"], "ITi")

    def test_legacy_marker_stems_align_without_silent_relabeling(self):
        stems = [name[:-4] for name in self.names]
        self.joined.write_text("".join(f"0 Neuron {n}\n" for n in stems))
        raw, _, meta, aligned = clustering.load_data(self.dist, self.types, return_metadata=True)
        self.assertEqual(raw.index.tolist(), stems)
        self.assertEqual(aligned.NeuronID.tolist(), self.names)
        self.assertEqual(meta["annotation_identity_source"], "NeuronID_stem")

    def test_missing_or_duplicate_annotation_is_error(self):
        frame = pd.read_csv(self.types)
        frame.iloc[:-1].to_csv(self.types, index=False)
        with self.assertRaisesRegex(ValueError, "Missing annotation identities"):
            clustering.load_data(self.dist, self.types)
        pd.concat([frame, frame.iloc[[0]]]).to_csv(self.types, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate annotation identities"):
            clustering.load_data(self.dist, self.types)

    def test_global_marker_alignment_uses_sample_and_neuron_identity(self):
        global_names = ["251637_001", "252383_001", "252384_001"]
        frame = pd.DataFrame({"SampleID": [252383, 252384, 251637],
                              "NeuronID": ["001.swc"] * 3, "Neuron_Type": ["ITi"] * 3})
        aligned, source = clustering._align_annotations(frame, global_names)
        self.assertEqual(source, "SampleID_NeuronID")
        self.assertEqual(aligned.SampleID.tolist(), [251637, 252383, 252384])

    def test_raw_default_has_no_type_penalty_and_ward_is_rejected(self):
        types = dict(zip(self.names, ["ITi"] * 4 + ["PT"] * 4))
        result = clustering.process_matrix(self.matrix, types)
        np.testing.assert_array_equal(result, self.matrix)
        penalized = clustering.process_matrix(self.matrix, types, mode="raw", use_penalty=True)
        self.assertGreater(penalized.iloc[0, 4], self.matrix.iloc[0, 4])
        with self.assertRaisesRegex(ValueError, "Ward"):
            clustering.compute_linkage(self.matrix, "ward")

    def test_constant_spearman_profile_is_not_imputed(self):
        flat = pd.DataFrame(np.zeros((3, 3)), index=list("abc"), columns=list("abc"))
        with self.assertRaisesRegex(ValueError, "Constant"):
            clustering.process_matrix(flat, {}, mode="spearman-profile")

    def test_c_index_is_bounded_and_records_duplicate_realized_cuts(self):
        n = len(self.names)
        flat = self.matrix.copy()
        flat.iloc[:, :] = 1.0
        np.fill_diagonal(flat.values, 0)
        tree = clustering.compute_linkage(flat)
        metrics = clustering.c_index_diagnostics(flat, tree, max_k=65)
        self.assertEqual(metrics.requested_k.max(), n - 1)
        self.assertTrue(metrics.duplicate_realized_k.iloc[1:].all())
        self.assertTrue(metrics.c_index.isna().all())

    def test_noninteractive_cli_exports_auditable_exploratory_result(self):
        before = hashlib.sha256(self.types.read_bytes()).hexdigest()
        out = self.root / "outputs"
        metadata = clustering.main(["--dist-file", str(self.dist), "--type-file", str(self.types),
                                    "--output-dir", str(out), "--no-plots", "--min-k", "2",
                                    "--max-k", "3", "--repeats", "5", "--k", "2"])
        self.assertEqual(before, hashlib.sha256(self.types.read_bytes()).hexdigest())
        self.assertEqual(metadata["selected_k"], 2)
        self.assertFalse(metadata["type_penalty"])
        self.assertEqual(metadata["mode"], "raw")
        self.assertEqual(metadata["linkage"], "average")
        self.assertIn("exploratory", metadata["k_status"])
        for name in ["clustering_diagnostics.csv", "cluster_assignments.csv", "candidate_cluster_assignments.csv",
                     "cluster_stability_k2.csv", "group_holdout_diagnostics.csv", "clustering_metadata.json"]:
            self.assertTrue((out / name).is_file(), name)
        saved = json.loads((out / "clustering_metadata.json").read_text())
        self.assertEqual(saved["selected_k_source"], "explicit")
        self.assertEqual(saved["selected_realized_k"], 2)
        self.assertIn("scikit_learn", saved["software_versions"])
        self.assertIn("clustering_validation.py", saved["source_sha256"])
        self.assertEqual(saved["annotation_sheet"], None)
        self.assertEqual(len(pd.read_csv(out / "cluster_assignments.csv")), 8)
        candidates = pd.read_csv(out / "candidate_cluster_assignments.csv")
        self.assertEqual(candidates.columns.tolist(), ["FNT_NeuronID", "k_2", "k_3"])
        self.assertEqual(candidates.FNT_NeuronID.tolist(), self.names)


if __name__ == "__main__":
    unittest.main()
