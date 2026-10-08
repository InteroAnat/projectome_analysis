"""Independent contracts for inventory selection, provenance and publication."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
spec = importlib.util.spec_from_file_location(
    "inventory_preparer_under_test", ROOT / "group_analysis/scripts/prepare_inventory_projection_manifest.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class InventoryFixture:
    def __init__(self, root, reference):
        self.root, self.reference = root, reference
        self.sources = root / "map_ready_sources.csv"
        self.provenance = root / "provenance.json"
        self.delivery = root / "delivery.json"
        self.registry = root / "animals.csv"
        self.output = root / "prepared"
        self.rows = []
        self.animals = {"251637": "936", "252790": "331"}

    def add(self, sample="251637", neuron="001.swc", group="HumanINS_R", **changes):
        folder = self.root / sample
        folder.mkdir(exist_ok=True)
        source = folder / neuron
        source.write_text("1 1 0 0 0 1 -1\n2 2 250 0 0 1 1\n", encoding="utf-8")
        row = dict(sample=sample, neuron_id=neuron, uid=f"{sample}|{neuron}",
            exclusive_map_group=group, registry_animal=self.animals.get(sample, ""),
            excluded="False", animal_aggregation_eligible=str(sample in self.animals),
            map_selected_source=str(source.relative_to(self.root)), map_selected_sha256=adapter.sha256(source),
            map_source_graph_status="numeric_graph_verified_all_copies",
            map_source_selection_rule="deterministic_equal_numeric_graph", portal_region="Unknown_0",
            Henry_coarse_INS_visual_evidence="True" if group.startswith("HumanINS") else "False",
            atlas_INS="True" if group.startswith("AtlasINS") else "False",
            potential_INS_candidate="True" if group.startswith("Candidate") else "False",
            atlas_G="True" if group.startswith("G_exploratory") else "False",
            original_Henry_folder="cube_data/Region_Unknown_0", fine_parcel="unresolved")
        row.update(changes)
        self.rows.append(row)
        return row

    def seal(self):
        pd.DataFrame(self.rows).to_csv(self.sources, index=False)
        pd.DataFrame([dict(fmost_id=s, animal=a) for s, a in self.animals.items()]).to_csv(self.registry, index=False)
        evidence = {"inputs": {"animal_registry": {"path": str(self.registry), "sha256": adapter.sha256(self.registry)},
                               "reference": {"path": str(self.reference), "sha256": adapter.sha256(self.reference)}}}
        self.provenance.write_text(json.dumps(evidence), encoding="utf-8")
        self.delivery.write_text(json.dumps({"artifacts": {
            self.sources.name: adapter.sha256(self.sources), self.provenance.name: adapter.sha256(self.provenance)}}),
            encoding="utf-8")

    def run(self):
        return adapter.prepare(self.sources, self.provenance, self.registry, self.reference, self.output,
                               input_root=self.root, inventory_delivery=self.delivery)

    def manifest(self):
        return pd.read_csv(self.output / "projection_manifest.csv", dtype=str, keep_default_na=False)


class InventoryProjectionManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_temp = tempfile.TemporaryDirectory()
        cls.reference = Path(cls.reference_temp.name) / "reference.nii.gz"
        image = nib.Nifti1Image(np.zeros((256, 312, 200), dtype=np.uint8), np.diag([.25, .25, .25, 1]))
        image.header.set_xyzt_units("mm")
        image.set_qform(image.affine, 5)
        image.set_sform(image.affine, 5)
        nib.save(image, cls.reference)

    @classmethod
    def tearDownClass(cls):
        cls.reference_temp.cleanup()

    def fixture(self, temp):
        return InventoryFixture(Path(temp), self.reference)

    def test_human_unknown_prco_and_overlapping_atlas_evidence_stay_human_once(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            for number, label in enumerate(("Unknown_0", "PrCO_49"), 1):
                f.add(neuron=f"{number:03d}.swc", portal_region=label,
                      atlas_INS="True", potential_INS_candidate="True", atlas_G="True")
            f.seal()
            run = f.run()
            frame = f.manifest()
            self.assertEqual(run["selected_neurons"], 2)
            self.assertEqual(frame.Subregion.tolist(), ["HumanINS_R"] * 2)
            self.assertEqual(frame.EvidenceStratum.tolist(), ["HumanINS"] * 2)
            self.assertEqual(frame.AnatomyStatus.tolist(), ["Henry_visual_coarse_INS"] * 2)
            self.assertEqual(frame.OriginalSourceLabel.tolist(), ["Unknown_0", "PrCO_49"])
            self.assertEqual(frame.fine_parcel.tolist(), ["unresolved"] * 2)
            self.assertFalse(frame.uid.duplicated().any())
            self.assertEqual(set(frame.ReferenceSHA256), {adapter.sha256(self.reference)})
            self.assertTrue(all(adapter.sha256(Path(r.SWCPath)) == r.SWCSHA256 for r in frame.itertuples()))

    def test_missing_animal_and_excluded_identity_remain_only_in_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            f.add()
            f.add(sample="250432", group="AtlasINS_L")
            f.add(neuron="002.swc", excluded="True")
            f.seal()
            run = f.run()
            ledger = pd.read_csv(f.output / "selection_ledger.csv", dtype=str, keep_default_na=False)
            self.assertEqual((run["selection_ledger_rows"], run["selected_neurons"]), (3, 1))
            self.assertEqual(run["selected_animals"], ["936"])
            self.assertEqual(dict(zip(ledger.uid, ledger.mapping_selection_status)), {
                "251637|001.swc": "selected", "250432|001.swc": "missing_established_animal_identity",
                "251637|002.swc": "excluded_source_identity"})
            self.assertEqual(f.manifest().uid.tolist(), ["251637|001.swc"])

    def test_duplicate_exact_identity_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            row = f.add()
            f.rows.append({**row, "exclusive_map_group": "AtlasINS_R", "atlas_INS": "True"})
            f.seal()
            with self.assertRaises(ValueError):
                f.run()
            self.assertFalse(f.output.exists())

    def test_weaker_strata_cannot_overwrite_higher_evidence(self):
        cases = [("AtlasINS_R", {"Henry_coarse_INS_visual_evidence": "True"}),
                 ("Candidate_R", {"atlas_INS": "True"}),
                 ("G_exploratory_R", {"potential_INS_candidate": "True"})]
        for group, changes in cases:
            with self.subTest(group=group), tempfile.TemporaryDirectory() as temp:
                f = self.fixture(temp)
                f.add(group=group, **changes)
                f.seal()
                with self.assertRaises(ValueError):
                    f.run()
                self.assertFalse(f.output.exists())

    def test_changed_swc_hash_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            row = f.add()
            f.seal()
            (f.root / row["map_selected_source"]).write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                f.run()
            self.assertFalse(f.output.exists())

    def test_delivery_binding_rejects_changed_selection_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            f.add()
            f.seal()
            frame = pd.read_csv(f.sources, dtype=str, keep_default_na=False)
            frame.loc[0, "portal_region"] = "post_audit_edited"
            frame.to_csv(f.sources, index=False)
            with self.assertRaises(ValueError):
                f.run()
            self.assertFalse(f.output.exists())

    def test_delivery_binding_rejects_changed_inventory_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            f.add()
            f.seal()
            f.provenance.write_text(f.provenance.read_text() + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                f.run()
            self.assertFalse(f.output.exists())

    def test_changed_registry_and_invalid_animal_identity_fail(self):
        for identity in ("331", "unknown"):
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as temp:
                f = self.fixture(temp)
                f.add()
                f.seal()
                pd.DataFrame([dict(fmost_id="251637", animal=identity)]).to_csv(f.registry, index=False)
                with self.assertRaises(ValueError):
                    f.run()
                self.assertFalse(f.output.exists())

    def test_reused_source_path_cannot_create_second_neuron_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            first = f.add()
            f.add(sample="252790", map_selected_source=first["map_selected_source"])
            f.seal()
            with self.assertRaisesRegex(ValueError, "source path"):
                f.run()
            self.assertFalse(f.output.exists())

    def test_no_overwrite_preserves_prior_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            f.add()
            f.seal()
            f.run()
            original = {p.name: p.read_bytes() for p in f.output.iterdir()}
            with self.assertRaises(FileExistsError):
                f.run()
            self.assertEqual(original, {p.name: p.read_bytes() for p in f.output.iterdir()})

    def test_public_cli_creates_readable_bound_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            f.add(portal_region="PrCO_49")
            f.seal()
            result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(adapter.__file__)),
                "--sources", str(f.sources), "--inventory-provenance", str(f.provenance),
                "--inventory-delivery", str(f.delivery), "--animal-map", str(f.registry),
                "--reference", str(f.reference), "--output", str(f.output), "--input-root", str(f.root)],
                text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 neurons, 1 animals", result.stdout)
            report = json.loads((f.output / "preparation_provenance.json").read_text())
            self.assertEqual(adapter.sha256(f.output / "projection_manifest.csv"), report["manifest_sha256"])
            self.assertEqual(report["inputs"]["inventory_delivery"]["sha256"], adapter.sha256(f.delivery))
            self.assertEqual(f.manifest().OriginalSourceLabel.tolist(), ["PrCO_49"])

    def test_all_audited_input_hashes_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            f = self.fixture(temp)
            f.add()
            f.seal()
            extra = f.root / "Henry_immutable_source.txt"
            extra.write_text("original source", encoding="utf-8")
            evidence = json.loads(f.provenance.read_text())
            evidence["inputs"]["Henry"] = {"path": str(extra), "sha256": adapter.sha256(extra)}
            f.provenance.write_text(json.dumps(evidence), encoding="utf-8")
            delivery = json.loads(f.delivery.read_text())
            delivery["artifacts"][f.provenance.name] = adapter.sha256(f.provenance)
            f.delivery.write_text(json.dumps(delivery), encoding="utf-8")
            extra.write_text("changed source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Audited input changed: Henry"):
                f.run()
            self.assertFalse(f.output.exists())


if __name__ == "__main__":
    unittest.main()
