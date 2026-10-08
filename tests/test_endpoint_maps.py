"""Saved-pixel, atlas-level and provenance contracts for endpoint mapping."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
spec = importlib.util.spec_from_file_location("endpoint_builder", ROOT / "group_analysis/scripts/build_endpoint_maps.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def trace(points, kind=2):
    return "1 1 0 0 0 1 -1\n" + "".join(
        f"{index+2} {kind} {float(x)!r} {float(y)!r} {float(z)!r} 1 1\n"
        for index, (x, y, z) in enumerate(points))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.reference = root / "reference.nii.gz"
        self.atlas_path = root / "atlas.nii.gz"
        self.atlas_key = root / "key.tsv"
        self.hemisphere_mask = root / "hemisphere.nii.gz"
        self.brain_mask = root / "brain.nii.gz"
        self.manifest = root / "manifest.csv"
        self.output = root / "output"
        self.affine = np.diag([2., 2., 2., 1.])
        self.shape = (4, 2, 2)
        self.save(self.reference, np.ones(self.shape, dtype=np.uint8))
        self.atlas = np.zeros((*self.shape, 1, 6), dtype=np.int16)
        keys = []
        for level in range(1, 7):
            self.atlas[:2, ..., 0, level-1] = level
            self.atlas[2:, ..., 0, level-1] = 100 + level
            for index, side in [(level, "CL"), (100+level, "CR")]:
                keys.append(dict(Index=index, Abbreviation=f"{side}_level{level}",
                                 Full_Name=f"{side}_actual_level{level}", First_Level=level, Last_Level=level))
        self.save(self.atlas_path, self.atlas)
        pd.DataFrame(keys).to_csv(self.atlas_key, sep="\t", index=False)
        hemisphere = np.ones(self.shape, dtype=np.uint8)
        hemisphere[:2] = 2  # Explicit label validation infers 2=L and 1=R.
        self.save(self.hemisphere_mask, hemisphere)
        brain = np.ones(self.shape, dtype=np.uint8)
        brain[1] = 0
        self.save(self.brain_mask, brain)
        self.records = []

    def save(self, path, data, *, affine=None, units="mm"):
        image = nib.Nifti1Image(data, self.affine if affine is None else affine)
        image.header.set_xyzt_units(units)
        nib.save(image, path)

    def add(self, text, animal="a", subregion="INS"):
        sample = f"sample{len(self.records)+1:03d}"
        folder = self.root / sample
        folder.mkdir()
        source = folder / "001.swc"
        source.write_text(text, encoding="utf-8")
        self.records.append(dict(AnimalID=animal, SampleID=sample, NeuronID="001.swc", Subregion=subregion,
            OriginalSourceLabel="original_INS", SWCPath=str(source), SWCSHA256=builder.sha256(source),
            ReferenceSHA256=builder.sha256(self.reference), CoordinateFrame="atlas_index_um", IndexScaleUm="1;1;1",
            AnatomyStatus="synthetic_unreviewed", RegistrationStatus="synthetic", ExtraSourceStatus="retained"))

    def run(self, **kwargs):
        pd.DataFrame(self.records).to_csv(self.manifest, index=False)
        arguments = dict(atlas_path=self.atlas_path, atlas_key=self.atlas_key,
                         hemisphere_mask=self.hemisphere_mask, brain_mask=self.brain_mask, input_root=self.root)
        arguments.update(kwargs)
        with redirect_stdout(io.StringIO()):
            return builder.build(self.manifest, self.reference, self.output, **arguments)

    def pixels(self, entry, metric):
        return nib.load(self.output / entry[metric + "_path"]).get_fdata()


class EndpointBuilderTests(unittest.TestCase):
    def test_saved_count_occupancy_density_and_full_leaf_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1.1, 0, 0), (1.2, 0, 0)]) + "4 3 2 0 0 1 1\n")
            run = fixture.run()
            self.assertEqual(run["status"], "software_verified_candidate_endpoints")
            entry = run["animal_maps"][0]
            self.assertEqual((entry["n_selected"], entry["n_computable"]), (1, 1))
            count, occupancy, density = [fixture.pixels(entry, metric) for metric in ("count", "occupancy", "density")]
            self.assertEqual(count[1, 0, 0], 2)
            self.assertEqual(count.sum(), 2)
            self.assertEqual(occupancy[1, 0, 0], 1)
            self.assertEqual(density[1, 0, 0], .25)  # Two tips / eight mm3.
            leaves = [json.loads(line) for line in (fixture.output / "leaf_records.jsonl").read_text().splitlines()]
            self.assertEqual(len(leaves), 3)
            self.assertEqual({leaf["node_type"] for leaf in leaves}, {2, 3})
            self.assertEqual(leaves[0]["source_metadata"]["ExtraSourceStatus"], "retained")
            self.assertEqual(leaves[0]["hemisphere_side"], "L")
            self.assertEqual(leaves[0]["brain_mask_status"], "outside")
            self.assertEqual(leaves[0]["source_sha256"], fixture.records[0]["SWCSHA256"])
            self.assertEqual(leaves[0]["SampleID"], fixture.records[0]["SampleID"])
            self.assertEqual(len(leaves[0]["atlas_levels"]), 6)
            qc = pd.read_csv(fixture.output / "per_neuron_qc.csv")
            self.assertEqual(qc.loc[0, "outside_brain_mask_in_FOV_candidate_count"], 2)
            targets = pd.read_csv(fixture.output / "per_neuron_target_counts.csv")
            self.assertEqual(len(targets), 6)
            self.assertTrue(targets.endpoint_count.eq(2).all())
            self.assertTrue(targets.neurons_with_target_evidence.eq(1).all())
            regional = pd.read_csv(fixture.output / "animal_region_summaries.csv")
            self.assertTrue(regional.frequency_per_computable_neuron.eq(1).all())
            np.testing.assert_allclose(regional.target_volume_mm3, 64)
            self.assertEqual((fixture.output / "input_manifest.csv").read_bytes(), fixture.manifest.read_bytes())
            for metric in ("count", "density", "occupancy"):
                image = nib.load(fixture.output / entry[metric + "_path"])
                self.assertIn(b"Candidate endpoint", image.header["descrip"].tobytes())
                self.assertEqual(builder.sha256(fixture.output / entry[metric + "_path"]), entry[metric + "_sha256"])

    def test_equal_animal_weights_with_unequal_neuron_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), animal="a")
            for _ in range(3):
                fixture.add(trace([(1, 0, 0), (1.1, 0, 0), (1.2, 0, 0)]), animal="b")
            run = fixture.run()
            entry = run["group_maps"][0]
            self.assertEqual(entry["n_animals"], 2)
            self.assertEqual(entry["n_computable"], 4)
            self.assertEqual(fixture.pixels(entry, "count")[1, 0, 0], 2)  # (1 + 3)/2; pooled would be 2.5.
            self.assertEqual(fixture.pixels(entry, "occupancy")[1, 0, 0], 1)

    def test_conditional_denominator_and_zero_computable_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.add("1 1 0 0 0 1 -1\n")
            fixture.add("1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 99 2 0 0 1 2\n")
            fixture.add("1 2 0 0 0 1 -1\n", subregion="unassessed")
            run = fixture.run()
            self.assertEqual((run["n_selected"], run["n_computable"]), (4, 1))
            entry = next(item for item in run["animal_maps"] if item["Subregion"] == "INS")
            self.assertEqual(fixture.pixels(entry, "count")[1, 0, 0], 1)
            self.assertEqual((entry["n_selected"], entry["n_computable"]), (3, 1))
            unavailable = next(item for item in run["animal_maps"] if item["Subregion"] == "unassessed")
            self.assertFalse(unavailable["map_available"])
            self.assertNotIn("count_path", unavailable)
            self.assertEqual(len(run["unassessed_neurons"]), 3)
            self.assertEqual(len(run["unresolved_compartment_neurons"]), 1)
            qc = pd.read_csv(fixture.output / "per_neuron_qc.csv")
            self.assertEqual(len(qc), 4)
            self.assertTrue(qc.biological_innervation_state.eq("unassessed").all())
            self.assertEqual(len(list((fixture.output / "animal_maps").glob("*.nii.gz"))), 3)
            regional = pd.read_csv(fixture.output / "animal_region_summaries.csv")
            blank = regional[regional.Subregion.eq("unassessed")].iloc[0]
            self.assertTrue(pd.isna(blank.frequency_per_computable_neuron))
            self.assertEqual(blank.target_status, "unavailable_no_computable_neurons")
            self.assertFalse(next(item for item in run["group_maps"] if item["Subregion"] == "unassessed")["map_available"])

    def test_fractional_occupancy_is_per_neuron_then_per_animal(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]), animal="a")
            for _ in range(2):
                fixture.add(trace([(1, 0, 0), (1.1, 0, 0)]), animal="b")
            fixture.add(trace([(2, 0, 0)]), animal="b")
            run = fixture.run()
            group = run["group_maps"][0]
            self.assertAlmostEqual(fixture.pixels(group, "count")[1, 0, 0], 7/6, places=6)
            self.assertAlmostEqual(fixture.pixels(group, "occupancy")[1, 0, 0], 5/6, places=6)
            animal_b = next(item for item in run["animal_maps"] if item["AnimalID"] == "b")
            self.assertAlmostEqual(fixture.pixels(animal_b, "occupancy")[1, 0, 0], 2/3, places=6)
            regional = pd.read_csv(fixture.output / "animal_region_summaries.csv")
            left = regional[(regional.AnimalID.eq("b")) & regional.level.eq(1) & regional.target_index.eq(1)].iloc[0]
            self.assertEqual(left.endpoint_count, 4)
            self.assertEqual(left.neurons_with_target_evidence, 2)
            self.assertAlmostEqual(left.frequency_per_computable_neuron, 2/3)

    def test_existing_run_and_long_paths_are_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            fixture.output.mkdir()
            sentinel = fixture.output / "prior.bin"
            sentinel.write_bytes(b"retain")
            with self.assertRaises(FileExistsError):
                fixture.run()
            self.assertEqual(sentinel.read_bytes(), b"retain")
            fixture.output = fixture.root / "uncreated"
            fixture.records[0]["Subregion"] = "x" * 240
            with self.assertRaisesRegex(ValueError, "240-character"):
                fixture.run()
            self.assertFalse(fixture.output.exists())

    def test_actual_levels_unknowns_hemisphere_conflicts_and_half_open_faces(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.atlas[1, 0, 0, 0, 1] = 77  # Unknown label at actual level 2.
            fixture.atlas[1, 0, 0, 0, 2] = 0
            fixture.atlas[0, 0, 0, 0, 0] = 101  # Coarse-level right label in validated left mask.
            fixture.save(fixture.atlas_path, fixture.atlas)
            fixture.add(trace([(.5, 0, 0), (-.5, 0, 0), (3.5, 0, 0)]))
            run = fixture.run()
            leaves = [json.loads(line) for line in (fixture.output / "leaf_records.jsonl").read_text().splitlines()]
            self.assertEqual(leaves[0]["voxel"], [1, 0, 0])
            self.assertEqual(leaves[0]["atlas_levels"][0]["index"], 1)
            self.assertEqual(leaves[0]["atlas_levels"][1]["target_status"], "unmapped_label")
            self.assertIsNone(leaves[0]["atlas_levels"][1]["abbreviation"])
            self.assertEqual(leaves[0]["atlas_levels"][2]["target_status"], "zero_unassigned")
            self.assertEqual(leaves[1]["voxel"], [0, 0, 0])
            self.assertTrue(leaves[1]["atlas_levels"][0]["hemisphere_conflict"])
            self.assertIsNone(leaves[2]["voxel"])
            self.assertTrue(all(target["target_status"] == "out_of_FOV" for target in leaves[2]["atlas_levels"]))
            self.assertEqual(run["accounting"]["outside_reference_candidates"], 1)
            self.assertEqual(fixture.pixels(run["animal_maps"][0], "count").sum(), 2)
            regional = pd.read_csv(fixture.output / "animal_region_summaries.csv")
            self.assertEqual(regional[regional.target_status.eq("out_of_FOV")].endpoint_count.sum(), 6)

    def test_every_endpoint_outside_still_has_computable_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(10, 0, 0)]))
            run = fixture.run(brain_mask=None)
            self.assertEqual(run["n_computable"], 1)
            self.assertTrue(run["animal_maps"][0]["map_available"])
            self.assertEqual(fixture.pixels(run["animal_maps"][0], "count").sum(), 0)
            regional = pd.read_csv(fixture.output / "animal_region_summaries.csv")
            self.assertTrue(regional.target_status.eq("out_of_FOV").all())
            self.assertTrue(regional.frequency_per_computable_neuron.eq(1).all())

    def test_hash_binding_and_output_collision_fail_before_destination(self):
        for change in ("source_hash", "reference_hash", "case_collision", "delimiter_collision"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(Path(temp))
                fixture.add(trace([(1, 0, 0)]))
                if change == "source_hash":
                    fixture.records[0]["SWCSHA256"] = "0" * 64
                elif change == "reference_hash":
                    fixture.records[0]["ReferenceSHA256"] = "0" * 64
                else:
                    fixture.add(trace([(2, 0, 0)]), animal="A" if change == "case_collision" else "a_space-S_region-x", subregion="INS" if change == "case_collision" else "y")
                    if change == "delimiter_collision":
                        fixture.records[0]["Subregion"] = "x_space-S_region-y"
                with self.assertRaises(ValueError):
                    fixture.run(space="S")
                self.assertFalse(fixture.output.exists())

    def test_mask_geometry_semantics_units_and_atlas_shape_fail_closed(self):
        for change in ("brain_geometry", "hemisphere_semantics", "reference_units", "atlas_shape", "atlas_units"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                fixture = Fixture(Path(temp))
                fixture.add(trace([(1, 0, 0)]))
                if change == "brain_geometry":
                    affine = fixture.affine.copy()
                    affine[0, 3] = 1
                    fixture.save(fixture.brain_mask, np.ones(fixture.shape, dtype=np.uint8), affine=affine)
                elif change == "hemisphere_semantics":
                    fixture.save(fixture.hemisphere_mask, np.ones(fixture.shape, dtype=np.uint8))
                elif change == "reference_units":
                    fixture.save(fixture.reference, np.ones(fixture.shape, dtype=np.uint8), units="micron")
                    fixture.records[0]["ReferenceSHA256"] = builder.sha256(fixture.reference)
                elif change == "atlas_units":
                    fixture.save(fixture.atlas_path, fixture.atlas, units="micron")
                else:
                    fixture.save(fixture.atlas_path, fixture.atlas[..., 0, 0])
                with self.assertRaises(ValueError):
                    fixture.run()
                self.assertFalse(fixture.output.exists())

    def test_bad_graph_records_failed_run_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add("1 1 0 0 0 1 -1\n2 2 1 0 0 1 9\n")
            with self.assertRaises(ValueError):
                fixture.run()
            failed = json.loads((fixture.output / "run_provenance.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertIn("missing parent", failed["error"])
            self.assertEqual(builder.sha256(fixture.records[0]["SWCPath"]), fixture.records[0]["SWCSHA256"])

    def test_concurrent_map_publication_does_not_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            link = builder.os.link
            created = []

            def concurrent(source, destination):
                if not created:
                    Path(destination).write_bytes(b"concurrent prior bytes")
                    created.append(Path(destination))
                return link(source, destination)

            with patch.object(builder.os, "link", side_effect=concurrent), self.assertRaises(FileExistsError):
                fixture.run()
            self.assertEqual(created[0].read_bytes(), b"concurrent prior bytes")
            self.assertFalse(list((fixture.output / "animal_maps").glob(".endpoint-map-*")))
            self.assertEqual(json.loads((fixture.output / "run_provenance.json").read_text())["status"], "failed")

    def test_public_cli_runs_isolated_synthetic_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = Fixture(Path(temp))
            fixture.add(trace([(1, 0, 0)]))
            pd.DataFrame(fixture.records).to_csv(fixture.manifest, index=False)
            arguments = [sys.executable, "-B", str(ROOT / "group_analysis/scripts/build_endpoint_maps.py")]
            for name in ("manifest", "reference", "atlas_path", "atlas_key", "hemisphere_mask", "brain_mask", "output"):
                arguments.extend(["--" + name.replace("_", "-"), str(getattr(fixture, name))])
            arguments.extend(["--input-root", str(fixture.root)])
            completed = subprocess.run(arguments, capture_output=True, text=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("software_verified_candidate_endpoints", completed.stdout)
            run = json.loads((fixture.output / "run_provenance.json").read_text())
            self.assertEqual(run["input_root"], str(fixture.root.resolve()))


if __name__ == "__main__":
    unittest.main()
