"""Saved-input contracts for additive subject and end-branch MSTIM summaries."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "group_analysis/scripts"))
import summarize_mstim_arm as summary


class SubjectEndBranchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def branch_fixture(self, references=None, selected=3, eligible=2):
        run = self.root / "branches"
        run.mkdir()
        refs = references or {}
        inputs = {}
        for name in ("reference", "atlas", "atlas_key"):
            path = refs.get(name, self.root / name)
            if not path.exists():
                path.write_bytes(name.encode())
            inputs[name] = {"path": str(path), "sha256": summary.sha256(path)}
        swc = self.root / "source.swc"
        swc.write_text("1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n", encoding="utf-8")
        frame = pd.DataFrame([
            {"SampleID": "s", "NeuronID": str(i), "SWCPath": str(swc),
             "SWCSHA256": summary.sha256(swc)}
            for i in range(selected)
        ])
        manifest = self.root / "manifest.csv"
        frame.to_csv(manifest, index=False)
        inputs["manifest"] = {"path": str(manifest), "sha256": summary.sha256(manifest)}
        ledger = run / "input_ledger.csv"
        ledger.write_bytes(manifest.read_bytes())
        entry = {"Subregion": "source", "selected_neurons": selected,
                 "eligible_neurons": eligible, "n_animals": 2 if eligible else 0,
                 "contributing_animals": ["a", "b"] if eligible else []}
        artifacts = {"input_ledger.csv": summary.sha256(ledger)}
        if eligible:
            image_path = run / "length.nii.gz"
            image = nib.Nifti1Image(np.full((2, 2, 2), 6., dtype=np.float32), np.eye(4))
            image.header.set_xyzt_units("mm")
            image.set_qform(np.eye(4), 5)
            image.set_sform(np.eye(4), 5)
            nib.save(image, image_path)
            entry.update(length_path=image_path.name, length_sha256=summary.sha256(image_path))
            artifacts[image_path.name] = entry["length_sha256"]
        record = {"status": "software_verified_descriptive_end_branches", "inputs": inputs,
                  "artifacts": artifacts, "selected_neurons": selected, "eligible_neurons": eligible,
                  "animal_aggregation": "eligible neuron mean within animal/source group",
                  "group_aggregation": "equal mean of animals with at least one eligible neuron; absent/unassessed groups not zero-filled",
                  "length_value_units": "reference-template mm per end-eligible neuron", "group_maps": [entry]}
        record_path = run / "run_provenance.json"
        self.save_json(record_path, record)
        review = {"status": summary.END_BRANCH_REVIEW_STATUS,
                  "run_provenance": {"path": str(record_path), "sha256": summary.sha256(record_path)},
                  "source_hashes_unchanged": True, "input_ledger_exactly_preserved": True,
                  "source_bindings": inputs, "verified_artifacts": artifacts,
                  "selected_neurons": selected, "eligible_neurons": eligible}
        review_path = self.root / "branch_review.json"
        self.save_json(review_path, review)
        return run, review_path, record, review, swc

    def load_branch(self, run, review):
        record = json.loads((run / "run_provenance.json").read_text())
        paths = [Path(record["inputs"][name]["path"]) for name in ("reference", "atlas", "atlas_key")]
        return summary.reviewed_end_branches(run, review, *paths)

    def repin(self, run, review_path, record, review):
        self.save_json(run / "run_provenance.json", record)
        review["run_provenance"]["sha256"] = summary.sha256(run / "run_provenance.json")
        self.save_json(review_path, review)

    def test_default_subject_and_optional_pair_cli(self):
        flags = ["--" + name for name in (
            "warp-receipt", "statistic", "normalized-anatomy", "endpoint-run",
            "endpoint-review", "axon-run", "axon-review", "label-map", "output")]
        arguments = [item for flag in flags for item in (flag, "placeholder")]
        parsed = summary.argument_parser().parse_args(arguments)
        self.assertEqual(parsed.subject, "cm032")
        self.assertIsNone(parsed.end_branch_run)
        self.assertEqual(summary.argument_parser().parse_args(arguments + ["--subject", "cm033"]).subject, "cm033")
        with self.assertRaisesRegex(ValueError, "supplied together"):
            summary.main(argparse.Namespace(subject="cm033", end_branch_run=self.root, end_branch_review=None))

    def test_review_binds_sources_and_keeps_conditional_mean(self):
        run, review, _, _, _ = self.branch_fixture()
        record, paths = self.load_branch(run, review)
        entry = record["group_maps"][0]
        self.assertEqual(entry["n_selected"], 3)
        self.assertEqual(entry["n_computable"], 2)
        self.assertTrue(entry["map_available"])
        self.assertIn(run / "input_ledger.csv", paths)
        sums = summary.covered_projection(np.array([6., 6.]), np.array([1, 1]), np.ones(2))
        fields = summary.projection_fields("end_branch", entry, float(sums[1]), 2)
        self.assertEqual(fields["end_branch_length_mm_per_eligible_neuron_in_coverage"], 12.)
        self.assertEqual(fields["end_branch_eligible_neurons"], 2)
        self.assertEqual(fields["end_branch_contributing_animals"], 2)

    def test_zero_eligible_and_uncovered_remain_missing(self):
        run, review, _, _, _ = self.branch_fixture(eligible=0)
        record, _ = self.load_branch(run, review)
        entry = record["group_maps"][0]
        self.assertFalse(entry["map_available"])
        self.assertIsNone(summary.projection_fields("end_branch", entry, 0., 10)["end_branch_length_mm_per_eligible_neuron_in_coverage"])
        entry.update(map_available=True, n_computable=2, n_animals=2)
        self.assertIsNone(summary.projection_fields("end_branch", entry, 8., 0)["end_branch_length_mm_per_eligible_neuron_in_coverage"])
        self.assertEqual(summary.projection_fields("end_branch", entry, 0., 10)["end_branch_length_mm_per_eligible_neuron_in_coverage"], 0.)

    def test_changed_swc_fails_even_when_run_and_review_still_match(self):
        run, review, _, _, swc = self.branch_fixture()
        swc.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SWC source changed"):
            self.load_branch(run, review)

    def test_custom_status_run_hash_and_source_binding_fail_closed(self):
        run, path, record, review, _ = self.branch_fixture()
        for change in (lambda r: r.update(status="passed"),
                       lambda r: r["run_provenance"].update(sha256="0" * 64),
                       lambda r: r.update(source_bindings={})):
            copy = json.loads(json.dumps(review))
            change(copy)
            self.save_json(path, copy)
            with self.assertRaises(ValueError):
                self.load_branch(run, path)

    def test_invalid_denominator_or_unavailable_map_fails_even_after_repin(self):
        run, path, record, review, _ = self.branch_fixture()
        for change in (lambda e: e.update(eligible_neurons=4),
                       lambda e: e.update(n_animals=0),
                       lambda e: e.pop("length_sha256")):
            copy = json.loads(json.dumps(record))
            change(copy["group_maps"][0])
            self.repin(run, path, copy, review)
            with self.assertRaises(ValueError):
                self.load_branch(run, path)

    def test_subject_visible_in_plot_title_default_unchanged(self):
        axes = np.empty((2, 3), dtype=object)
        for index in np.ndindex(axes.shape):
            axes[index] = MagicMock()
        figure = MagicMock()
        with patch.object(summary.plt, "subplots", return_value=(figure, axes)), \
                patch.object(summary, "matching_planes", return_value=[np.ones((2, 2))] * 3):
            reference = MagicMock()
            reference.get_fdata.return_value = np.ones((2, 2, 2))
            for subject in ("cm032", "cm033"):
                summary.plot_qc(reference, np.ones((2, 2, 2)), np.ones((2, 2, 2)),
                                np.ones((2, 2, 2)), self.root / "plot.png", subject=subject)
                self.assertIn(subject.upper(), figure.suptitle.call_args.args[0])

    def test_cm033_requires_genuine_independent_chain_binding(self):
        outputs = {}
        for name in ("contrast", "T", "analysis_coverage"):
            path = self.root / name
            path.write_bytes(name.encode())
            outputs[name] = {"path": str(path), "sha256": summary.sha256(path)}
        source = self.root / "native"
        source.write_bytes(b"actual source")
        path = self.root / "chain.json"
        receipt = {"status": summary.CM033_CHAIN_STATUS, "subject": "cm033", "outputs": outputs,
                   "source_sha256": {str(source): summary.sha256(source)}}
        self.save_json(path, receipt)
        review_path = self.root / "warp_review.json"
        review = {"status": summary.CM033_REVIEW_STATUS,
                  "chain_provenance": {"path": str(path), "sha256": summary.sha256(path)},
                  "source_sha256": receipt["source_sha256"],
                  "output_sha256": {key: value["sha256"] for key, value in outputs.items()},
                  "anatomical_acceptance": False}
        self.save_json(review_path, review)
        self.assertEqual(summary.reviewed_warp(path, "cm033", review_path)["outputs"]["reproduced_contrast"], outputs["contrast"])
        with self.assertRaisesRegex(ValueError, "genuine completed"):
            summary.reviewed_warp(path, "cm033")
        for change in (lambda r: r.update(status="passed"),
                       lambda r: r["chain_provenance"].update(sha256="0" * 64),
                       lambda r: r.update(source_sha256={}),
                       lambda r: r["output_sha256"].update(T="0" * 64),
                       lambda r: r.update(anatomical_acceptance=True)):
            copy = json.loads(json.dumps(review))
            change(copy)
            self.save_json(review_path, copy)
            with self.assertRaises(ValueError):
                summary.reviewed_warp(path, "cm033", review_path)
        self.save_json(review_path, review)
        source.write_bytes(b"changed actual source")
        with self.assertRaisesRegex(ValueError, "source changed"):
            summary.reviewed_warp(path, "cm033", review_path)

    @unittest.skipUnless(sys.platform == "win32", "Native Windows WSL-drive transport contract")
    def test_wsl_receipt_transport_preserves_original_binding_strings(self):
        def wsl(path):
            text = str(path.resolve()).replace("\\", "/")
            return "/mnt/" + text[0].lower() + text[2:]

        outputs = {}
        for name in ("contrast", "T", "analysis_coverage"):
            path = self.root / name
            path.write_bytes(name.encode())
            outputs[name] = {"path": wsl(path), "sha256": summary.sha256(path)}
        native = self.root / "native"
        native.write_bytes(b"native source payload")
        bindings = {wsl(native): summary.sha256(native)}
        chain_path = self.root / "chain.json"
        self.save_json(chain_path, {"status": summary.CM033_CHAIN_STATUS, "subject": "cm033",
                                    "source_sha256": bindings, "outputs": outputs})
        original_chain_bytes = chain_path.read_bytes()
        review_path = self.root / "review.json"
        self.save_json(review_path, {
            "status": summary.CM033_REVIEW_STATUS,
            "chain_provenance": {"path": wsl(chain_path), "sha256": summary.sha256(chain_path)},
            "source_sha256": bindings,
            "output_sha256": {key: value["sha256"] for key, value in outputs.items()},
            "anatomical_acceptance": False})
        original_review_bytes = review_path.read_bytes()
        adapted = summary.reviewed_warp(chain_path, "cm033", review_path)
        self.assertEqual(adapted["source_sha256"], bindings)
        for key in outputs:
            self.assertEqual(Path(adapted["outputs"][key]["path"]).resolve(), (self.root / key).resolve())
            self.assertEqual(adapted["outputs"][key]["sha256"], outputs[key]["sha256"])
        self.assertEqual(chain_path.read_bytes(), original_chain_bytes)
        self.assertEqual(review_path.read_bytes(), original_review_bytes)
        self.assertEqual(summary.local_source_path("/mnt/d/file.nii"), Path("D:/file.nii"))
        for unsupported in ("/home/user/source.nii", "/mnt/network/source", "/mnt/d", "/mnt/d/../source", "/mnt/d//source"):
            with self.assertRaisesRegex(ValueError, "Unsupported Unix"):
                summary.local_source_path(unsupported)

    def test_small_saved_inputs_default_and_cm033_all_six_levels(self):
        base = self.root / "atlas/NMT_v2.1_sym/NMT_v2.1_sym"
        base.mkdir(parents=True)
        reference = base / "NMT_v2.1_sym_SS.nii.gz"
        atlas = base / "ARM_in_NMT_v2.1_sym.nii.gz"
        key = self.root / "atlas/ARM_key_all.txt"
        atlas.write_bytes(b"bound catalog fixture")
        key.write_bytes(b"bound key fixture")
        image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
        image.header.set_xyzt_units("mm")
        image.set_qform(np.eye(4), 5)
        image.set_sform(np.eye(4), 5)
        for path in (reference, self.root / "effect.nii", self.root / "statistic.nii", self.root / "coverage.nii", self.root / "anatomy.nii"):
            nib.save(image, path)
        receipt_path = self.root / "warp.json"
        self.save_json(receipt_path, {
            "status": "software_reproduction_passed_registration_unaccepted", "source_sha256": {},
            "spm_contrast": "test signed contrast", "outputs": {
                "analysis_coverage": {"path": str(self.root / "coverage.nii"), "sha256": summary.sha256(self.root / "coverage.nii")},
                "reproduced_contrast": {"path": str(self.root / "effect.nii"), "sha256": summary.sha256(self.root / "effect.nii")}}})
        args = argparse.Namespace(warp_receipt=receipt_path, statistic=self.root / "statistic.nii",
                                  normalized_anatomy=self.root / "anatomy.nii", label_map=self.root / "labels.csv")
        args.label_map.write_bytes(b"reviewed source names")
        for kind in ("endpoint", "axon"):
            run = self.root / kind
            run.mkdir()
            (run / "manifest.csv").write_bytes(b"preserved identities")
            field = "count" if kind == "endpoint" else "length"
            path = run / "map.nii.gz"
            nib.save(image, path)
            record = {"reference_sha256": summary.sha256(reference), "preserved_manifest": "manifest.csv",
                      "group_maps": [{"Subregion": "source", "n_selected": 3, "n_computable": 2,
                                      "n_animals": 2, field + "_path": path.name, field + "_sha256": summary.sha256(path)}]}
            self.save_json(run / "run_provenance.json", record)
            review = self.root / (kind + "_review.json")
            self.save_json(review, {"status": "passed", "run_provenance_sha256": summary.sha256(run / "run_provenance.json")})
            setattr(args, kind + "_run", run)
            setattr(args, kind + "_review", review)
        target_rows = [{"Level": level, "ARMIndex": 1, "TargetID": str(level) + ":1", "TargetStatus": "mapped"} for level in range(1, 7)]
        target_rows += [{"Level": level, "ARMIndex": 2, "TargetID": str(level) + ":2", "TargetStatus": "mapped"} for level in range(1, 7)]
        labels = np.ones(8, dtype=int)
        labels[-1] = 2
        source_receipt = {"atlas_sha256": summary.sha256(atlas), "atlas_key_sha256": summary.sha256(key),
                          "names": [{"Subregion": "source", "ARMIndex": 1, "ARMFullName": "full source", "Hemisphere": "L", "SourceARMStatus": "mapped"}]}
        branches, branch_review, _, _, _ = self.branch_fixture({"reference": reference, "atlas": atlas, "atlas_key": key})
        with patch.object(summary, "ROOT", self.root), \
                patch.object(summary, "catalog", return_value=(pd.DataFrame(target_rows), [labels] * 6)), \
                patch.object(summary, "arm_source_labels", return_value=({"source": "full source (Left)"}, source_receipt)), \
                patch.object(summary, "plot_qc", return_value={}) as plot:
            args.output = self.root / "cm032_output"
            summary.main(args)
            old = pd.read_csv(args.output / "cm032_projectome_common_coverage.csv")
            self.assertEqual(len(old), 12)
            self.assertEqual(set(old.subject), {"cm032"})
            np.testing.assert_array_equal(old.candidate_ends_per_eligible_neuron_in_coverage, [7., 1.] * 6)
            np.testing.assert_array_equal(old.axon_length_mm_per_selected_neuron_in_coverage, [7., 1.] * 6)
            self.assertNotIn("end_branch_eligible_neurons", old)
            args.subject = "cm033"
            receipt = json.loads(receipt_path.read_text())
            receipt.update(status=summary.CM033_CHAIN_STATUS, subject="cm033",
                           limitations=["Provisional bridge; historical payload identity and physical laterality unaccepted"])
            receipt["outputs"]["contrast"] = receipt["outputs"].pop("reproduced_contrast")
            receipt["outputs"]["T"] = {"path": str(args.statistic), "sha256": summary.sha256(args.statistic)}
            receipt["source_sha256"] = {str(reference): summary.sha256(reference)}
            self.save_json(receipt_path, receipt)
            args.warp_review = self.root / "cm033_warp_review.json"
            self.save_json(args.warp_review, {
                "status": summary.CM033_REVIEW_STATUS,
                "chain_provenance": {"path": str(receipt_path), "sha256": summary.sha256(receipt_path)},
                "source_sha256": receipt["source_sha256"],
                "output_sha256": {key: receipt["outputs"][key]["sha256"] for key in ("contrast", "T", "analysis_coverage")},
                "anatomical_acceptance": False})
            args.end_branch_run, args.end_branch_review = branches, branch_review
            args.output = self.root / "cm033_output"
            summary.main(args)
            new = pd.read_csv(args.output / "cm033_projectome_common_coverage.csv")
            self.assertEqual(set(new.subject), {"cm033"})
            np.testing.assert_array_equal(new.candidate_ends_per_eligible_neuron_in_coverage, old.candidate_ends_per_eligible_neuron_in_coverage)
            np.testing.assert_array_equal(new.end_branch_length_mm_per_eligible_neuron_in_coverage, [42., 6.] * 6)
            self.assertTrue(new.end_branch_eligible_neurons.eq(2).all())
            self.assertEqual(plot.call_args.kwargs["subject"], "cm033")
            provenance = json.loads((args.output / "integration_provenance.json").read_text())
            self.assertEqual(provenance["subject"], "cm033")
            self.assertNotIn("CM033", provenance)
            self.assertFalse(provenance["anatomical_acceptance"])
            with self.assertRaises(FileExistsError):
                summary.main(args)


if __name__ == "__main__":
    unittest.main()
