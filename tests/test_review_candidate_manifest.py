"""Spatial preview selection must not promote ID cues, overlap or stale sources."""
import importlib.util
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
spec = importlib.util.spec_from_file_location("candidate_preparer", ROOT / "group_analysis/scripts/prepare_review_candidate_manifest.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class Fixture:
    def __init__(self, root, reference):
        self.root, self.reference = root, reference
        self.sources, self.delivery = root / "map_ready_sources.csv", root / "delivery.json"
        self.triage, self.correction = root / "corrected.csv", root / "correction.json"
        self.main, self.registry = root / "main.csv", root / "registry.csv"
        self.output = root / "preview"
        self.rows, self.triage_rows = [], []
        self.old = root / "old" / "001.swc"
        self.old.parent.mkdir()
        self.old.write_text("1 1 0 0 0 1 -1\n", encoding="utf-8")
        self.main_rows = [dict(AnimalID="a", SampleID="sample001", NeuronID="001.swc", Subregion="HumanINS_L",
            SWCPath=str(self.old), SWCSHA256=adapter.sha256(self.old), ReferenceSHA256=adapter.sha256(reference),
            CoordinateFrame="atlas_index_um", IndexScaleUm="250;250;250", AnatomyStatus="synthetic", RegistrationStatus="synthetic")]

    def add(self, neuron="002.swc", *, current="False", edge="True", distance="0.4", side="L", sample="sample001"):
        folder = self.root / sample
        folder.mkdir(exist_ok=True)
        path = folder / neuron
        path.write_text("1 1 0 0 0 1 -1\n2 2 250 0 0 1 1\n", encoding="utf-8")
        row = dict(sample=sample, neuron_id=neuron, uid=f"{sample}|{neuron}", excluded="False", registry_animal="a",
            animal_aggregation_eligible="True", map_source_graph_status="numeric_graph_verified_all_copies",
            map_selected_source=str(path.relative_to(self.root)), map_selected_sha256=adapter.sha256(path),
            map_source_selection_rule="equal_numeric_graphs", portal_region="Unknown_0", own_mask_side=side,
            exclusive_map_group=f"Other_{side}", Henry_coarse_INS_visual_evidence="False",
            original_source_note="preserve original evidence")
        triage = {**row, adapter.CURRENT:current, adapter.EDGE:edge, adapter.DISTANCE:distance, "nearest_INS_anchor_uid":f"{sample}|100.swc",
            "nearest_INS_anchor_evidence_basis":'["original_portal_INS_label"]',
            "anchor_distance_status":"other_established_anchor_found", "neighboring_numeric_INS_ID_retrieval_cues":'["100.swc"]'}
        self.rows.append(row)
        self.triage_rows.append(triage)
        return row, triage

    def seal(self):
        pd.DataFrame(self.rows).to_csv(self.sources, index=False)
        pd.DataFrame(self.triage_rows).to_csv(self.triage, index=False)
        pd.DataFrame(self.main_rows).to_csv(self.main, index=False)
        pd.DataFrame([dict(fmost_id="sample001", animal="a"), dict(fmost_id="sample002", animal="a")]).to_csv(self.registry,index=False)
        provenance = self.root / "provenance.json"
        provenance.write_text(json.dumps({"inputs":{
            "reference":{"path":str(self.reference),"sha256":adapter.sha256(self.reference)},
            "animal_registry":{"path":str(self.registry),"sha256":adapter.sha256(self.registry)}}}),encoding="utf-8")
        prior = {self.sources.name:adapter.sha256(self.sources), provenance.name:adapter.sha256(provenance)}
        self.delivery.write_text(json.dumps({"artifacts":prior}),encoding="utf-8")
        self.correction.write_text(json.dumps({"input_delivery":{"path":str(self.delivery),"sha256":adapter.sha256(self.delivery)},
            "outputs":{self.triage.name:adapter.sha256(self.triage)},"protected_prior_file_sha256":prior}),encoding="utf-8")

    def run(self):
        return adapter.prepare(self.sources,self.delivery,self.triage,self.correction,self.main,self.registry,
                               self.reference,self.output,input_root=self.root)


class CandidateManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.reference = Path(cls.temp.name) / "reference.nii.gz"
        image = nib.Nifti1Image(np.zeros((256,312,200),dtype=np.uint8), np.diag([.25,.25,.25,1]))
        image.header.set_xyzt_units("mm")
        nib.save(image,cls.reference)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_origin_near_inclusive_boundary_and_id_only_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp),self.reference)
            f.add(edge="True",distance="3.0")  # Origin sensitivity, independent of ID/distance.
            f.add("003.swc",edge="False",distance="2.0",side="R")
            f.add("004.swc",edge="False",distance="2.00001")
            f.add("005.swc",edge="False",distance="nan")
            f.add("006.swc",current="True",edge="True")
            f.seal()
            report=f.run()
            frame=pd.read_csv(f.output/"projection_manifest.csv",dtype=str,keep_default_na=False)
            self.assertEqual(report["selected_neurons"],2)
            self.assertEqual(frame.Subregion.tolist(),["OriginCandidate_L","NearINS_R"])
            self.assertEqual(frame.AnatomyStatus.tolist(),["candidate_only_not_anatomically_accepted"]*2)
            self.assertEqual(frame.portal_region.tolist(),["Unknown_0"]*2)
            self.assertEqual(frame.original_source_note.tolist(),["preserve original evidence"]*2)
            self.assertEqual(frame.Triage_nearest_same_exact_sample_INS_anchor_mm.tolist(),["3.0","2.0"])
            self.assertEqual(report["selection_ledger_status_counts"]["no_spatial_selection_evidence_ID_cue_not_sufficient"],3)

    def test_main_exact_identity_is_ledger_only(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp),self.reference)
            f.add()
            f.add("001.swc")
            f.seal()
            before=f.main.read_bytes()
            report=f.run()
            self.assertEqual((report["selected_neurons"],report["main_overlap_neurons"]),(1,0))
            self.assertEqual(report["selection_ledger_status_counts"]["already_in_main_manifest"],1)
            self.assertEqual(f.main.read_bytes(),before)

    def test_main_source_alias_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp),self.reference)
            row,triage=f.add("001.swc",sample="sample002")
            for item in (row,triage):
                item.update(map_selected_source=str(f.old),map_selected_sha256=adapter.sha256(f.old))
            f.seal()
            with self.assertRaisesRegex(ValueError,"overlaps"):
                f.run()
            self.assertFalse(f.output.exists())

    def test_changed_selected_source_hash_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp),self.reference)
            row,_=f.add()
            f.seal()
            (f.root/row["map_selected_source"]).write_text("1 1 0 0 0 1 -1\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                f.run()
            self.assertFalse(f.output.exists())

    def test_corrected_triage_byte_hash_is_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp),self.reference)
            f.add()
            f.seal()
            f.triage.write_text(f.triage.read_text()+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"triage artifact changed"):
                f.run()
            self.assertFalse(f.output.exists())

    def test_near_anchor_cannot_be_self_or_another_sample(self):
        for anchor in ("sample001|002.swc","otherSample|100.swc"):
            with self.subTest(anchor=anchor),tempfile.TemporaryDirectory() as temp:
                f=Fixture(Path(temp),self.reference)
                _,triage=f.add(edge="False")
                triage["nearest_INS_anchor_uid"]=anchor
                f.seal()
                with self.assertRaisesRegex(ValueError,"another established same-sample"):
                    f.run()
                self.assertFalse(f.output.exists())

    def test_original_source_evidence_cannot_be_replaced_by_triage(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp),self.reference)
            _,triage=f.add()
            triage["portal_region"]="edited_INS"
            f.seal()
            with self.assertRaisesRegex(ValueError,"changed original source evidence"):
                f.run()
            self.assertFalse(f.output.exists())

    def test_no_clobber_and_public_cli_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            f=Fixture(Path(temp),self.reference)
            f.add()
            f.seal()
            command=[sys.executable,"-X","utf8","-B",str(Path(adapter.__file__))]
            for name,path in (("sources",f.sources),("inventory-delivery",f.delivery),("triage",f.triage),
                ("triage-provenance",f.correction),("main-manifest",f.main),("animal-map",f.registry),
                ("reference",f.reference),("output",f.output),("input-root",f.root)):
                command += ["--"+name,str(path)]
            completed=subprocess.run(command,text=True,capture_output=True,check=False)
            self.assertEqual(completed.returncode,0,completed.stderr)
            report=json.loads((f.output/"preparation_provenance.json").read_text())
            self.assertEqual(report["manifest_sha256"],adapter.sha256(f.output/"projection_manifest.csv"))
            old={p.name:p.read_bytes() for p in f.output.iterdir()}
            with self.assertRaises(FileExistsError):
                f.run()
            self.assertEqual(old,{p.name:p.read_bytes() for p in f.output.iterdir()})


if __name__=="__main__":
    unittest.main()
