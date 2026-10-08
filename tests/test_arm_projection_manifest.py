"""ARM-only grouping preserves selection/evidence and rejects stale inputs."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('arm_preparer',ROOT/'group_analysis/scripts/prepare_arm_labeled_projection_manifest.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def image(path,data,affine):
    img=nib.Nifti1Image(data,affine);img.header.set_xyzt_units('mm');img.set_sform(affine,code=2);img.set_qform(None,code=0);nib.save(img,path)


class Fixture:
    def __init__(self,root):
        self.root=root;self.output=root/'out';self.reference=root/'reference.nii.gz';self.atlas=root/'ARM.nii.gz';self.key=root/'ARM_key.txt';self.mask=root/'LR.nii.gz'
        self.affine=np.diag([.25,.25,.25,1]);self.affine[:3,3]=[10,20,30]
        self.data=np.zeros((3,3,3,1,6),dtype=np.int16)
        self.data[0,...]=726;self.data[1:,...]=226
        self.data[0,0,0,0,5]=0
        self.hem=np.ones((3,3,3),dtype=np.uint8);self.hem[1:,...]=2
        image(self.reference,np.zeros((3,3,3),dtype=np.uint8),self.affine)
        image(self.atlas,self.data,self.affine);image(self.mask,self.hem,self.affine)
        self.key.write_text('Index\tAbbreviation\tFull_Name\tFirst_Level\tLast_Level\n226\tCL_Pi\tCL_parainsula\t1\t6\n726\tCR_Pi\tCR_parainsula\t1\t6\n',encoding='utf-8')
        self.manifests={};self.runs={};self.reviews={};self.records={}
        for cohort,sample,xyz in [('main','exactSample',(250,250,250)),('preview','exactSamplech2',(0,250,250))]:
            folder=root/sample;folder.mkdir();source=folder/'001.swc'
            source.write_text(f'1 1 {xyz[0]} {xyz[1]} {xyz[2]} 1 -1\n2 2 {xyz[0]} {xyz[1]} {xyz[2]} 1 1\n',encoding='utf-8')
            self.records[cohort]=[dict(AnimalID='animalA',SampleID=sample,NeuronID='001.swc',Subregion='HumanINS_R' if cohort=='main' else 'OriginCandidate_R',SWCPath=str(source),SWCSHA256=adapter.sha256(source),ReferenceSHA256=adapter.sha256(self.reference),CoordinateFrame='atlas_index_um',IndexScaleUm='250;250;250',AnatomyStatus='original_unaccepted',RegistrationStatus='pending',EvidenceStratum='original_human' if cohort=='main' else 'original_candidate',portal_region='Unknown_0',sample=sample,neuron_id='001.swc',uid=sample+'|001.swc')]
        self.seal()

    def seal(self):
        for cohort in ('main','preview'):
            manifest=self.root/(cohort+'.csv');pd.DataFrame(self.records[cohort]).to_csv(manifest,index=False);self.manifests[cohort]=manifest
            assets={'manifest':manifest,'reference':self.reference,'atlas':self.atlas,'atlas_key':self.key,'hemisphere_mask':self.mask}
            run=self.root/(cohort+'_run.json');run.write_text(json.dumps({'status':'software_verified_candidate_endpoints','inputs':{k:{'path':str(v),'sha256':adapter.sha256(v)} for k,v in assets.items()}}),encoding='utf-8');self.runs[cohort]=run
            review=self.root/(cohort+'_review.json');review.write_text(json.dumps({'status':'passed','run_provenance_sha256':adapter.sha256(run),'manifest_sha256':adapter.sha256(manifest)}),encoding='utf-8');self.reviews[cohort]=review

    def change_source(self,cohort,text):
        row=self.records[cohort][0];path=Path(row['SWCPath']);path.write_text(text,encoding='utf-8');row['SWCSHA256']=adapter.sha256(path)

    def run(self,**kwargs):
        return adapter.prepare(self.manifests['main'],self.manifests['preview'],self.runs['main'],self.reviews['main'],self.runs['preview'],self.reviews['preview'],self.output,input_root=self.root,**kwargs)


class ARMManifestTests(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);return Fixture(Path(temp.name))

    def test_ARM_full_names_selection_exact_channels_and_evidence_preserved(self):
        f=self.fixture();report=f.run();frame=pd.read_csv(f.output/'combined/projection_manifest.csv',dtype=str,keep_default_na=False)
        self.assertEqual(frame.Subregion.tolist(),['ARM6_226_L','ARM6_726_R'])
        self.assertEqual(frame.DisplayLabel.tolist(),['CL_parainsula (Left)','CR_parainsula (Right)'])
        self.assertEqual(frame.SourceCohort.tolist(),['main','preview'])
        self.assertEqual(frame.EvidenceSourceGroup.tolist(),['HumanINS_R','OriginCandidate_R'])
        self.assertEqual(frame.EvidenceStratum.tolist(),['original_human','original_candidate'])
        self.assertEqual(frame.AnatomyStatus.tolist(),['original_unaccepted']*2)
        self.assertEqual(frame.SampleID.tolist(),['exactSample','exactSamplech2'])
        self.assertEqual(report['selection']['combined']['neurons'],2)
        self.assertEqual(report['published_origin_label_differences'],2)
        lm=pd.read_csv(f.output/'combined/label_map.csv',dtype=str,keep_default_na=False)
        for column in adapter.LABEL_COLUMNS:self.assertIn(column,lm)
        self.assertEqual(set(lm.AtlasPath),{str(f.atlas.resolve())})

    def test_background_and_outside_stay_unresolved(self):
        f=self.fixture();f.change_source('main','1 1 0 0 0 1 -1\n');f.change_source('preview','1 1 -1000 0 0 1 -1\n');f.seal();f.run()
        frame=pd.read_csv(f.output/'combined/projection_manifest.csv',dtype=str,keep_default_na=False)
        self.assertEqual(frame.SourceARMStatus.tolist(),['zero_unassigned','out_of_FOV'])
        self.assertEqual(frame.ARMIndex.tolist(),['0','-1'])
        self.assertEqual(frame.ARMFullName.tolist(),['Atlas background','Outside reference'])
        self.assertEqual(frame.Hemisphere.tolist(),['R','Unknown'])
        self.assertEqual(frame.portal_region.tolist(),['Unknown_0']*2)

    def test_exact_sample_alias_mismatch_rejected_before_destination(self):
        f=self.fixture();f.records['preview'][0]['sample']='exactSample';f.seal()
        with self.assertRaisesRegex(ValueError,'alias mismatch'):f.run()
        self.assertFalse(f.output.exists())

    def test_main_preview_identity_overlap_rejected(self):
        f=self.fixture();f.records['preview'][0]=dict(f.records['main'][0]);f.seal()
        with self.assertRaisesRegex(ValueError,'identity overlap'):f.run()
        self.assertFalse(f.output.exists())

    def test_changed_key_receipt_rejected(self):
        f=self.fixture();f.key.write_text(f.key.read_text()+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Changed endpoint input'):f.run()
        self.assertFalse(f.output.exists())

    def test_grid_mismatch_rejected(self):
        f=self.fixture();affine=f.affine.copy();affine[0,3]+=.25;image(f.mask,f.hem,affine);f.seal()
        with self.assertRaisesRegex(ValueError,'geometry'):f.run()
        self.assertFalse(f.output.exists())

    def test_unknown_positive_ARM_label_fails_before_output(self):
        f=self.fixture();f.data[1,1,1,0,5]=9999;image(f.atlas,f.data,f.affine);f.seal()
        with self.assertRaisesRegex(ValueError,'Positive source ARM6 label'):f.run()
        self.assertFalse(f.output.exists())

    def test_primary_ARM_level_conflict_fails_before_output(self):
        f=self.fixture();f.key.write_text(f.key.read_text().replace('CL_parainsula\t1\t6','CL_parainsula\t1\t5'),encoding='utf-8');f.seal()
        with self.assertRaisesRegex(ValueError,'key_level_conflict'):f.run()
        self.assertFalse(f.output.exists())

    def test_world_mm_not_interpreted_as_index_um(self):
        f=self.fixture()
        for cohort,xyz in [('main',(10.25,20.25,30.25)),('preview',(10,20.25,30.25))]:
            f.change_source(cohort,f'1 1 {xyz[0]} {xyz[1]} {xyz[2]} 1 -1\n');f.records[cohort][0].update(CoordinateFrame='nifti_world_mm',IndexScaleUm='')
        f.seal();f.run();frame=pd.read_csv(f.output/'combined/projection_manifest.csv',dtype=str)
        self.assertEqual(frame.ARMAbbreviation.tolist(),['CL_Pi','CR_Pi'])

    def test_alternative_cannot_silently_become_primary(self):
        f=self.fixture()
        with self.assertRaisesRegex(ValueError,'sensitivity only'):f.run(root_policy=adapter.POLICIES[1])
        self.assertFalse(f.output.exists())

    def test_invalid_graph_fails_before_output_creation(self):
        f=self.fixture();f.change_source('main','1 1 250 250 250 1 -1\n2 2 250 250 250 1 99\n');f.seal()
        with self.assertRaisesRegex(ValueError,'missing parent'):f.run()
        self.assertFalse(f.output.exists())

    def test_existing_destination_is_preserved(self):
        f=self.fixture();f.output.mkdir();marker=f.output/'original';marker.write_text('keep')
        with self.assertRaises(FileExistsError):f.run()
        self.assertEqual(marker.read_text(),'keep')

    def test_half_voxel_tie_remains_explicit(self):
        self.assertEqual(adapter.policy_voxel((.5,1,1),adapter.PRIMARY),(0,1,1))
        self.assertEqual(adapter.policy_voxel((.5,1,1),adapter.POLICIES[2]),(1,1,1))
        self.assertEqual(adapter.policy_voxel((1,1,1),adapter.POLICIES[1]),(0,0,0))


if __name__=='__main__':unittest.main()
