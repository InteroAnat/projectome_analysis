"""Small genuine saved map runs test ARM export eligibility and denominators."""
from pathlib import Path
import contextlib,importlib.util,io,json,sys,tempfile,unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'main_scripts'))
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'group_analysis/scripts'/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
exporter=module('arm_table_exporter','export_arm_projection_tables.py')
endpoint=module('arm_table_endpoint_builder','build_endpoint_maps.py')
axon=module('arm_table_axon_builder','build_projection_maps.py')

def image(path,data):
    img=nib.Nifti1Image(data,np.eye(4));img.header.set_xyzt_units('mm');img.set_qform(np.eye(4),1);img.set_sform(np.eye(4),1);nib.save(img,path)

class ARMTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.base=Path(cls.temp.name)
        base=cls.base;shape=(6,2,2);reference=base/'reference.nii.gz';image(reference,np.zeros(shape,dtype=np.float32))
        atlas=np.ones((*shape,1,6),dtype=np.int16);atlas[3:,...]=501
        atlas[2,...,0,3:5]=45;atlas[0,1,1,0,:]=0
        atlas_path=base/'ARM.nii.gz';image(atlas_path,atlas)
        hemisphere=np.ones(shape,dtype=np.int16);hemisphere[3:]=2
        mask=base/'hemisphere.nii.gz';image(mask,hemisphere)
        key=base/'ARM_key.txt'
        pd.DataFrame({'Index':[1,501,45],'Abbreviation':['CL_A','CR_A','CL_Conflict'],
            'Full_Name':['CL_official_left','CR_official_right','CL_official_conflict'],
            'First_Level':[1,1,6],'Last_Level':[6,6,6]}).to_csv(key,sep='\t',index=False)
        traces={
            'a':'1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 2 2 0 0 1 1',
            'b':'1 1 0 0 0 1 -1\n2 2 4 0 0 1 1',
            'c':'1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n3 3 2 0 0 1 2',
            'e':'1 1 0 0 0 1 -1\n2 2 8 0 0 1 1',
            'd':'1 1 0 0 0 1 -1\n2 2 1 0 0 1 1'}
        rows=[]
        for name,text in traces.items():
            folder=base/name;folder.mkdir();source=folder/f'{name}.swc';source.write_text(text,encoding='utf-8')
            animal='B' if name=='d' else 'A'
            rows.append(dict(AnimalID=animal,SampleID=animal,NeuronID=source.name,Subregion='OldExtra' if animal=='B' else 'OldMain',
                SWCPath=str(source),SWCSHA256=exporter.sha256(source),ReferenceSHA256=exporter.sha256(reference),
                CoordinateFrame='nifti_world_mm',IndexScaleUm='',AnatomyStatus='candidate',RegistrationStatus='unresolved',OriginalSourceLabel=''))
        cls.paths=[]
        for number,animal in enumerate(('A','B')):
            manifest=base/f'input{number}.csv';pd.DataFrame([r for r in rows if r['AnimalID']==animal]).to_csv(manifest,index=False)
            ep=base/f'endpoint{number}';ax=base/f'axon{number}'
            with contextlib.redirect_stdout(io.StringIO()):
                endpoint.build(manifest,reference,ep,atlas_path=atlas_path,atlas_key=key,hemisphere_mask=mask,input_root=base)
                axon.build(manifest,reference,ax,input_root=base)
            cls.paths.append((ep,ax))
        final=pd.DataFrame(rows);final['Subregion']='ARM6_1_L'
        for field,value in dict(ARMLevel=6,ARMIndex=1,ARMAbbreviation='CL_A',ARMFullName='CL_official_left',Hemisphere='L',SourceARMStatus='mapped').items():final[field]=value
        cls.manifest=base/'combined.csv';final.to_csv(cls.manifest,index=False)
        cls.output=base/'export'
        with contextlib.redirect_stdout(io.StringIO()),patch.object(exporter,'axon_length_map',wraps=exporter.axon_length_map) as raster:
            cls.receipt=exporter.export(*cls.paths[0],cls.output,manifest=cls.manifest,
                additional_endpoint_run=cls.paths[1][0],additional_axon_run=cls.paths[1][1],input_root=base)
            cls.raster_calls=raster.call_count

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_one_raster_per_neuron_and_all_actual_levels_reconciled(self):
        self.assertEqual(self.raster_calls,5)
        self.assertEqual(self.receipt['n_selected'],5);self.assertEqual(self.receipt['n_endpoint_eligible'],4)
        self.assertEqual(self.receipt['regional_reconciliation_maps'],8)
        self.assertFalse(self.receipt['scientific_anatomical_acceptance'])

    def test_distinct_neuron_presence_and_equal_animal_eligible_denominators(self):
        animal=pd.read_excel(self.output/'arm_projection_hierarchy_tables.xlsx',sheet_name='Animal_Means')
        group=pd.read_excel(self.output/'arm_projection_hierarchy_tables.xlsx',sheet_name='Group_Means')
        left=animal[animal.TargetID.eq('L1:mapped:1')].set_index('AnimalID')
        self.assertEqual(left.loc['A','NSelected'],4);self.assertEqual(left.loc['A','NEndpointEligible'],3)
        self.assertAlmostEqual(left.loc['A','EndpointNeuronFrequency'],1/3)
        self.assertAlmostEqual(left.loc['A','CandidateEndpointMean'],2/3)
        self.assertAlmostEqual(group.loc[group.TargetID.eq('L1:mapped:1'),'EndpointNeuronFrequency'].iloc[0],2/3)

    def test_unassessed_endpoints_na_but_axon_observed_and_outside_explicit(self):
        book=self.output/'arm_projection_hierarchy_tables.xlsx'
        counts=pd.read_excel(book,sheet_name='L1_EP_Count').set_index('NeuronID')
        lengths=pd.read_excel(book,sheet_name='L1_AxonLen_mm').set_index('NeuronID')
        target_columns=[c for c in counts if c.startswith('L1:')]
        self.assertTrue(counts.loc['c.swc',target_columns].isna().all())
        self.assertGreater(lengths.loc['c.swc',target_columns].sum(),0)
        outside=next(c for c in counts if c.startswith('L1:out_of_FOV:'))
        self.assertEqual(counts.loc['e.swc',outside],1)
        self.assertAlmostEqual(lengths.loc['e.swc',outside],2.5)

    def test_actual_key_conflicts_flagged_and_official_names_domain_side_retained(self):
        targets=pd.read_csv(self.output/'targets.csv')
        conflict=targets[targets.ARMIndex.eq(45)]
        self.assertEqual(conflict.Level.tolist(),[4,5])
        self.assertTrue(conflict.KeyRangeConflict.all())
        self.assertTrue(conflict.OfficialFullName.eq('CL_official_conflict').all())
        self.assertTrue(conflict.Domain.eq('Cortex').all());self.assertTrue(conflict.Hemisphere.eq('L').all())
        long=pd.read_csv(self.output/'per_neuron_regional_measures.csv')
        self.assertTrue(long.TargetID.eq('L4:key_level_conflict:45').any())

    def test_output_bound_to_sources_and_no_overwrite(self):
        for name,expected in self.receipt['artifacts'].items():self.assertEqual(exporter.sha256(self.output/name),expected)
        self.assertEqual(self.receipt['manifest_sha256'],exporter.sha256(self.manifest))
        with self.assertRaises(FileExistsError):exporter.export(Path('missing'),Path('missing'),self.output)

    def test_common_sheets_expose_official_source_and_target_metadata(self):
        book=self.output/'arm_projection_hierarchy_tables.xlsx'
        for name in ('Animal_Means','Group_Means','L6_EP_Count','L6_AxonLen_mm'):
            table=pd.read_excel(book,sheet_name=name)
            self.assertTrue(table.ARMFullName.eq('CL_official_left').all())
            self.assertTrue(table.Hemisphere.eq('L').all())
            if name.endswith('Means'):
                self.assertIn('OfficialFullName',table);self.assertIn('TargetHemisphere',table)

    def test_fabricated_source_names_and_groups_rejected_before_output(self):
        for field in ('ARMFullName','Subregion','SourceARMStatus'):
            bad=pd.read_csv(self.manifest,dtype=str,keep_default_na=False);bad.loc[0,field]='fabricated'
            path=self.base/f'bad_{field}.csv';bad.to_csv(path,index=False)
            destination=self.base/f'bad_output_{field}'
            with self.assertRaises(ValueError):
                exporter.export(*self.paths[0],destination,manifest=path,
                    additional_endpoint_run=self.paths[1][0],additional_axon_run=self.paths[1][1],input_root=self.base)
            self.assertFalse(destination.exists())

    def test_swapped_positive_source_side_rejected_but_zero_mask_side_preserved(self):
        targets=exporter.read_csv(self.output/'targets.csv')
        bad=pd.read_csv(self.manifest,dtype=str,keep_default_na=False)
        bad['Hemisphere']='R';bad['Subregion']='ARM6_1_R'
        with self.assertRaisesRegex(ValueError,'hemisphere'):
            exporter.validate_source_labels(bad,targets,'unused','unused')
        bad['ARMIndex']='0';bad['Subregion']='ARM6_0_R';bad['ARMFullName']='Atlas background'
        bad['ARMAbbreviation']='';bad['SourceARMStatus']='zero_unassigned'
        self.assertEqual(exporter.validate_source_labels(bad,targets,'unused','unused').Hemisphere.tolist(),['R'])

    def test_rehashed_target_identity_corruption_rejected(self):
        source=self.paths[0][0]/'per_neuron_target_counts.csv'
        provenance=self.paths[0][0]/'run_provenance.json'
        original_source=source.read_bytes();original_provenance=provenance.read_bytes()
        try:
            bad=exporter.read_csv(source);bad.loc[0,'AnimalID']='wrong';bad.to_csv(source,index=False)
            record=json.loads(original_provenance)
            record['artifacts'][source.name]['sha256']=exporter.sha256(source)
            provenance.write_text(json.dumps(record),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Target row identity metadata'):
                exporter.export(*self.paths[0],self.base/'rehashed_bad_identity',manifest=self.manifest,
                    additional_endpoint_run=self.paths[1][0],additional_axon_run=self.paths[1][1],input_root=self.base)
            self.assertFalse((self.base/'rehashed_bad_identity').exists())
        finally:
            source.write_bytes(original_source);provenance.write_bytes(original_provenance)

    def test_recorded_geometry_checked_against_actual_reference(self):
        provenance=self.paths[0][0]/'run_provenance.json';before=provenance.read_bytes()
        try:
            record=json.loads(before);record['voxel_volume_mm3']=2
            provenance.write_text(json.dumps(record),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'geometry'):
                exporter.export(*self.paths[0],self.base/'bad_geometry',manifest=self.manifest,
                    additional_endpoint_run=self.paths[1][0],additional_axon_run=self.paths[1][1],input_root=self.base)
        finally:provenance.write_bytes(before)

    def test_integral_nullable_csv_label_notation_and_invalid_numbers(self):
        self.assertEqual(exporter.integer('45.0'),45);self.assertIsNone(exporter.integer('',True))
        for value in ('1.5','nan','inf'):
            with self.assertRaises(ValueError):exporter.integer(value)

    def test_changed_artifact_rejected_before_output(self):
        source=self.paths[0][0]/'per_neuron_qc.csv';before=source.read_bytes()
        try:
            source.write_bytes(before+b'\n')
            with self.assertRaisesRegex(ValueError,'artifact hash'):
                exporter.export(*self.paths[0],self.base/'invalid',manifest=self.manifest,
                    additional_endpoint_run=self.paths[1][0],additional_axon_run=self.paths[1][1],input_root=self.base)
            self.assertFalse((self.base/'invalid').exists())
        finally:source.write_bytes(before)

if __name__=='__main__':unittest.main(verbosity=2)
