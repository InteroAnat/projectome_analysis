"""Fail-closed workbook/atlas contracts and order-independent SWC root checks."""
from pathlib import Path
import contextlib,importlib.util,io,sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'main_scripts'))
from region_analysis.neuron_analysis import RegionAnalysisPerNeuron

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'group_analysis/scripts'/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
combined=load('combined_contracts','05a_build_combined_table.py')
coordinates=load('coordinate_contracts','05b_audit_coord_frames.py')
harmonizer=load('harmonizer_contracts','06_harmonize_atlas_to_manual.py')

def write(path,sheets):
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for name,frame in sheets.items():frame.to_excel(writer,sheet_name=name,index=False)

class CombinedContracts(unittest.TestCase):
    def setup_case(self,d):
        base=Path(d);recovery=base/'recovery';recovery.mkdir();out=base/'output';out.mkdir()
        ref=base/'reference.xlsx';src=base/'step1.xlsx';keepers=recovery/'252385_INS_HE_coord_inferred.xlsx'
        summary=pd.DataFrame({'NeuronID':['ref.swc'],'Soma_Region':['CL_Ial'],'Soma_Side':['L']})
        ref_sheets={'Summary':summary}
        for name in combined.PROJ_SHEETS:
            ref_sheets[name]=pd.DataFrame({'NeuronID':['ref.swc'],'Neuron_Type':['ITi'],'A':[1.25]})
        src_sheets={name:pd.DataFrame({'NeuronID':['new.swc','unselected.swc'],'SampleID':[252385,252385],
            'NeuronUID':['252385::new.swc','252385::unselected.swc'],'Neuron_Type':['ITi','ITi'],'B':[2.5,9.]}) for name in combined.PROJ_SHEETS}
        keeper_frame=pd.DataFrame({'SampleID':[252385],'NeuronID':['new.swc'],'Soma_Region_Refined':['IAL'],
            'Soma_Region_Source':['auto_atlas_insula'],'Soma_Side':['L']})
        write(ref,ref_sheets);write(src,src_sheets);write(keepers,{'Insula_keepers':keeper_frame})
        return ref,src,recovery,out,keepers,ref_sheets,src_sheets,keeper_frame

    def run_case(self,case):
        ref,src,recovery,out,*_=case
        with patch.multiple(combined,REF_INS_XLSX=str(ref),RECOVERY_DIR=str(recovery),OUT_DIR=str(out),NEW_SAMPLES=['252385']), \
             patch.object(combined,'find_results_xlsx',return_value=str(src)),contextlib.redirect_stdout(io.StringIO()):
            return combined.main()

    def test_valid_combined_derivative_preserves_inputs_full_membership_and_values(self):
        with tempfile.TemporaryDirectory() as d:
            case=self.setup_case(d);ref,src,_,out,keepers,*_=case
            inputs={p:p.read_bytes() for p in (ref,src,keepers)}
            self.assertEqual(self.run_case(case),0)
            sheets=pd.read_excel(out/'multi_monkey_INS_combined.xlsx',sheet_name=None)
            expected=['251637::ref.swc','252385::new.swc']
            self.assertEqual(sheets['Summary'].NeuronUID.tolist(),expected)
            for name in combined.PROJ_SHEETS:
                self.assertEqual(sheets[name].NeuronUID.tolist(),expected)
                self.assertEqual(sheets[name].A.tolist(),[1.25,0.])
                self.assertEqual(sheets[name].B.tolist(),[0.,2.5])
            for p,data in inputs.items():self.assertEqual(p.read_bytes(),data)

    def test_existing_destination_refused_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'multi_monkey_INS_combined.xlsx';output.write_bytes(b'protected')
            with patch.object(combined,'OUT_DIR',d),patch.object(combined.pd,'read_excel',side_effect=AssertionError('must not read')):
                with self.assertRaises(FileExistsError):combined.main()
            self.assertEqual(output.read_bytes(),b'protected')

    def test_missing_required_sheet_never_publishes(self):
        for source_index,sheets_index in ((0,5),(1,6)):
            with self.subTest(source_index=source_index),tempfile.TemporaryDirectory() as d:
                case=self.setup_case(d);sheets=case[sheets_index];del sheets[combined.PROJ_SHEETS[0]]
                write(case[source_index],sheets)
                with self.assertRaises((ValueError,KeyError)):self.run_case(case)
                self.assertFalse((case[3]/'multi_monkey_INS_combined.xlsx').exists())

    def test_missing_duplicate_foreign_selected_identities_fail_closed(self):
        for defect in ('missing_source_id','duplicate_source_id','foreign_sample','wrong_uid','duplicate_keeper'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as d:
                case=self.setup_case(d);sheets=case[6];frame=sheets[combined.PROJ_SHEETS[0]]
                if defect=='missing_source_id':frame=frame.iloc[1:].copy()
                elif defect=='duplicate_source_id':frame=pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
                elif defect=='foreign_sample':frame.loc[0,'SampleID']=999
                elif defect=='wrong_uid':frame.loc[0,'NeuronUID']='999::new.swc'
                else:write(case[4],{'Insula_keepers':pd.concat([case[7],case[7]],ignore_index=True)})
                sheets[combined.PROJ_SHEETS[0]]=frame;write(case[1],sheets)
                with self.assertRaises(ValueError):self.run_case(case)
                self.assertFalse((case[3]/'multi_monkey_INS_combined.xlsx').exists())

    def test_missing_configured_input_files_fail_closed(self):
        for which in (1,4):
            with self.subTest(which=which),tempfile.TemporaryDirectory() as d:
                case=self.setup_case(d);case[which].unlink()
                with self.assertRaises(FileNotFoundError):self.run_case(case)
                self.assertFalse((case[3]/'multi_monkey_INS_combined.xlsx').exists())

    def test_source_selection_reorders_explicitly_to_keeper_identity_order(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'input.xlsx'
            write(source,{'Projection_Length_ipsi':pd.DataFrame({'NeuronID':['b','a'],'value':[2.,1.]})})
            result=combined.union_projection_sheet(pd.DataFrame(),source,'Projection_Length_ipsi','S',['a','b'])
            self.assertEqual(result.NeuronID.tolist(),['a','b'])
            self.assertEqual(result.value.tolist(),[1.,2.])

    def test_reference_extra_missing_and_duplicate_projection_identities_rejected(self):
        for defect in ('extra','missing','duplicate','blank'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as d:
                case=self.setup_case(d);sheets=case[5];key=combined.PROJ_SHEETS[0];frame=sheets[key]
                if defect=='extra':frame=pd.concat([frame,pd.DataFrame({'NeuronID':['extra'],'A':[99.]})],ignore_index=True)
                elif defect=='missing':frame=frame.iloc[:0]
                elif defect=='duplicate':frame=pd.concat([frame,frame],ignore_index=True)
                else:frame.loc[0,'NeuronID']=None
                sheets[key]=frame;write(case[0],sheets)
                with self.assertRaises(ValueError):self.run_case(case)
                self.assertFalse((case[3]/'multi_monkey_INS_combined.xlsx').exists())

    def test_keeper_requires_sample_identity(self):
        with tempfile.TemporaryDirectory() as d:
            case=self.setup_case(d)
            write(case[4],{'Insula_keepers':case[7].drop(columns='SampleID')})
            with self.assertRaisesRegex(ValueError,'SampleID'):self.run_case(case)

    def test_concurrently_created_destination_is_preserved_and_temporary_cleaned(self):
        with tempfile.TemporaryDirectory() as d:
            case=self.setup_case(d);real_link=combined.os.link
            def competing_link(source,destination):
                Path(destination).write_bytes(b'concurrent protected file')
                return real_link(source,destination)
            with patch.object(combined.os,'link',side_effect=competing_link):
                with self.assertRaises(FileExistsError):self.run_case(case)
            self.assertEqual((case[3]/'multi_monkey_INS_combined.xlsx').read_bytes(),b'concurrent protected file')
            self.assertFalse(list(case[3].glob('.combined_build_*')))

class CoordinateAndAtlasContracts(unittest.TestCase):
    def test_root_selected_by_parent_marker_not_row_order(self):
        text='# shuffled\n2 2 9 8 7 1 1\n1 1 1 2 3 1 -1\n'
        self.assertEqual(coordinates.parse_swc_root(text),((1.,2.,3.),2))

    def test_invalid_root_topology_and_nonfinite_coordinates_are_rejected(self):
        for text in ('1 1 0 0 0 1 -1\n2 2 1 0 0 1 -1',
                     '1 1 0 0 0 1 2\n2 2 1 0 0 1 1','1 1 nan 0 0 1 -1','1 1 0 0 0 1 -1\n2 2 1 0 0 1 3'):
            with self.subTest(text=text),self.assertRaises(ValueError):coordinates.parse_swc_root(text)

    def test_malformed_atlas_index_and_label_values_rejected(self):
        for index,label in ((1.5,'CL_A'),(np.nan,'CL_A'),(np.inf,'CL_A'),(True,'CL_A'),(-1,'CL_A'),(1,''),(1,None),(1,7)):
            with self.subTest(index=index,label=label),self.assertRaises(ValueError):
                RegionAnalysisPerNeuron(None,np.ones((2,2,2)),pd.DataFrame({'Index':[index],'Abbreviation':[label]}))

    def test_duplicate_anatomical_labels_cannot_overwrite_lengths(self):
        with self.assertRaisesRegex(ValueError,'unique'):
            RegionAnalysisPerNeuron(None,np.ones((2,2,2)),pd.DataFrame({'Index':[1,2],'Abbreviation':['CL_A','CL_A']}))

    def test_valid_key_preserves_proximal_rounding_total_and_leaf_gate(self):
        a=SimpleNamespace(x_nii=0.,y_nii=0.,z_nii=0.)
        b=SimpleNamespace(x_nii=1.0004,y_nii=0.,z_nii=0.)
        tracer=SimpleNamespace(root=a,terminal_nodes=[a,b],branches=[[a,b]])
        atlas=np.zeros((2,1,1));atlas[0,0,0]=1;atlas[1,0,0]=2
        obj=RegionAnalysisPerNeuron(tracer,atlas,pd.DataFrame({'Index':[2,1],'Abbreviation':['CR_B','CL_A']}));obj.run()
        self.assertEqual(obj.neuron_total_length,1.)
        self.assertEqual(obj.mapped_brain_region_lengths,{'CL_A':1.})

    def test_actual_atlas_key_passes_without_mutating_source_table(self):
        key=pd.read_csv(ROOT/'atlas/ARM_key_all.txt',sep='\t');original=key.copy(deep=True)
        obj=RegionAnalysisPerNeuron(None,np.zeros((1,1,1)),key)
        self.assertEqual(len(obj.brain_region_map),len(key))
        pd.testing.assert_frame_equal(key,original)

class HarmonizerSchemaContracts(unittest.TestCase):
    def test_summary_only_or_missing_quantitative_sheet_rejected_before_enrichment(self):
        summary=pd.DataFrame({'SampleID':['S'],'NeuronID':['n'],'NeuronUID':['S::n'],
            'Soma_Region_Refined':['IAL'],'Soma_Region_Source':['curated_251637']})
        for names in ([],list(combined.PROJ_SHEETS[:-1])):
            with self.subTest(names=names),tempfile.TemporaryDirectory() as d:
                source=Path(d)/'source.xlsx';output=Path(d)/'output.xlsx'
                sheets={'Summary':summary}
                sheets.update({name:summary[['SampleID','NeuronID','NeuronUID']] for name in names})
                write(source,sheets);before=source.read_bytes()
                with patch.object(harmonizer,'_enrich_henry_layers',side_effect=AssertionError('must not enrich')):
                    with self.assertRaisesRegex(ValueError,'required.*sheet|sheet.*required'):harmonizer.harmonize(source,output)
                self.assertEqual(source.read_bytes(),before);self.assertFalse(output.exists())

if __name__=='__main__':unittest.main(verbosity=2)
