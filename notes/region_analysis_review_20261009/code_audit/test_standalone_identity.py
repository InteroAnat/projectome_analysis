"""Independent identity/laterality regressions against staged standalone code."""
from pathlib import Path
import os,sys,tempfile,unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'main_scripts'))
sys.path.insert(0,os.environ.get('PROJECTOME_CONSUMER_MODULE_ROOT',str(ROOT/'main_scripts')))
from region_analysis.laterality_projection_analysis import LateralityProjectionAnalyzer as L


def frame(**extra):
    return pd.DataFrame({'SampleID':['A'],'NeuronID':['001.swc'],'Neuron_Type':['ITi'],
        'Soma_Region':['Unknown_0'],'Soma_Side':['R'],'Length_Unit':['voxel'],**extra})


class StandaloneIdentityTests(unittest.TestCase):
    def test_stale_precomputed_only_reclassified_by_explicit_side(self):
        result=L(frame(Ipsilateral_Projection_Length=[{'CL_A':9.}],
                       Contralateral_Projection_Length=[{}])).analyze()
        self.assertTrue(result['ipsilateral'].empty)
        self.assertEqual(result['contralateral'].A_length.iloc[0],9.)

    def test_precomputed_overlap_and_nonabsolute_anatomy_rejected(self):
        for ipsi,contra in [({'CL_A':1.},{'CL_A':1.}),({'A':1.},{})]:
            with self.subTest(ipsi=ipsi),self.assertRaises(ValueError):
                L(frame(Ipsilateral_Projection_Length=[ipsi],Contralateral_Projection_Length=[contra])).analyze()

    def test_missing_projection_source_fails(self):
        with self.assertRaises(ValueError):L(frame()).analyze()

    def test_missing_duplicate_or_mismatched_exact_identity_fails(self):
        for data in [frame(Region_projection_length=[{}]).drop(columns='SampleID'),
                     pd.concat([frame(Region_projection_length=[{}])]*2,ignore_index=True),
                     frame(NeuronUID=['B::001.swc'],Region_projection_length=[{}])]:
            with self.subTest(columns=list(data)),self.assertRaises(ValueError):L(data)

    def test_cross_sample_same_bare_id_remains_distinct_in_export(self):
        f=pd.concat([frame(Region_projection_length=[{'CR_A':3.}]),
                     frame(SampleID=['B'],Region_projection_length=[{'CR_A':5.}])],ignore_index=True)
        a=L(f);r=a.analyze()['ipsilateral']
        self.assertEqual(r.NeuronUID.tolist(),['A::001.swc','B::001.swc'])
        self.assertEqual(r.Length_Unit.tolist(),['voxel','voxel'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'exact.xlsx';a.save_excel(str(p))
            x=pd.read_excel(p,'Ipsilateral_Length')
            self.assertEqual(x.NeuronUID.tolist(),['A::001.swc','B::001.swc'])
            self.assertEqual(x.SampleID.tolist(),['A','B'])

    def test_uid_only_identity_and_unknown_lengths_preserved(self):
        f=frame(NeuronUID=['A::001.swc'],Soma_Side=['Unknown'],
            Ipsilateral_Projection_Length=[{'CL_A':3.}],Contralateral_Projection_Length=[{}]).drop(columns='SampleID')
        a=L(f);r=a.analyze()
        self.assertTrue(r['ipsilateral'].empty);self.assertTrue(r['contralateral'].empty)
        self.assertEqual(r['unknown_laterality'].Length.iloc[0],3.)
        self.assertEqual(r['unknown_laterality'].NeuronUID.iloc[0],'A::001.swc')

    def test_all_lengths_conserved_and_unknown_export_is_explicit(self):
        a=L(frame(Region_projection_length=[{'CR_A':3.,'CL_B':5.,'Unknown_0':7.,'SL_Unknown_9':11.}]))
        r=a.analyze()
        self.assertEqual(r['ipsilateral'].ipsilateral_total_length.sum(),3.)
        self.assertEqual(r['contralateral'].contralateral_total_length.sum(),5.)
        self.assertEqual(r['unknown_laterality'].Length.sum(),18.)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unknown.xlsx';a.save_excel(str(p),r)
            unknown=pd.read_excel(p,'Unknown_Laterality')
            self.assertEqual(unknown.Length.sum(),18.)
            self.assertTrue(unknown.Length_Unit.eq('voxel').all())

    def test_mixed_units_fail_before_export(self):
        f=pd.concat([frame(Region_projection_length=[{'CR_A':3.}]),
            frame(SampleID=['B'],Length_Unit=['mm'],Region_projection_length=[{'CR_A':5.}])],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'mixed source units'):L(f)


if __name__=='__main__':unittest.main(verbosity=2)
