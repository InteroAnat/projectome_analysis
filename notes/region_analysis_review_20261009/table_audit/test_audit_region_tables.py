import unittest
import pandas as pd
from audit_region_tables import ROOT, inspect_frame


class TableAuditRegressionTests(unittest.TestCase):
    def test_explicit_combined_uid_is_valid(self):
        df=pd.DataFrame({'SampleID':['252385'],'NeuronID':['021.swc'],'NeuronUID':['252385::021.swc']})
        _,findings=inspect_frame(df,ROOT/'group_analysis/combined/example.xlsx','Summary')
        self.assertFalse(findings)

    def test_channel_collision_stays_identity_error(self):
        df=pd.DataFrame({'SampleID':['252385ch2'],'NeuronID':['021.swc'],'NeuronUID':['252385::021.swc']})
        _,findings=inspect_frame(df,ROOT/'group_analysis/combined/example.xlsx','Summary')
        self.assertEqual(findings[0]['check'],'composite_UID_mismatch')

    def test_fnt_uid_uses_explicit_basename_contract(self):
        df=pd.DataFrame({'SampleID':['251637'],'NeuronID':['021.swc'],'NeuronUID':['251637_021']})
        _,findings=inspect_frame(df,ROOT/'group_analysis/fnt/example.csv','__csv__')
        self.assertFalse(findings)

    def test_target_longform_is_not_duplicate_neuron_error(self):
        df=pd.DataFrame({'SampleID':['251637','251637'],'NeuronID':['021.swc','021.swc'],'target_index':['1','2']})
        schema,findings=inspect_frame(df,ROOT/'group_analysis/endpoint_maps/example.csv','__csv__')
        self.assertEqual(schema['identity_grain'],'longform_or_source_record')
        self.assertFalse(findings)

    def test_laterality_counts_must_conserve(self):
        df=pd.DataFrame({'NeuronID':['021.swc'],'Terminal_Count':[4],'N_Ipsilateral':[2],'N_Contralateral':[1],'N_Laterality_Unknown':[0]})
        _,findings=inspect_frame(df,ROOT/'neuron_tables/example.xlsx','Summary')
        self.assertEqual(findings[0]['check'],'terminal_laterality_count_conservation')


if __name__=='__main__':unittest.main()
