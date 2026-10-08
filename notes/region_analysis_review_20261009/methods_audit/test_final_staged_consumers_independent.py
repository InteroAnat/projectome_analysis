"""Independent counterexamples for combined staged consumers; temp files only."""
from pathlib import Path
import contextlib
import io
import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGED = Path(os.environ.get('PROJECTOME_AUDIT_MODULE_ROOT',
                            str(HERE.parent/'validated_repairs/staged/main_scripts')))
sys.path.insert(0, str(STAGED))
sys.path.append(str(ROOT/'main_scripts'))
from region_analysis.utils import neuron_metadata, _aligned_sheet, _long_sheet_records, load_processed_df
from region_analysis.population import PopulationRegionAnalysis
from region_analysis.laterality import add_laterality_columns
from region_analysis.laterality_projection_analysis import LateralityProjectionAnalyzer


def subjects():
    return pd.DataFrame({'SampleID':['A','B'], 'NeuronID':['001.swc','001.swc'],
                         'NeuronUID':['A::001.swc','B::001.swc'],
                         'Neuron_Type':['ITi','CT'], 'Soma_Region':['CL_Ig','CR_Ig']})


def population(frame):
    p = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
    p.plot_dataframe = frame
    p.atlas_table = pd.DataFrame({'Abbreviation':['CL_Pi','SL_Pi','CR_Pi','SR_Pi']})
    return p


class FinalStagedConsumers(unittest.TestCase):
    def test_uid_contradictions_and_blank_identity_fail(self):
        summary = subjects()
        wrong = summary.copy(); wrong.loc[0, 'NeuronUID'] = 'B::001.swc'
        for action in (lambda: neuron_metadata(wrong),
                       lambda: _aligned_sheet(summary, wrong, 'Length')):
            with self.assertRaises(ValueError): action()
        for column in ('NeuronID','SampleID','NeuronUID'):
            wrong = summary.copy(); wrong.loc[0,column] = ''
            with self.subTest(column=column), self.assertRaises(ValueError):
                neuron_metadata(wrong)

    def test_shuffled_multi_subject_identity_and_repeated_records(self):
        summary = subjects()
        sheet = summary.iloc[::-1].copy(); sheet['value'] = [11., 7.]
        self.assertEqual(_aligned_sheet(summary,sheet,'Length').value.tolist(),[7.,11.])
        records = pd.concat([sheet.iloc[[0]], sheet.iloc[[0]], sheet.iloc[[1]]], ignore_index=True)
        self.assertEqual(_long_sheet_records(summary,records,'Terminals',['value']),
                         [[{'value':7.}], [{'value':11.},{'value':11.}]])
        uid_only = summary.drop(columns='SampleID')
        self.assertEqual(neuron_metadata(uid_only).NeuronUID.tolist(),summary.NeuronUID.tolist())
        with self.assertRaises(ValueError):
            _aligned_sheet(summary,sheet.iloc[[0]],'Length')
        with self.assertRaises(ValueError):
            _aligned_sheet(summary,pd.concat([sheet,sheet.iloc[[0]]]),'Length')

    def test_population_split_metadata_collision_and_unknown_side(self):
        frame = subjects()
        frame['Soma_Side'] = ['L','Unknown']
        frame['Region_projection_length'] = [{'CL_Pi':2.,'SL_Pi':7.,'CR_Pi':3.,'_Unmapped':5.},
                                              {'CL_Pi':11.,'SR_Pi':13.}]
        p = population(frame)
        ipsi, contra = p.get_projection_matrix_split()
        for table in (ipsi,contra):
            self.assertEqual(table.NeuronUID.tolist(),frame.NeuronUID.tolist())
            self.assertEqual(table.SampleID.tolist(),frame.SampleID.tolist())
            self.assertEqual(table.Neuron_Type.tolist(),frame.Neuron_Type.tolist())
        self.assertEqual(ipsi.loc[0,'C_Pi'],2.)
        self.assertEqual(ipsi.loc[0,'S_Pi'],7.)
        self.assertEqual(contra.loc[0,'C_Pi'],3.)
        self.assertEqual(ipsi.iloc[1,4:].sum(),0.)
        self.assertEqual(contra.iloc[1,4:].sum(),0.)
        strength, _ = p.get_projection_strength_split()
        self.assertEqual(strength.loc[0,'S_Pi'],round(np.log10(8.),4))

    def test_population_explicit_legacy_hierarchy_conflict(self):
        frame = subjects()
        frame['Soma_Level_3'] = ['CL_A','CR_A']
        frame['Soma_Region_Hierarchy'] = [{'L3':'CL_B'},{'L3':'CR_A'}]
        with self.assertRaises(ValueError):
            population(frame).get_soma_hierarchy_df()
        frame.loc[0,'Soma_Level_3'] = None
        self.assertEqual(population(frame).get_soma_hierarchy_df().L3.tolist(),['CL_B','CR_A'])

    def test_conflicting_summary_and_hierarchy_sheet_fail(self):
        summary = subjects()
        summary['Soma_Level_3'] = ['CL_A','CR_A']
        hierarchy = subjects()
        hierarchy['L3'] = ['CL_B','CR_A']
        with tempfile.TemporaryDirectory(prefix='projectome-consumer-review-') as directory:
            path = Path(directory)/'conflicting_hierarchy.xlsx'
            with pd.ExcelWriter(path) as writer:
                summary.to_excel(writer,sheet_name='Summary',index=False)
                hierarchy.to_excel(writer,sheet_name='Soma_Hierarchy',index=False)
            with self.assertRaises(ValueError):
                load_processed_df(path)

    def test_unknown_side_length_conservation_and_stale_reclassification(self):
        frame = subjects()
        frame['Soma_Side'] = ['R','Unknown']
        frame['Ipsilateral_Projection_Length'] = [{'CL_Pi':2.,'SL_Pi':7.},{'CL_Pi':11.}]
        frame['Contralateral_Projection_Length'] = [{'CR_Pi':3.},{'SR_Pi':13.}]
        frame['Unknown_Laterality_Projection_Length'] = [{'_Unmapped':5.},{}]
        frame['Length_Unit'] = 'voxel'
        with contextlib.redirect_stdout(io.StringIO()):
            analyzer = LateralityProjectionAnalyzer(frame)
            results = analyzer.analyze()
        self.assertEqual(results['ipsilateral'].NeuronUID.tolist(),['A::001.swc'])
        self.assertEqual(results['ipsilateral'].iloc[0].ipsilateral_total_length,3.)
        self.assertEqual(results['contralateral'].iloc[0].contralateral_total_length,9.)
        unknown = results['unknown_laterality']
        self.assertEqual(dict(unknown.groupby('NeuronUID').Length.sum()),
                         {'A::001.swc':5.,'B::001.swc':24.})
        self.assertEqual(3.+9.+unknown.Length.sum(),41.)
        overlap = frame.copy()
        overlap.at[0,'Contralateral_Projection_Length'] = {'CL_Pi':3.}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            LateralityProjectionAnalyzer(overlap).analyze()

    def test_unknown_target_control_consistent_between_consumers(self):
        frame = subjects().iloc[[0]].copy()
        frame['Terminal_Regions'] = [['CL_Unknown_0','CR_Unknown_0','CL_Ig']]
        frame['Region_projection_length'] = [{'CL_Unknown_0':5.,'CR_Unknown_0':7.,'CL_Ig':2.}]
        with contextlib.redirect_stdout(io.StringIO()):
            joined = add_laterality_columns(frame)
            standalone = LateralityProjectionAnalyzer(joined).analyze()
        self.assertEqual(standalone['unknown_laterality'].Length.sum(),12.)
        self.assertEqual(joined.Total_Unknown_Laterality_Length.iloc[0],12.)
        self.assertEqual(joined.N_Laterality_Unknown.iloc[0],2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
