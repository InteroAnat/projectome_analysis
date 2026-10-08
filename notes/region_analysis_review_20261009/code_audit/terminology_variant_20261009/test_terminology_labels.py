"""Independent visible-label and code-contract checks for staged terminology."""
from pathlib import Path
import ast,contextlib,io,json,os,re,subprocess,sys,unittest
from unittest.mock import patch
import pandas as pd
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
STAGE=BASE/'staged/main_scripts'
sys.path.insert(0,str(STAGE))
sys.path.append(str(ROOT/'main_scripts'))
from region_analysis import plotting
from region_analysis.population import PopulationRegionAnalysis
import region_analysis.population as population_module
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

class LiteralOnly(ast.NodeTransformer):
    def generic_visit(self,node):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body=node.body[1:]
        return super().generic_visit(node)
    def visit_Constant(self,node):
        if isinstance(node.value,str):return ast.copy_location(ast.Constant(value='__DISPLAY_TEXT__'),node)
        return node

def protected_strings(tree):
    result=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and isinstance(n.slice.value,str):
            result.append(('subscript',n.slice.value))
        if isinstance(n,ast.Dict):
            result.extend(('dictionary_key',k.value) for k in n.keys if isinstance(k,ast.Constant) and isinstance(k.value,str))
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+',n.value):
            result.append(('schema_token',n.value))
        if isinstance(n,ast.Call):
            func=n.func.attr if isinstance(n.func,ast.Attribute) else ''
            if func in ('save_report','get_plot_path','get_report_path','get_table_path'):
                result.extend(('filename',arg.value) for arg in n.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str))
            result.extend(('sheet_name',k.value.value) for k in n.keywords if k.arg=='sheet_name' and isinstance(k.value,ast.Constant))
    return sorted(result)

class TerminologyTests(unittest.TestCase):
    def tearDown(self):plt.close('all')

    def test_only_text_changes_and_no_api_schema_or_filename_changes(self):
        for name in ('population.py','plotting.py','laterality_projection_analysis.py'):
            with self.subTest(name=name):
                original=ast.parse((ROOT/'main_scripts/region_analysis'/name).read_text(encoding='utf-8'))
                staged=ast.parse((STAGE/'region_analysis'/name).read_text(encoding='utf-8'))
                self.assertEqual(protected_strings(original),protected_strings(staged))
                self.assertEqual(ast.dump(LiteralOnly().visit(original)),ast.dump(LiteralOnly().visit(staged)))

    def frame(self,typed=True):
        df=pd.DataFrame({'NeuronID':['cell'], 'SampleID':['A'], 'Neuron_Type':['ITi'],
            'Soma_Region':['CL_A'],'Soma_Side':['L'],'Total_Length':[8.0],
            'Terminal_Count':[2],'Terminal_Regions':[['CL_A','Unknown_0']],
            'Outlier_Count':[0],'Region_projection_length':[{'CL_A':4.0,'CR_B':2.0}],
            'N_Ipsilateral':[1],'N_Contralateral':[1], 'N_Laterality_Unknown':[1],
            'Laterality_Index':[1/3], 'Total_Ipsilateral_Length':[4.0],
            'Total_Contralateral_Length':[2.0],'Length_Unit':['voxel']})
        return df if typed else df.drop(columns='Neuron_Type')

    def test_laterality_labels_define_fraction_in_both_display_branches(self):
        for typed in (True,False):
            with self.subTest(typed=typed):
                fig=plotting.plot_laterality_summary_df(self.frame(typed),show=False)
                label=fig.axes[1].get_ylabel() if typed else fig.axes[1].get_xlabel()
                self.assertIn('Contra/(Ipsi+Contra)',label)
                self.assertIn('0 to 1',label)
                self.assertEqual(fig.axes[2].get_ylabel(),'Retained reconstruction length (voxel)')
                self.assertIn('endpoint-target',fig.axes[0].get_title())
        docs=plotting.plot_laterality_summary_df.__doc__
        self.assertIn('(Contra-Ipsi)/(Contra+Ipsi)',docs)
        self.assertIn('-1..1',docs)

    def test_partial_missing_units_still_rejected(self):
        df=pd.concat([self.frame(),self.frame()],ignore_index=True)
        df.loc[1,'Length_Unit']=None
        with self.assertRaisesRegex(ValueError,'mixed Length_Unit'):
            plotting.plot_laterality_summary_df(df,show=False)

    def test_endpoint_region_figures_and_reports_have_no_site_claim(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report=plotting.plot_terminal_distribution_df(self.frame(),show=False)
        self.assertIn('legacy all-compartment leaves',report)
        self.assertIn('Total endpoint-target entries: 2',report)
        self.assertIn('Known endpoint-target entries: 1',report)
        self.assertIn('Unresolved endpoint-target entries: 1',report)
        self.assertIn('Endpoint-Target',plt.gcf().axes[1].get_title())
        with contextlib.redirect_stdout(io.StringIO()):
            report=plotting.plot_projection_sites_count_df(self.frame(),show=False)
        self.assertIn('Mean known endpoint-target regions per neuron: 1.00',report)
        self.assertIn('Endpoint-Target',plt.gcf()._suptitle.get_text())

    def test_population_reports_and_inspection_use_source_units_and_regions(self):
        p=PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
        p.plot_dataframe=self.frame();p.sample_id='A';p.output=None;p.show_plots=False
        reports={};p.save_report=lambda report,name:reports.__setitem__(name,report)
        with contextlib.redirect_stdout(io.StringIO()):
            p._save_analysis_summary_report();p._save_terminal_report();p._save_projection_sites_report()
        summary=reports['analysis_summary']
        self.assertNotIn('atlas voxels',summary)
        self.assertIn('Distinct endpoint-target regions (legacy all compartments)',summary)
        self.assertIn('Contra/(Ipsi+Contra), 0..1',summary)
        self.assertIn('source units (see Length_Unit)',summary)
        self.assertIn('endpoint-target entries',reports['terminal_report'])
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream),patch.object(population_module,'plot_neuron_projections'):
            p.inspect_neuron('cell')
        self.assertIn('Distinct endpoint-target regions',stream.getvalue())
        self.assertNotIn('atlas voxels',stream.getvalue())

    def test_cli_help_states_identity_units_and_fresh_output(self):
        env=os.environ.copy();env['PYTHONPATH']=str(STAGE)+os.pathsep+str(ROOT/'main_scripts')
        result=subprocess.run([sys.executable,'-B',str(STAGE/'region_analysis/laterality_projection_analysis.py'),'--help'],
            capture_output=True,text=True,env=env,check=True)
        compact=' '.join(result.stdout.split())
        self.assertIn('retained reconstruction lengths',compact)
        self.assertIn('NeuronUID or SampleID+NeuronID',compact)
        self.assertIn('Fresh output Excel path',compact)
        self.assertIn('source units are preserved',compact)

if __name__=='__main__':unittest.main(verbosity=2)
