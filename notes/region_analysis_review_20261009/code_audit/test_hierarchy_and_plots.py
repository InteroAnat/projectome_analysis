"""Independent focused public-API regressions for the 2026-10-09 audit."""
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
import hashlib
import io
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
import os
sys.path.insert(0, os.environ.get('PROJECTOME_AUDIT_MODULE_ROOT', str(ROOT / 'main_scripts')))
sys.path.append(str(ROOT / 'main_scripts'))
from region_analysis.hierarchy_table import HierarchyTable, DualHierarchyTable
from region_analysis.hierarchy import _aggregate_length_dict_to_level
from region_analysis.population import PopulationRegionAnalysis
from region_analysis.output_manager import OutputManager
import region_analysis.population as population_module
from region_analysis.plotting import (plot_region_distribution,
    plot_region_distribution_stacked, plot_laterality_summary_df)
import matplotlib.pyplot as plt
import neuro_tracer
plt.switch_backend('Agg')


def row(leaf='CL_A', index=4):
    return {'Level_1': 'Parent', 'Level_1_abbr': 'CL_P', 'Level_1_index': 1,
            'Level_3': 'Leaf', 'Level_3_abbr': leaf, 'Level_3_index': index}


class HierarchyPublicTests(unittest.TestCase):
    def test_xlsx_initial_and_incremental_load_preserves_sparse_ancestry(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a.xlsx', Path(d)/'b.XLSX'
            pd.DataFrame([row()]).to_excel(a, index=False)
            pd.DataFrame([row('CL_B',5)]).to_excel(b, index=False)
            p = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
            p._load_hierarchy_csv(a)
            p.load_hierarchy_csv(b)
            self.assertEqual(p.hierarchy_table.get_at_level('CL_A',1),'CL_P')
            self.assertIsNone(p.hierarchy_table.get_at_level('CL_A',2))
            self.assertEqual(p.hierarchy_table.get_at_level('CL_B',3),'CL_B')
            self.assertIsNone(p.hierarchy_table.get_at_level('CL_P',3))

    def test_single_file_initial_and_incremental_load(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a.csv', Path(d)/'b.csv'
            pd.DataFrame([row()]).to_csv(a, index=False)
            pd.DataFrame([row('CL_B',5)]).to_csv(b, index=False)
            p = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
            p._load_hierarchy_csv(a)
            self.assertEqual(p.hierarchy_table.get_at_level('CL_A',3),'CL_A')
            p.load_hierarchy_csv(b)
            self.assertEqual(p.hierarchy_table.get_at_level('CL_A',3),'CL_A')
            self.assertEqual(p.hierarchy_table.get_at_level('CL_B',3),'CL_B')
            with self.assertRaises(FileNotFoundError):
                p.load_hierarchy_csv(Path(d)/'absent.csv')
            self.assertEqual(p.hierarchy_table.get_at_level('CL_A',3),'CL_A')

    def test_single_aggregation_and_sparse_levels(self):
        t=HierarchyTable(pd.DataFrame([row()]))
        self.assertIsNone(t.get_at_level('CL_A',2))
        self.assertEqual(t.get_at_level('CL_A',3),'CL_A')
        self.assertEqual(_aggregate_length_dict_to_level({'CL_A':2},None,3,t,True),({'A':2},[]))

    def test_parent_has_no_arbitrary_descendant(self):
        t=HierarchyTable(pd.DataFrame([row(),row('CL_B',5)]))
        self.assertEqual(t.get_at_level('1',1),'CL_P')
        self.assertIsNone(t.get_at_level('1',3))
        self.assertIsNone(t.get_at_level('CL_P',3))

    def test_domain_routing_and_real_complete_table_baseline(self):
        t=DualHierarchyTable.load(ROOT/'atlas/CHARM_key_table_v2.csv',ROOT/'atlas/SARM_key_table_v2.csv')
        self.assertEqual(t.get_at_level('CL_Pi',6),'CL_Pi')
        self.assertEqual(t.get_at_level('SL_Pi',6),'SL_Pi')
        self.assertNotEqual(t.get_at_level('CL_Pi',3),t.get_at_level('SL_Pi',3))
        for table in (t.cortex,t.subcortex):
            for _,r in table.df.iterrows():
                leaf=r['Level_6_abbr']
                for level in range(1,7):
                    self.assertEqual(table.get_at_level(leaf,level),r[f'Level_{level}_abbr'])

    def test_conflicting_paths_fail(self):
        bad=row();bad['Level_1_abbr']='CL_Q'
        with self.assertRaises(ValueError):
            HierarchyTable(pd.DataFrame([row(),bad]))


class PlotContractTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_current_producer_hierarchy_columns_render(self):
        df=pd.DataFrame({'Soma_Level_3':['CL_A','CL_A','CR_B']})
        f=plot_region_distribution(df,3,show=False)
        self.assertIsNotNone(f)
        self.assertEqual(sorted(p.get_width() for p in f.axes[0].patches),[1,2])
        g=plot_region_distribution_stacked(df,levels=[3],show=False)
        self.assertIsNotNone(g)

    def test_automatic_plot_entrypoint_accepts_current_producer_columns(self):
        p=PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
        p.plot_dataframe=pd.DataFrame({'Soma_Level_3':['CL_A'], 'Soma_Region':['CL_A']})
        p.show_plots=False
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            p.output=OutputManager(d,'audit',create_timestamp=False)
            for name in ('plot_type_distribution_df','plot_soma_distribution_df',
                         'plot_terminal_distribution_df','plot_projection_sites_count_df'):
                stack.enter_context(patch.object(population_module,name))
            # Exercise the actual plotting helper, not only the dispatch condition.
            actual=population_module.plot_region_distribution_stacked
            called=stack.enter_context(patch.object(population_module,
                'plot_region_distribution_stacked', wraps=actual))
            p._generate_all_plots()
            called.assert_called_once()
            self.assertTrue((p.output.plots_dir/'region_dist_stacked.png').is_file())

    def test_length_plot_declares_observed_units_and_reconstruction_semantics(self):
        df=pd.DataFrame({'N_Ipsilateral':[1],'N_Contralateral':[0],
            'Laterality_Index':[0.0],'Total_Ipsilateral_Length':[4.0],
            'Total_Contralateral_Length':[0.0], 'Neuron_Type':['ITi'],
            'Length_Unit':['voxel']})
        f=plot_laterality_summary_df(df,show=False)
        self.assertIn('voxel',f.axes[2].get_ylabel())
        self.assertNotIn('Axon',f.axes[2].get_ylabel())
        self.assertIn('region',f.axes[0].get_title().lower())

    def test_zero_laterality_targets_render(self):
        df=pd.DataFrame({'N_Ipsilateral':[0],'N_Contralateral':[0]})
        self.assertIsNotNone(plot_laterality_summary_df(df,show=False))


class InputAndGraphTests(unittest.TestCase):
    def population(self, entries):
        return PopulationRegionAnalysis('A',np.zeros((2,2,2,1,6)),
            pd.DataFrame({'Index':[1],'Abbreviation':['CL_A']}),
            neuron_list=entries,auto_hierarchy=False,auto_laterality=False)

    def test_bad_levels_rejected_before_any_source_processing(self):
        p=self.population([])
        for level in [0,7,-1,1.5,True]:
            with self.subTest(level=level),self.assertRaises(ValueError):
                p.process(level=level)

    def test_duplicates_and_foreign_samples_rejected(self):
        for entries in [[{'sampleid':'B','name':'001.swc'}],
                        [{'sampleid':'A','name':'001.swc'}]*2]:
            with self.subTest(entries=entries),self.assertRaises(ValueError):
                self.population(entries).process()
        with self.assertRaises(ValueError):
            self.population([]).process(neuron_id=['001.swc','001.swc'])

    def test_diagnostics_use_supported_table_apis(self):
        p=self.population([])
        p.dual_hierarchy=DualHierarchyTable.load(ROOT/'atlas/CHARM_key_table_v2.csv',ROOT/'atlas/SARM_key_table_v2.csv')
        p.plot_dataframe=pd.DataFrame({'Region_projection_length':[{'CL_Ial':2.0}]})
        with contextlib.redirect_stdout(io.StringIO()):
            p.debug_hierarchy('CL_Ial');p.debug_region_resolution('SL_Pi')
            self.assertEqual(p.diagnose_projection_regions(),[])

    def test_public_projection_hierarchy_preserves_dual_source(self):
        p=self.population([])
        p.dual_hierarchy=DualHierarchyTable.load(ROOT/'atlas/CHARM_key_table_v2.csv',ROOT/'atlas/SARM_key_table_v2.csv')
        p.plot_dataframe=pd.DataFrame({'Soma_Region':['CL_Ial'],'Terminal_Regions':[['CL_Ial']],
            'Region_projection_length':[{'CL_Ial':2.0}]})
        p.add_projection_hierarchy_columns(min_level=1)
        self.assertEqual(p.plot_dataframe.Region_Projection_Length_L3.iloc[0],{'CL_caudal_OFC':2.0})

    def trace(self, text, directory):
        n=neuro_tracer.neuro_tracer();n.output_dir=directory
        with contextlib.redirect_stdout(io.StringIO()):
            n._loadSWC('A','001.swc',text);n._acquire_nii_nodes('monkey')
            n._find_children();n._mark_branch_terminals();n._construct_branches()
        return n

    def test_iterative_branches_preserve_edges_orders_and_dfs_order(self):
        text='1 1 0 0 0 1 -1\n2 2 250 0 0 1 1\n3 2 0 250 0 1 1\n4 2 500 0 0 1 2\n5 2 750 0 0 1 4\n6 2 500 250 0 1 4'
        with tempfile.TemporaryDirectory() as d:n=self.trace(text,d)
        self.assertEqual([[x.id for x in b] for b in n.branches],[[1,2,4],[4,5],[4,6],[1,3]])
        self.assertEqual({i:x.order for i,x in n.nodes.items()},{1:0,2:1,3:1,4:1,5:2,6:2})

    def test_deep_valid_swc_preserves_bytes_and_each_edge_once(self):
        lines=['1 1 0 0 0 1 -1'];parent=1
        for i in range(1200):
            child,leaf=2*i+2,2*i+3
            lines.extend([f'{child} 2 {i+1} 0 0 1 {parent}',f'{leaf} 2 {i+1} 1 0 1 {parent}'])
            parent=child
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'deep.swc';source.write_text('\n'.join(lines),encoding='utf-8')
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            n=self.trace(str(source),d)
            edges=[(a.id,b.id) for branch in n.branches for a,b in zip(branch,branch[1:])]
            self.assertEqual(len(edges),2400);self.assertEqual(len(set(edges)),2400)
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())


if __name__=='__main__':
    unittest.main(verbosity=2)
