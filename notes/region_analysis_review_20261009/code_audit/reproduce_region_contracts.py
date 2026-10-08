"""Read-only/in-memory reproductions; generated workbooks stay in temp dirs."""
from pathlib import Path
import sys, json, hashlib, contextlib, io, tempfile, importlib.util
from types import SimpleNamespace as N
from unittest.mock import patch
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'main_scripts'))
from region_analysis.neuron_analysis import RegionAnalysisPerNeuron as Analysis
from region_analysis.population import PopulationRegionAnalysis as Pop
from region_analysis.hierarchy_table import HierarchyTable,DualHierarchyTable
import neuro_tracer


def node(i,x,t=2,parent=-1):
    return N(id=i,x=x*250,y=0,z=0,x_nii=x,y_nii=0,z_nii=0,
             type=t,parent=parent,children=[],order=0,is_terminal=False)


def execute():
    results={}
    atlas=np.array([1,2,2]).reshape(3,1,1)
    key=pd.DataFrame({'Index':[1,2], 'Abbreviation':['CL_A','SL_PrC']})
    root,leaf=node(1,0,1),node(2,2,3,1)
    a=Analysis(N(root=root,branches=[[root,leaf]],terminal_nodes=[leaf]),atlas,key);a.run()
    results['dendrite_leaf_is_classification_target']={'terminal_regions':a.terminal_regions,
        'retained_lengths':a.mapped_brain_region_lengths,'total_voxels':a.neuron_total_length}
    short=node(2,0.0004,2,1)
    a=Analysis(N(root=root,branches=[[root,short]],terminal_nodes=[short]),atlas,key);a.run()
    results['per_edge_rounding_zeroes_nonzero_edge']={'true_voxel_length':0.0004,'reported':a.neuron_total_length}
    n=neuro_tracer.neuro_tracer();n.nodes={1:root};n.root=root
    with contextlib.redirect_stdout(io.StringIO()):n._mark_branch_terminals()
    a=Analysis(n,atlas,key)
    results['root_only_leaf']={'terminal_count':len(n.terminal_nodes),'root_id':n.terminal_nodes[0].id}
    # Two distinct labels with the same abbreviation lose length in dict conversion.
    second=node(2,1,2,1);last=node(3,2,2,2)
    a=Analysis(N(root=root,branches=[[root,second,last]],terminal_nodes=[last]),atlas,
        pd.DataFrame({'Index':[1,2],'Abbreviation':['CL_X','CL_X']}));a.run()
    results['duplicate_abbreviation_loses_length']={'total_length':a.neuron_total_length,'retained':a.mapped_brain_region_lengths}
    # Valid SWC branch depth can exceed the recursive branch constructor's limit.
    n=neuro_tracer.neuro_tracer();n.root=node(1,0,1)
    cur=n.root
    for i in range(1200):
        nxt=node(2*i+2,i+1);cur.children=[nxt,node(2*i+3,i+1)];cur=nxt
    try:
        with contextlib.redirect_stdout(io.StringIO()):n._construct_branches()
        results['deep_fork_tree']='accepted'
    except Exception as e:results['deep_fork_tree']=type(e).__name__+': '+str(e)

    class Fake:
        def process(self,sample,name,**kwargs):
            self.swc_filename=name;self.root=node(1,0,1);self.branches=[]
            self.terminal_nodes=[self.root]
    p=Pop('sampleA',atlas,key,neuron_list=[{'sampleid':'sampleB','name':'001.swc'}]*2,
          auto_hierarchy=False,auto_laterality=False)
    with patch('region_analysis.population.nt.neuro_tracer',Fake),contextlib.redirect_stdout(io.StringIO()):p.process()
    results['duplicate_cross_sample_entry']={'rows':p.plot_dataframe[['SampleID','NeuronID']].to_dict('records'),
        'neuron_objects':len(p.neurons)}
    p=Pop('sampleA',np.zeros((3,1,1,1,6)),key,neuron_list=[],auto_hierarchy=False,auto_laterality=False)
    try:
        with contextlib.redirect_stdout(io.StringIO()):p.process(level=0)
        results['invalid_level_zero']='accepted (selects last volume through -1 indexing)'
    except Exception as e:results['invalid_level_zero']=repr(e)
    t=DualHierarchyTable.load(ROOT/'atlas/CHARM_key_table_v2.csv',ROOT/'atlas/SARM_key_table_v2.csv')
    for k,fn in [('debug_dual',lambda:Pop.debug_hierarchy(N(dual_hierarchy=t),'CL_Ial')),
                 ('debug_resolution',lambda:Pop.debug_region_resolution(N(dual_hierarchy=t),'CL_Ial'))]:
        try:
            with contextlib.redirect_stdout(io.StringIO()):fn()
            results[k]='accepted'
        except Exception as e:results[k]=type(e).__name__+': '+str(e)
    # Current real table numeric parent resolves an arbitrary descendant at L6.
    table=t.cortex
    results['real_numeric_parent_descendant']={'name':table.get_at_level('CL_Frontal',6),
        'index':table.get_at_level('1',6)}
    p=Pop('sampleA',atlas,key,neuron_list=[],auto_hierarchy=False,auto_laterality=False,
          arm_key_path=str(ROOT/'atlas/ARM_key_all.txt'),
          cortex_hierarchy_csv=str(ROOT/'atlas/CHARM_key_table_v2.csv'),
          subcortical_hierarchy_csv=str(ROOT/'atlas/SARM_key_table_v2.csv'))
    p.plot_dataframe=pd.DataFrame({'NeuronID':['001.swc'],'Soma_Region':['CL_Ial'],
        'Terminal_Regions':[['CL_Ial']],'Region_projection_length':[{'CL_Ial':2.0}]})
    with contextlib.redirect_stdout(io.StringIO()):p._apply_hierarchy_columns(projection_min_level=1)
    before=p.plot_dataframe.Region_Projection_Length_L3.iloc[0]
    with contextlib.redirect_stdout(io.StringIO()):p.add_projection_hierarchy_columns(min_level=1)
    results['public_projection_hierarchy_ignores_dual']={'before_L3':before,
        'after_L3':p.plot_dataframe.Region_Projection_Length_L3.iloc[0]}
    try:
        p.plot_dataframe['Region_Projection_Length_finest']=[{'CL_Ial':2.0}]
        with contextlib.redirect_stdout(io.StringIO()):p.diagnose_projection_regions()
        results['diagnostic_category_case']='accepted'
    except Exception as e:results['diagnostic_category_case']=type(e).__name__+': '+str(e)
    spec=importlib.util.spec_from_file_location('harmonize_audit',ROOT/'group_analysis/scripts/06_harmonize_atlas_to_manual.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    with tempfile.TemporaryDirectory() as d:
        source,dest=Path(d)/'input.xlsx',Path(d)/'output.xlsx'
        summary=pd.DataFrame({'SampleID':[252385],'NeuronID':['001.swc'],'NeuronUID':['252385::001.swc'],
            'Soma_Region_Refined':['IAL'],'Soma_Region_Source':['coord_candidate']})
        with pd.ExcelWriter(source) as w:
            summary.to_excel(w,sheet_name='Summary',index=False)
            pd.DataFrame({'formula':['=1+1']}).to_excel(w,sheet_name='Notes',index=False)
        before=hashlib.sha256(source.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):h.harmonize(source,dest)
        import openpyxl
        a=openpyxl.load_workbook(source,data_only=False);b=openpyxl.load_workbook(dest,data_only=False)
        results['harmonizer_metadata_formula']={'source_formula':a['Notes']['A2'].value,
            'derivative_value':b['Notes']['A2'].value,'source_unchanged':before==hashlib.sha256(source.read_bytes()).hexdigest()}
        a.close();b.close()
    # Quantify broad legacy CT range against explicit SARM L2 ancestry.
    sarm=pd.read_csv(ROOT/'atlas/SARM_key_table_v2.csv');bad=set()
    for _,r in sarm.iterrows():
        idx=int(r.Level_6_index)
        if ((1111<=idx<=1168) or (1611<=idx<=1668)) and r.Level_2_abbr not in {'SL_Thal','SR_Thal'}:
            bad.add(r.Level_6_abbr)
    sources=sorted((ROOT/'group_analysis/step1_results').glob('*/tables/*results*.xlsx'))
    records=[];labels={};total=0;table_checks=[]
    for path in sources:
        summary=pd.read_excel(path,sheet_name='Summary');terms=pd.read_excel(path,sheet_name='Terminal_Sites')
        sheets=pd.read_excel(path,sheet_name=None)
        proj_checks={name:{'rows':len(frame),'missing_neurons':len(set(summary.NeuronID)-set(frame.NeuronID)),
             'extra_neurons':len(set(frame.NeuronID)-set(summary.NeuronID)),
             'duplicate_neurons':int(frame.NeuronID.duplicated().sum())}
             for name,frame in sheets.items() if name.startswith('Projection_') and 'NeuronID' in frame}
        table_checks.append({'path':str(path.relative_to(ROOT)),'rows':len(summary),
            'duplicate_neurons':int(summary.NeuronID.duplicated().sum()),
            'missing_Length_Unit':'Length_Unit' not in summary,
            'terminal_L3_nonnull':int(terms.Terminal_L3.notna().sum()) if 'Terminal_L3' in terms else None,
            'terminal_rows':len(terms),'projection_sheets':proj_checks})
        total+=len(summary);hit=terms[terms.Terminal_Region.isin(bad)]
        counts=hit.Terminal_Region.value_counts().to_dict()
        for label,count in counts.items():labels[label]=labels.get(label,0)+int(count)
        ids=set(hit.NeuronID)
        for _,r in summary[summary.NeuronID.isin(ids)].iterrows():
            targets=terms.loc[terms.NeuronID.eq(r.NeuronID),'Terminal_Region'].dropna().tolist()
            records.append({'SampleID':str(int(r.SampleID)),'NeuronID':r.NeuronID,
                'saved_type':r.Neuron_Type,'bad_targets':[x for x in targets if x in bad],
                'other_thalamic_targets':[x for x in targets if x not in bad and x in set(sarm.loc[sarm.Level_2_abbr.isin(['SL_Thal','SR_Thal']),'Level_6_abbr'])]})
    results['cached_legacy_CT_range']={'scope':'nine existing step1 workbooks only; reference251637 not included',
        'total_neurons':total,'affected_neurons':len(records),'bad_target_counts':labels,'records':records,
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    results['cached_table_membership_checks']=table_checks
    out=ROOT/'notes/region_analysis_review_20261009/code_audit/region_contract_reproductions.json'
    convert=lambda x: x.item() if isinstance(x,np.generic) else str(x)
    out.write_text(json.dumps(results,indent=2,default=convert),encoding='utf-8')
    print(json.dumps({k:v for k,v in results.items() if k!='cached_legacy_CT_range'},indent=2,default=convert))
    print(json.dumps({k:v for k,v in results['cached_legacy_CT_range'].items() if k not in {'records','source_hashes'}},indent=2))


if __name__=='__main__':execute()
