"""Rebase auditor staging on current source without writing production files."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil

AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[2]
STAGED=AUDIT/'staged'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def method_span(text,name):
    tree=ast.parse(text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PopulationRegionAnalysis')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return node.lineno-1,node.end_lineno

relative=Path('main_scripts/region_analysis/population.py')
stage_text=(STAGED/relative).read_text(encoding='utf-8')
start,end=method_span(stage_text,'_generate_all_plots')
method=''.join(stage_text.splitlines(keepends=True)[start:end])
before={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'main_scripts/region_analysis').glob('*.py')}
for source in (ROOT/'main_scripts/region_analysis').glob('*.py'):
    shutil.copyfile(source,STAGED/source.relative_to(ROOT))
for name in ('neuro_tracer.py','region_labels.py','swc_validation.py'):
    source=ROOT/'main_scripts'/name
    if source.is_file():shutil.copyfile(source,STAGED/'main_scripts'/name)
current=(ROOT/relative).read_text(encoding='utf-8')
start,end=method_span(current,'_generate_all_plots')
lines=current.splitlines(keepends=True)
(STAGED/relative).write_text(''.join(lines[:start])+method+''.join(lines[end:]),encoding='utf-8')
assert before=={rel:sha(ROOT/rel) for rel in before},'Production changed during read-only stage preparation'

files=['main_scripts/region_analysis/population.py',
       'main_scripts/region_analysis/hierarchy_table.py',
       'group_analysis/scripts/06_harmonize_atlas_to_manual.py']
patch=[];hashes={}
for rel in files:
    source,staged=ROOT/rel,STAGED/rel
    hashes[rel]={'source_sha256':sha(source),'staged_sha256':sha(staged)}
    patch.extend(difflib.unified_diff(source.read_text(encoding='utf-8').splitlines(keepends=True),
        staged.read_text(encoding='utf-8').splitlines(keepends=True),
        fromfile='a/'+rel,tofile='b/'+rel))
proposal=AUDIT/'proposed_final_preservation_repairs.patch'
proposal.write_text(''.join(patch),encoding='utf-8')
receipt={
    'status':'staged_only_no_production_writes_by_auditor',
    'purpose':'Current-source follow-up; earlier hierarchy/graph repairs were integrated by parent',
    'files':hashes,'population_method_scope':['_generate_all_plots'],
    'hierarchy_xlsx_status':'already identical in current production and staged; independent public API regression added',
    'harmonizer_scope':'Original workbook objects retained; only documented harmonization columns and generated rule/provenance sheets are changed; existing Phase6a scoped enrichment remains',
    'historical_formula_incidence':{'workbooks':12,'formula_cells':0,'inventory':'actual_source_formula_inventory.json'},
    'production_snapshot_unchanged_during_preparation':True,
    'patch_sha256':sha(proposal),
    'tests':{'hierarchy_graph_plot':{'count':16,'status':'pending_final_rebase_run'},
             'harmonizer':{'count':1,'status':'pending_final_rebase_run'}},
    'test_source_hashes':{name:sha(AUDIT/name) for name in
        ('test_hierarchy_and_plots.py','test_harmonizer_preservation.py')},
    'validation_scope':'Software contracts on synthetic/temp inputs; no canonical rebuild or anatomical acceptance',
}
(AUDIT/'preservation_followup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
old_path=AUDIT/'expanded_staged_repair_receipt.json'
old=json.loads(old_path.read_text(encoding='utf-8'))
if '_generate_all_plots' not in old['population_method_scope']:
    old['population_method_scope'].append('_generate_all_plots')
old['followup_receipt']='preservation_followup_receipt.json'
old['snapshot_note']='Original 14-test hashes are retained as historical; follow-up receipt binds current-source narrow proposal and 17 regressions.'
old_path.write_text(json.dumps(old,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
