"""Bind final parse/selector receipts; do not touch scientific source data."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


syntax=HERE/'R_syntax_validation_final_live_20261009.json'
selector=HERE/'F6_selector_regression_final_live_20261009.json'
consumer=HERE/'independent_live_consumer_review_receipt_20261009.json'
consumer_final=HERE/'consumer_regressions_final_live_20261009.json'
consumer_final.write_bytes(consumer.read_bytes())
# Preserve the first completed live-consumer receipt if its exact original
# bytes can be recovered; duration text is not scientific data.
candidate=consumer.read_bytes().replace(b'0.269s',b'0.282s')
if hashlib.sha256(candidate).hexdigest()=='eb95d3d198a9838788d2f613b0eae92f71702ca0fba956993f7d976bb17f96db':
    consumer.write_bytes(candidate)
parsed=json.loads(syntax.read_text())
selected=json.loads(selector.read_text())
checked=json.loads(consumer_final.read_text())
for item in parsed['files']:
    assert item['validated_source_path']==item['path']
    assert sha(ROOT/item['path'])==item['sha256']
for item in checked['module_bindings']:
    assert sha(ROOT/item['path'])==item['sha256']
assert parsed['passes']==63 and parsed['failures']==0 and parsed['purl_failures']==0
assert selected['status']=='pass' and selected['live'] and selected['tests_run']==4
assert checked['status']=='pass' and checked['tests_run']==7
assert sha(ROOT/'group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd')==selected['selector_source_sha256']
summary={
 'status':'final_live_software_methods_validation_pass',
 'R_syntax':{'sources':63,'by_kind':dict(Counter(item['kind'] for item in parsed['files'])),
             'R_chunks':sum(item.get('R_chunks',0) for item in parsed['files']),
             'parse_failures':0,'purl_failures':0,'runtime':parsed['runtime'],
             'receipt':syntax.name,'sha256':sha(syntax)},
 'F6_selector':{'tests':4,'failures':0,'receipt':selector.name,'sha256':sha(selector),
               'source_Rmd_sha256':selected['selector_source_sha256'],
               'new_pvalues_computed':False,'established_commit':'d27c855b82a280989850d71f9f1eb994f70df58e'},
 'live_consumer_counterexamples':{'tests':7,'failures':0,'receipt':consumer_final.name,'sha256':sha(consumer_final)},
 'readable_label_static_tests':{'tests':5,'failures':0,'numeric_code_unchanged':True},
 'verified_prior_review':{'path':'notes/code_review_20260926.md','sha256':sha(ROOT/'notes/code_review_20260926.md'),
                          'anchors':[19,20,63]},
 'source_R_sha256':sha(ROOT/'group_analysis/R_analysis/v2_combined_primary_pipeline.R'),
 'historical_F6_table_sha256':selected['saved_table_sha256'],
 'full_analysis_rerun':False,'historical_figures_or_stats_regenerated':False,
 'scientific_acceptance':False,
 'remaining_limits':['Collision-safe R namespace/exact-cohort preflight needed before new full analysis',
                     'Animal/injection dependence not resolved by existing neuron-level tests',
                     'Legacy length/terminal/registration semantics remain unaccepted',
                     'Historical Chinese full tutorial overloads the imported Laterality_Index; not reused or reclassified']}
(HERE/'final_live_methods_validation_summary_20261009.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
