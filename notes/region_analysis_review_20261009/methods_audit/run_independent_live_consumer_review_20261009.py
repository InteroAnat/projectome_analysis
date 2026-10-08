"""Run focused live consumer counterexamples and bind actual imported sources."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.environ['PROJECTOME_AUDIT_MODULE_ROOT']=str(ROOT/'main_scripts')
path=HERE/'test_final_staged_consumers_independent.py'
spec=importlib.util.spec_from_file_location('independent_live_consumer_cases',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
names=['region_analysis.utils','region_analysis.laterality',
       'region_analysis.laterality_projection_analysis','region_analysis.population']
bindings=[]
for name in names:
    source=Path(sys.modules[name].__file__).resolve()
    assert source.is_relative_to(ROOT/'main_scripts')
    bindings.append({'module':name,'path':str(source.relative_to(ROOT)),
                     'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
for item in bindings:
    assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
receipt={'status':'pass' if result.wasSuccessful() else 'fail','live':True,
  'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
  'module_bindings':bindings,'test_path':str(path.relative_to(ROOT)),
  'test_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'log':stream.getvalue(),
  'counterexamples_resolved':['Contradictory composite UID now rejected',
    'Summary/Soma_Hierarchy conflict now rejected',
    'Prefixed Unknown_0 targets retained consistently as unresolved'],
  'test_scope':['Multi-subject same bare ID joins and repeated long records',
    'Exact membership/duplicate/blank/UID checks','Metadata retention and C/S collision-safe split',
    'Explicit/legacy hierarchy conflict','Standalone stale-side reclassification',
    'Unknown side/target length conservation'],
  'production_writes_by_this_runner':False,'scientific_acceptance':False}
(HERE/'independent_live_consumer_review_receipt_20261009.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(stream.getvalue())
raise SystemExit(0 if result.wasSuccessful() else 1)
