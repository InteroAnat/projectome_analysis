"""Recover exact recorded label-patch bases from reversible staged edits."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
receipt_path = HERE/'additional_factual_labels_staging_receipt_20261009.json'
receipt = json.loads(receipt_path.read_text())
tree = ast.parse((HERE/'prepare_additional_factual_label_patch_20261009.py').read_text())
functions = {node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
for record in receipt['files']:
    if record['path'].startswith('group_analysis/'):
        continue
    after = (HERE/'proposed_additional_factual_labels_20261009'/record['path']).read_text(encoding='utf-8')
    function = functions['comparison' if record['path'].startswith('notes/') else 'tutorial']
    replacements = []
    for statement in function.body:
        value = getattr(statement,'value',None)
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='replace':
            replacements.append(tuple(ast.literal_eval(arg) for arg in value.args[1:3]))
    for old,new in reversed(replacements):
        assert new in after
        after = after.replace(new,old)
    candidates = [after.encode('utf-8'),after.replace('\n','\r\n').encode('utf-8')]
    matches = [data for data in candidates if hashlib.sha256(data).hexdigest()==record['base_sha256']]
    assert len(matches)>=1,record['path']
    path = HERE/'additional_factual_labels_base_20261009'/record['path']
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(matches[0])
    record['base_path'] = str(path.relative_to(ROOT))
receipt['base_freeze_method'] = 'Inverse of recorded string-only substitutions, accepted only on exact prepatch SHA256 match; production never changed'
receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('Four exact original bases recovered and frozen; patch bytes unchanged')
