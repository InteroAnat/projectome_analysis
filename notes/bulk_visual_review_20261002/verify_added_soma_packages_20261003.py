"""Read all added soma packages against validated roots; no source downloads."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'group_analysis/visual_review_20261002'
validation = json.loads((HERE / 'unresolved_full_source_validation_20261003.json').read_text())
records = []
for validated in validation['records']:
    uid = validated['uid']
    sample, neuron = uid.split('|')
    folder = OUT / sample / Path(neuron).stem
    provenance_path = folder / 'provenance.json'
    before = provenance_path.read_bytes()
    provenance = json.loads(before)
    meta = json.loads((folder / 'soma.nii.gz.json').read_text())
    acquisition = meta['acquisition']
    raw = OUT / 'swc_raw' / sample / neuron
    assert provenance['UID'] == uid
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == provenance['swc_sha256'] == validated['sha256']
    assert provenance['swc_url'] == validated['url']
    native = validated['raw_root_xyz']
    np.testing.assert_allclose(meta['center_xyz_um'], native, atol=1e-9)
    np.testing.assert_allclose(acquisition['center_xyz_um'], native, atol=1e-9)
    np.testing.assert_allclose([provenance['native_soma_' + axis + '_um'] for axis in 'xyz'], native, atol=1e-9)
    assert meta['sample_id'] == acquisition['sample_id'] == sample
    assert meta['array_axis_order'] == meta['nifti_axis_order'] == 'XYZ'
    assert meta['source_array_axis_order'] == acquisition['array_axis_order'] == 'ZYX'
    image = nib.load(str(folder / 'soma.nii.gz'))
    assert image.header.get_xyzt_units()[0] == 'micron'
    assert image.get_data_dtype() == np.dtype('uint16')
    assert list(image.shape) == meta['shape_xyz'] == meta['shape_zyx'][::-1]
    np.testing.assert_allclose(image.affine[:3, 3], meta['origin_xyz_um'], atol=1e-3)
    np.testing.assert_allclose(np.diag(image.affine)[:3], meta['spacing_xyz_um'], atol=1e-6)
    local = (np.array(native) - meta['origin_xyz_um']) / meta['spacing_xyz_um']
    assert np.all(local >= 0) and np.all(local < image.shape)
    requested = {tuple(cube) for cube in acquisition['requested']}
    loaded = {tuple(cube) for cube in acquisition['loaded']}
    missing = {tuple(cube) for cube in acquisition['missing']}
    assert loaded.isdisjoint(missing) and requested == loaded | missing
    center_cube = tuple(np.floor(np.array(native) / [.65, .65, 3] / [360, 360, 90]).astype(int))
    assert center_cube in loaded
    assert meta['complete'] == acquisition['complete'] == (not missing)
    assert provenance['soma_complete'] == (not missing)
    assert provenance['coverage_status'] == ('partial_neighbor' if missing else 'complete')
    assert provenance['production_status'] == ('partial' if missing else 'soma_only')
    assert all(url.startswith(f'http://bap.cebsit.ac.cn/monkeydata/{sample}/cube/') for url in acquisition['source_paths'])
    assert len(acquisition['source_paths']) == len(requested)
    panels = {}
    for name in ('soma_unmarked.png', 'soma_marked.png', 'soma_trace.png', 'soma_stack.png', 'soma_ortho.png'):
        path = folder / name
        with Image.open(path) as panel:
            panel.load()
            size = list(panel.size)
        panels[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size': size}
    assert provenance_path.read_bytes() == before, 'Provenance changed during readback'
    records.append({'uid': uid, 'status': provenance['production_status'],
                    'coverage': provenance['coverage_status'], 'loaded_cubes': len(loaded),
                    'missing_cubes': len(missing), 'raw_sha256': validated['sha256'],
                    'provenance_sha256': hashlib.sha256(before).hexdigest(), 'panels': panels})
    print(json.dumps({'checked_uid': uid}), flush=True)
assert len(records) == 86 and len({row['uid'] for row in records}) == 86
result = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
          'scope': 'All 86 added packages: own raw hashes and roots, saved NIfTI geometry/headers, complete PNG decode, source identity and coverage accounting; source-pixel equivalence and anatomical acceptance remain separate final checks',
          'checked_identities': len(records), 'pngs_fully_decoded': len(records) * 5,
          'coverage_counts': dict(Counter(row['coverage'] for row in records)),
          'network_requests': 0, 'anatomical_acceptance': False, 'records': records}
(HERE / 'added_soma_package_readback_20261003.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'records'}))
