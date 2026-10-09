"""Independent final byte, marker-plane and document-link readback."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import nibabel as nib
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    base = Path(__file__).resolve().parent
    root = base.parents[1]
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh final readback receipt')
    inspection_path = base / 'reproduction_checked/inspection/inspection_receipt.json'
    inspection = json.loads(inspection_path.read_text())
    assert inspection['status'] == 'passed_full_voxel_relations_and_direct_graph_endpoints'
    assert inspection['map_files_checked'] == 452 and inspection['unique_source_neurons'] == 462
    assert sha(base / 'inspect_projection_maps.py') == inspection['inspector_sha256']
    assert sha(inspection_path.parent / 'all_map_value_checks.csv') == inspection['map_value_checks_sha256']
    for binding in inspection['runs'].values():
        assert sha(binding['path']) == binding['sha256']
    display_path = base / 'reproduction_checked/endpoint_markers/display_provenance.json'
    display = json.loads(display_path.read_text())
    assert display['inspection_sha256'] == sha(inspection_path)
    assert display['renderer_sha256'] == sha(base / 'render_endpoint_markers.py')
    assert display['MIP'] is False and display['smoothing'] is None
    assert display['source_maps_modified'] is False
    run_path = Path(inspection['runs']['main_endpoints']['path'])
    run = json.loads(run_path.read_text())
    group = {entry['Subregion']: entry for entry in run['group_maps']}
    for entry in display['panel_counts']:
        data = np.asarray(nib.load(run_path.parent / group[entry['source']]['density_path']).dataobj)
        plane = np.take(data, entry['cut'], axis='XYZ'.index(entry['axis']))
        assert int(np.count_nonzero(plane)) == entry['occupied_voxels_drawn']
        assert np.isclose(float(plane.sum(dtype=np.float64)), entry['slice_density_sum'], rtol=1e-12, atol=1e-10)
    figures = []
    for entry in display['figures']:
        path = display_path.parent / entry['path']
        assert sha(path) == entry['sha256']
        with Image.open(path) as image:
            image.load(); figures.append({'path': entry['path'], 'pixels': list(image.size)})
    links = []
    for document in (base / 'README.md', base / 'reproduction.md'):
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            path = (document.parent / target.split('#', 1)[0]).resolve()
            assert path.exists(), (document, target)
            links.append(target)
    tests = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', 'test_map_inspection'], cwd=base, capture_output=True, encoding='utf-8')
    assert tests.returncode == 0, tests.stderr
    preflight = json.loads((base / 'procedure_preflight.json').read_text())
    assert preflight['driver_sha256'] == sha(base / 'reproduce_maps.ps1')
    for item in preflight['cli_entrypoints']:
        assert sha(root / item['script']) == item['script_sha256']
    # The publication helper updates its catalog. Bind the sorting input to
    # its exact earlier Git snapshot rather than overwrite the dated receipt.
    snapshot = '515f7334eb293279700984c1de21460dde4b525f'
    locator = snapshot + ':notes/region_analysis_review_20261009/publication_local_artifacts.json'
    old = subprocess.check_output(['git', '-c', 'safe.directory=' + root.as_posix(), 'show', locator], cwd=root)
    sorting = json.loads((base / 'sorting/sorting_receipt.json').read_text())
    assert hashlib.sha256(old).hexdigest() == sorting['catalog_sha256']
    for file in ('archive_readback.json', 'development_archive_readback.json'):
        for item in json.loads((base / 'sorting' / file).read_text())['files']:
            assert not (root / item['old_path']).exists()
            assert sha(root / item['new_path']) == item['sha256']
    result = {'status': 'passed_final_source_script_marker_plane_archive_and_link_readback', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'verifier_sha256': sha(__file__), 'inspection_sha256': sha(inspection_path), 'display_sha256': sha(display_path),
              'figures': figures, 'marker_panels_checked': len(display['panel_counts']), 'resolved_document_links': len(links),
              'focused_tests': {'exit_code': tests.returncode, 'output': tests.stderr}, 'sorting_catalog_git_snapshot': locator,
              'full_numerical_rebuild_this_round': False, 'anatomical_acceptance': False}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
