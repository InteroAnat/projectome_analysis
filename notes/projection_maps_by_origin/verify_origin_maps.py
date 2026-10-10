"""Independent readback of source somata and candidate ends from original SWCs.

No map producer imports. Passing this check establishes computational agreement,
not accepted NMT registration, soma anatomy, or biological terminal fields.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'group_analysis/evolution_20261008/soma_origin_maps_20261010'
LEDGER = ROOT / 'notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv'
INS = {41, 42, 43, 228, 229, 541, 542, 728, 729}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def category(row):
    index, human = int(row['ARMIndex']), row['Henry_coarse_INS_visual_evidence'] == 'True'
    if index in INS:
        return 'atlasAndHenry' if human else 'atlasOnly'
    if index == 0:
        return 'HenryUnassigned' if human else 'unassignedCandidate'
    return 'HenryPrCOConflict' if human else 'neighborCandidate'


def verify(output=OUTPUT, receipt=None):
    output = Path(output)
    named_dir, evidence_dir = output / 'named_ARM_origins', output / 'evidence_categories'
    named = json.loads((named_dir / 'run_provenance.json').read_text())
    evidence = json.loads((evidence_dir / 'run_provenance.json').read_text())
    for binding in named['inputs'].values():
        check(sha(binding['path']) == binding['sha256'], 'Named map input changed: ' + binding['path'])
    for key in ('background', 'atlas', 'atlas_key'):
        check(sha(named[key]['path']) == named[key]['sha256'], 'Image/key input changed')
    check(sha(ROOT / 'group_analysis/scripts/map_insula_origin_evidence.py') == evidence['producer_sha256'], 'Evidence producer changed')
    check(sha(ROOT / 'group_analysis/scripts/map_projections_by_soma_origin.py') == evidence['classification_helper_sha256'], 'Classification helper changed')
    for binding in evidence['leaf_records']:
        check(sha(binding['path']) == binding['sha256'] and sha(binding['run_path']) == binding['run_sha256'], 'Cached leaves changed')
    check(sha(LEDGER) == evidence['ledger']['sha256'], 'Evidence ledger changed')
    table = pd.read_csv(LEDGER, dtype=str, keep_default_na=False)
    check(len(table) == 462 and table.NeuronUID.nunique() == 462, 'Full source identities differ')
    ref = nib.load(named['background']['path'])
    shape, affine = ref.shape, ref.affine
    volume = abs(np.linalg.det(affine[:3, :3]))
    atlas = np.asarray(nib.load(named['atlas']['path']).dataobj)[..., 0, 5]
    source_voxels, category_members, graph_ends = {}, defaultdict(list), {}
    checked_sources, total_ends = [], 0
    for row in table.to_dict('records'):
        check(sha(row['SWCPath']) == row['SWCSHA256'], 'Source changed')
        data = np.loadtxt(row['SWCPath'], comments='#', ndmin=2)
        check(data.shape[1] == 7 and np.isfinite(data).all(), 'Invalid SWC')
        ids, kinds, parents = (data[:, i].astype(np.int64) for i in (0, 1, 6))
        check(np.all(data[:, [0, 1, 6]] == np.rint(data[:, [0, 1, 6]])), 'Nonintegral node metadata')
        check(len(np.unique(ids)) == len(ids) and sum(parents == -1) == 1 and np.isin(parents[parents != -1], ids).all(), 'Invalid graph identities')
        soma = data[kinds == 1]
        check(len(soma) == 1 and int(soma[0, 0]) == int(row['SourceRootNodeID']) and soma[0, 6] == -1, 'Soma/root differs')
        scale = np.array(row['IndexScaleUm'].split(';'), float)
        point = soma[0, 2:5] / scale
        voxel = np.rint(point).astype(int)
        check(np.array_equal(voxel, json.loads(row['SourceRootVoxelXYZ'])) and np.allclose(point, json.loads(row['SourceRootIndexXYZ']), rtol=0, atol=1e-10), 'Source coordinate record differs')
        check(((voxel >= 0) & (voxel < shape)).all() and int(atlas[tuple(voxel)]) == int(row['ARMIndex']), 'Direct source ARM lookup differs')
        uid = row['NeuronUID']
        source_voxels[uid] = tuple(voxel)
        source_category = category(row)
        category_members[source_category].append(row)
        leaves = (kinds == 2) & (parents != -1) & ~np.isin(ids, parents)
        n = int(leaves.sum())
        check(n == int(row['Endpoint_candidate_axon_endpoint_count']) and (n > 0) == (row['EndpointEligible'] == 'True'), 'Direct graph eligibility differs')
        points = data[leaves, 2:5] / scale
        ends = np.column_stack([np.searchsorted(np.arange(size + 1) - .5, points[:, a], side='right') - 1 for a, size in enumerate(shape)])
        inside = ((ends >= 0) & (ends < shape)).all(axis=1)
        flat, counts = np.unique(np.ravel_multi_index(ends[inside].T, shape), return_counts=True)
        graph_ends[uid] = (n, flat, counts)
        total_ends += n
        checked_sources.append({'NeuronUID': uid, 'sha256': row['SWCSHA256'], 'candidate_ends': n, 'in_reference_ends': int(inside.sum())})
    check(total_ends == 75625, 'Full direct endpoint census differs')
    map_checks = []

    def read_map(directory, binding):
        path = directory / binding['path']
        check(sha(path) == binding['sha256'], 'Output map changed')
        image = nib.load(path)
        values = np.asarray(image.dataobj)
        check(image.shape == shape and np.array_equal(image.affine, affine) and image.header.get_xyzt_units()[0] == 'mm', 'Output image grid/units differ')
        check(int(image.header['sform_code']) == 2 and int(image.header['qform_code']) == 0, 'Output coded transform differs')
        check(np.isfinite(values).all() and (values >= 0).all(), 'Invalid output map values')
        return values

    for directory, entries in ((named_dir, named['outputs']), (evidence_dir, evidence['outputs'])):
        for view, entry in entries.items():
            rows = (table[table.ARMIndex.eq(str(entry['ARMIndex']))].to_dict('records')
                    if directory == named_dir else category_members[view])
            check(len(rows) == entry['selected_neurons'], 'Source view membership count differs')
            expected_soma = np.zeros(shape, dtype=np.int32)
            for row in rows:
                expected_soma[source_voxels[row['NeuronUID']]] += 1
            actual_soma = read_map(directory, entry['artifacts']['soma_count'])
            check(np.array_equal(actual_soma, expected_soma), 'Source-derived soma count differs')
            actual_density = read_map(directory, entry['artifacts']['projection_density'])
            if directory == named_dir:
                binding = entry['original_projection']
                check(sha(binding['path']) == binding['sha256'], 'Original projection map changed')
                check(np.array_equal(actual_density, np.asarray(nib.load(binding['path']).dataobj)), 'Origin map alters established projection values')
            sums, denominators = defaultdict(Counter), Counter()
            for row in rows:
                n, flat, counts = graph_ends[row['NeuronUID']]
                if n:
                    denominators[row['AnimalID']] += 1
                    sums[row['AnimalID']].update(dict(zip(map(int, flat), map(int, counts))))
            expected_density = np.zeros(int(np.prod(shape)), dtype=np.float64)
            for animal, denominator in denominators.items():
                for flat, count in sums[animal].items():
                    expected_density[flat] += count / denominator / len(denominators) / volume
            expected_density = expected_density.reshape(shape).astype(np.float32)
            check(np.allclose(actual_density, expected_density, rtol=2e-7, atol=1e-7), 'Direct full-graph animal-mean map differs')
            check(sum(denominators.values()) == entry['eligible_neurons'], 'View denominator differs')
            for panel in entry['panels']:
                if panel.get('kind') == 'soma':
                    check(sum(source_voxels[r['NeuronUID']][panel['axis']] == panel['cut'] for r in rows) == panel['visible_neurons'], 'Soma locator count differs')
                    target = table[table.SampleID.eq(panel['arrow_target_SampleID']) & table.NeuronID.eq(panel['arrow_target_NeuronID'])].iloc[0]
                    point = nib.affines.apply_affine(affine, json.loads(target.SourceRootIndexXYZ))
                    in_plane = [a for a in range(3) if a != panel['axis']]
                    check(target.NeuronUID in {r['NeuronUID'] for r in rows} and source_voxels[target.NeuronUID][panel['axis']] == panel['cut'], 'Origin arrow points outside displayed source selection')
                    check(np.allclose(point[in_plane], panel['arrow_target_template_mm'], rtol=0, atol=1e-12), 'Origin arrow target moved from actual soma')
                else:
                    values = np.take(actual_density, panel['cut'], axis=panel['axis'])
                    check(np.count_nonzero(values) == panel['occupied_voxels'] and np.isclose(values.sum(dtype=np.float64), panel['slice_density_sum'], rtol=1e-12), 'Projection slice data differs')
            figure = directory / entry['artifacts']['figure']['path']
            check(sha(figure) == entry['artifacts']['figure']['sha256'], 'Figure changed')
            map_checks.append({'view': view, 'selected': len(rows), 'eligible': sum(denominators.values()), 'animals': len(denominators), 'soma_sum': int(actual_soma.sum()), 'density_sum': float(actual_density.sum(dtype=np.float64))})
            print(view, 'independent graph/map agreement', flush=True)
    check(Counter({key: len(rows) for key, rows in category_members.items()}) == Counter({'atlasAndHenry': 212, 'atlasOnly': 89, 'HenryPrCOConflict': 35, 'HenryUnassigned': 13, 'neighborCandidate': 74, 'unassignedCandidate': 39}), 'Six-category partition differs')
    for key, value in named['artifacts'].items():
        check(sha(named_dir / key) == value, 'Named map index changed')
    history = output / 'count_reconciliation'
    historical = json.loads((history / 'reconciliation_provenance.json').read_text())
    for binding in historical['sources'].values():
        check(sha(binding['path']) == binding['sha256'], 'Historical source changed')
    for file, value in historical['artifacts'].items():
        check(sha(history / file) == value, 'Historical comparison changed')
    old = pd.read_excel(historical['sources']['older_harmonized353']['path'], sheet_name='Summary', dtype=str)
    september = pd.read_excel(historical['sources']['September26_staging420']['path'], sheet_name='Summary', dtype=str)
    check(len(set(old.NeuronUID) & set(table.NeuronUID)) == 353 and len(set(table.NeuronUID) - set(old.NeuronUID)) == 109, 'Older identity reconciliation differs')
    check(len(set(september.NeuronUID) & set(table.NeuronUID)) == 412 and len(set(table.NeuronUID) - set(september.NeuronUID)) == 50, 'September identity reconciliation differs')
    result = {'status': 'passed_direct_source_graph_and_full_voxel_origin_map_readback', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'checker_sha256': sha(__file__), 'checked_sources': checked_sources, 'candidate_ends': total_ends, 'checked_maps': 30, 'checked_views': map_checks,
              'producer_receipts': {str(p.resolve()): sha(p) for p in (named_dir / 'run_provenance.json', evidence_dir / 'run_provenance.json', history / 'reconciliation_provenance.json')},
              'software_agreement': True, 'anatomical_acceptance': False, 'registration_acceptance': False, 'terminal_field_acceptance': False}
    path = Path(receipt) if receipt is not None else Path(__file__).with_name('independent_readback.json')
    check(not path.exists(), 'Preserve prior readback')
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    print(verify(args.output_root, args.receipt)['status'])
