"""Fresh orthogonal map checks, with direct full-graph endpoint reconstruction.

No producer/detector imports. Preserve historical partitions and all inputs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'notes/region_analysis_review_20261009'
EVO = ROOT / 'group_analysis/evolution_20261008'
RUNS = {
    'main_endpoints': EVO / 'arm_mapping_20261009/main/endpoints',
    'additional_endpoints': EVO / 'endpoint_maps/additional_candidates_20261009',
    'main_axons': EVO / 'arm_mapping_20261009/main/axons',
    'additional_axons': EVO / 'projection_maps/additional_candidates_20261009',
    'end_branches': REVIEW / 'axon_end_branches_20261009/selected462_ARM',
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def endpoint_indices(rows, scale, shape):
    """Full-graph leaves first; independent right-sided voxel-face lookup."""
    require(rows.ndim == 2 and rows.shape[1] == 7 and np.isfinite(rows).all(), 'Invalid SWC matrix')
    require(np.asarray(scale).shape == (3,) and np.isfinite(scale).all() and np.all(np.asarray(scale) > 0), 'Positive XYZ coordinate scale required')
    require(np.all(rows[:, [0, 1, 6]] == np.rint(rows[:, [0, 1, 6]])), 'Nonintegral identities')
    ids, kinds, parents = (rows[:, i].astype(np.int64) for i in (0, 1, 6))
    require(len(np.unique(ids)) == len(ids), 'Duplicate node IDs')
    require(np.sum(parents == -1) == 1, 'SWC root count')
    require(np.isin(parents[parents != -1], ids).all(), 'Missing SWC parents')
    selected = (kinds == 2) & (parents != -1) & ~np.isin(ids, parents)
    points = rows[selected, 2:5] / scale
    voxels = np.column_stack([np.searchsorted(np.arange(n + 1) - .5, points[:, a], side='right') - 1
                              for a, n in enumerate(shape)])
    inside = ((voxels >= 0) & (voxels < shape)).all(axis=1)
    return int(selected.sum()), np.ravel_multi_index(voxels[inside].T, shape)


def check_values(count, density, occupancy, volume):
    require(np.allclose(density * volume, count, rtol=2e-7, atol=1e-7), 'Density/count conversion differs')
    require(np.all((occupancy >= 0) & (occupancy <= 1)), 'Occupancy outside [0,1]')
    require(np.array_equal(count > 0, occupancy > 0), 'Count/occupancy support differs')
    require(np.all(occupancy <= count + 1e-7), 'Occupancy exceeds endpoint count')


def inspect(output, runs=RUNS):
    output = Path(output)
    require(not output.exists(), 'Use a fresh inspection destination')
    output.mkdir(parents=True)
    summaries, bindings, identities, cache, map_checks = [], {}, set(), {}, []
    source_paths = {}
    common_grid = None
    for name, directory in runs.items():
        directory = Path(directory).resolve()
        run_path = directory / 'run_provenance.json'
        run = json.loads(run_path.read_text())
        bindings[name] = {'path': str(run_path.resolve()), 'sha256': digest(run_path)}
        shape, affine = tuple(run['shape']), np.array(run['affine_mm'])
        if common_grid is None:
            common_grid = (shape, affine.copy())
        require(shape == common_grid[0] and np.array_equal(affine, common_grid[1]), 'Cross-run reference grid differs')
        volume = abs(np.linalg.det(affine[:3, :3]))
        require(np.isclose(volume, run['voxel_volume_mm3'], rtol=0, atol=1e-12), 'Voxel determinant differs')
        reference = Path(run.get('reference') or run['inputs']['reference']['path'])
        require(digest(reference) == (run.get('reference_sha256') or run['inputs']['reference']['sha256']), 'Reference changed')
        image = nib.load(reference)
        require(image.shape == shape and np.allclose(image.affine, affine, rtol=0, atol=1e-7), 'Reference grid differs')
        if 'inputs' in run:
            for value in run['inputs'].values():
                require(digest(value['path']) == value['sha256'], 'Bound input changed')
        else:
            require(digest(run['input_manifest']) == run['manifest_sha256'], 'Manifest changed')
        manifest = pd.read_csv(directory / ('input_ledger.csv' if name == 'end_branches' else 'input_manifest.csv'), dtype=str, keep_default_na=False)
        require(not manifest.duplicated(['SampleID', 'NeuronID']).any(), 'Duplicate neuron identity')
        sparse = defaultdict(lambda: [defaultdict(float), defaultdict(float), 0, 0])
        total_ends = eligible = 0
        for record in manifest.to_dict('records'):
            uid = (record['SampleID'], record['NeuronID'])
            path = Path(record['SWCPath'])
            if not path.is_absolute():
                path = ROOT / path
            require(record['CoordinateFrame'] == 'atlas_index_um', 'Unsupported coordinates')
            if uid not in cache:
                require(digest(path) == record['SWCSHA256'], 'SWC changed')
                rows = np.loadtxt(path, comments='#', ndmin=2)
                number, indices = endpoint_indices(rows, np.array([float(v) for v in record['IndexScaleUm'].split(';')]), shape)
                cache[uid] = (record['SWCSHA256'], number, indices)
                source_paths[uid] = path
            saved_hash, number, indices = cache[uid]
            require(saved_hash == record['SWCSHA256'], 'Cross-run source hash differs')
            total_ends += number
            eligible += number > 0
            if name.endswith('endpoints'):
                require(uid not in identities, 'Endpoint partitions overlap')
                identities.add(uid)
                item = sparse[(record['AnimalID'], record['Subregion'])]
                item[3] += 1
                if number:
                    item[2] += 1
                    unique, counts = np.unique(indices, return_counts=True)
                    for index, count in zip(unique, counts):
                        item[0][int(index)] += int(count)
                        item[1][int(index)] += 1

        def load(entry, metric):
            path = directory / entry[metric + '_path']
            require(digest(path) == entry[metric + '_sha256'], 'Map bytes changed')
            image = nib.load(path)
            require(image.shape == shape and np.allclose(image.affine, affine, rtol=0, atol=1e-7), 'Map grid differs')
            require(image.header.get_xyzt_units()[0] == 'mm', 'Map units differ')
            for matrix, code in (image.get_sform(coded=True), image.get_qform(coded=True)):
                if code:
                    require(np.allclose(matrix, affine, rtol=0, atol=1e-7), 'Coded transform differs')
            require(int(image.header['sform_code']) or int(image.header['qform_code']), 'Uncoded transform')
            data = np.asarray(image.dataobj)
            require(np.isfinite(data).all() and np.all(data >= 0), 'Invalid map values')
            map_checks.append({'run': name, 'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': digest(path),
                               'sum': float(data.sum(dtype=np.float64)), 'nonzero_voxels': int(np.count_nonzero(data)), 'maximum': float(data.max())})
            return data

        metrics = ['count', 'density', 'occupancy'] if name.endswith('endpoints') else ['length'] + (['density'] if name != 'end_branches' else [])
        for entry in run['animal_maps']:
            if entry.get('map_available') is False:
                require(entry.get('n_computable', 0) == 0, 'Missing map has eligible neurons')
                continue
            data = load(entry, metrics[0])
            if name.endswith('endpoints'):
                density, occupancy = load(entry, 'density'), load(entry, 'occupancy')
                check_values(data, density, occupancy, volume)
                counts, occupied, denominator, selected = sparse[(entry['AnimalID'], entry['Subregion'])]
                require(denominator == entry['n_computable'] and selected == entry['n_selected'], 'Animal denominator differs')
                for values, expected in [(data, counts), (occupancy, occupied)]:
                    ids = np.array(sorted(expected), dtype=np.int64)
                    require(np.array_equal(np.flatnonzero(values), ids), 'Source/map endpoint support differs')
                    require(np.allclose(values.ravel()[ids], [expected[int(i)] / denominator for i in ids], rtol=2e-7, atol=1e-7), 'Source endpoint/occupancy values differ')
            elif name != 'end_branches':
                require(np.allclose(load(entry, 'density') * volume, data, rtol=2e-7, atol=1e-7), 'Axon density differs')
            if 'mean_in_reference_length_mm' in entry:
                require(np.isclose(data.sum(dtype=np.float64), entry['mean_in_reference_length_mm'], rtol=2e-7, atol=1e-6), 'Integrated animal length differs')
        for entry in run['group_maps']:
            if entry.get('map_available') is False:
                continue
            animals = [a for a in run['animal_maps'] if a['Subregion'] == entry['Subregion'] and a.get('map_available') is not False]
            require(sorted(a['AnimalID'] for a in animals) == sorted(entry['contributing_animals']), 'Contributing animals differ')
            require(len(animals) == entry['n_animals'], 'Animal count differs')
            if name.endswith('endpoints'):
                require(sum(a['n_computable'] for a in animals) == entry['n_computable'], 'Group eligible count differs')
            for metric in metrics:
                data = load(entry, metric)
                expected = np.zeros(shape, dtype=np.float64)
                for animal in animals:
                    expected += np.asarray(nib.load(directory / animal[metric + '_path']).dataobj)
                expected /= len(animals)
                require(np.allclose(data, expected, rtol=3e-7, atol=1e-7), 'Equal-animal voxel mean differs')
        summaries.append({'run': name, 'selected': len(manifest), 'eligible_original_ends': eligible, 'original_axon_ends': total_ends,
                          'animal_entries': len(run['animal_maps']), 'group_entries': len(run['group_maps'])})
        print(name, 'passed', flush=True)
    ledger = pd.read_csv(REVIEW / 'hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv', dtype=str, keep_default_na=False)
    require(identities == set(zip(ledger.SampleID, ledger.NeuronID)) and len(identities) == 462, 'Combined membership differs')
    branches = pd.read_csv(runs['end_branches'] / 'per_neuron_measurements.csv', dtype=str, keep_default_na=False)
    whole = ledger.set_index(['SampleID', 'NeuronID'])
    for row in branches.to_dict('records'):
        source = whole.loc[(row['SampleID'], row['NeuronID'])]
        require(source.AnimalID == row['AnimalID'] and source.Subregion == row['Subregion'], 'Branch membership differs')
        require(int(source.Endpoint_candidate_axon_endpoint_count) == cache[(row['SampleID'], row['NeuronID'])][1], 'Combined endpoint census differs')
        if row['computable'] == 'True':
            require(float(row['total_length_mm']) <= float(source.Axon_selected_axon_length_mm) + 1e-7, 'Branches exceed whole axon')
        else:
            require(row['total_length_mm'] == '', 'Unassessed branch length must be NA')
    pd.DataFrame(map_checks).to_csv(output / 'all_map_value_checks.csv', index=False)
    for uid, path in source_paths.items():
        require(digest(path) == cache[uid][0], 'Source changed during inspection')
    result = {'status': 'passed_full_voxel_relations_and_direct_graph_endpoints', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'inspector_sha256': digest(__file__), 'runs': bindings, 'run_summaries': summaries, 'unique_source_neurons': len(cache),
              'map_files_checked': len(map_checks), 'versions': {'python': sys.version, 'numpy': np.__version__, 'nibabel': nib.__version__, 'pandas': pd.__version__},
              'map_value_checks_sha256': digest(output / 'all_map_value_checks.csv'),
              'checks': ['coded transforms, mm units, finite nonnegative voxels', 'density-times-volume conversion', 'occupancy support, range and count inequality', 'direct original endpoint counts and animal occupancy', 'all group voxels equal contributing animal means', 'integrated animal lengths', 'exact partition union/no overlap', 'per-neuron end branches bounded by whole axon'],
              'source_checks': [{'SampleID': uid[0], 'NeuronID': uid[1], 'sha256': item[0], 'original_axon_ends': item[1]} for uid, item in sorted(cache.items())],
              'anatomical_acceptance': False, 'limitations': ['No new native-image terminal-field review', 'Authoritative fMOST-to-NMT transform and export origin unavailable', 'Graph ends may include unfinished tracing', 'Independent full-edge raster and six-level cells were checked previously; this round adds orthogonal relations']}
    (output / 'inspection_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs-json', type=Path, help='Optional object with the five RUNS keys and fresh run paths')
    args = parser.parse_args()
    runs = RUNS
    if args.runs_json:
        runs = {name: Path(path) for name, path in json.loads(args.runs_json.read_text(encoding='utf-8-sig')).items()}
        require(set(runs) == set(RUNS), 'All five map families required')
    result = inspect(args.output, runs)
    print(result['status'], result['map_files_checked'], 'maps', flush=True)
