"""Map all 462 selected neurons in six explicit source-evidence categories.

Reuse hashed complete-graph leaf records, retaining equal-animal means and
the existing conditional endpoint denominator. Categories are evidence views,
not additional cohorts or anatomical parcels. Preserve prior numerical maps.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd

from map_projections_by_soma_origin import (ROOT, EVO, DEFAULT_LEDGER, DEFAULT_INSPECTION, DEFAULT_DISPLAY,
                                          CATEGORY_NAMES, evidence_categories, soma_count_map, save_volume, digest, require, slice_markers)


def build(output):
    output = Path(output)
    require(not output.exists(), 'Use a fresh evidence-view destination')
    ledger = pd.read_csv(DEFAULT_LEDGER, dtype=str, keep_default_na=False)
    ledger['OriginEvidenceCategory'] = evidence_categories(ledger)
    inspection = json.loads(DEFAULT_INSPECTION.read_text())
    display = json.loads(DEFAULT_DISPLAY.read_text())
    require(inspection['status'] == 'passed_full_voxel_relations_and_direct_graph_endpoints', 'Completed source inspection required')
    require(display['inspection_sha256'] == digest(DEFAULT_INSPECTION), 'Display lineage differs')
    source_checks = {(r['SampleID'], r['NeuronID']): r for r in inspection['source_checks']}
    require(set(zip(ledger.SampleID, ledger.NeuronID)) == set(source_checks) and len(ledger) == 462, 'Exact 462 sources required')
    records = {(r['SampleID'], r['NeuronID']): r for r in ledger.to_dict('records')}
    ends = defaultdict(list)
    seen_counts, seen_leaf_ids = Counter(), set()
    leaf_bindings = []
    for name, directory in [('main_endpoints', EVO / 'arm_mapping_20261009/main/endpoints'),
                            ('additional_endpoints', EVO / 'endpoint_maps/additional_candidates_20261009')]:
        run_path = directory / 'run_provenance.json'
        require(digest(run_path) == inspection['runs'][name]['sha256'], 'Endpoint run changed')
        run = json.loads(run_path.read_text())
        path = directory / 'leaf_records.jsonl'
        leaf_hash = run['artifacts']['leaf_records.jsonl']['sha256']
        require(digest(path) == leaf_hash, 'Bound full-graph leaf record changed')
        leaf_bindings.append({'path': str(path.resolve()), 'sha256': leaf_hash, 'run_path': str(run_path.resolve()), 'run_sha256': digest(run_path)})
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                leaf = json.loads(line)
                if not leaf['candidate_axon_endpoint']:
                    continue
                uid = (leaf['SampleID'], leaf['NeuronID'])
                require(uid in records and leaf['source_sha256'] == records[uid]['SWCSHA256'] and leaf['AnimalID'] == records[uid]['AnimalID'], 'Leaf/source identity differs')
                identity = uid + (leaf['node_id'],)
                require(identity not in seen_leaf_ids, 'Duplicated original ending')
                seen_leaf_ids.add(identity)
                seen_counts[uid] += 1
                if leaf['voxel'] is not None:
                    point = leaf['index_xyz']
                    expected = [int(np.searchsorted(np.arange(n + 1) - .5, point[a], side='right') - 1) for a, n in enumerate(run['shape'])]
                    require(expected == leaf['voxel'], 'Leaf voxel-face lookup differs')
                    ends[uid].append(leaf['voxel'])
        reference_binding = run['inputs']['reference']
        require(digest(reference_binding['path']) == reference_binding['sha256'], 'Reference changed')
    for uid, row in source_checks.items():
        require(seen_counts[uid] == row['original_axon_ends'], 'Fresh graph census and saved leaves differ')
        require(digest(records[uid]['SWCPath']) == row['sha256'], 'Source changed after inspection')
    require(sum(seen_counts.values()) == 75625, 'Original endpoint census differs')
    reference_image = nib.load(reference_binding['path'])
    reference, affine = reference_image.get_fdata(), reference_image.affine
    shape = reference.shape
    volume = abs(np.linalg.det(affine[:3, :3]))
    output.mkdir(parents=True)
    for folder in ('nifti', 'figures'):
        (output / folder).mkdir()
    results, summaries = {}, []
    for category, label in CATEGORY_NAMES.items():
        rows = ledger[ledger.OriginEvidenceCategory.eq(category)]
        require(rows.Endpoint_soma_anchor_state.eq('unique_type1_node').all(), 'Unconfirmed soma anchor')
        voxels = np.array([json.loads(v) for v in rows.SourceRootVoxelXYZ])
        soma_counts = soma_count_map(voxels, shape)
        sums, denominators = defaultdict(Counter), Counter()
        for row in rows.to_dict('records'):
            uid = (row['SampleID'], row['NeuronID'])
            if seen_counts[uid]:
                denominators[row['AnimalID']] += 1
                sums[row['AnimalID']].update(map(tuple, ends[uid]))
        density = np.zeros(shape, dtype=np.float64)
        for animal, denominator in denominators.items():
            for voxel, count in sums[animal].items():
                density[voxel] += count / denominator / len(denominators) / volume
        density = density.astype(np.float32)
        stem = f'space-NMTv2p1_sourceEvidence-{category}'
        soma_path = output / 'nifti' / f'{stem}_desc-selectedSomaCount_map.nii.gz'
        projection_path = output / 'nifti' / f'{stem}_to-wholeBrain_desc-candidateAxonEndDensity_map.nii.gz'
        save_volume(soma_path, soma_counts, affine, 'Selected soma count by evidence category; not accepted anatomical parcel')
        save_volume(projection_path, density, affine, 'Candidate axon-end density by source evidence; not terminal-field acceptance')
        figure, axes = plt.subplots(1, 3, figsize=(14, 5.1), layout='constrained')
        panels = []
        for axis, panel in zip((2, 1, 0), axes):
            cut = display['cuts_XYZ_voxels'][axis]
            in_plane = [a for a in range(3) if a != axis]
            extent = [affine[a, 3] + (edge - .5) * affine[a, a] for a in in_plane for edge in (0, shape[a])]
            panel.imshow(np.take(reference, cut, axis=axis).T, cmap='gray', norm=Normalize(*display['MRI_intensity_limits'], clip=True), extent=extent, origin='lower', interpolation='nearest')
            x, y, values = slice_markers(density, axis, cut, affine)
            panel.scatter(x, y, s=17, c='white', linewidths=0)
            plotted = panel.scatter(x, y, s=9, c=np.log10(1 + values), cmap='viridis', norm=Normalize(0, display['shared_colour_upper'], clip=True), edgecolors='black', linewidths=.25)
            panel.set_xlim(extent[:2])
            panel.set_ylim(extent[2:])
            panel.set_title(f'Projection targets: {"XYZ"[axis]} voxel {cut}', fontsize=9)
            panel.set_xlabel(f'{"XYZ"[in_plane[0]]} (template mm)')
            panel.set_ylabel(f'{"XYZ"[in_plane[1]]} (template mm)')
            panels.append({'axis': axis, 'cut': cut, 'occupied_voxels': len(x), 'slice_density_sum': float(values.sum(dtype=np.float64))})
        figure.suptitle('Projections from neurons classified by source evidence\n' + label +
                         f'\nSelected {len(rows)} neurons; end eligible {sum(denominators.values())}; {len(denominators)} contributing animals\n' +
                         'Evidence category, not an accepted anatomical parcel; equal-animal candidate-end density', fontsize=11)
        figure.colorbar(plotted, ax=axes.tolist(), shrink=.7, label='log10(1 + end density)\nends / mm³ / eligible neuron')
        figure_path = output / 'figures' / f'{stem}_to-wholeBrain_desc-candidateEnds.png'
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        artifacts = {name: {'path': path.relative_to(output).as_posix(), 'sha256': digest(path)} for name, path in
                     [('soma_count', soma_path), ('projection_density', projection_path), ('figure', figure_path)]}
        entry = {'category': category, 'title': label, 'selected_neurons': len(rows), 'eligible_neurons': sum(denominators.values()),
                 'selected_animals': rows.AnimalID.nunique(), 'contributing_animals': len(denominators), 'per_animal_eligible': dict(denominators),
                 'ARM_origins': sorted(rows.ARMFullName.unique()), 'panels': panels, 'artifacts': artifacts}
        results[category] = entry
        summaries.append({key: value for key, value in entry.items() if key not in ('panels', 'artifacts')})
        for metric, binding in artifacts.items():
            summaries[-1][metric + '_path'] = binding['path']
        print(category, 'created', flush=True)
    pd.DataFrame(summaries).to_csv(output / 'category_counts_and_map_index.csv', index=False)
    result = {'status': 'created_all462_source_evidence_maps', 'created_utc': datetime.now(timezone.utc).isoformat(), 'producer_sha256': digest(__file__),
              'classification_helper_sha256': digest(Path(__file__).with_name('map_projections_by_soma_origin.py')),
              'ledger': {'path': str(DEFAULT_LEDGER), 'sha256': digest(DEFAULT_LEDGER)}, 'inspection_sha256': digest(DEFAULT_INSPECTION),
              'leaf_records': leaf_bindings, 'background': reference_binding, 'selected_neurons': 462, 'eligible_neurons': 429,
              'INS_supported_by_atlas_or_Henry': 349, 'remaining_candidates': 113, 'original_candidate_endings': 75625,
              'category_partition': 'six exhaustive non-overlapping groups; no new independent cohorts', 'aggregation': 'eligible-neuron mean per animal/category, then equal contributing-animal mean',
              'MRI_intensity_limits': display['MRI_intensity_limits'], 'projection_colour_upper': display['shared_colour_upper'],
              'MIP': False, 'smoothing': None, 'anatomical_acceptance': False, 'source_data_modified': False, 'outputs': results}
    (output / 'run_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(build(parser.parse_args().output)['status'])
