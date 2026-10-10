"""Create origin-specific soma/projection NIfTIs and clearly titled MRI cards.

Existing endpoint values and source assignments are preserved. Soma count is
the number of selected reconstructions in a voxel, not a population density.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import patheffects
import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'notes/projection_map_review_round2'))
from inspect_projection_maps import digest, require
from render_endpoint_markers import slice_markers

NAMED_INSULA_INDICES = (41, 42, 43, 228, 229, 541, 542, 728, 729)
EVO = ROOT / 'group_analysis/evolution_20261008'
REVIEW = ROOT / 'notes/region_analysis_review_20261009'
DEFAULT_LEDGER = REVIEW / 'hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv'
DEFAULT_RUN = EVO / 'arm_mapping_20261009/main/endpoints'
DEFAULT_INSPECTION = ROOT / 'notes/projection_map_review_round2/reproduction_checked/inspection/inspection_receipt.json'
DEFAULT_DISPLAY = ROOT / 'notes/projection_map_review_round2/reproduction_checked/endpoint_markers/display_provenance.json'

CATEGORY_NAMES = {
    'atlasAndHenry': 'ARM insula assignment and Henry coarse INS evidence',
    'atlasOnly': 'ARM insula assignment without a Henry note',
    'HenryPrCOConflict': 'Henry coarse INS evidence; ARM precentral operular assignment conflicts',
    'HenryUnassigned': 'Henry coarse INS evidence; ARM source label unassigned',
    'neighborCandidate': 'Neighboring ARM origin; candidate without a Henry note',
    'unassignedCandidate': 'Unassigned ARM origin; candidate without a Henry note',
}


def evidence_categories(ledger):
    """An exhaustive non-overlapping evidence view of the unchanged ledger."""
    insula = ledger.ARMIndex.astype(int).isin(NAMED_INSULA_INDICES)
    manual = ledger.Henry_coarse_INS_visual_evidence.eq('True')
    unassigned = ledger.ARMIndex.eq('0')
    masks = [insula & manual, insula & ~manual, ~insula & ~unassigned & manual,
             unassigned & manual, ~insula & ~unassigned & ~manual, unassigned & ~manual]
    require(np.all(np.stack([mask.to_numpy() for mask in masks]).sum(axis=0) == 1), 'Evidence categories must partition the ledger')
    categories = pd.Series('', index=ledger.index, dtype=str)
    for name, mask in zip(CATEGORY_NAMES, masks):
        categories.loc[mask] = name
    if masks[2].any():
        require(ledger.loc[masks[2], 'ARMFullName'].eq('CR_precentral_operular_area').all(), 'Manual/atlas conflict includes an unexpected parcel')
    return categories


def soma_count_map(voxels, shape):
    voxels = np.asarray(voxels)
    require(voxels.ndim == 2 and voxels.shape[1] == 3 and np.isfinite(voxels).all(), 'Finite XYZ soma voxels required')
    require(np.array_equal(voxels, np.rint(voxels)), 'Saved source voxels must be integers')
    voxels = voxels.astype(int)
    require(((voxels >= 0) & (voxels < shape)).all(), 'Soma outside reference')
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, tuple(voxels.T), 1)
    return counts


def source_title(full_name, index, hemisphere):
    side = {'L': 'Left', 'R': 'Right'}[hemisphere]
    return f'Projections from neurons with somata assigned to\n{full_name.replace("_", " ")} ({side}; ARM level 6 index {index})'


def save_volume(path, data, affine, description):
    require(len(description.encode('ascii')) <= 79, 'NIfTI description too long')
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units('mm')
    image.set_sform(affine, code=2)
    image.set_qform(None, code=0)
    image.header['descrip'] = description
    nib.save(image, path)
    saved = nib.load(path)
    require(np.array_equal(np.asarray(saved.dataobj), data) and np.allclose(saved.affine, affine), 'NIfTI readback differs')


def build(output, ledger_path=DEFAULT_LEDGER, endpoint_run=DEFAULT_RUN,
          inspection_path=DEFAULT_INSPECTION, display_path=DEFAULT_DISPLAY):
    output, ledger_path, endpoint_run, inspection_path, display_path = map(Path, (output, ledger_path, endpoint_run, inspection_path, display_path))
    require(not output.exists(), 'Preserve existing maps; use a fresh output directory')
    inspection = json.loads(inspection_path.read_text())
    display = json.loads(display_path.read_text())
    run_path = endpoint_run / 'run_provenance.json'
    run = json.loads(run_path.read_text())
    require(inspection['status'] == 'passed_full_voxel_relations_and_direct_graph_endpoints', 'Completed map inspection required')
    require(digest(run_path) == inspection['runs']['main_endpoints']['sha256'], 'Run changed after inspection')
    require(display['inspection_sha256'] == digest(inspection_path) and display['run_provenance_sha256'] == digest(run_path), 'Display/inspection lineage differs')
    for binding in run['inputs'].values():
        require(digest(binding['path']) == binding['sha256'], 'Bound endpoint input changed')
    prior_sources = {(r['SampleID'], r['NeuronID']): r['sha256'] for r in inspection['source_checks']}
    ledger = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    require(len(ledger) == 462 and not ledger.duplicated(['SampleID', 'NeuronID']).any(), 'Exact 462-neuron ledger required')
    require(set(zip(ledger.SampleID, ledger.NeuronID)) == set(prior_sources), 'Ledger membership changed')
    ledger['IncludedNamedInsulaOriginMap'] = ledger.ARMIndex.astype(int).isin(NAMED_INSULA_INDICES)
    ledger['OriginEvidenceCategory'] = evidence_categories(ledger)
    selected = ledger[ledger.IncludedNamedInsulaOriginMap].copy()
    require(len(selected) == 301, 'Named-origin selection differs from established 301 neurons')
    require(selected.SourceARMStatus.eq('mapped').all() and selected.SourceRootLookupPolicy.eq('current_rint_zero_center').all(), 'Source assignment policy differs')
    require(selected.Endpoint_soma_anchor_state.eq('unique_type1_node').all() and selected.SourceRootNodeID.eq(selected.Endpoint_soma_anchor_node_id).all(), 'Source anchor is not the unique soma node')
    require(selected.SourceARMHemisphereConflict.eq('False').all(), 'Source hemisphere conflict')
    require(selected.CoordinateFrame.eq('atlas_index_um').all(), 'Unsupported source coordinates')
    key = pd.read_csv(run['inputs']['atlas_key']['path'], sep='\t', dtype=str).set_index('Index')
    reference_image = nib.load(run['inputs']['reference']['path'])
    atlas_image = nib.load(run['inputs']['atlas']['path'])
    reference, affine = reference_image.get_fdata(), reference_image.affine
    shape = reference.shape
    require(atlas_image.shape == shape + (1, 6) and np.allclose(atlas_image.affine, affine), 'Pinned ARM requires a singleton axis followed by six hierarchy volumes')
    atlas = np.asarray(atlas_image.dataobj)[..., 0, 5]
    require(np.allclose(affine[:3, :3], np.diag(np.diag(affine[:3, :3]))) and np.all(np.diag(affine[:3, :3]) > 0), 'Positive axis-aligned display grid required')
    groups = {entry['Subregion']: entry for entry in run['group_maps']}
    endpoint_manifest = pd.read_csv(endpoint_run / 'input_manifest.csv', dtype=str, keep_default_na=False)
    selected['SomaVoxelXYZ'] = selected.SourceRootVoxelXYZ
    selected['SomaContinuousIndexXYZ'] = selected.SourceRootIndexXYZ
    for record in selected.to_dict('records'):
        uid = (record['SampleID'], record['NeuronID'])
        require(record['SWCSHA256'] == prior_sources[uid] and digest(record['SWCPath']) == prior_sources[uid], 'Soma source hash differs')
        voxel = np.asarray(json.loads(record['SomaVoxelXYZ']), dtype=int)
        point = np.asarray(json.loads(record['SomaContinuousIndexXYZ']), dtype=float)
        require(np.array_equal(np.rint(point).astype(int), voxel), 'Saved source lookup and coordinate disagree')
        require(((voxel >= 0) & (voxel < shape)).all() and int(atlas[tuple(voxel)]) == int(record['ARMIndex']), 'Soma voxel/ARM source differs')
        actual = key.loc[record['ARMIndex']]
        require(record['ARMFullName'] == actual.Full_Name and int(actual.First_Level) <= 6 <= int(actual.Last_Level), 'Official full ARM source name differs')
    output.mkdir(parents=True)
    for folder in ('nifti', 'figures'):
        (output / folder).mkdir()
    membership_fields = ['SampleID', 'NeuronID', 'AnimalID', 'Subregion', 'ARMIndex', 'ARMFullName', 'Hemisphere', 'SourceARMStatus',
                         'SourceRootVoxelXYZ', 'SourceRootIndexXYZ', 'SourceRootNodeID', 'SourceRootLookupPolicy', 'AnatomyStatus', 'FineParcelStatus',
                         'EvidenceStratum', 'Henry_coarse_INS_visual_evidence', 'CoordinateOriginStatus', 'RegistrationStatus', 'EndpointEligible',
                         'SWCPath', 'SWCSHA256', 'IncludedNamedInsulaOriginMap', 'OriginEvidenceCategory']
    ledger[membership_fields].to_csv(output / 'neuron_origin_membership.csv', index=False)
    outputs, index_rows = {}, []
    cuts = display['cuts_XYZ_voxels']
    gray = Normalize(*display['MRI_intensity_limits'], clip=True)
    colour = Normalize(0, display['shared_colour_upper'], clip=True)
    for index in NAMED_INSULA_INDICES:
        rows = selected[selected.ARMIndex.eq(str(index))]
        require(not rows.empty, 'Expected source parcel has no selected neuron')
        metadata = rows.iloc[0]
        group = groups[metadata.Subregion]
        expected_ids = set(zip(rows.SampleID, rows.NeuronID))
        original = endpoint_manifest[endpoint_manifest.Subregion.eq(metadata.Subregion)]
        require(expected_ids == set(zip(original.SampleID, original.NeuronID)), 'Projection/source membership mismatch')
        require(len(rows) == group['n_selected'] and sum(rows.EndpointEligible.eq('True')) == group['n_computable'], 'Projection/source denominator mismatch')
        voxels = np.array([json.loads(value) for value in rows.SomaVoxelXYZ])
        points = np.array([json.loads(value) for value in rows.SomaContinuousIndexXYZ])
        somas = soma_count_map(voxels, shape)
        source_cuts = [int(np.argmax(somas.sum(axis=tuple(a for a in range(3) if a != axis)))) for axis in range(3)]
        original_path = endpoint_run / group['density_path']
        require(digest(original_path) == group['density_sha256'], 'Projection map changed')
        original_image = nib.load(original_path)
        require(original_image.shape == shape and np.allclose(original_image.affine, affine), 'Projection grid differs')
        density = np.asarray(original_image.dataobj)
        stem = f'space-NMTv2p1_from-ARM6_{index}_{metadata.Hemisphere}'
        origin_path = output / 'nifti' / f'{stem}_desc-selectedSomaCount_map.nii.gz'
        projection_path = output / 'nifti' / f'{stem}_to-wholeBrain_desc-candidateAxonEndDensity_map.nii.gz'
        save_volume(origin_path, somas, affine, f'ARM6 {index} {metadata.Hemisphere} selected soma count; not population density')
        save_volume(projection_path, density, affine, f'From ARM6 {index} {metadata.Hemisphere} somata; candidate axon-end density; descriptive')
        manual = rows.Henry_coarse_INS_visual_evidence.eq('True').to_numpy()
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
        panels = []
        for col, axis in enumerate((2, 1, 0)):
            in_plane = [a for a in range(3) if a != axis]
            extent = [affine[a, 3] + (edge - .5) * affine[a, a] for a in in_plane for edge in (0, shape[a])]
            for row_number, cut in enumerate((source_cuts[axis], cuts[axis])):
                panel = axes[row_number, col]
                panel.imshow(np.take(reference, cut, axis=axis).T, cmap='gray', norm=gray, extent=extent, origin='lower', interpolation='nearest')
                if row_number == 0:
                    source_plane = np.take(atlas == index, cut, axis=axis).T
                    if source_plane.any() and not source_plane.all():
                        xx = affine[in_plane[0], 3] + np.arange(source_plane.shape[1]) * affine[in_plane[0], in_plane[0]]
                        yy = affine[in_plane[1], 3] + np.arange(source_plane.shape[0]) * affine[in_plane[1], in_plane[1]]
                        panel.contour(xx, yy, source_plane, levels=[.5], colors=['#f5d453'], linewidths=.8)
                    visible = voxels[:, axis] == cut
                    mm = nib.affines.apply_affine(affine, points)
                    for condition, mark_colour, label in ((manual, '#26d9e9', 'Henry coarse INS evidence'), (~manual, '#ed99db', 'Atlas assignment; no Henry note')):
                        mask = visible & condition
                        panel.scatter(mm[mask, in_plane[0]], mm[mask, in_plane[1]], s=86, c='white', marker='D', linewidths=0)
                        panel.scatter(mm[mask, in_plane[0]], mm[mask, in_plane[1]], s=48, c=mark_colour, marker='D', edgecolors='black', linewidths=.65, label=label)
                    visible_indices = np.flatnonzero(visible)
                    centre = np.median(mm[visible][:, in_plane], axis=0)
                    representative = visible_indices[np.argmin(np.linalg.norm(mm[visible][:, in_plane] - centre, axis=1))]
                    arrow_target = mm[representative, in_plane]
                    arrow_label = [float(np.clip(arrow_target[0] + 12, extent[0] + 8, extent[1] - 8)),
                                   float(np.clip(arrow_target[1] + 12, extent[2] + 8, extent[3] - 8))]
                    annotation = panel.annotate('Source somata', xy=arrow_target, xytext=arrow_label, fontsize=8, fontweight='bold', color='black',
                                   ha='center', bbox=dict(boxstyle='round,pad=.3', facecolor='#ffe063', edgecolor='black'),
                                   arrowprops=dict(arrowstyle='-|>', facecolor='#ffe063', edgecolor='#ffe063', linewidth=2, mutation_scale=20))
                    annotation.arrow_patch.set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])
                    if col == 0:
                        panel.legend(loc='lower left', fontsize=6)
                    title = 'Soma origin locator (yellow = source ARM boundary)'
                    panels.append({'kind': 'soma', 'axis': axis, 'cut': cut, 'visible_neurons': int(visible.sum()),
                                   'arrow_target_SampleID': rows.iloc[representative].SampleID, 'arrow_target_NeuronID': rows.iloc[representative].NeuronID,
                                   'arrow_target_template_mm': arrow_target.tolist(), 'arrow_label_template_mm': arrow_label})
                else:
                    x, y, values = slice_markers(density, axis, cut, affine)
                    panel.scatter(x, y, s=17, c='white', linewidths=0)
                    plotted = panel.scatter(x, y, s=9, c=np.log10(1 + values), cmap='viridis', norm=colour, edgecolors='black', linewidths=.25)
                    title = 'Projection targets: candidate axon-end voxels'
                    panels.append({'kind': 'projection', 'axis': axis, 'cut': cut, 'occupied_voxels': len(x), 'slice_density_sum': float(values.sum(dtype=np.float64))})
                panel.set_xlim(extent[:2])
                panel.set_ylim(extent[2:])
                panel.set_title(title + f'\n{"XYZ"[axis]} voxel {cut} ({affine[axis, 3] + cut * affine[axis, axis]:.2f} template mm)', fontsize=9)
                panel.set_xlabel(f'{"XYZ"[in_plane[0]]} (template mm)', fontsize=8)
                panel.set_ylabel(f'{"XYZ"[in_plane[1]]} (template mm)', fontsize=8)
                panel.tick_params(labelsize=7)
        n_animals = rows.AnimalID.nunique()
        summary = f'Selected: {len(rows)} neurons / {n_animals} animals; end eligible: {group["n_computable"]} / {group["n_animals"]} contributing animals'
        fig.suptitle(source_title(metadata.ARMFullName, index, metadata.Hemisphere) + '\n' + summary +
                     '\nTop: source-specific single-slice locator. Bottom: common target cuts, equal-animal mean.\nAtlas parcel assignment and candidate endings are not anatomical/terminal-field acceptance.', fontsize=11)
        fig.colorbar(plotted, ax=axes[1, :].tolist(), shrink=.7, label='log10(1 + end density)\nends / mm³ / eligible neuron')
        figure = output / 'figures' / f'{stem}_to-wholeBrain_desc-somaOriginAndCandidateEnds.png'
        fig.savefig(figure, dpi=160)
        plt.close(fig)
        paths = {'soma_count': origin_path, 'projection_density': projection_path, 'figure': figure}
        artifacts = {name: {'path': path.relative_to(output).as_posix(), 'sha256': digest(path)} for name, path in paths.items()}
        entry = {'Subregion': metadata.Subregion, 'ARMIndex': index, 'ARMFullName': metadata.ARMFullName, 'Hemisphere': metadata.Hemisphere,
                 'selected_neurons': len(rows), 'eligible_neurons': group['n_computable'], 'selected_animals': n_animals, 'contributing_animals': group['n_animals'],
                 'Henry_coarse_INS_evidence_neurons': int(manual.sum()), 'atlas_only_neurons': int((~manual).sum()), 'soma_locator_cuts_XYZ': source_cuts,
                 'projection_cuts_XYZ': cuts, 'original_projection': {'path': str(original_path.resolve()), 'sha256': group['density_sha256']}, 'panels': panels, 'artifacts': artifacts}
        outputs[metadata.Subregion] = entry
        index_rows.append({key: value for key, value in entry.items() if key not in ('panels', 'artifacts', 'original_projection')})
        for metric, binding in artifacts.items():
            index_rows[-1][metric + '_path'] = binding['path']
        print(metadata.Subregion, 'created', flush=True)
    pd.DataFrame(index_rows).to_csv(output / 'origin_map_index.csv', index=False)
    source_bindings = {'ledger': ledger_path, 'endpoint_run': run_path, 'inspection': inspection_path, 'prior_display': display_path,
                       'producer': Path(__file__), 'slice_helper': ROOT / 'notes/projection_map_review_round2/render_endpoint_markers.py',
                       'hash_helper': ROOT / 'notes/projection_map_review_round2/inspect_projection_maps.py'}
    result = {'status': 'created_origin_specific_descriptive_maps', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'inputs': {name: {'path': str(path.resolve()), 'sha256': digest(path)} for name, path in source_bindings.items()},
              'background': run['inputs']['reference'], 'atlas': run['inputs']['atlas'], 'atlas_key': run['inputs']['atlas_key'],
              'selected_named_origin_neurons': len(selected), 'full_membership_ledger_neurons': len(ledger), 'named_origin_parcels': len(outputs),
              'other_named_source_neurons_retained_in_ledger': 109, 'unassigned_source_neurons_retained_in_ledger': 52,
              'MRI_intensity_limits': display['MRI_intensity_limits'], 'projection_colour_upper': display['shared_colour_upper'],
              'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'nibabel': nib.__version__, 'matplotlib': matplotlib.__version__},
              'scope': 'Nine exact soma-assigned ARM6 insula parcels; no reclassification or pooling of neighboring/unknown source parcels',
              'soma_measure': 'raw selected reconstruction count per saved rint source voxel; sampling map, not population density',
              'projection_measure': 'existing equal-animal candidate-end density; each new map has identical voxel values to its original source-group map',
              'soma_locator_cut_rule': 'per-source argmax raw soma count per axis; first voxel breaks ties', 'MIP': False, 'smoothing': None,
              'origin_marker': 'solid diamond, 48 pt2, black rim and white halo; no coordinate displacement',
              'origin_arrow': 'solid arrow targets the displayed actual soma nearest the in-plane median; UID and template coordinates recorded per panel',
              'anatomical_acceptance': False, 'source_data_modified': False, 'outputs': outputs,
              'artifacts': {path.name: digest(path) for path in (output / 'neuron_origin_membership.csv', output / 'origin_map_index.csv')},
              'regional_tables': {'path': str(DEFAULT_LEDGER.parent / 'arm_projection_hierarchy_tables.xlsx'), 'sha256': digest(DEFAULT_LEDGER.parent / 'arm_projection_hierarchy_tables.xlsx'), 'levels': [1, 2, 3, 4, 5, 6], 'method': 'existing 462-neuron export unchanged; select exact source identities for source-specific summaries'}}
    (output / 'run_provenance.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output)['status'])
