"""One additive CM032 end-branch coverage table; old modalities are read only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
from summarize_mstim_arm import reviewed_end_branches, covered_projection, projection_fields
from export_arm_projection_tables import catalog, sha256
from endpoint_atlas import coded_mm_grid, matching_grid
from render_projection_slices import arm_source_labels


def identities(frame):
    required = {'SampleID', 'NeuronID', 'Subregion'}
    if not required <= set(frame) or frame.empty:
        raise ValueError('Missing exact identity/source columns')
    if frame[['SampleID', 'NeuronID']].duplicated().any():
        raise ValueError('Duplicate exact composite identity')
    return set(zip(frame.SampleID, frame.NeuronID))


def require_target_coverage(target, counts, covered_counts):
    index = int(float(target['ARMIndex']))
    count = int(counts[index]) if index < len(counts) else 0
    covered = int(covered_counts[index]) if index < len(covered_counts) else 0
    if count != int(target['region_voxels']) or covered != int(target['covered_voxels']):
        raise ValueError('Saved response coverage differs from actual pinned ARM/mask')
    return covered


def coverage_counts(labels, coverage):
    """Compute actual target footprints once per level, retaining background."""
    if labels.shape != coverage.shape or not np.isin(np.unique(coverage), [0, 1]).all():
        raise ValueError('Coverage must be a same-grid binary mask')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('ARM indices must be nonnegative integers')
    return (np.bincount(labels.ravel()),
            np.bincount(labels[coverage.astype(bool)].ravel()))


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh table destination')
    old = HERE.parent / 'cm032_regional_comparison'
    old_record_path = old / 'integration_provenance.json'
    old_record = json.loads(old_record_path.read_text())
    if old_record['status'] != 'provisional_regional_integration_completed':
        raise ValueError('Completed historical CM032 integration required')
    bound = {}
    def check(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Changed bound input: {path}')
        bound[str(path)] = digest
        return path
    check(old_record_path)
    response_path = check(old / 'cm032_signed_response_by_ARM_level.csv', old_record['outputs']['cm032_signed_response_by_ARM_level.csv'])
    joint_path = check(old / 'cm032_projectome_common_coverage.csv', old_record['outputs']['cm032_projectome_common_coverage.csv'])
    old_review = check(HERE.parent / 'independent_cm032_regional_readback_20261009.json')
    review = json.loads(old_review.read_text())
    if (review.get('status') != 'independent_CM032_saved_regional_readback_passed'
            or review.get('integration_provenance_sha256') != sha256(old_record_path)):
        raise ValueError('Successful old regional independent readback required')
    helpers = [Path(sys.modules[name].__file__) for name in ['summarize_mstim_arm', 'export_arm_projection_tables', 'endpoint_atlas', 'render_projection_slices']]
    for path in [__file__, *helpers]:
        check(path)
    base = ROOT / 'atlas/NMT_v2.1_sym/NMT_v2.1_sym'
    reference = base / 'NMT_v2.1_sym_SS.nii.gz'
    atlas = base / 'ARM_in_NMT_v2.1_sym.nii.gz'
    key = ROOT / 'atlas/ARM_key_all.txt'
    coverage_path = HERE.parent / 'cm032_warp_reproduction/analysis_coverage_space-NMTv2p1.nii.gz'
    label_map = ROOT / 'group_analysis/evolution_20261008/projection_inputs/arm_labels_20261009/main/label_map.csv'
    main_manifest = ROOT / 'group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints/input_manifest.csv'
    for path in [reference, atlas, key, coverage_path, label_map, main_manifest]:
        check(path, old_record['source_sha256'][str(path)])
    end_run = ROOT / 'notes/region_analysis_review_20261009/axon_end_branches_20261009/selected462_ARM'
    end_review = end_run.parent / 'independent_end_branch_readback.json'
    record, extra = reviewed_end_branches(end_run, end_review, reference, atlas, key)
    for path in extra:
        check(path)
    measurement_path = check(end_run / 'per_neuron_measurements.csv', record['artifacts']['per_neuron_measurements.csv'])
    measures = pd.read_csv(measurement_path, dtype=str, keep_default_na=False)
    main = pd.read_csv(main_manifest, dtype=str, keep_default_na=False)
    ledger = pd.read_csv(end_run / 'input_ledger.csv', dtype=str, keep_default_na=False)
    identities(main)
    identities(ledger)
    identities(measures)
    mapped_main = main.loc[main.SourceARMStatus.eq('mapped')]
    mapped_ledger = ledger.loc[ledger.SourceARMStatus.eq('mapped')]
    mapped_ids = identities(mapped_main)
    if mapped_ids != identities(mapped_ledger) or len(mapped_main) != 410:
        raise ValueError('Mapped end-branch ledger must be exactly the existing 410 identities')
    if not set(measures.computable) <= {'True', 'False'} or identities(measures) != identities(ledger):
        raise ValueError('End-branch eligibility requires exact complete ledger and explicit booleans')
    mapped_measures = measures.loc[measures.apply(lambda row: (row.SampleID, row.NeuronID) in mapped_ids, axis=1)]
    eligible_ids = identities(mapped_measures.loc[mapped_measures.computable.eq('True')])
    if len(eligible_ids) != 378:
        raise ValueError('Expected exact 378 mapped end-eligible identities')
    _, source_receipt = arm_source_labels(label_map, main_manifest)
    source_names = {row['Subregion']: row for row in source_receipt['names'] if row['SourceARMStatus'] == 'mapped'}
    response = pd.read_csv(response_path, keep_default_na=False)
    joint = pd.read_csv(joint_path, keep_default_na=False)
    if response.TargetID.duplicated().any() or joint[['SourceARMGroup', 'TargetID']].duplicated().any():
        raise ValueError('Historical response/joint keys must be unique')
    if len(response) != 1676 or len(joint) != 25140 or set(joint.SourceARMGroup) != set(source_names):
        raise ValueError('Historical CM032 complete target/source scope differs')
    grid = coded_mm_grid(nib.load(reference))
    mask_image = nib.load(coverage_path)
    matching_grid(mask_image, grid)
    coverage = np.asanyarray(mask_image.dataobj)
    if not np.isin(np.unique(coverage), [0, 1]).all():
        raise ValueError('Actual transformed model mask must be binary')
    targets, levels = catalog(atlas, key, grid)
    target_catalog = targets.set_index('TargetID')
    footprints = [coverage_counts(labels.reshape(grid.shape), coverage) for labels in levels]
    for target in response.to_dict('records'):
        official = target_catalog.loc[target['TargetID']]
        for field in ['Level', 'OfficialFullName', 'Abbreviation', 'Domain', 'Hemisphere', 'TargetStatus']:
            if str(target[field]) != str(official[field]):
                raise ValueError('Saved target metadata differs from actual ARM')
        require_target_coverage(target, *footprints[int(target['Level']) - 1])
    joint_lookup = joint.set_index(['SourceARMGroup', 'TargetID'])
    rows = []
    groups = []
    for entry in record['group_maps']:
        source = source_names.get(entry['Subregion'])
        if source is None:
            continue
        selected_group = mapped_measures.loc[mapped_measures.Subregion.eq(entry['Subregion'])]
        selected, eligible = len(selected_group), int(selected_group.computable.eq('True').sum())
        eligible_animals = sorted(selected_group.loc[selected_group.computable.eq('True'), 'AnimalID'].unique())
        if (selected, eligible, eligible_animals) != (entry['n_selected'], entry['n_computable'], sorted(entry['contributing_animals'])):
            raise ValueError('Group denominator/animal identities differ from exact selected cohort')
        groups.append(entry['Subregion'])
        available = entry['map_available']
        values = None
        if available:
            path = check(end_run / entry['length_path'], entry['length_sha256'])
            image = nib.load(path)
            matching_grid(image, grid)
            values = image.get_fdata()
        for level, labels in enumerate(levels, 1):
            sums = covered_projection(values, labels.reshape(grid.shape), coverage) if available else None
            for target in response.loc[response.Level == level].to_dict('records'):
                prior = joint_lookup.loc[(entry['Subregion'], target['TargetID'])]
                if (int(prior.endpoint_selected_neurons), int(prior.endpoint_eligible_neurons), int(prior.endpoint_contributing_animals)) != (selected, eligible, entry['n_animals']):
                    raise ValueError('New end-branch denominators differ from historical eligible-end comparison')
                if prior.SourceARMFullName != source['ARMFullName']:
                    raise ValueError('Historical source full name differs')
                index = int(float(target['ARMIndex']))
                value = float(sums[index]) if available and index < len(sums) else 0.
                row = {field: target[field] for field in ['TargetID', 'Level', 'ARMIndex', 'OfficialFullName', 'Abbreviation', 'Domain', 'Hemisphere', 'TargetStatus', 'KeyRangeConflict', 'region_voxels', 'covered_voxels', 'coverage_fraction']}
                row.update(subject='cm032', SourceARMIndex=source['ARMIndex'], SourceARMFullName=source['ARMFullName'],
                           SourceARMGroup=entry['Subregion'], SourceHemisphere=source['Hemisphere'],
                           comparison_status='conditional_end_branch_summary_registration_unaccepted', same_animal_pairing='unknown')
                row.update(projection_fields('end_branch', entry, value, target['covered_voxels']))
                rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 25140 or set(groups) != set(source_names):
        raise ValueError('Incomplete supplement source-target scope')
    if set(zip(frame.SourceARMGroup, frame.TargetID)) != set(zip(joint.SourceARMGroup, joint.TargetID)):
        raise ValueError('Supplement must losslessly join existing joint-table keys')
    if any(sha256(path) != expected for path, expected in bound.items()):
        raise RuntimeError('Bound input changed during supplement')
    output.mkdir(parents=True)
    table = output / 'cm032_end_branch_common_coverage.csv'
    frame.to_csv(table, index=False)
    saved = pd.read_csv(table)
    if len(saved) != len(frame):
        raise RuntimeError('Saved row count differs')
    provenance = dict(status='software_completed_provisional_cm032_end_branch_supplement', created_utc=datetime.now(timezone.utc).isoformat(),
                      subject='cm032', source_sha256=bound, code_sha256=sha256(__file__), outputs={table.name: sha256(table)},
                      rows=len(frame), targets=len(response), source_groups=len(groups), atlas_levels=[1, 2, 3, 4, 5, 6],
                      mapped_selected_neurons=410, mapped_end_eligible_neurons=378, selected_neuron_UIDs=sorted('::'.join(uid) for uid in identities(mapped_main)),
                      end_eligible_UIDs=sorted('::'.join(uid) for uid in eligible_ids), coverage_voxels=int(coverage.sum()),
                      measurement='template-space reconstructed original end-branch length, eligible mean within animal then equal contributing-animal mean',
                      missing_policy='No coverage or no eligible means NA; covered zero remains zero; no missing-animal zero imputation',
                      old_response_axon_endpoint_recomputed=False, new_NIfTI_or_figures=False, biological_terminal_arbor_acceptance=False,
                      anatomical_acceptance=False, causal_claim=False, independent_saved_table_review='pending')
    (output / 'supplement_provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': provenance['status'], 'rows': len(frame), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'table')
    build(parser.parse_args().output)
