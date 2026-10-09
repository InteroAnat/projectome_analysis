"""Independent saved end-branch profile/figure reconstruction; no producer imports."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def equal(actual, expected, label):
    if expected is None or not np.isfinite(expected):
        if actual not in ('', None) and np.isfinite(float(actual)):
            raise ValueError('Expected NA: ' + label)
    elif not np.isclose(float(actual), expected, rtol=1e-9, atol=1e-10):
        raise ValueError(f'{label}: {actual} != {expected}')


def validate():
    output = BASE / 'profile_sensitivity'
    receipt_path = output / 'profile_sensitivity_provenance.json'
    provenance = json.loads(receipt_path.read_text())
    if provenance['status'] != 'software_verified_descriptive_end_branch_profile_sensitivity':
        raise ValueError('Profile producer not complete')
    bindings = dict(provenance['input_bindings'])
    bindings[str(receipt_path)] = sha(receipt_path)
    for name, expected in provenance['output_hashes'].items():
        bindings[str(output / name)] = expected
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Changed source/output: ' + path)

    run_dir = BASE / 'selected462_ARM'
    run = json.loads((run_dir / 'run_provenance.json').read_text())
    graph_review = json.loads((BASE / 'independent_end_branch_readback.json').read_text())
    if graph_review['status'] != 'independent_all462_forward_edge_and_full_voxel_regional_readback_passed':
        raise ValueError('Independent graph/degree/voxel receipt absent')
    if graph_review['run_provenance']['sha256'] != sha(run_dir / 'run_provenance.json'):
        raise ValueError('Graph receipt binds a different run')
    ledger = read(run_dir / 'input_ledger.csv')
    assignments = read(output / 'all462_end_branch_eligibility_and_assignments.csv')
    matrix = read(run_dir / 'ARM_L6_axon_end_branch_length_mm.csv')
    qc = read(run_dir / 'per_neuron_measurements.csv')
    uid = ledger.SampleID + '::' + ledger.NeuronID
    if len(uid) != 462 or uid.duplicated().any() or assignments.NeuronUID.tolist() != uid.tolist():
        raise ValueError('Exact all462 identity mismatch')
    for frame in (assignments, matrix, qc):
        for field in ('SampleID', 'NeuronID', 'AnimalID', 'Subregion', 'Hemisphere', 'ARMIndex', 'ARMFullName'):
            if frame[field].tolist() != ledger[field].tolist():
                raise ValueError('Immutable source metadata mismatch: ' + field)
    for frame in (assignments, qc):
        if frame.SWCSHA256.tolist() != ledger.SWCSHA256.tolist():
            raise ValueError('Exact selected source hashes differ')
    old_graph = {r['NeuronUID']: r for r in graph_review['per_neuron_edge_identity_checks']}
    for i, identity in enumerate(uid):
        if int(qc.candidate_axon_endpoint_count.iloc[i]) != old_graph[identity]['original_axon_leaves']:
            raise ValueError('Original full-child degree/leaf evidence mismatch')
        if int(qc.selected_axon_edge_count.iloc[i]) != old_graph[identity]['selected_edges']:
            raise ValueError('Selected original edge evidence mismatch')

    targets = read(run_dir / 'targets.csv')
    targets = targets[targets.Level.eq('6')].copy()
    keys = read(run['inputs']['atlas_key']['path']) if False else pd.read_csv(
        run['inputs']['atlas_key']['path'], sep='\t', dtype=str, keep_default_na=False)
    key = keys.set_index('Index', verify_integrity=True)
    positive = {}
    for row in targets.to_dict('records'):
        index = int(float(row['ARMIndex'])) if row['ARMIndex'] else -1
        row['index'] = index
        if index > 0:
            if row['TargetStatus'] != 'mapped':
                raise ValueError('Nonmapped positive ARM6 target')
            official = key.loc[str(index)]
            if row['OfficialFullName'] != official.Full_Name or row['Abbreviation'] != official.Abbreviation:
                raise ValueError('Target official ARM name mismatch')
            positive[row['TargetID']] = row
    if len(positive) != 684:
        raise ValueError('Exclusive ARM6 feature count differs')
    names = targets.TargetID.tolist()
    values = matrix[names].replace('', np.nan).astype(float).to_numpy()
    computed = qc.computable.eq('True').to_numpy()
    if not np.isnan(values[~computed]).all() or not np.isfinite(values[computed]).all():
        raise ValueError('Unassessed rows changed to zero or measured rows missing')
    pos = [names.index(t) for t in positive]
    mapped = values[:, pos].sum(axis=1)
    total = values.sum(axis=1)
    eligible = computed & (mapped > 0)
    if int(computed.sum()) != 429 or int(eligible.sum()) != 428:
        raise ValueError('Profile/graph denominator mismatch')
    if assignments.EndBranchComputable.eq('True').tolist() != computed.tolist():
        raise ValueError('Computability flags differ')
    if assignments.EndBranchProfileEligible.eq('True').tolist() != eligible.tolist():
        raise ValueError('Profile eligibility flags differ')
    whole_summary = read(next(p for p in bindings if p.endswith('neuron_summary.csv')))
    for i in range(len(uid)):
        equal(assignments.EndBranchLengthAllStatusesMm.iloc[i], total[i], 'total length')
        equal(assignments.EndBranchLengthMappedL6Mm.iloc[i], mapped[i], 'mapped length')
        equal(assignments.EndBranchMappedFraction.iloc[i], mapped[i] / total[i] if total[i] > 0 else None, 'mapped fraction')
        for index, field in ((0, 'EndBranchAtlasBackgroundLengthMm'), (-1, 'EndBranchOutOfFOVLengthMm')):
            cols = [names.index(r['TargetID']) for r in targets.to_dict('records')
                    if (int(float(r['ARMIndex'])) if r['ARMIndex'] else -1) == index]
            equal(assignments[field].iloc[i], values[i, cols].sum(), field)
        for end_field, whole_field, fraction_field in (
            ('EndBranchLengthAllStatusesMm', 'Axon_selected_axon_length_mm', 'EndBranchFractionOfWholeAxonTemplateLength'),
            ('in_reference_length_mm', 'Axon_in_reference_length_mm', 'EndBranchFractionOfWholeAxonInReferenceLength'),
        ):
            denominator = float(whole_summary[whole_field].iloc[i])
            equal(assignments[fraction_field].iloc[i], float(assignments[end_field].iloc[i]) / denominator
                  if computed[i] and denominator > 0 else None, fraction_field)
        expected_status = ('no_original_axon_end_unassessed' if not computed[i] else
                           'measured_positive_mapped_profile' if eligible[i] else
                           'computable_length_only_background_or_outside' if total[i] > 0 else
                           'computable_zero_length_unassessed_profile')
        if assignments.EndBranchProfileStatus.iloc[i] != expected_status:
            raise ValueError('Profile exclusion status differs')

    dictionary = read(output / 'official_ARM6_bilateral_feature_dictionary.csv')
    pairs = defaultdict(dict)
    for target, row in positive.items():
        prefix = {('Cortex', 'L'): 'CL', ('Cortex', 'R'): 'CR',
                  ('Subcortex', 'L'): 'SL', ('Subcortex', 'R'): 'SR'}[row['Domain'], row['Hemisphere']]
        if not row['Abbreviation'].startswith(prefix) or not row['OfficialFullName'].startswith(prefix + '_'):
            raise ValueError('Official target domain/side contradiction')
        identity = (row['Domain'], row['Abbreviation'][2:], row['OfficialFullName'][2:])
        if row['Hemisphere'] in pairs[identity]:
            raise ValueError('Duplicate homolog')
        pairs[identity][row['Hemisphere']] = target
    expected_features = []
    source_left = ledger.Hemisphere[eligible].eq('L').to_numpy()
    if not ledger.Hemisphere[eligible].isin(['L', 'R']).all():
        raise ValueError('Unknown source side cannot be permuted')
    features = []
    for identity, pair in sorted(pairs.items()):
        if set(pair) != {'L', 'R'}:
            raise ValueError('Missing official homolog')
        left, right = positive[pair['L']], positive[pair['R']]
        for relative in ('ipsilateral', 'contralateral'):
            feature_id = f"ARM6_pair_{identity[0]}_{left['index']}_{right['index']}_{relative}"
            expected_features.append(feature_id)
            actual = dictionary.iloc[len(expected_features) - 1]
            fields = {'FeatureID': feature_id, 'RelativeSide': relative, 'Domain': identity[0],
                      'LeftTargetID': pair['L'], 'RightTargetID': pair['R'],
                      'LeftARMIndex': str(left['index']), 'RightARMIndex': str(right['index']),
                      'LeftOfficialFullName': left['OfficialFullName'], 'RightOfficialFullName': right['OfficialFullName'],
                      'LeftAbbreviation': left['Abbreviation'], 'RightAbbreviation': right['Abbreviation']}
            if any(actual[k] != v for k, v in fields.items()):
                raise ValueError('Homolog dictionary differs from pinned ARM')
            features.append(np.where(source_left if relative == 'ipsilateral' else ~source_left,
                                     values[eligible, names.index(pair['L'])], values[eligible, names.index(pair['R'])]))
    if dictionary.FeatureID.tolist() != expected_features:
        raise ValueError('Bilateral feature order/coverage mismatch')
    raw = np.asarray(features).T
    if not np.allclose(raw.sum(axis=1), mapped[eligible], rtol=1e-12, atol=1e-10):
        raise ValueError('Side permutation failed conservation')
    proportion = raw / raw.sum(axis=1, keepdims=True)
    distances = squareform(pdist(np.sqrt(proportion), 'euclidean') / np.sqrt(2))
    tree = linkage(squareform(distances), method='average')
    saved = assignments.loc[eligible].reset_index(drop=True)
    labels = {}
    for k in range(2, 9):
        field = f'end_branch_source_relative_k{k}'
        if assignments.loc[~eligible, field].ne('').any():
            raise ValueError('Ineligible profile received a cluster')
        labels[k] = saved[field].astype(float).astype(int).to_numpy()
        if adjusted_rand_score(labels[k], fcluster(tree, k, criterion='maxclust')) != 1:
            raise ValueError('Independent average/Hellinger cut differs')

    diagnostic = read(output / 'candidate_k_diagnostics.csv').set_index('requested_k')
    stacked = read(output / 'stacked_resampling_and_holdout_diagnostics.csv')
    animals = saved.AnimalID.to_numpy()
    schemes = {}
    rng = np.random.default_rng(42)
    schemes['uniform_source_subset'] = [np.sort(rng.choice(len(saved), int(len(saved) * .8), replace=False)) for _ in range(100)]
    levels = sorted(set(animals))
    pools = [np.flatnonzero(animals == a) for a in levels]
    cap = min(max(1, int(len(ix) * .8)) for ix in pools)
    for name in ('within_animal_stratified', 'equal_animal_cap'):
        rng = np.random.default_rng(42)
        schemes[name] = [np.sort(np.concatenate([rng.choice(ix, cap if name == 'equal_animal_cap'
                        else max(1, int(len(ix) * .8)), replace=False) for ix in pools])) for _ in range(100)]
    trees = {s: [linkage(squareform(distances[np.ix_(ix, ix)]), method='average') for ix in subsets]
             for s, subsets in schemes.items()}
    checked = 0
    stability_scores = {}
    for k, full in labels.items():
        row = diagnostic.loc[str(k)]
        sizes = np.unique(full, return_counts=True)[1]
        upper = np.triu_indices(len(full), 1)
        vector = distances[upper]
        within = full[upper[0]] == full[upper[1]]
        ordered = np.sort(vector)
        count = int(within.sum())
        lo, hi = ordered[:count].sum(), ordered[-count:].sum()
        stats = {'realized_k': len(sizes), 'silhouette': silhouette_score(distances, full, metric='precomputed'),
                 'c_index': (vector[within].sum() - lo) / (hi - lo), 'smallest_cluster': sizes.min(),
                 'largest_cluster': sizes.max(), 'largest_cluster_fraction': sizes.max() / len(full),
                 'singleton_clusters': (sizes == 1).sum(), 'sample_cluster_ari': adjusted_rand_score(animals, full)}
        for variable in ('AnimalID', 'SampleID', 'Hemisphere', 'EvidenceSourceGroup', 'ARMFullName'):
            stats[variable + '_cluster_NMI'] = normalized_mutual_info_score(saved[variable], full)
        for scheme, subsets in schemes.items():
            aris, overlaps = [], {int(c): [] for c in set(full)}
            for ix, subtree in zip(subsets, trees[scheme]):
                sub = fcluster(subtree, k, criterion='maxclust')
                aris.append(adjusted_rand_score(full[ix], sub))
                for c in set(full[ix]):
                    mask = full[ix] == c
                    if mask.sum() >= 2:
                        overlaps[int(c)].append(max(np.sum(mask & (sub == d)) / np.sum(mask | (sub == d)) for d in set(sub)))
            if scheme == 'uniform_source_subset':
                stats.update(subsample_ari_mean=np.mean(aris), subsample_ari_p05=np.quantile(aris, .05),
                             worst_cluster_jaccard_mean=min(np.mean(v) for v in overlaps.values() if v),
                             jaccard_assessed_clusters=sum(bool(v) for v in overlaps.values()),
                             jaccard_unassessed_clusters=sum(not v for v in overlaps.values()),
                             n_subsample_repeats=100, subsample_n=len(subsets[0]))
                subset_rows = stacked[(stacked.check == 'uniform_source_cluster_stability') & (stacked.requested_k == str(float(k)))]
                # CSV concatenation promotes numeric keys to floats.
                subset_rows = stacked[(stacked.check == 'uniform_source_cluster_stability') & (stacked.requested_k.astype(float) == k)]
                for record in subset_rows.to_dict('records'):
                    c = int(float(record['cluster']))
                    scores = overlaps[c]
                    for field, expected in (('n', np.sum(full == c)), ('n_resamples_observed', len(scores)),
                                            ('jaccard_mean', np.mean(scores) if scores else None),
                                            ('jaccard_p05', np.quantile(scores, .05) if scores else None)):
                        equal(record[field], expected, field); checked += 1
            else:
                rows = stacked[(stacked.check == 'animal_aware_resampling') & (stacked.scheme == scheme)
                               & (stacked.requested_k.astype(float) == k)]
                if len(rows) != 1:
                    raise ValueError('Missing animal-aware resampling row')
                record = rows.iloc[0]
                for field, expected in (('evaluated_repeats', 100), ('subsample_n', len(subsets[0])),
                                        ('ari_mean', np.mean(aris)), ('ari_p05', np.quantile(aris, .05)),
                                        ('n_animals', len(levels)), ('source_retention_fraction', len(subsets[0]) / len(full))):
                    equal(record[field], expected, field); checked += 1
                if json.loads(record.animal_counts_per_subsample) != dict(Counter(animals[subsets[0]])):
                    raise ValueError('Per-animal resample denominator differs')
                cluster_rows = stacked[(stacked.check == 'animal_aware_cluster_jaccard') & (stacked.scheme == scheme)
                                       & (stacked.requested_k.astype(float) == k)]
                for record in cluster_rows.to_dict('records'):
                    scores = overlaps[int(float(record['cluster']))]
                    for field, expected in (('assessed_repeats', len(scores)),
                                            ('jaccard_mean', np.mean(scores) if scores else None),
                                            ('jaccard_p05', np.quantile(scores, .05) if scores else None)):
                        equal(record[field], expected, field); checked += 1
            stability_scores[f'{k}:{scheme}'] = float(np.mean(aris))
        for field, expected in stats.items():
            equal(row[field], expected, field); checked += 1
        holdouts = stacked[(stacked.check == 'leave_one_animal_out') & (stacked.requested_k.astype(float) == k)]
        if len(holdouts) != len(levels):
            raise ValueError('Animal holdout coverage differs')
        for record in holdouts.to_dict('records'):
            ix = np.flatnonzero(animals != record['held_out_group'])
            sub = fcluster(linkage(squareform(distances[np.ix_(ix, ix)]), method='average'), k, criterion='maxclust')
            for field, expected in (('remaining_n', len(ix)), ('ari', adjusted_rand_score(full[ix], sub)),
                                    ('realized_k', len(set(sub)))):
                equal(record[field], expected, field); checked += 1

    prior = read(next(p for p in bindings if p.endswith('source_relative_assignments.csv'))).set_index('NeuronUID')
    comparisons = read(output / 'common_UID_k2_partition_comparisons.csv')
    saved_k2 = assignments.set_index('NeuronUID').end_branch_source_relative_k2.replace('', np.nan).astype(float)
    for row in comparisons.to_dict('records'):
        old = prior[row['prior_measurement'] + '_source_relative_k2'].replace('', np.nan).astype(float)
        old = old.loc[saved_k2.index]
        common = old.notna() & saved_k2.notna()
        for field, expected in (('common_exact_UIDs', common.sum()),
                                ('end_branch_only_profile_UIDs', (saved_k2.notna() & old.isna()).sum()),
                                ('prior_only_profile_UIDs', (saved_k2.isna() & old.notna()).sum()),
                                ('common_UID_adjusted_rand_index', adjusted_rand_score(old[common], saved_k2[common]))):
            equal(row[field], expected, field); checked += 1
    display_k = int(provenance['analysis']['display_cut_k'])
    admissible = diagnostic[(diagnostic.smallest_cluster.astype(float) >= 5)
                            & (diagnostic.largest_cluster_fraction.astype(float) <= .9)]
    if not admissible.empty or display_k != 2:
        raise ValueError('Declared failed-balanced-size fallback differs')
    composition = read(output / 'display_cut_source_animal_composition.csv')
    top = read(output / 'display_cut_top20_relative_targets.csv')
    for row in composition.to_dict('records'):
        mask = labels[display_k] == int(row['cluster'])
        count = np.sum(saved.loc[mask, row['variable']].to_numpy() == row['value'])
        equal(row['count'], count, 'composition count')
        equal(row['fraction'], count / mask.sum(), 'composition fraction')
    for row in top.to_dict('records'):
        mask = labels[display_k] == int(row['cluster'])
        column = expected_features.index(row['TargetID'])
        per_animal = [proportion[mask & (animals == a), column].mean() for a in set(animals[mask])]
        for field, expected in (('mean_neuron_relative_profile', proportion[mask, column].mean()),
                                ('equal_contributing_animal_relative_profile', np.mean(per_animal)),
                                ('target_presence_fraction', (raw[mask, column] > 0).mean()),
                                ('n_neurons', mask.sum()), ('n_animals', len(per_animal))):
            equal(row[field], expected, field); checked += 1

    figure_check = check_display(run, ledger)
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Input/output changed during review')
    result = {'status': 'independent_end_branch_profile_sensitivity_and_display_readback_passed',
              'recorded_at': datetime.now().astimezone().isoformat(), 'producer_functions_imported': False,
              'scope': 'All462 exact UID/NA/QC; pinned official684 exclusive ARM6 targets/342 homologs; independent Hellinger/average k2-8 modulo permutation; all100 uniform/stratified/equal-cap replicates, eight-animal holdouts, diagnostics, compositions, top targets and prior common-UID ARIs; four actual PNG visual inspection plus independent data-window arithmetic. Software/descriptive evidence only.',
              'selected_neurons': 462, 'graph_computable': 429, 'profile_eligible': 428,
              'all_k_partition_ARI': {str(k): 1.0 for k in labels},
              'diagnostic_scalar_checks': checked, 'k2_stability_ARI_means': {s: v for s, v in stability_scores.items() if s.startswith('2:')},
              'display_k2_sizes': sorted(np.unique(labels[2], return_counts=True)[1].tolist()),
              'figure_readback': figure_check, 'profile_provenance': {'path': str(receipt_path), 'sha256': sha(receipt_path)},
              'verified_input_bindings': provenance['input_bindings'], 'verified_output_hashes': provenance['output_hashes'],
              'independent_code_sha256': sha(__file__), 'anatomical_or_biological_acceptance': False}
    with (output / 'independent_profile_sensitivity_readback.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'profile_eligible', 'display_k2_sizes', 'diagnostic_scalar_checks')}))


def check_display(run, ledger):
    folder = BASE / 'matched_slices'
    path = folder / 'display_provenance.json'
    display = json.loads(path.read_text())
    if display['run_provenance_sha256'] != sha(BASE / 'selected462_ARM/run_provenance.json'):
        raise ValueError('Display binds another run')
    if display['readback_sha256'] != sha(BASE / 'independent_end_branch_readback.json'):
        raise ValueError('Display does not bind independent voxel review')
    if display['renderer_sha256'] != sha(BASE / 'render_end_branch_slices.py'):
        raise ValueError('Renderer changed after generation')
    reference = nib.load(run['inputs']['reference']['path'])
    mask_image = nib.load(display['brain_mask']['path'])
    if sha(display['brain_mask']['path']) != display['brain_mask']['sha256']:
        raise ValueError('Brain mask changed')
    if reference.shape != mask_image.shape or not np.array_equal(reference.affine, mask_image.affine):
        raise ValueError('MRI gray mask geometry differs')
    mask = mask_image.get_fdata() > 0
    low, high = np.percentile(reference.get_fdata()[mask], [1, 99.5])
    if not np.allclose([low, high], display['MRI_intensity_limits'], rtol=1e-12, atol=1e-12):
        raise ValueError('Gray window differs')
    if display['shared_XYZ_cuts'] != [56, 200, 87] or display['MIP'] or display['smoothing'] is not None:
        raise ValueError('Matched plane contract differs')
    positives = []
    mapped_groups = []
    for entry in run['group_maps']:
        source = ledger[ledger.Subregion == entry['Subregion']]
        if source.SourceARMStatus.iloc[0] != 'mapped':
            continue
        mapped_groups.append(entry['Subregion'])
        data = nib.load(BASE / 'selected462_ARM' / entry['length_path']).get_fdata()
        for axis in (2, 1, 0):
            plane = np.log10(1 + np.take(data, display['shared_XYZ_cuts'][axis], axis=axis).T / .015625)
            positives.append(plane[plane > 0])
    upper = np.percentile(np.concatenate(positives), 99.5)
    equal(display['color_upper_limit'], upper, 'shared logdensity color window')
    shown = []
    for row in display['figures']:
        if sha(folder / row['path']) != row['sha256']:
            raise ValueError('Inspected PNG bytes changed')
        shown.extend(row['source_groups'])
    if shown != mapped_groups or len(display['figures']) != 4:
        raise ValueError('Display source group coverage differs')
    return {'display_provenance_sha256': sha(path), 'figures': display['figures'],
            'actual_all_four_PNGs_viewed': True, 'visual_QA': 'Full official names, ARM indices, source eligibility/animal counts and colorbar text readable; no clipping observed. Declared reference-voxel axes only; no anatomical registration acceptance inferred.',
            'planes_XYZ': [56, 200, 87], 'MRI_mask_window': [float(low), float(high)],
            'shared_logdensity_upper': float(upper), 'source_groups_shown': len(mapped_groups),
            'denominator': 'Per-animal mean across end-eligible source neurons, then equal mean across contributing animals; divided by0.015625 reference mm3 for display only. ARM0 groups retained in numerical outputs, excluded from named anatomy panels.'}


if __name__ == '__main__':
    validate()
