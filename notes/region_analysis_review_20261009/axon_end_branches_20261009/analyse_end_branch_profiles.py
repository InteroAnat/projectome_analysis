"""Source-relative ARM6 end-branch profile sensitivity on the selected ledger.

Consumes a completed, independently checked graph-length run. It neither
detects biological terminals nor changes atlas labels or reconstruction data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
from cluster_arm_projection_profiles import (
    bools, digest, exact_uids, prepare_target_catalog,
    profile_distance, run_analysis, source_relative_profiles,
)

IDENTITY = ['SampleID', 'NeuronID', 'AnimalID', 'Subregion', 'ARMIndex', 'ARMFullName', 'Hemisphere']
REVIEW_STATUS = 'independent_all462_forward_edge_and_full_voxel_regional_readback_passed'


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def verify_hash(path, expected):
    if not isinstance(expected, str) or digest(path) != expected:
        raise ValueError('Missing or mismatched hash: ' + str(path))


def completed_run(run_dir, review_path, prior_dir):
    """Gate every consumed artifact against completed producer/readback receipts."""
    provenance = run_dir / 'run_provenance.json'
    run = json.loads(provenance.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    if run.get('status') != 'software_verified_descriptive_end_branches':
        raise ValueError('End-branch run is not completed')
    if review.get('status') != REVIEW_STATUS:
        raise ValueError('Independent all-neuron end-branch review has not passed')
    if Path(review['run_provenance']['path']).resolve() != provenance.resolve():
        raise ValueError('Independent review refers to another run')
    verify_hash(provenance, review['run_provenance']['sha256'])
    if review.get('verified_artifacts') != run.get('artifacts') or review.get('source_bindings') != run.get('inputs'):
        raise ValueError('Review artifact/source scope differs from completed run')
    if run.get('selected_neurons') != 462 or review.get('selected_neurons') != 462:
        raise ValueError('Expected the unchanged selected462 ledger')
    for name in ['input_ledger.csv', 'per_neuron_measurements.csv', 'targets.csv', 'ARM_L6_axon_end_branch_length_mm.csv']:
        verify_hash(run_dir / name, run['artifacts'].get(name))
    for value in run['inputs'].values():
        verify_hash(value['path'], value['sha256'])
    for relative, expected in run['source_code_sha256'].items():
        verify_hash(ROOT / relative, expected)
    prior_path = prior_dir / 'sensitivity_provenance.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if prior.get('status') != 'software_verified_feature_transformation_sensitivity':
        raise ValueError('Prior source-relative sensitivity is not completed')
    verify_hash(prior['manifest'], prior['manifest_SHA256'])
    verify_hash(prior_dir / 'source_relative_assignments.csv', prior['output_hashes'].get('source_relative_assignments.csv'))
    verify_hash(prior['export_provenance'], prior['export_provenance_SHA256'])
    export = json.loads(Path(prior['export_provenance']).read_text(encoding='utf-8'))
    if export.get('status') != 'software_verified_descriptive_tables':
        raise ValueError('Whole-axon source summary export is not completed')
    if export.get('manifest_sha256') != prior['manifest_SHA256']:
        raise ValueError('Whole-axon export and prior partitions bind different manifests')
    for name in ['reference', 'atlas', 'atlas_key']:
        if export.get(name + '_sha256') != run['inputs'][name]['sha256']:
            raise ValueError('Whole-axon/end-branch input binding differs: ' + name)
    summary_path = Path(prior['export_provenance']).parent / 'neuron_summary.csv'
    verify_hash(summary_path, export['artifacts'].get('neuron_summary.csv'))
    # The end-branch producer accepts the completed enriched summary as its ledger.
    accepted_ledgers = {prior['manifest_SHA256'], export['artifacts']['neuron_summary.csv']}
    if run['inputs']['manifest']['sha256'] not in accepted_ledgers:
        raise ValueError('End-branch run does not bind the prior manifest or its exact completed summary')
    current = read_csv(run_dir / 'input_ledger.csv')
    original = read_csv(prior['manifest'])
    if not exact_uids(current).equals(exact_uids(original)):
        raise ValueError('Enriched/original ledger identity order differs')
    source_contract = IDENTITY + ['SWCSHA256', 'SWCPath', 'ReferenceSHA256', 'CoordinateFrame',
        'IndexScaleUm', 'ARMLevel', 'AtlasSHA256', 'AtlasKeySHA256', 'SourceARMStatus',
        'OriginalSourceLabel', 'EvidenceSourceGroup', 'SourceCohort']
    for field in source_contract:
        if field not in current or field not in original or not current[field].equals(original[field]):
            raise ValueError('Enriched/original source contract differs: ' + field)
    return run, review, prior


def prepare_profiles(manifest, qc, matrix, targets):
    """Measured missing rows stay NA; only positive mapped ARM6 is compositional."""
    uid = exact_uids(manifest)
    for frame in [qc, matrix]:
        if not exact_uids(frame).equals(uid):
            raise ValueError('Exact identity/order differs from manifest')
        for field in IDENTITY:
            if field not in frame or not frame[field].equals(manifest[field]):
                raise ValueError('Source metadata differs: ' + field)
    if 'SWCSHA256' not in qc or not qc.SWCSHA256.equals(manifest.SWCSHA256):
        raise ValueError('QC/source reconstruction hashes differ')
    targets = prepare_target_catalog(targets)
    if targets.TargetID.duplicated().any():
        raise ValueError('Duplicate ARM target token')
    l6 = targets[targets.Level.eq(6)].copy().set_index('TargetID')
    positive = l6[(l6.ARMIndex > 0) & l6.TargetStatus.eq('mapped')]
    if positive.empty or ((l6.ARMIndex > 0) & ~l6.TargetStatus.eq('mapped')).any():
        raise ValueError('Positive ARM6 target absent, conflicted, or unassigned')
    if matrix.columns.tolist() != IDENTITY + l6.index.tolist():
        raise ValueError('ARM6 matrix target schema/order differs from official target catalog')
    values = matrix[l6.index].replace('', np.nan).apply(pd.to_numeric, errors='raise')
    values.index = uid
    eligible = bools(qc.computable)
    a = values.to_numpy(dtype=float)
    if not np.isnan(a[~eligible]).all():
        raise ValueError('Uncomputable rows must be all NA, not zero')
    if not np.isfinite(a[eligible]).all() or (a[eligible] < 0).any():
        raise ValueError('Computable rows require finite nonnegative measured lengths')
    source_columns = IDENTITY + [
        'SWCSHA256', 'EvidenceSourceGroup', 'SourceCohort',
        'Henry_coarse_INS_visual_evidence', 'SourceARMStatus', 'CoordinateFrame',
        'CoordinateOriginStatus', 'RegistrationStatus', 'ImageTerminalStatus',
        'SourceRootLookupPolicy', 'SourceARMHemisphereConflict',
    ]
    # Keep a readable QC ledger; the full source metadata remain hash-bound inputs.
    ledger = manifest[[field for field in source_columns if field in manifest]].copy()
    ledger.index = uid
    ledger.index.name = 'NeuronUID'
    if 'NeuronUID' in ledger:
        ledger = ledger.drop(columns='NeuronUID')
    for field in ['candidate_status', 'candidate_axon_endpoint_count', 'selected_axon_edge_count',
                  'total_length_mm', 'in_reference_length_mm', 'outside_reference_length_mm']:
        if field not in qc:
            raise ValueError('Required graph QC absent: ' + field)
        ledger[field] = qc[field].to_numpy()
    for field in ['candidate_axon_endpoint_count', 'selected_axon_edge_count']:
        numbers = pd.to_numeric(ledger[field], errors='raise')
        if not np.isfinite(numbers).all() or (numbers < 0).any() or (numbers != np.floor(numbers)).any():
            raise ValueError('Invalid original graph count')
        ledger[field] = numbers.astype(int)
    total = values.sum(axis=1, min_count=1)
    for field in ['total_length_mm', 'in_reference_length_mm', 'outside_reference_length_mm']:
        ledger[field] = pd.to_numeric(ledger[field].replace('', np.nan), errors='raise')
        if ledger.loc[~eligible, field].notna().any():
            raise ValueError('Missing graph length must stay NA')
    if not np.allclose(total[eligible], ledger.loc[eligible, 'total_length_mm'], rtol=1e-9, atol=1e-9):
        raise ValueError('ARM6 regional lengths do not conserve absolute graph length')
    mapped = values[positive.index].sum(axis=1, min_count=1)
    ledger['EndBranchComputable'] = eligible
    ledger['EndBranchLengthAllStatusesMm'] = total
    ledger['EndBranchLengthMappedL6Mm'] = mapped
    ledger['EndBranchMappedFraction'] = mapped / total.replace(0, np.nan)
    for index, label in [(0, 'AtlasBackground'), (-1, 'OutOfFOV')]:
        columns = l6.index[l6.ARMIndex.eq(index)]
        ledger['EndBranch' + label + 'LengthMm'] = values[columns].sum(axis=1, min_count=1) if len(columns) else np.where(eligible, 0, np.nan)
    if not np.allclose(ledger.loc[eligible, 'EndBranchOutOfFOVLengthMm'],
                       ledger.loc[eligible, 'outside_reference_length_mm'], rtol=1e-9, atol=1e-9):
        raise ValueError('Regional outside-FOV length differs from source QC')
    if not np.allclose(ledger.loc[eligible, 'total_length_mm'],
                       ledger.loc[eligible, 'in_reference_length_mm'] + ledger.loc[eligible, 'outside_reference_length_mm'],
                       rtol=1e-9, atol=1e-9):
        raise ValueError('Source inside/outside QC does not conserve absolute length')
    profile_eligible = eligible & mapped.gt(0).to_numpy()
    ledger['EndBranchProfileEligible'] = profile_eligible
    ledger['EndBranchProfileStatus'] = np.select(
        [~eligible, profile_eligible, total.gt(0).to_numpy()],
        ['no_original_axon_end_unassessed', 'measured_positive_mapped_profile',
         'computable_length_only_background_or_outside'],
        default='computable_zero_length_unassessed_profile',
    )
    raw = values.loc[profile_eligible, positive.index]
    relative, feature_map = source_relative_profiles(raw, ledger.loc[profile_eligible], positive)
    return ledger, relative, feature_map


def compare_partitions(ledger, labels, previous):
    previous = previous.set_index('NeuronUID', verify_integrity=True)
    if set(previous.index) != set(ledger.index):
        raise ValueError('Prior source-relative all462 identity coverage differs')
    for field in ['SampleID', 'NeuronID', 'AnimalID', 'Hemisphere']:
        if not previous.loc[ledger.index, field].equals(ledger[field]):
            raise ValueError('Prior exact source metadata differs: ' + field)
    mask = ledger.EndBranchProfileEligible
    rows = []
    for family in ['axon', 'endpoint']:
        column = family + '_source_relative_k2'
        old_values = previous.loc[ledger.index, column]
        old = pd.to_numeric(old_values.where(old_values.ne(''), np.nan), errors='raise')
        expected = bools(previous.loc[ledger.index, 'AxonProfileEligible' if family == 'axon' else 'EndpointProfileEligible'])
        if not np.array_equal(old.notna().to_numpy(), expected):
            raise ValueError('Prior partition missingness differs from prior eligibility')
        if (old.dropna() <= 0).any() or not np.equal(old.dropna(), np.floor(old.dropna())).all():
            raise ValueError('Prior cluster labels must be positive integers')
        common = mask & old.notna()
        new = pd.Series(labels, index=ledger.index[mask])
        rows.append({'prior_measurement': family, 'requested_k': 2,
            'common_exact_UIDs': int(common.sum()),
            'end_branch_only_profile_UIDs': int((mask & old.isna()).sum()),
            'prior_only_profile_UIDs': int((~mask & old.notna()).sum()),
            'common_UID_adjusted_rand_index': float(adjusted_rand_score(new.loc[common[common].index], old.loc[common])) if common.sum() >= 2 else None,
            'comparison_scope': 'Fitted descriptive source-relative partitions on intersecting eligible identities; not prediction or biological-class validation.'})
    return pd.DataFrame(rows)


def whole_axon_coverage(ledger, summary):
    """Join cached measured lengths by exact identity, never turning NA into zero."""
    summary = summary.copy()
    summary.index = exact_uids(summary)
    if set(summary.index) != set(ledger.index):
        raise ValueError('Whole-axon summary identity coverage differs')
    summary = summary.loc[ledger.index]
    for field in ['AnimalID', 'SWCSHA256', 'Hemisphere']:
        if not summary[field].equals(ledger[field]):
            raise ValueError('Whole-axon source metadata differs: ' + field)
    result = ledger.copy()
    for source, destination in [('Axon_selected_axon_length_mm', 'WholeAxonTemplateLengthMm'),
                                ('Axon_in_reference_length_mm', 'WholeAxonInReferenceLengthMm')]:
        numbers = pd.to_numeric(summary[source], errors='raise')
        if not np.isfinite(numbers).all() or (numbers < 0).any():
            raise ValueError('Cached whole-axon lengths must be measured nonnegative values')
        result[destination] = numbers
    eligible = result.EndBranchComputable
    for numerator, denominator, output in [
        ('EndBranchLengthAllStatusesMm', 'WholeAxonTemplateLengthMm', 'EndBranchFractionOfWholeAxonTemplateLength'),
        ('in_reference_length_mm', 'WholeAxonInReferenceLengthMm', 'EndBranchFractionOfWholeAxonInReferenceLength'),
    ]:
        if (result.loc[eligible, numerator] > result.loc[eligible, denominator] + 1e-8).any():
            raise ValueError('End-branch subset length exceeds measured whole-axon length')
        result[output] = result[numerator] / result[denominator].replace(0, np.nan)
        result.loc[~eligible, output] = np.nan
    return result


def self_check():
    """Independent tiny conservation, NA, identity and shared-method checks."""
    import unittest
    from unittest.mock import patch

    class Checks(unittest.TestCase):
        def fixture(self):
            n = 8
            frame = pd.DataFrame({'SampleID': ['s'] * n, 'NeuronID': [str(i) + '.swc' for i in range(n)],
                'AnimalID': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c'], 'Subregion': ['source'] * n,
                'ARMIndex': ['1'] * n, 'ARMFullName': ['CL_x'] * n,
                'Hemisphere': ['L', 'R'] * 4, 'SWCSHA256': ['hash'] * n})
            targets = pd.DataFrame({'TargetID': ['left', 'right', 'zero'], 'Level': ['6'] * 3,
                'ARMIndex': ['1', '2', '0'], 'TargetStatus': ['mapped', 'mapped', 'zero_unassigned'],
                'Domain': ['Cortex'] * 3, 'Abbreviation': ['CL_x', 'CR_x', ''],
                'OfficialFullName': ['CL_region_x', 'CR_region_x', 'Unassigned'], 'Hemisphere': ['L', 'R', '']})
            values = np.array([[1., 3., 0.], [3., 1., 0.], [2., 2., 0.], [4., 1., 0.],
                               [1., 4., 0.], [2., 1., 0.], [1., 2., 0.], [np.nan] * 3])
            matrix = pd.concat([frame[IDENTITY], pd.DataFrame(values, columns=targets.TargetID)], axis=1)
            qc = frame.copy()
            qc['computable'] = ['True'] * 7 + ['False']
            qc['candidate_status'] = ['candidate'] * 7 + ['none']
            qc['candidate_axon_endpoint_count'] = ['1'] * 7 + ['0']
            qc['selected_axon_edge_count'] = ['1'] * 7 + ['0']
            qc['total_length_mm'] = np.sum(values, axis=1)
            qc['in_reference_length_mm'] = qc.total_length_mm
            qc['outside_reference_length_mm'] = [0.] * 7 + [np.nan]
            return frame, qc, matrix, targets

        def test_relative_and_missing(self):
            ledger, raw, _ = prepare_profiles(*self.fixture())
            self.assertEqual(int(ledger.EndBranchProfileEligible.sum()), 7)
            np.testing.assert_array_equal(raw.iloc[0], raw.iloc[1])
            self.assertTrue(np.isnan(ledger.EndBranchLengthAllStatusesMm.iloc[-1]))
            np.testing.assert_allclose(raw.sum(axis=1), ledger.EndBranchLengthMappedL6Mm.iloc[:7])

        def test_missing_is_not_zero(self):
            frame, qc, matrix, targets = self.fixture()
            matrix.loc[7, ['left', 'right', 'zero']] = 0.
            with self.assertRaisesRegex(ValueError, 'all NA'):
                prepare_profiles(frame, qc, matrix, targets)

        def test_identity(self):
            frame, qc, matrix, targets = self.fixture()
            matrix.loc[0, 'AnimalID'] = 'foreign'
            with self.assertRaisesRegex(ValueError, 'metadata'):
                prepare_profiles(frame, qc, matrix, targets)

        def test_conservation(self):
            frame, qc, matrix, targets = self.fixture()
            matrix.loc[0, 'left'] = 99.
            with self.assertRaisesRegex(ValueError, 'conserve'):
                prepare_profiles(frame, qc, matrix, targets)

        def test_shared_analysis(self):
            ledger, raw, targets = prepare_profiles(*self.fixture())
            _, distance, _ = profile_distance(raw)
            self.assertEqual(distance[0, 1], 0.)
            result = run_analysis(raw, ledger.loc[raw.index], targets, None, [2, 3], 2, 0.8, 42,
                                  'hellinger', 'synthetic_end_branch_source_relative')
            self.assertEqual(result['summary']['n_profiles'], 7)
            self.assertIn('leave_one_animal_out', result['tables'])

        def test_measured_zero_is_unclustered(self):
            frame, qc, matrix, targets = self.fixture()
            matrix.loc[0, ['left', 'right', 'zero']] = 0.
            qc.loc[0, ['total_length_mm', 'in_reference_length_mm']] = 0.
            ledger, _, _ = prepare_profiles(frame, qc, matrix, targets)
            self.assertTrue(ledger.EndBranchComputable.iloc[0])
            self.assertFalse(ledger.EndBranchProfileEligible.iloc[0])
            self.assertEqual(ledger.EndBranchLengthAllStatusesMm.iloc[0], 0.)

        def test_key_conflict(self):
            frame, qc, matrix, targets = self.fixture()
            targets.loc[0, 'TargetStatus'] = 'key_level_conflict'
            with self.assertRaisesRegex(ValueError, 'conflicted'):
                prepare_profiles(frame, qc, matrix, targets)

        def test_unknown_side(self):
            frame, qc, matrix, targets = self.fixture()
            for table in [frame, qc, matrix]:
                table.loc[0, 'Hemisphere'] = ''
            with self.assertRaisesRegex(ValueError, 'known current source'):
                prepare_profiles(frame, qc, matrix, targets)

        def test_running_producer_gate(self):
            with patch.object(Path, 'read_text', side_effect=[json.dumps({'status': 'running'}), json.dumps({'status': REVIEW_STATUS})]):
                with self.assertRaisesRegex(ValueError, 'not completed'):
                    completed_run(Path('synthetic_run'), Path('synthetic_review'), Path('synthetic_prior'))

        def test_independent_review_gate(self):
            with patch.object(Path, 'read_text', side_effect=[json.dumps({'status': 'software_verified_descriptive_end_branches'}), json.dumps({'status': 'running'})]):
                with self.assertRaisesRegex(ValueError, 'has not passed'):
                    completed_run(Path('synthetic_run'), Path('synthetic_review'), Path('synthetic_prior'))

        def test_whole_axon_fraction_preserves_NA(self):
            frame, qc, matrix, targets = self.fixture()
            ledger, _, _ = prepare_profiles(frame, qc, matrix, targets)
            summary = frame.copy()
            summary['Axon_selected_axon_length_mm'] = [8., 8., 8., 10., 10., 6., 6., 0.]
            summary['Axon_in_reference_length_mm'] = summary.Axon_selected_axon_length_mm
            joined = whole_axon_coverage(ledger, summary)
            self.assertEqual(joined.EndBranchFractionOfWholeAxonTemplateLength.iloc[0], 0.5)
            self.assertTrue(np.isnan(joined.EndBranchFractionOfWholeAxonTemplateLength.iloc[-1]))

        def test_common_UID_partition_comparison(self):
            ledger, _, _ = prepare_profiles(*self.fixture())
            previous = ledger.reset_index()[['NeuronUID', 'SampleID', 'NeuronID', 'AnimalID', 'Hemisphere']]
            previous['AxonProfileEligible'] = ['True'] * 7 + ['False']
            previous['EndpointProfileEligible'] = ['True'] * 4 + ['False', 'True', 'True', 'False']
            previous['axon_source_relative_k2'] = [1, 1, 1, 2, 2, 2, 2, '']
            previous['endpoint_source_relative_k2'] = [1, 1, 1, 2, '', 2, 2, '']
            result = compare_partitions(ledger, np.array([1, 1, 1, 2, 2, 2, 2]), previous)
            self.assertEqual(result.common_exact_UIDs.tolist(), [7, 6])
            self.assertEqual(result.common_UID_adjusted_rand_index.tolist(), [1., 1.])

        def test_subset_exceeds_whole_rejected(self):
            frame, qc, matrix, targets = self.fixture()
            ledger, _, _ = prepare_profiles(frame, qc, matrix, targets)
            summary = frame.copy()
            summary['Axon_selected_axon_length_mm'] = [1.] * 8
            summary['Axon_in_reference_length_mm'] = [1.] * 8
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                whole_axon_coverage(ledger, summary)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    return {'passed': result.testsRun, 'failures': 0}


def analyse(args):
    run_dir, review_path, prior_dir = args.run.resolve(), args.review.resolve(), args.prior_relative.resolve()
    output = args.output.resolve()
    output.relative_to(Path(__file__).resolve().parent)
    if output.exists():
        raise FileExistsError('Use a fresh sensitivity output directory')
    run, review, prior = completed_run(run_dir, review_path, prior_dir)
    inputs = [run_dir / name for name in ['run_provenance.json', 'input_ledger.csv', 'per_neuron_measurements.csv',
              'targets.csv', 'ARM_L6_axon_end_branch_length_mm.csv']]
    inputs += [review_path, prior_dir / 'sensitivity_provenance.json', prior_dir / 'source_relative_assignments.csv']
    whole_summary = Path(prior['export_provenance']).parent / 'neuron_summary.csv'
    inputs += [Path(prior['export_provenance']), whole_summary, Path(prior['manifest'])]
    codes = [Path(__file__), ROOT / 'group_analysis/scripts/cluster_arm_projection_profiles.py', ROOT / 'main_scripts/clustering_validation.py']
    inputs += codes
    protected = {str(path): digest(path) for path in inputs}
    ledger, raw, targets = prepare_profiles(read_csv(run_dir / 'input_ledger.csv'),
        read_csv(run_dir / 'per_neuron_measurements.csv'), read_csv(run_dir / 'ARM_L6_axon_end_branch_length_mm.csv'),
        read_csv(run_dir / 'targets.csv'))
    ledger = whole_axon_coverage(ledger, read_csv(whole_summary))
    if len(ledger) != 462 or int(ledger.EndBranchComputable.sum()) != run['eligible_neurons']:
        raise ValueError('All462 or graph denominator accounting differs from run')
    if not np.isclose(ledger.EndBranchLengthAllStatusesMm.sum(), run['total_end_branch_length_mm'], rtol=1e-9, atol=1e-9):
        raise ValueError('Absolute end-branch length total differs from producer')
    if (int(ledger.candidate_axon_endpoint_count.sum()) != run['original_axon_ends']
            or int(ledger.selected_axon_edge_count.sum()) != run['selected_original_axon_edges']):
        raise ValueError('Absolute original endpoint/edge counts differ from producer')
    result = run_analysis(raw, ledger.loc[raw.index], targets, None, list(range(2, 9)), 100, 0.8, 42,
                          'hellinger', 'all462_eligible_source_relative_end_branch_length')
    for k, labels in result['labels'].items():
        ledger.loc[raw.index, f'end_branch_source_relative_k{k}'] = labels
    comparisons = compare_partitions(ledger, result['labels'][2], read_csv(prior_dir / 'source_relative_assignments.csv'))
    output.mkdir(parents=True, exist_ok=False)
    ledger.to_csv(output / 'all462_end_branch_eligibility_and_assignments.csv', index_label='NeuronUID')
    diagnostics = result['diagnostics'].copy()
    diagnostics.insert(0, 'measurement', 'reconstructed_axon_end_branch_length')
    diagnostics.to_csv(output / 'candidate_k_diagnostics.csv', index=False)
    stacked = []
    for name in ['animal_aware_resampling', 'animal_aware_cluster_jaccard', 'uniform_source_cluster_stability', 'leave_one_animal_out']:
        table = result['tables'][name].copy()
        table.insert(0, 'check', name)
        stacked.append(table)
    pd.concat(stacked, ignore_index=True).to_csv(output / 'stacked_resampling_and_holdout_diagnostics.csv', index=False)
    comparisons.to_csv(output / 'common_UID_k2_partition_comparisons.csv', index=False)
    result['tables']['cluster_source_animal_composition'].to_csv(output / 'display_cut_source_animal_composition.csv', index=False)
    if not result['profiles'].empty:
        top_targets = result['profiles'].sort_values(
            ['cluster', 'equal_contributing_animal_relative_profile', 'TargetID'],
            ascending=[True, False, True],
        ).groupby('cluster', sort=True).head(20)
        top_targets.to_csv(output / 'display_cut_top20_relative_targets.csv', index=False)
    targets.to_csv(output / 'official_ARM6_bilateral_feature_dictionary.csv', index_label='FeatureID')
    for path, expected in protected.items():
        verify_hash(path, expected)
    coverage = {}
    for field in ['EndBranchMappedFraction', 'EndBranchFractionOfWholeAxonTemplateLength',
                  'EndBranchFractionOfWholeAxonInReferenceLength']:
        assessed = ledger[field].dropna()
        coverage[field] = {
            'n_assessed': len(assessed), 'n_NA': int(ledger[field].isna().sum()),
            'median': float(assessed.median()) if len(assessed) else None,
            'q05': float(assessed.quantile(0.05)) if len(assessed) else None,
            'q95': float(assessed.quantile(0.95)) if len(assessed) else None,
        }
    report = {'status': 'software_verified_descriptive_end_branch_profile_sensitivity',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'input_bindings': protected,
        'producer_source_code_sha256': run['source_code_sha256'], 'analysis_source_code': [bind(path) for path in codes],
        'enriched_input_adapter': 'Producer ledger is original manifest or exact completed export summary; exact ordered source identities/hash/coordinates/ARM/evidence fields independently matched to original manifest.',
        'selected_neurons': len(ledger), 'graph_computable_neurons': int(ledger.EndBranchComputable.sum()),
        'mapped_positive_profile_neurons': len(raw), 'profile_exclusion_reasons': dict(Counter(ledger.EndBranchProfileStatus)),
        'absolute_end_branch_total_length_mm': float(ledger.EndBranchLengthAllStatusesMm.sum()),
        'absolute_end_branch_mapped_L6_length_mm': float(ledger.EndBranchLengthMappedL6Mm.sum()),
        'candidate_original_axon_ends': int(ledger.candidate_axon_endpoint_count.sum()),
        'selected_original_axon_edges': int(ledger.selected_axon_edge_count.sum()),
        'coverage_summary': coverage,
        'official_bilateral_pairs': len(targets) // 2, 'analysis': result['summary'],
        'comparisons': comparisons.astype(object).where(pd.notna(comparisons), None).to_dict('records'),
        'parameters': {'k_values': list(range(2, 9)), 'repeats': 100, 'fraction': 0.8, 'seed': 42,
            'representation': 'sqrt(relative mapped positive actual ARM6 end-branch template length), Hellinger distance, average linkage',
            'relative_sides': 'Official same-domain bilateral pair permutation using each current source hemisphere',
            'resampling': 'Uniform source subsets, within-animal stratified subsets, equal-animal cap, remaining-cohort animal holdout; fitted partition remains neuron-weighted'},
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn']},
        'python': sys.version, 'self_checks': self_check(),
        'limitations': ['Graph end-branches are not accepted biological terminal arbors, boutons or synapses.',
            'Unfinished unbranched shafts can qualify; reconstruction coverage can change profiles.',
            'Reference-template length is not calibrated native tissue length.',
            'Source-relative transformation uses pinned source side; export origin/registration acceptance remains unresolved.',
            'Conditional mapped-target proportions exclude background/outside from clustering, with coverage retained in the ledger.',
            'Partition comparisons use intersecting eligible neurons and do not establish biological classes or prediction.',
            'All Henry visual cases arise from one animal; no separate Henry-only clustering duplicated here.'],
        'anatomical_or_biological_acceptance': False, 'new_cohort_or_maps': False,
        'output_hashes': {path.name: digest(path) for path in output.iterdir() if path.is_file()}}
    with (output / 'profile_sensitivity_provenance.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ['status', 'selected_neurons', 'graph_computable_neurons',
          'mapped_positive_profile_neurons', 'profile_exclusion_reasons', 'comparisons']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--prior-relative', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    elif all([args.run, args.review, args.prior_relative, args.output]):
        analyse(args)
    else:
        parser.error('Use --self-check or all of --run --review --prior-relative --output')
