"""Focused regressions for exact-cohort projection profiles and missingness."""
import sys
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'group_analysis/scripts'))
import cluster_arm_projection_profiles as c

class ProfileTests(unittest.TestCase):

    def fixture(self):
        manifest = pd.DataFrame({
            'SampleID': ['A', 'B', 'C', 'D'],
            'NeuronID': ['001.swc'] * 4,
            'AnimalID': ['a', 'b', 'c', 'd'],
            'SWCSHA256': ['h1', 'h2', 'h3', 'h4'],
        })
        summary = manifest.copy()
        summary['NeuronUID'] = summary.SampleID + '::' + summary.NeuronID
        summary['EndpointEligible'] = ['True', 'False', 'True', 'True']
        targets = pd.DataFrame({
            'TargetID': ['L6_A', 'L6_B', 'L6_zero', 'L3_parent'],
            'Level': ['6', '6', '6', '3'],
            'ARMIndex': ['1', '501', '0', '10'],
            'OfficialFullName': ['CL_A', 'CR_B', 'Atlas background', 'CL_parent'],
            'Hemisphere': ['L', 'R', 'Unknown', 'L'],
            'TargetStatus': ['mapped', 'mapped', 'zero_unassigned', 'mapped'],
        })
        measures = pd.DataFrame(
            [
                ['A::001.swc', 'L6_A', '6', '10', '2', '1'],
                ['A::001.swc', 'L6_zero', '6', '2', '1', '1'],
                ['B::001.swc', 'L6_B', '6', '8', '', ''],
                ['C::001.swc', 'L6_zero', '6', '1', '3', '1'],
                ['D::001.swc', 'L6_B', '6', '5', '1', '1'],
                ['A::001.swc', 'L3_parent', '3', '100', '3', '1'],
            ],
            columns=[
                'NeuronUID',
                'TargetID',
                'Level',
                'AxonTemplateLengthMm',
                'CandidateEndpointCount',
                'EndpointPresence',
            ],
        )
        return (summary, targets, measures, manifest)

    def test_identity_and_exclusive_level6(self):
        (ledger, targets, axon, ends, _, _) = c.build_profiles(*self.fixture())
        self.assertEqual(
            list(ledger.index),
            ['A::001.swc', 'B::001.swc', 'C::001.swc', 'D::001.swc'],
        )
        self.assertEqual(list(axon.columns), ['L6_A', 'L6_B'])
        self.assertEqual(axon.loc['A::001.swc'].sum(), 10)
        self.assertAlmostEqual(ledger.loc['A::001.swc', 'AxonMappedFraction'], 10 / 12)

    def test_no_endpoint_vs_all_unassigned_distinct(self):
        (ledger, _, _, ends, _, _) = c.build_profiles(*self.fixture())
        self.assertTrue(ends.loc['B::001.swc'].isna().all())
        self.assertEqual(ends.loc['C::001.swc'].sum(), 0)
        self.assertFalse(ledger.loc['C::001.swc', 'EndpointProfileEligible'])
        self.assertIn('no_mapped_target', ledger.loc['C::001.swc', 'EndpointProfileStatus'])
        self.assertIn('only_unassigned_or_outside', ledger.loc['C::001.swc', 'AxonProfileStatus'])
        self.assertIn('no_eligible', ledger.loc['B::001.swc', 'EndpointProfileStatus'])

    def test_duplicate_exact_identity_and_wrong_hash_rejected(self):
        (s, t, m, f) = self.fixture()
        s = pd.concat([s, s.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate exact'):
            c.build_profiles(s, t, m, f)
        (s, t, m, f) = self.fixture()
        s.loc[0, 'SWCSHA256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'hash binding'):
            c.build_profiles(s, t, m, f)

    def test_composite_uid_and_source_channel_cannot_silently_join(self):
        (s, t, m, f) = self.fixture()
        s.loc[0, 'SampleID'] = 'A_ch2'
        with self.assertRaisesRegex(ValueError, 'NeuronUID'):
            c.build_profiles(s, t, m, f)

    def test_duplicate_target_rows_and_missing_eligibility_fail(self):
        (s, t, m, f) = self.fixture()
        m = pd.concat([m, m.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate neuron-target'):
            c.build_profiles(s, t, m, f)
        (s, t, m, f) = self.fixture()
        s.loc[0, 'EndpointEligible'] = ''
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            c.build_profiles(s, t, m, f)

    def test_no_eligible_endpoint_zero_not_accepted(self):
        (s, t, m, f) = self.fixture()
        m.loc[2, 'CandidateEndpointCount'] = '0'
        m.loc[2, 'EndpointPresence'] = '0'
        with self.assertRaisesRegex(ValueError, 'must be NA'):
            c.build_profiles(s, t, m, f)

    def test_negative_length_count_presence_and_level_rejected(self):
        for (
            col,
            value,
        ) in [('AxonTemplateLengthMm', '-1'), ('CandidateEndpointCount', '2.5'), ('EndpointPresence', '0'), ('Level', '3')]:
            (s, t, m, f) = self.fixture()
            m.loc[0, col] = value
            with self.assertRaises(ValueError):
                c.build_profiles(s, t, m, f)

    def test_hellinger_scale_invariant_bounded_and_exact(self):
        a = np.array([[1.0, 0], [0, 2.0], [4, 4.0]])
        (_, d, p) = c.profile_distance(a)
        (_, scaled, _) = c.profile_distance(a * np.array([100, 2, 0.3])[:, None])
        np.testing.assert_allclose(d, scaled)
        self.assertAlmostEqual(d[0, 1], 1)
        self.assertAlmostEqual(d[0, 2], np.linalg.norm(np.sqrt(p[0]) - np.sqrt(p[2])) / np.sqrt(2))

    def test_zero_missing_profiles_and_unjustified_ward_rejected(self):
        for a in [np.array([[0.0, 0], [1, 0], [0, 1]]), np.array([[np.nan, 0], [1, 0], [0, 1]])]:
            with self.assertRaises(ValueError):
                c.profile_distance(a)
        with self.assertRaisesRegex(ValueError, 'Ward'):
            c.profile_distance(np.eye(3), method='ward')

    def test_presence_jaccard_and_logrelative_finite(self):
        a = np.array([[2.0, 0], [3, 1], [0, 1]])
        (_, d, _) = c.profile_distance(a, 'presence_jaccard')
        self.assertAlmostEqual(d[0, 1], 0.5)
        (_, d, _) = c.profile_distance(a, 'logrelative')
        self.assertTrue(np.isfinite(d).all())

    def test_animal_stratified_and_equal_cap_reproducible(self):
        groups = np.array(['large'] * 12 + ['small'] * 4)
        a = c.stratified_subsets(groups, repeats=4, seed=3)
        b = c.stratified_subsets(groups, repeats=4, seed=3)
        for (x, y) in zip(a, b):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(len(set(x)), len(x))
            self.assertEqual(sum(groups[x] == 'small'), 3)
        equal = c.stratified_subsets(groups, repeats=4, seed=3, balanced=True)
        for x in equal:
            self.assertEqual(sum(groups[x] == 'large'), sum(groups[x] == 'small'))

    def test_singletons_not_claimed_stable_in_animal_resampling(self):
        (
            _,
            d,
            _,
        ) = c.profile_distance(np.array([[10, 0], [9, 1], [8, 2], [0, 10], [1, 9], [2, 8.0]]))
        (
            r,
            j,
        ) = c.conditional_resampling(d, ['a', 'a', 'a', 'b', 'b', 'b'], {5: np.arange(6)}, 3, 0.8, 42)
        self.assertTrue((r.evaluated_repeats == 0).all())
        self.assertTrue(j.jaccard_mean.isna().all())

    def test_saved_analysis_preserves_identity_and_one_animal_limit(self):
        ids = pd.Index(['S::%03d.swc' % i for i in range(12)])
        raw = pd.DataFrame(
            np.r_[np.tile([[9.0, 1]], (6, 1)), np.tile([[1.0, 9]], (6, 1))],
            index=ids,
            columns=['a', 'b'],
        )
        meta = pd.DataFrame(
            {
                'AnimalID': ['animal'] * 12,
                'SampleID': ['S'] * 12,
                'Hemisphere': ['L'] * 12,
                'EvidenceSourceGroup': ['candidate'] * 12,
                'ARMFullName': ['Atlas background'] * 12,
                'AxonMappedFraction': [0.8] * 12,
            },
            index=ids,
        )
        targets = pd.DataFrame(
            {
                'ARMIndex': [1, 501],
                'OfficialFullName': ['CL_target', 'CR_target'],
                'Hemisphere': ['L', 'R'],
            },
            index=['a', 'b'],
        )
        parent = ROOT / 'notes/region_analysis_review_20261009/clustering_20261009'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            result = c.run_analysis(
                raw,
                meta,
                targets,
                Path(tmp) / 'analysis',
                [2],
                3,
                0.8,
                42,
                'hellinger',
                'Henry_only',
            )
            saved = pd.read_csv(
                Path(tmp) / 'analysis/candidate_cluster_assignments.csv',
                dtype={'NeuronUID': str},
            )
            self.assertEqual(saved.NeuronUID.tolist(), ids.tolist())
            self.assertIn('one_animal_only', result['summary']['animal_replication_status'])
            held = pd.read_csv(Path(tmp) / 'analysis/leave_one_animal_out.csv')
            self.assertTrue((held.status == 'insufficient_remaining_neurons').all())
            self.assertTrue((Path(tmp) / 'analysis/animal_aware_resampling.csv').exists())

    def test_outside_index_absent_only_with_explicit_status(self):
        targets = pd.DataFrame({
            'TargetID': ['mapped', 'outside'], 'Level': [6, 6],
            'ARMIndex': ['1', ''], 'TargetStatus': ['mapped', 'out_of_FOV'],
        })
        result = c.prepare_target_catalog(targets)
        self.assertEqual(result.loc[1, 'ARMIndex'], -1)
        targets.loc[1, 'TargetStatus'] = 'mapped'
        with self.assertRaisesRegex(ValueError, 'Only explicit'):
            c.prepare_target_catalog(targets)
        summary, targets, measures, manifest = self.fixture()
        targets.loc[0, 'TargetStatus'] = 'key_level_conflict'
        with self.assertRaisesRegex(ValueError, 'Positive level6'):
            c.build_profiles(summary, targets, measures, manifest)

    def test_export_receipt_requires_completed_named_bindings(self):
        parent = ROOT / 'notes/region_analysis_review_20261009/clustering_20261009'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            directory = Path(tmp)
            names = ['neuron_summary.csv', 'targets.csv', 'per_neuron_regional_measures.csv']
            manifest = directory / 'manifest.csv'
            manifest.write_text('bound manifest')
            for name in names:
                (directory / name).write_text(name)
            record = {
                'status': 'failed', 'manifest_sha256': c.digest(manifest),
                'artifacts': {name: c.digest(directory / name) for name in names},
            }
            receipt = directory / 'export_provenance.json'
            receipt.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'not completed'):
                c.validate_completed_export(directory, receipt, manifest)
            record['status'] = 'software_verified_descriptive_tables'
            receipt.write_text(json.dumps(record))
            c.validate_completed_export(directory, receipt, manifest)
            (directory / 'targets.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'artifact hash'):
                c.validate_completed_export(directory, receipt, manifest)

    def test_source_relative_uses_exact_official_bilateral_pairs(self):
        targets = pd.DataFrame({
            'Domain': ['Cortex', 'Cortex', 'Subcortex', 'Subcortex'],
            'Abbreviation': ['CL_Pi', 'CR_Pi', 'SL_Pi', 'SR_Pi'],
            'OfficialFullName': ['CL_parainsula', 'CR_parainsula',
                                 'SL_pineal_gland', 'SR_pineal_gland'],
            'Hemisphere': ['L', 'R', 'L', 'R'],
            'ARMIndex': [1, 501, 2, 502],
        }, index=['cL', 'cR', 'sL', 'sR'])
        raw = pd.DataFrame([[1., 2., 3., 4.], [10., 20., 30., 40.]],
                           index=['S::001', 'S::002'], columns=targets.index)
        meta = pd.DataFrame({'Hemisphere': ['L', 'R']}, index=raw.index)
        result, mapping = c.source_relative_profiles(raw, meta, targets)
        np.testing.assert_allclose(result.sum(axis=1), raw.sum(axis=1))
        cortex_ipsi = mapping[(mapping.Domain == 'Cortex') &
                             (mapping.RelativeSide == 'ipsilateral')].index[0]
        np.testing.assert_allclose(result[cortex_ipsi], [1., 20.])
        self.assertEqual(len(mapping), 4)
        self.assertEqual(set(mapping.Domain), {'Cortex', 'Subcortex'})
        with self.assertRaisesRegex(ValueError, 'Missing official'):
            c.source_relative_profiles(raw.iloc[:, :-1], meta, targets.iloc[:-1])
        bad = targets.copy()
        bad.loc['cR', 'OfficialFullName'] = 'CR_different_target'
        with self.assertRaisesRegex(ValueError, 'Missing official'):
            c.source_relative_profiles(raw, meta, bad)
        bad = targets.copy()
        bad.loc['sL', 'Domain'] = 'Cortex'
        with self.assertRaisesRegex(ValueError, 'prefix mismatch'):
            c.source_relative_profiles(raw, meta, bad)
        with self.assertRaisesRegex(ValueError, 'Exact source UID'):
            c.source_relative_profiles(raw, meta.iloc[::-1], targets)
        bad = meta.copy()
        bad.loc['S::002', 'Hemisphere'] = 'Unknown'
        with self.assertRaisesRegex(ValueError, 'known current source'):
            c.source_relative_profiles(raw, bad, targets)
        duplicate = pd.concat([targets, targets.loc[['cL']].rename(index={'cL': 'extra'})])
        with self.assertRaisesRegex(ValueError, 'Nonunique official'):
            c.source_relative_profiles(raw.reindex(columns=duplicate.index), meta, duplicate)

    def test_display_cut_does_not_accept_unbalanced_partition(self):
        d = pd.DataFrame({
            'requested_k': [2, 3],
            'realized_k': [2, 3],
            'smallest_cluster': [1, 5],
            'largest_cluster_fraction': [0.99, 0.8],
            'silhouette': [0.9, 0.2],
        })
        (k, status) = c.cut_choice(d)
        self.assertEqual(k, 3)
        self.assertEqual(status, 'balanced_size_diagnostic_cut')
if __name__ == '__main__':
    unittest.main()
