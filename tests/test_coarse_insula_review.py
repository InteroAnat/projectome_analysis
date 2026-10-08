import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location('coarse_review', Path(__file__).resolve().parents[1] / 'group_analysis/scripts/build_coarse_insula_review.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class CoarseReviewTests(unittest.TestCase):
    def test_segmentation_reference_geometry_mismatch_raises(self):
        reference = m.nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), np.eye(4))
        matching = m.nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), np.eye(4))
        m.validate_tissue_reference_geometry(matching, reference)
        shifted = np.eye(4)
        shifted[0, 3] = .25
        for mismatch in (m.nib.Nifti1Image(np.zeros((2, 3, 5), dtype=np.uint8), np.eye(4)),
                         m.nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), shifted)):
            with self.subTest(shape=mismatch.shape, affine=mismatch.affine.tolist()):
                with self.assertRaisesRegex(ValueError, 'identical shape and affine'):
                    m.validate_tissue_reference_geometry(mismatch, reference)

    @staticmethod
    def anchor_row(sample, nid, xyz, source_ins=False, human_ins=False, alternate_ins=False):
        return {'sample': sample, 'neuron_id': nid, 'portal_coordinate_index_xyz': __import__('json').dumps(xyz) if xyz is not None else '',
                'atlas_INS': str(source_ins), 'Henry_coarse_INS_visual_evidence': str(human_ins),
                'original_manual_rows': '[]', 'excluded': 'False',
                'portal_coordinate_published_ceil_edge_one_based_to_zero_insula_candidate': str(alternate_ins)}

    def test_nearest_other_excludes_self_even_when_established_INS(self):
        row = self.anchor_row('A', '001.swc', [0, 0, 0], source_ins=True)
        other = self.anchor_row('A', '002.swc', [4, 0, 0], human_ins=True)
        anchors = m.established_anchors([row, other], set())
        distance, nid, uid, basis = m.nearest_other_anchor(row, anchors)
        self.assertEqual((distance, nid, uid), (1.0, '002.swc', 'A|002.swc'))
        self.assertEqual(basis, ['Henry_coarse_INS_visual_annotation'])
        self.assertEqual(m.nearest_other_anchor(row, anchors[:1]), (None, '', '', []))

    def test_alternative_origin_candidate_not_anchor_or_numeric_neighbor_basis(self):
        query = self.anchor_row('A', '001.swc', [0, 0, 0], alternate_ins=True)
        alternative = self.anchor_row('A', '002.swc', [0.1, 0, 0], alternate_ins=True)
        source = self.anchor_row('A', '010.swc', [8, 0, 0], source_ins=True)
        anchors = m.established_anchors([query, alternative, source], set())
        self.assertEqual([x['neuron_id'] for x in anchors], ['010.swc'])
        result = m.corrected_priority_fields(query, anchors)
        self.assertEqual(result['nearest_same_exact_sample_INS_anchor_mm'], 2.0)
        self.assertEqual(result['nearest_INS_anchor_uid'], 'A|010.swc')
        self.assertEqual(result['nearest_INS_anchor_evidence_basis'], ['original_portal_INS_label'])
        self.assertEqual(result['neighboring_numeric_INS_ID_retrieval_cues'], [])
        self.assertTrue(result['alternative_policy_INS_sensitivity_candidate'])
        self.assertFalse(result['alternative_policy_candidate_used_as_established_anchor'])

    def test_anchor_missing_coordinates_and_exact_channel(self):
        query = self.anchor_row('A', '001.swc', [0, 0, 0])
        channel = self.anchor_row('Ach2', '002.swc', [0, 0, 0], source_ins=True)
        missing = self.anchor_row('A', '003.swc', None, source_ins=True)
        anchors = m.established_anchors([channel, missing], set())
        self.assertEqual(m.nearest_other_anchor(query, anchors), (None, '', '', []))
        result = m.corrected_priority_fields(query, anchors)
        self.assertEqual(result['neighboring_numeric_INS_ID_retrieval_cues'], ['003.swc'])
        self.assertFalse(result['ID_adjacency_anatomical_evidence'])

    def test_exact_identity_and_numeric_retrieval(self):
        self.assertNotEqual(m.identity({'sample': '233001', 'neuron_id': '001.swc'}), m.identity({'sample': '233001ch2', 'neuron_id': '001.swc'}))
        self.assertEqual(m.numeric_id('001.swc'), 1)
        self.assertIsNone(m.numeric_id('qroot1.swc'))

    def test_tissue_class_not_atlas_background(self):
        names = {1: 'CSF', 2: 'GM', 3: 'scGM', 4: 'WM', 5: 'BV'}
        self.assertEqual(m.tissue_flag(4, 4, names), 'persistent_WM')
        self.assertEqual(m.tissue_flag(2, 4, names), 'origin_sensitive_GM_WM_transition')
        self.assertEqual(m.tissue_flag(3, 3, names), 'persistent_scGM')
        self.assertEqual(m.tissue_flag(None, 4, names), 'missing_or_out_of_bounds_unassessed')

    def test_coordinate_lookup_absence_and_bounds(self):
        seg = np.ones((2, 2, 2), dtype=int) * 4
        self.assertEqual(m.tissue_at(seg, '', {4: 'WM'}), (None, 'missing_unassessed'))
        self.assertEqual(m.tissue_at(seg, '[-1, 0, 0]', {4: 'WM'}), (None, 'out_of_bounds'))
        self.assertEqual(m.tissue_at(seg, '[0, 0, 0]', {4: 'WM'}), (4, 'WM'))

    def test_human_state_separate_from_candidates(self):
        self.assertEqual(m.coarse_state(True, False, 'persistent_WM', True), 'reviewed_INS')
        self.assertEqual(m.coarse_state(False, False, 'persistent_WM', True), 'WM_or_boundary_uncertain')
        self.assertEqual(m.coarse_state(False, False, 'persistent_GM', True), 'candidate-only')
        self.assertEqual(m.coarse_state(False, False, 'missing_or_out_of_bounds_unassessed', False), 'unresolved')

    def test_serialization_graph_equal_but_coordinate_change_not_equal(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, c = (Path(d) / x for x in ('a.swc', 'b.swc', 'c.swc'))
            a.write_text('1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n')
            b.write_text('# reordered serialization\n2 2 1.0 0 0 1 1\n1 1 0 0 0 1 -1\n')
            c.write_text('1 1 0 0 0 1 -1\n2 2 1.01 0 0 1 1\n')
            self.assertTrue(m.graph_equivalent(m.graph(a), m.graph(b)))
            self.assertFalse(m.graph_equivalent(m.graph(a), m.graph(c)))

    def test_missing_parent_and_disconnected_cycle_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.swc'
            for content in ('1 1 0 0 0 1 -1\n2 2 1 0 0 1 99\n', '1 1 0 0 0 1 -1\n2 2 1 0 0 1 3\n3 2 2 0 0 1 2\n'):
                p.write_text(content)
                with self.assertRaises(ValueError):
                    m.graph(p)


if __name__ == '__main__':
    unittest.main()
