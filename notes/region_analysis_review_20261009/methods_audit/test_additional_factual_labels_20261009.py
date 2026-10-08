"""Static label-only follow-up checks; no R execution or production writes."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGED = HERE/'proposed_additional_factual_labels_20261009'
RECEIPT = json.loads((HERE/'additional_factual_labels_staging_receipt_20261009.json').read_text())
tree = ast.parse((HERE/'prepare_label_corrections_patch_v2.py').read_text())
functions = [node for node in tree.body if isinstance(node,ast.FunctionDef)]
namespace = {'re':re}
exec(compile(ast.Module(body=functions,type_ignores=[]),'label-validator','exec'),namespace)


class AdditionalFactualLabels(unittest.TestCase):
    def test_code_tokens_output_names_and_hashes_preserved(self):
        for item in RECEIPT['files']:
            with self.subTest(path=item['path']):
                before_path = ROOT/item['base_path']
                after_path = STAGED/item['path']
                self.assertEqual(hashlib.sha256(before_path.read_bytes()).hexdigest(),item['base_sha256'])
                self.assertEqual(hashlib.sha256(after_path.read_bytes()).hexdigest(),item['proposed_sha256'])
                before = before_path.read_text(encoding='utf-8')
                after = after_path.read_text(encoding='utf-8')
                if item['path'].endswith('.Rmd'):
                    self.assertEqual(namespace['nontext_code'](before,True),namespace['nontext_code'](after,True))
                    pattern = r'[A-Za-z0-9_/.-]+\.(?:csv|xlsx|png|pdf)'
                    self.assertEqual(Counter(re.findall(pattern,before)),Counter(re.findall(pattern,after)))

    def test_three_tutorials_link_demonstration_names_consistently(self):
        paths = [item['path'] for item in RECEIPT['files'] if item['path'].startswith('R_analysis/')]
        self.assertEqual(len(paths),3)
        for path in paths:
            text = (STAGED/path).read_text(encoding='utf-8')
            self.assertEqual(text.count('"IPSI_DEMO_COPY"'),4)
            self.assertNotIn('name = "CONTRA"',text)
            self.assertNotIn('decorate_row_title("CONTRA"',text)
            self.assertIn('h2 <- build_projection_ht(mat_strength_ipsi_t, "IPSI_DEMO_COPY", "Duplicated ipsilateral example; not contralateral data", order = 2)',text)
            self.assertIn('mat_strength_ipsi_t,  # 使用相同矩阵作为示例',text)

    def test_statistical_unit_and_relative_hemisphere_corrections(self):
        text = (STAGED/'notes/hemispheric_asymmetry_methods_comparison.md').read_text(encoding='utf-8')
        self.assertNotIn('every statistical unit is a brain region',text)
        self.assertNotIn('more right-lateralized',text)
        self.assertIn('tested observations are neuron values',text)
        self.assertIn('relatively more contralateral than ipsilateral',text)
        self.assertIn('one hypothesis per target region',text)
        self.assertIn('arbor-count laterality',text)

    def test_cohort_is_versioned_without_animal_count_inference(self):
        text = (STAGED/'group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd').read_text(encoding='utf-8')
        self.assertNotIn('4 macaques',text)
        self.assertIn('Historical 306-neuron cohort summary',text)
        self.assertIn('353 unique neurons across six SampleIDs',text)
        self.assertIn('not asserted to be independent',text)
        self.assertIn('Historical combined',text)
        self.assertFalse(RECEIPT['cohort_readback']['animal_count_asserted'])
        self.assertEqual(sum(RECEIPT['cohort_readback']['SampleIDs'].values()),353)


if __name__ == '__main__':
    unittest.main(verbosity=2)
