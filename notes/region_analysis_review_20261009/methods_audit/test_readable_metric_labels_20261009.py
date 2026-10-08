"""Formula and static invariants for readable labels; no analyses run."""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'proposed_readable_metric_labels_20261009'
RECEIPT=json.loads((HERE/'readable_metric_labels_staging_receipt_20261009.json').read_text())
tree=ast.parse((HERE/'prepare_label_corrections_patch_v2.py').read_text())
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
namespace={'re':re}
exec(compile(ast.Module(body=functions,type_ignores=[]),'metric-label-validator','exec'),namespace)


class ReadableMetricLabels(unittest.TestCase):
    def test_all_14_sources_preserve_numeric_code_and_filenames(self):
        self.assertEqual(len(RECEIPT['files']),14)
        for item in RECEIPT['files']:
            with self.subTest(path=item['path']):
                before_path=ROOT/item['base_path']; after_path=OUT/item['path']
                self.assertEqual(hashlib.sha256(before_path.read_bytes()).hexdigest(),item['base_sha256'])
                self.assertEqual(hashlib.sha256(after_path.read_bytes()).hexdigest(),item['proposed_sha256'])
                before=before_path.read_text(encoding='utf-8-sig'); after=after_path.read_text(encoding='utf-8')
                rmd=item['path'].lower().endswith('.rmd')
                self.assertEqual(namespace['nontext_code'](before,rmd),namespace['nontext_code'](after,rmd))
                pattern=r'[A-Za-z0-9_/.-]+\.(?:csv|xlsx|png|pdf|rds)'
                self.assertEqual(Counter(re.findall(pattern,before)),Counter(re.findall(pattern,after)))
                for identifier in ('"IDD5_plus_IDM_balanced"','"IDD5_balanced"','"IDM_balanced"'):
                    self.assertEqual(before.count(identifier),after.count(identifier))

    def test_signs_and_ranges_are_distinct_exact_arithmetic_oracle(self):
        # I=3,C=1: contra fraction=1/4; signed contra-minus-ipsi=-1/2.
        ipsi,contra=Fraction(3),Fraction(1)
        share=contra/(ipsi+contra)
        ibias=(contra-ipsi)/(contra+ipsi)
        self.assertEqual(share,Fraction(1,4))
        self.assertEqual(ibias,Fraction(-1,2))
        self.assertEqual(ibias,2*share-1)
        self.assertEqual((ipsi-contra)/(ipsi+contra),-ibias)
        # Source-group contrast uses source means, not ipsi/contra totals.
        left,right=Fraction(4,5),Fraction(1,5)
        self.assertEqual((left-right)/(left+right),Fraction(3,5))
        self.assertGreater(left-right,0)

    def test_active_plot_balance_labels_and_formulas(self):
        for suffix in ('R','Rmd'):
            text=(OUT/f'group_analysis/R_analysis/v2_combined_primary_pipeline.{suffix}').read_text(encoding='utf-8')
            self.assertIn('Retained length balance (Ibias)\\n-1 ipsi; +1 contra',text)
            self.assertIn('Left vs right SOURCE contrast',text)
            self.assertIn('positive for a',text)
            self.assertIn('higher LEFT-source neuron mean',text)
            self.assertIn('(total_contra - total_ipsi) / (total_contra + total_ipsi)',text)
            self.assertIn('(mean_L - mean_R) / (mean_L + mean_R + eps_li)',text)
            self.assertNotIn('if both zero we treat as bias = -1',text)

    def test_six_imported_summary_plots_have_correct_zero_one_contract(self):
        selected=[item for item in RECEIPT['files'] if item['path'].startswith('R_analysis/') and 'Novel_' not in item['path']]
        self.assertEqual(len(selected),6)
        for item in selected:
            text=(OUT/item['path']).read_text(encoding='utf-8')
            self.assertIn('Contra/(Ipsi+Contra): 0 ipsilateral, 1 contralateral',text)
            self.assertNotIn('1 = Purely Ipsilateral, -1 = Purely Contralateral',text)
            self.assertIn('Contralateral retained length share',text)

    def test_historical_strength_balance_keeps_opposite_sign_explicit(self):
        text=(OUT/'R_analysis/Novel_Projectome_Analysis_Tutorial.Rmd').read_text(encoding='utf-8')
        self.assertIn('opposite sign to Ibias',text)
        self.assertIn('log10(retained length + 1)',text)
        self.assertIn('Length_Asymmetry = (Total_Ipsi - Total_Contra)',text)
        self.assertIn('Log-scaled length balance (ipsi-contra)/(ipsi+contra+eps)',text)


if __name__=='__main__': unittest.main(verbosity=2)
