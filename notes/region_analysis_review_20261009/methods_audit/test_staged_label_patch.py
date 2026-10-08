"""Regression checks for the unapplied factual-label patch; no R execution."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGED = HERE / "proposed_label_corrections_v2"
REPORT = json.loads((HERE / "proposed_label_patch_validation_v2.json").read_text())
tree = ast.parse((HERE / "prepare_label_corrections_patch_v2.py").read_text())
namespace = {"re": re}
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(ast.Module(body=functions, type_ignores=[]), "staged_validator", "exec"), namespace)


class LabelPatchTests(unittest.TestCase):
    def test_all_nontext_R_tokens_and_output_references_are_preserved(self):
        for item in REPORT["changed_files"]:
            if not item["path"].endswith((".R", ".Rmd")):
                continue
            with self.subTest(path=item["path"]):
                original = (ROOT / item["path"]).read_text(encoding="utf-8").replace("\r\n", "\n")
                proposed = (STAGED / item["path"]).read_text(encoding="utf-8").replace("\r\n", "\n")
                is_rmd = item["path"].endswith(".Rmd")
                self.assertEqual(namespace["nontext_code"](original, is_rmd),
                                 namespace["nontext_code"](proposed, is_rmd))
                outputs = r"[A-Za-z0-9_/.-]+\.(?:csv|xlsx|png)"
                self.assertEqual(Counter(re.findall(outputs, original)), Counter(re.findall(outputs, proposed)))
                for name in ("IDD5_plus_IDM_balanced", "IDD5_balanced", "IDM_balanced"):
                    self.assertEqual(original.count('"' + name + '"'), proposed.count('"' + name + '"'))

    def test_primary_spec_labels_corrected_without_relabeling_pvalues(self):
        for name in ("v2_combined_primary_pipeline.R", "v2_combined_primary_pipeline.Rmd"):
            text = (STAGED / "group_analysis/R_analysis" / name).read_text(encoding="utf-8")
            self.assertNotIn("with permutation p", text)
            self.assertIn("ordinary t-test p", text)
            self.assertIn("normalized ipsi log-strength share", text)
            self.assertIn("not animal balancing", text)
            self.assertIn('p     <- co[2, "Pr(>|t|)"]', text)

    def test_living_report_warning_is_append_only(self):
        relative = "notes/LR_insula_analysis_review.md"
        original = (ROOT / relative).read_bytes()
        proposed = (STAGED / relative).read_bytes()
        self.assertTrue(proposed.startswith(original))
        warning = proposed[len(original):].decode("utf-8")
        self.assertIn("normalized", warning)
        self.assertIn("not an axonal-length budget", warning)
        self.assertIn("collision-safe C_/S_ exports", warning)

    def test_patch_is_bound_and_no_historical_outputs_are_changed(self):
        patch = ROOT / REPORT["patch_path"]
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), REPORT["patch_sha256"])
        self.assertEqual(len(REPORT["changed_files"]), 8)
        self.assertFalse(REPORT["numeric_outputs_regenerated"])
        for item in REPORT["changed_files"]:
            self.assertNotIn("/outputs/", item["path"])
            self.assertEqual(hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest(), item["original_sha256"])
            self.assertEqual(hashlib.sha256((STAGED / item["path"]).read_bytes()).hexdigest(), item["proposed_sha256"])


if __name__ == "__main__":
    unittest.main()
