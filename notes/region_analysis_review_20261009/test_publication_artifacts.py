"""Focused selection checks for immutable CM033 fit evidence and local caches."""
import unittest

from prepare_publication import is_cm033_earlier_display, is_cm033_work_artifact


class PublicationArtifacts(unittest.TestCase):
    def test_both_candidate_workflows_keep_disposable_files_local(self):
        for name in ("cm033_bridge_candidate_20261009", "cm033_samecontrast_bridge_candidate_20261009"):
            for folder in ("nipype_work", "nipype_config", "mpl_config", "logs", "tmp"):
                path = f"notes/region_analysis_review_20261009/{name}/{folder}/nested/result.json"
                self.assertTrue(is_cm033_work_artifact(path))

    def test_named_evidence_and_transform_files_are_retained(self):
        prefix = "notes/region_analysis_review_20261009/cm033_bridge_candidate_20261009/"
        for name in ("README.md", "run_projectome_cm033_bridge.py", "rejected_candidate_review.json",
                     "fit_execution_log.txt", "fit/desc-epiToAnatAffine.aff12.1D",
                     "coarse_qc/readable/cm033_cachedSPM_mean_on_taskA_candidate.json"):
            self.assertFalse(is_cm033_work_artifact(prefix + name))
            self.assertFalse(is_cm033_earlier_display(prefix + name))

    def test_earlier_png_and_json_display_variants_remain_local(self):
        prefix = "notes/region_analysis_review_20261009/cm033_bridge_candidate_20261009/coarse_qc/"
        for suffix in ("png", "json"):
            self.assertTrue(is_cm033_earlier_display(prefix + "comparison." + suffix))
            self.assertFalse(is_cm033_earlier_display(prefix + "readable/comparison." + suffix))

    def test_scope_does_not_filter_unrelated_reports(self):
        for path in ("notes/another_review/cm033_fit/nipype_work/record.json",
                     "notes/region_analysis_review_20261009/clustering_20261009/logs/result.json",
                     "notes/region_analysis_review_20261009/cm033_followup_20261009/logs_summary.json"):
            self.assertFalse(is_cm033_work_artifact(path))

    def test_current_samecontrast_figures_are_primary_evidence(self):
        prefix = "notes/region_analysis_review_20261009/cm033_samecontrast_bridge_candidate_20261009/coarse_qc/"
        for name in ("cm033_forward_mean_intensity_correspondence", "cm033_inverse_mean_intensity_correspondence"):
            for suffix in ("png", "json"):
                self.assertFalse(is_cm033_earlier_display(prefix + name + "." + suffix))
                self.assertFalse(is_cm033_work_artifact(prefix + name + "." + suffix))


if __name__ == "__main__":
    unittest.main()
