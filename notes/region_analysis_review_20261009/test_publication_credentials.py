"""Token-scanner regressions use generated fixtures, never real credentials."""
import re
import unittest
from prepare_publication import ACCESS_TOKEN_PATTERN
from verify_publication_index import INDEX_CREDENTIAL_PATTERN


class PublicationCredentialBoundaries(unittest.TestCase):
    def setUp(self):
        self.text_pattern = re.compile(ACCESS_TOKEN_PATTERN)

    def test_bids_task_paths_are_not_tokens(self):
        for task in ("opto", "mstim"):
            path = ("D:/data/sub-cm033_ses-20160120_task-" + task +
                    "_acq-leftaic_run-39_desc-aligned_bold.nii")
            self.assertIsNone(self.text_pattern.search(path))
            self.assertIsNone(INDEX_CREDENTIAL_PATTERN.search(path.encode("ascii")))

    def test_standalone_supported_token_prefixes_remain_detected(self):
        for prefix in ("sk-", "sk-proj-", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"):
            fixture = prefix + "a" * 40
            for wrapper in ('"{}"', "key={}", "({})", "\n{}\n"):
                value = wrapper.format(fixture)
                self.assertIsNotNone(self.text_pattern.search(value))
                self.assertIsNotNone(INDEX_CREDENTIAL_PATTERN.search(value.encode("ascii")))

    def test_minimum_lengths_are_retained(self):
        for prefix, length in (("sk-", 25), ("ghp_", 24), ("github_pat_", 30)):
            self.assertIsNone(self.text_pattern.search(prefix + "a" * (length - 1)))
            self.assertIsNotNone(self.text_pattern.search(prefix + "a" * length))

    def test_index_private_key_detection_remains_enabled(self):
        value = ("-----BEGIN " + "PRIVATE KEY" + "-----").encode("ascii")
        self.assertIsNotNone(INDEX_CREDENTIAL_PATTERN.search(value))


if __name__ == "__main__":
    unittest.main()
