"""Run the existing live-suite harness into a fresh end-branch receipt."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import json
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
sys.argv.append("--live")
spec = importlib.util.spec_from_file_location("end_branch_live_harness", ROOT / "notes/region_analysis_review_20261009/validated_repairs/run_validation.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT)
    parser.add_argument("--live", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    log_path = args.output_directory / "live_suite_20261009.log"
    receipt_path = args.output_directory / "live_suite_20261009.json"
    if log_path.exists() or receipt_path.exists():
        raise FileExistsError("Preserve earlier test receipts; choose a fresh validation destination")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    test_files = list((ROOT / "tests").glob("*.py"))
    consumer_files = [harness.AUDIT / relative for relative in (
        "code_audit/test_hierarchy_and_plots.py", "code_audit/test_standalone_identity.py",
        "methods_audit/test_final_staged_consumers_independent.py", "code_audit/test_harmonizer_preservation.py")]
    sources = sorted(set(test_files + consumer_files + list(harness.MODULES.rglob("*.py")) + [Path(__file__)]))
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    suites = [unittest.defaultTestLoader.discover(str(ROOT / "tests"))]
    for relative in ("code_audit/test_hierarchy_and_plots.py", "code_audit/test_standalone_identity.py",
                     "methods_audit/test_final_staged_consumers_independent.py", "code_audit/test_harmonizer_preservation.py"):
        suites.append(harness.load_file(harness.AUDIT / relative))
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(suites))
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    if before != after:
        raise ValueError("Scientific/test source bytes changed during validation; preserve log and revalidate")
    receipt = {"status": "passed" if result.wasSuccessful() else "failed", "tests_run": result.testsRun,
               "errors": len(result.errors), "failures": len(result.failures), "skipped": len(result.skipped),
               "live_module_root": str(harness.MODULES), "scientific_acceptance": False,
               "runtime_seconds": time.perf_counter() - started,
               "source_hashes_before_and_after_equal": True, "source_sha256": after}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "source_sha256"}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
