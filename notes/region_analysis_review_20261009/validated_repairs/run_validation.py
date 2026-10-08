"""Run the complete suite and independent contracts against staged or live code."""
from pathlib import Path
import contextlib
import importlib.util
import json
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "notes/region_analysis_review_20261009"
LIVE = "--live" in sys.argv
MODULES = ROOT / "main_scripts" if LIVE else AUDIT / "validated_repairs/staged/main_scripts"
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(MODULES))
os.environ["PROJECTOME_CONSUMER_MODULE_ROOT"] = str(MODULES)
os.environ["PROJECTOME_AUDIT_MODULE_ROOT"] = str(MODULES)
os.environ["PYTHONPATH"] = os.pathsep.join((str(MODULES), str(ROOT / "main_scripts")))

# Pin the modules before tests that independently adjust their import paths.
import region_analysis.population
import region_analysis.laterality_projection_analysis
import neuro_tracer


def load_file(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def main():
    bindings = {name: str(sys.modules[name].__file__) for name in
                ("region_analysis.population", "region_analysis.utils",
                 "region_analysis.hierarchy_table", "region_analysis.plotting", "neuro_tracer")}
    assert all(Path(path).is_relative_to(MODULES) for path in bindings.values()), bindings
    suites = [unittest.defaultTestLoader.discover(str(ROOT / "tests"))]
    for relative in ("code_audit/test_hierarchy_and_plots.py", "code_audit/test_standalone_identity.py",
                     "methods_audit/test_final_staged_consumers_independent.py",
                     "code_audit/test_harmonizer_preservation.py"):
        suites.append(load_file(AUDIT / relative))
    output = AUDIT / "validated_repairs" / ("live_suite.log" if LIVE else "staged_suite.log")
    with output.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(suites))
    receipt = {"status": "pass" if result.wasSuccessful() else "fail", "live": LIVE,
               "tests_run": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
               "skipped": len(result.skipped),
               "module_bindings": bindings, "log": str(output.relative_to(ROOT)),
               "scientific_acceptance": False}
    (output.with_suffix(".json")).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
