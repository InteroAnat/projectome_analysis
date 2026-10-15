"""Offline checks of real source failures and rejection of altered evidence."""
import copy
import json
from datetime import datetime, timezone

from verify_sample_regional_milestone_20261004 import HERE, OUT, check_unavailable_root, digest, read_json


def main():
    cases, hashes = [], {}
    for stem in ("145", "169", "161"):
        path = OUT / "manifest" / f"derived_context_pilot_252718_{stem}.json"
        raw = OUT / "swc_raw/252718" / f"{stem}.swc"
        report = read_json(path)
        hashes[str(path)] = digest(path)
        hashes[str(raw)] = digest(raw)
        check_unavailable_root(report, raw)
        rejected = []
        changes = {
            "borrowed_root_coordinates": ("native_soma_xyz_um", [0, 0, 0]),
            "wrong_root_cube": ("central_cube_xyz", [0, 0, 0]),
            "false_root_coverage": ("root_pixel_covered", True),
            "acquired_pixels": ("coverage_fraction", 0.1),
            "missing_source_error": ("source_errors", []),
            "wrong_error_class": ("source_errors", [{"cube_xyz": report["central_cube_xyz"], "reason": "timeout"}]),
            "omitted_unattempted_tile": ("unattempted", report["unattempted"][:-1]),
            "duplicate_requested_tile": ("requested", report["requested"] + report["requested"][:1]),
        }
        for name, (key, value) in changes.items():
            altered = copy.deepcopy(report)
            altered[key] = value
            try:
                check_unavailable_root(altered, raw)
            except AssertionError:
                rejected.append(name)
            else:
                raise AssertionError(f"Altered evidence accepted: {stem} {name}")
        cases.append({"uid": report["UID"], "actual_failure_validated": True,
                      "altered_evidence_rejected": rejected})
    assert all(digest(path) == expected for path, expected in hashes.items())
    result = {"status": "passed", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "real_failure_cases": len(cases), "altered_evidence_rejections": 24,
              "network_requests": 0, "image_validation_passed": False,
              "verifier_sha256": digest(HERE / "verify_sample_regional_milestone_20261004.py"),
              "input_hashes": hashes, "cases": cases}
    target = HERE / "unavailable_root_milestone_diagnostic_20261004.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: value for key, value in result.items() if key not in {"input_hashes", "cases"}}))


if __name__ == "__main__":
    main()
