"""Resumable native regional fields with shared disk/request/runtime admission.

Copied contexts are preserved. Missing sources and resource stops remain explicit
per-UID results. A run budget does not redefine the selected review population.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from uuid import uuid4

import nibabel as nib
import numpy as np

import visual_review_20261002 as review
from fmost_derived_context import context_plan
from fmost_context_resources import BoundedCubeSource, atomic_json
from fmost_review_lock import exclusive_batch

PRIORITY = {"INS": 0, "G": 1, "PrCO": 2, "possible_insula": 3,
            "Unknown": 4, "metadata_unresolved": 5, "adjacent": 6}
# A thin 4 mm field can span two Z blocks while retaining the same canvas.
# Tiles are streamed one at a time; shared cache/request/free-space admission
# remains authoritative. These bounds admit at most 800 bounded responses.
FIELD_MAX_CUBES = 800
FIELD_MAX_SOURCE_BYTES = 26 * 1024**3


def requested_field(native, preferred=4000.0, fallback=2800.0):
    """Fallback changes declared FOV only; native calibration and stride stay fixed."""
    _, plan = context_plan(native, field_um=preferred, max_cubes=FIELD_MAX_CUBES,
                           max_source_bytes=FIELD_MAX_SOURCE_BYTES)
    if plan.get("status") != "not_run":
        return preferred, plan
    _, plan = context_plan(native, field_um=fallback, max_cubes=FIELD_MAX_CUBES,
                           max_source_bytes=FIELD_MAX_SOURCE_BYTES)
    plan["fallback_reason"] = "preferred field exceeded per-context resource admission"
    return fallback, plan


def selected_rows(out_root, samples=None):
    rows = review.load_manifest(out_root)
    counts = rows[rows["review_category"] == "INS"].groupby("SampleID").size()
    selected = list(samples or review.PENDING_SAMPLES)
    if any(sample not in review.NEW_SAMPLES for sample in selected):
        raise ValueError("Only selected review monkeys can be rendered")
    rows = rows[rows["SampleID"].isin(selected)].copy()
    rows["_priority"] = rows["review_category"].map(PRIORITY).fillna(7)
    rows["_sample_insula_count"] = rows["SampleID"].map(counts).fillna(0).astype(int)
    return rows.sort_values(["_priority", "_sample_insula_count", "SampleID", "NeuronID"],
                            ascending=[True, False, True, True], kind="stable")


def reusable_context(out_root, sample, neuron, minimum_field=4000.0):
    """Require source-grid/own-root provenance and all persisted intensity panels."""
    folder = out_root / sample / neuron.replace(".swc", "")
    report_path = out_root / "manifest" / f"derived_context_pilot_{sample}_{neuron.replace('.swc', '')}.json"
    names = ("context.nii.gz", "context.nii.gz.json", "context_unmarked.png",
             "context_marked.png", "context_trace.png", "context_stack.png", "context_ortho.png")
    if not report_path.is_file() or not all((folder / name).is_file() for name in names):
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        meta = json.loads((folder / "context.nii.gz.json").read_text(encoding="utf-8"))
        acquisition = meta["acquisition"]
        if report.get("status") == "derived_partial":
            failures = report.get("source_errors") or []
            if (report.get("unattempted") or report.get("invalid") or not failures
                    or any("HTTP Error 404" not in str(error.get("reason")) for error in failures)):
                return None  # Resource/transient stops must be retried on resume.
        field = report.get("requested_field_um_xyz") or []
        if len(field) != 3 or any(float(value) < minimum_field for value in field[:2]):
            return None
        if (report["status"] not in review.DERIVED_CONTEXT_STATES
                or not report["root_pixel_covered"] or not report["central_cube_loaded"]
                or acquisition["sample_id"] != sample
                or report["UID"] != f"{sample}|{neuron}"
                or not acquisition["sampling"].startswith("global source point decimation")):
            return None
        image = nib.load(str(folder / "context.nii.gz"))
        np.testing.assert_allclose(image.affine[:3, 3], report["origin_xyz_um"], atol=1e-3)
        np.testing.assert_allclose(np.diag(image.affine)[:3], report["derived_spacing_xyz_um"], atol=1e-6)
        if image.header.get_xyzt_units()[0] != "micron" or list(image.shape) != report["shape_zyx"][::-1]:
            return None
        np.testing.assert_allclose(np.array(report["source_first_index_xyz"]) * [.65, .65, 3],
                                   report["origin_xyz_um"], atol=1e-6)
        swc = out_root / "swc_raw" / sample / neuron
        if not swc.is_file() or review.sha256_file(swc) != report["swc_sha256"]:
            return None
        return report
    except (ValueError, KeyError, TypeError, OSError, AssertionError, nib.filebasedimages.ImageFileError):
        return None


def run(out_root=review.OUT_ROOT, samples=None, preferred_field=4000.0, fallback_field=2800.0,
        cache_bytes=300 * 1024**3, max_requests=60000, max_seconds=8 * 3600,
        min_free_bytes=50 * 1024**3, retry_missing=False, force=False):
    out_root = Path(out_root)
    manifest_path = out_root / "manifest" / "identity_manifest.csv"
    if not manifest_path.is_file():
        review.build_manifest(out_root)
    initial_hash = review.sha256_file(manifest_path)
    rows = selected_rows(out_root, samples)
    if rows["UID"].duplicated().any():
        raise ValueError("Identity manifest has duplicate selected UIDs")
    manifest_hash = review.sha256_file(manifest_path)
    if initial_hash != manifest_hash:
        raise RuntimeError("Identity manifest changed during producer selection")
    state_path = out_root / "manifest" / "regional_context_run_20261003.json"
    state = {"run_id": uuid4().hex, "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "state": "running", "manifest_sha256": manifest_hash,
             "preferred_field_um": preferred_field, "fallback_field_um": fallback_field,
             "priority_policy": "category, descending sample INS candidate count, sample, neuron",
             "sample_insula_candidate_counts": {
                 str(sample): int(count) for sample, count in rows.groupby("SampleID")["_sample_insula_count"].first().items()},
             "identities": {row["UID"]: {"state": "not_attempted", "category": row["review_category"]}
                            for row in rows.to_dict(orient="records")}}
    with exclusive_batch(out_root / "manifest" / "regional_context.lock"):
        if review.sha256_file(manifest_path) != manifest_hash:
            raise RuntimeError("Identity manifest changed before producer acquired ownership")
        review.guard_canonical_baseline(out_root)
        source = BoundedCubeSource(out_root, cache_bytes, max_requests, max_seconds,
                                   min_free_bytes, retry_missing=retry_missing)
        atomic_json(state_path, state)
        try:
            for row in rows.to_dict(orient="records"):
                uid, sample, neuron = row["UID"], str(row["SampleID"]), str(row["NeuronID"])
                if review.sha256_file(manifest_path) != manifest_hash:
                    state["stop_reason"] = "identity manifest changed during production"
                    break
                if source.runtime_expired():
                    state["stop_reason"] = "batch runtime budget"
                    break
                if shutil.disk_usage(out_root).free < source.min_free_bytes + 128 * 1024**2:
                    state["stop_reason"] = "output free-space margin"
                    break
                if sample in review.AVAILABLE_SAMPLES:
                    state["identities"][uid].update(state="copied_context_retained")
                    continue
                existing = None if force else reusable_context(out_root, sample, neuron, preferred_field)
                if retry_missing and existing and existing["status"] == "derived_partial":
                    existing = None
                if existing is not None:
                    folder = out_root / sample / neuron.replace(".swc", "")
                    review.write_derived_coverage(folder, existing, existing["native_soma_xyz_um"])
                    state["identities"][uid].update(state="reused_existing_derived", status=existing["status"])
                else:
                    try:
                        raw, _, _, _ = review.fetch_raw_swc(sample, neuron, str(row["project_id"]), out_root)
                        native = review._root_xyz(raw)
                        field, plan = requested_field(native, preferred_field, fallback_field)
                        if plan.get("status") == "not_run":
                            state["identities"][uid].update(state="resource_unavailable", reason=plan["reason"], plan=plan)
                            continue
                        remaining = max(1.0, max_seconds - (time.monotonic() - source.started))
                        report = review.pilot_derived_context(sample, neuron, out_root, field_um=field,
                                                             max_cubes=FIELD_MAX_CUBES,
                                                             max_source_bytes=FIELD_MAX_SOURCE_BYTES,
                                                             max_seconds=min(900.0, remaining), cube_download=source)
                        state["identities"][uid].update(state="attempt_finished", status=report["status"],
                                                       requested_field_um=field, coverage_fraction=report.get("coverage_fraction"),
                                                       reason=report.get("reason"), fallback_reason=plan.get("fallback_reason"),
                                                       source_errors=report.get("source_errors"),
                                                       missing_tiles=len(report.get("missing") or []),
                                                       unattempted_tiles=len(report.get("unattempted") or []))
                    except (OSError, ValueError, RuntimeError) as exc:
                        state["identities"][uid].update(state="source_or_render_error", reason=str(exc))
                state["resources"] = source.snapshot()
                atomic_json(state_path, state)
                print(f"[regional] {uid} {state['identities'][uid]['state']}", flush=True)
            state["state"] = "terminal_budget_stop" if state.get("stop_reason") else "terminal_scope_attempted"
        finally:
            if state["state"] == "running":
                state["state"] = "terminal_interrupted_or_error"
            state["resources"] = source.snapshot()
            state["ended_utc"] = datetime.now(timezone.utc).isoformat()
            terminal_path = out_root / "manifest" / "regional_runs" / (state["run_id"] + ".json")
            state["terminal_record"] = str(terminal_path)
            atomic_json(terminal_path, state)
            atomic_json(state_path, state)
            review.write_gallery(out_root)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="append")
    parser.add_argument("--out-root", type=Path, default=review.OUT_ROOT)
    parser.add_argument("--preferred-field-um", type=float, default=4000)
    parser.add_argument("--fallback-field-um", type=float, default=2800)
    parser.add_argument("--cache-gib", type=float, default=300)
    parser.add_argument("--min-free-gib", type=float, default=50)
    parser.add_argument("--max-requests", type=int, default=60000)
    parser.add_argument("--max-seconds", type=float, default=8 * 3600)
    parser.add_argument("--retry-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.out_root, args.sample, args.preferred_field_um, args.fallback_field_um,
        int(args.cache_gib * 1024**3), args.max_requests, args.max_seconds,
        int(args.min_free_gib * 1024**3), args.retry_missing, args.force)


if __name__ == "__main__":
    main()
