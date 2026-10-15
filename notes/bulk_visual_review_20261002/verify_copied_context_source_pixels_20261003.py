"""Compare every copied-context voxel to bounded reads of its own CH1 TIFF.

The real audit refuses to run before the regional producer has terminated.
Source fingerprints cover TIFF header bytes and every compressed segment read;
they are not whole-file hashes. Geometry remains nominal, not calibrated.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import time

import nibabel as nib
import numpy as np
import psutil
import tifffile

from verify_all_soma_pixels_20261003 import save, sha
from verify_copied_context_packages_20261003 import OUT, HERE, SAMPLES, SPACING

MAX_SEGMENT_BYTES = 64 * 1024**2
MAX_SECTION_BYTES = 256 * 1024**2


def fingerprint(path):
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_roi(path, x, y, width, height):
    """Read only intersecting lossless TIFF strips/tiles, with exact coverage."""
    path = Path(path)
    before = fingerprint(path)
    result = np.zeros((height, width), dtype=np.uint16)
    covered = np.zeros(result.shape, dtype=bool)
    segments = []
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 1, "Expected a single grayscale source section"
        page = tif.pages[0]
        assert page.ndim == 2 and page.dtype == np.dtype("uint16")
        assert page.samplesperpixel == 1 and int(page.compression) in {1, 5, 8, 32946}
        assert x >= 0 and y >= 0 and x + width <= page.imagewidth and y + height <= page.imagelength
        assert page.imagedepth == 1
        full_shape = [page.imagelength, page.imagewidth]
        if page.is_tiled:
            sx, sy = page.tilewidth, page.tilelength
            assert page.tiledepth == 1
        else:
            sx, sy = page.imagewidth, page.rowsperstrip
        assert sx * sy * 2 <= MAX_SEGMENT_BYTES, "Decoded segment exceeds memory bound"
        nx = (page.imagewidth + sx - 1) // sx
        ny = (page.imagelength + sy - 1) // sy
        assert len(page.dataoffsets) == len(page.databytecounts) == nx * ny
        selected = [iy * nx + ix for iy in range(y // sy, (y + height - 1) // sy + 1)
                    for ix in range(x // sx, (x + width - 1) // sx + 1)]
        counts = [page.databytecounts[i] for i in selected]
        assert max(counts) <= MAX_SEGMENT_BYTES and sum(counts) <= MAX_SECTION_BYTES
        first_offset = min(o for o in page.dataoffsets if o > 0)
        assert first_offset <= MAX_SEGMENT_BYTES
        fh = tif.filehandle
        fh.seek(0)
        header = fh.read(first_offset)
        assert len(header) == first_offset
        for index in sorted(selected, key=lambda i: page.dataoffsets[i]):
            offset, count = page.dataoffsets[index], page.databytecounts[index]
            assert offset > 0 and count > 0, "Sparse/empty TIFF segment cannot establish source coverage"
            fh.seek(offset)
            encoded = fh.read(count)
            assert len(encoded) == count, "Truncated compressed TIFF segment"
            decoded, position, shape = page.decode(encoded, index, _fullsize=page.is_tiled)
            assert decoded is not None and decoded.dtype == np.dtype("uint16")
            assert position[0] == position[1] == position[4] == 0
            assert shape[0] == shape[3] == 1
            expected_y, expected_x = (index // nx) * sy, (index % nx) * sx
            assert position[2:4] == (expected_y, expected_x)
            section = decoded[0, :, :, 0]
            y0, y1 = max(y, expected_y), min(y + height, expected_y + section.shape[0])
            x0, x1 = max(x, expected_x), min(x + width, expected_x + section.shape[1])
            assert y0 < y1 and x0 < x1
            target = np.s_[y0-y:y1-y, x0-x:x1-x]
            assert not covered[target].any(), "Overlapping decoded source coverage"
            result[target] = section[y0-expected_y:y1-expected_y, x0-expected_x:x1-expected_x]
            covered[target] = True
            segments.append({"index": index, "offset": page.dataoffsets[index],
                             "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()})
        assert covered.all(), "Source segments do not cover every requested pixel"
    assert before == fingerprint(path), "Source TIFF changed during readback"
    return result, {"path": str(path), "stat": before, "full_section_shape_yx": full_shape,
                    "roi_xywh": [x, y, width, height], "header_bytes": first_offset,
                    "header_sha256": hashlib.sha256(header).hexdigest(), "segments": segments,
                    "source_bytes_read": first_offset + sum(s["bytes"] for s in segments),
                    "whole_file_hash_checked": False, "stable_during_read": True}


def producer_guard(manifest_hash):
    ledger = json.loads((OUT / "manifest/regional_context_run_20261003.json").read_text())
    assert ledger["state"].startswith("terminal_"), "Regional source run is still active"
    assert not psutil.pid_exists(ledger["pid"]), "Regional producer PID still exists"
    watcher = json.loads((HERE / "regional_checkpoint_watcher_20261003.json").read_text())
    assert watcher["state"] == "terminal_verified", "Final regional verifier has not finished"
    assert sha(OUT / "manifest/identity_manifest.csv") == manifest_hash
    return ledger["run_id"]


def check(row, local_record, remaining_bytes):
    uid, sample = row["UID"], row["SampleID"]
    for path, digest in local_record["artifact_hashes"].items():
        assert sha(Path(path)) == digest, f"Previously audited package changed: {path}"
    if not local_record["usable_context"]:
        return {"uid": uid, "outcome": "swc_unavailable", "voxels_checked": 0, "source_bytes_read": 0}
    volume_path = OUT / sample / Path(row["NeuronID"]).stem / "context.nii.gz"
    meta = json.loads(Path(str(volume_path) + ".json").read_text())
    acq = meta["acquisition"]
    data = np.asanyarray(nib.load(str(volume_path)).dataobj)
    assert data.dtype == np.dtype("uint16") and list(data.shape) == meta["shape_xyz"]
    voxel_origin = np.array(meta["origin_xyz_um"]) / SPACING
    np.testing.assert_allclose(voxel_origin, np.round(voxel_origin), atol=1e-8)
    x, y, first_z = map(int, np.round(voxel_origin))
    assert not acq["missing"], "Unexpected missing copied section requires explicit review"
    sources = {entry["z_index"]: entry["path"] for entry in acq["source_paths"]}
    assert set(sources) == set(acq["loaded"]) == set(range(first_z, first_z + data.shape[2]))
    evidence, read_bytes = [], 0
    for iz in range(data.shape[2]):
        z = first_z + iz
        source = PureWindowsPath(sources[z])
        assert source.parent == PureWindowsPath(row["low_res_dir"])
        assert source.name == f"{sample}_{z:05d}_CH1_resample.tif"
        assert remaining_bytes - read_bytes >= MAX_SECTION_BYTES + MAX_SEGMENT_BYTES, "Source read admission budget exhausted"
        roi, record = read_roi(source, x, y, data.shape[0], data.shape[1])
        assert record["full_section_shape_yx"] == acq["full_section_shape_yx"]
        assert np.array_equal(roi, data[:, :, iz].T), f"Copied source pixels differ: {uid}, section {z}"
        record["z_index"] = z
        evidence.append(record)
        read_bytes += record["source_bytes_read"]
    for path, digest in local_record["artifact_hashes"].items():
        assert sha(Path(path)) == digest, f"Package changed during source readback: {path}"
    return {"uid": uid, "outcome": "source_pixels_equal", "voxels_checked": int(data.size),
            "source_bytes_read": read_bytes, "source_sections": evidence,
            "artifact_hashes": local_record["artifact_hashes"], "stable_during_read": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", action="append")
    parser.add_argument("--max-source-gib", type=float, default=256)
    parser.add_argument("--max-seconds", type=float, default=21600)
    parser.add_argument("--output", type=Path, default=HERE / "copied_context_source_pixel_readback_20261003.json")
    args = parser.parse_args()
    assert 0 < args.max_source_gib <= 512 and 0 < args.max_seconds <= 43200
    manifest = OUT / "manifest/identity_manifest.csv"
    manifest_hash = sha(manifest)
    local_path = HERE / "copied_context_package_readback_20261003.json"
    local_hash = sha(local_path)
    local = json.loads(local_path.read_text())
    assert local["status"] == "passed" and local["manifest_sha256"] == manifest_hash
    records = {record["uid"]: record for record in local["records"]}
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["SampleID"] in SAMPLES]
    assert len(rows) == len({row["UID"] for row in rows}) == len(records) == 301
    assert set(records) == {row["UID"] for row in rows}
    if args.uid:
        rows = [row for row in rows if row["UID"] in set(args.uid)]
        assert {row["UID"] for row in rows} == set(args.uid)
    run_id = producer_guard(manifest_hash)
    result = {"status": "running", "manifest_sha256": manifest_hash, "regional_run_id": run_id,
              "local_package_audit_sha256": local_hash, "script_sha256": sha(Path(__file__)),
              "selected_identities": len(rows), "scope": "all copied identities" if not args.uid else "diagnostic subset",
              "max_source_bytes": int(args.max_source_gib * 1024**3), "max_seconds": args.max_seconds,
              "source_bytes_read": 0, "anatomical_acceptance": False, "physical_calibration": False,
              "records": [], "failures": []}
    started = time.monotonic()
    save(args.output, result)
    for row in rows:
        assert producer_guard(manifest_hash) == run_id
        if time.monotonic() - started >= args.max_seconds:
            result["status"] = "budget_stop"
            break
        try:
            record = check(row, records[row["UID"]], result["max_source_bytes"] - result["source_bytes_read"])
            result["records"].append(record)
            result["source_bytes_read"] += record["source_bytes_read"]
            print(json.dumps({"uid": row["UID"], "outcome": record["outcome"]}), flush=True)
        except Exception as exc:
            result["failures"].append({"uid": row["UID"], "type": type(exc).__name__, "error": str(exc)})
            print(json.dumps(result["failures"][-1]), flush=True)
            result["status"] = "failed"
            break
        result["observed_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output, result)
    assert sha(local_path) == local_hash and sha(manifest) == manifest_hash
    if result["status"] == "running":
        result["status"] = "passed" if len(result["records"]) == len(rows) else "incomplete"
    result["checked_identities"] = len(result["records"])
    result["outcome_counts"] = dict(Counter(record["outcome"] for record in result["records"]))
    result["voxels_checked"] = sum(record["voxels_checked"] for record in result["records"])
    result["observed_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"records", "failures"}}), flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
