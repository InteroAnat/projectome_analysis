"""Exercise bounded TIFF ROI decoding locally, without reading server files."""
from pathlib import Path
import argparse
import json

import numpy as np
import tifffile

from verify_copied_context_source_pixels_20261003 import HERE, OUT, producer_guard, read_roi, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-active-guard", action="store_true")
    args = parser.parse_args()
    folder = HERE / "copied_source_roi_diagnostics"
    folder.mkdir(exist_ok=True)
    data = ((np.arange(79 * 103, dtype=np.uint32).reshape(79, 103) * 193 + 87) % 65536).astype(np.uint16)
    cases = [
        ("uncompressed_strips", {"rowsperstrip": 11}),
        ("lzw_strips", {"compression": "lzw", "rowsperstrip": 11}),
        ("lzw_predictor_strips", {"compression": "lzw", "predictor": True, "rowsperstrip": 11}),
        ("deflate_tiles", {"compression": "deflate", "tile": (32, 32)}),
        ("lzw_predictor_tiles", {"compression": "lzw", "predictor": True, "tile": (32, 32)}),
    ]
    records = []
    for name, options in cases:
        path = folder / (name + ".tif")
        tifffile.imwrite(path, data, photometric="minisblack", **options)
        for x, y, width, height in ((0, 0, 3, 7), (23, 19, 45, 39), (98, 75, 5, 4), (0, 0, 103, 79)):
            roi, evidence = read_roi(path, x, y, width, height)
            assert np.array_equal(roi, data[y:y+height, x:x+width])
            assert evidence["stable_during_read"] and not evidence["whole_file_hash_checked"]
            assert len({s["index"] for s in evidence["segments"]}) == len(evidence["segments"])
            records.append({"case": name, "roi_xywh": [x, y, width, height],
                            "pixels_equal": roi.size, "source_bytes_read": evidence["source_bytes_read"]})
        try:
            read_roi(path, 100, 78, 4, 2)
        except AssertionError:
            pass
        else:
            raise AssertionError("Out-of-bounds crop was accepted")
    script = HERE / "verify_copied_context_source_pixels_20261003.py"
    result = {"status": "passed", "network_requests": 0, "server_reads": 0,
              "source_verifier_sha256": sha(script), "cases": records,
              "successful_roi_cases": len(records), "out_of_bounds_rejections": len(cases)}
    if args.check_active_guard:
        try:
            producer_guard(sha(OUT / "manifest/identity_manifest.csv"))
        except AssertionError as exc:
            assert str(exc) == "Regional source run is still active"
            result["active_source_producer_rejected_before_server_access"] = True
        else:
            raise AssertionError("Active producer was admitted")
    (HERE / "copied_source_roi_diagnostic_20261003.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))


if __name__ == "__main__":
    main()
