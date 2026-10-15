"""Prepare validated extra-candidate raw caches without changing producer inputs."""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "group_analysis/scripts"))
import pandas as pd
import visual_review_20261002 as review

source_report = HERE / "unresolved_full_source_validation_20261003.json"
manifest_path = review.OUT_ROOT / "manifest/identity_manifest.csv"
manifest_sha = review.sha256_file(manifest_path)
canonical = review.guard_canonical_baseline(review.OUT_ROOT)
records = review.validated_unresolved_sources(source_report)
live = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
preview = pd.read_csv(HERE / "unresolved_candidate_preview/manifest/identity_manifest.csv",
                      dtype=str, keep_default_na=False)
expected = set(preview["UID"]) - set(live["UID"])
if len(records) != 86 or set(records) != expected:
    raise ValueError("Validated sources do not match the unactivated 86-candidate extension")

# Preflight every destination before copying any file; never replace different bytes.
for uid, record in records.items():
    sample, neuron = uid.split("|", 1)
    target = review.OUT_ROOT / "swc_raw" / sample / neuron
    if target.exists() and review.sha256_file(target) != record["sha256"]:
        raise ValueError(f"Existing raw cache differs from validated source: {uid}")

prepared = []
for uid, record in records.items():
    sample, neuron = uid.split("|", 1)
    source = ROOT / record["local_path"]
    target = review.OUT_ROOT / "swc_raw" / sample / neuron
    already_present = target.exists()
    if not already_present:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if review.sha256_file(temporary) != record["sha256"]:
                raise ValueError(f"Copied raw bytes differ: {uid}")
            temporary.rename(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    rows, digest, url, cache_source = review.fetch_raw_swc(
        sample, neuron, "Monkey", review.OUT_ROOT, attempts=0)
    if (digest != record["sha256"] or url != record["url"]
            or len(rows) != record["nodes"] or review._root_xyz(rows) != record["raw_root_xyz"]
            or target.stat().st_size != record["bytes"]):
        raise ValueError(f"Production cache readback differs: {uid}")
    prepared.append({"uid": uid, "path": str(target.relative_to(ROOT)),
                     "sha256": digest, "bytes": record["bytes"], "nodes": len(rows),
                     "raw_root_xyz": review._root_xyz(rows), "own_source_url": url,
                     "cache_source": cache_source, "already_present": already_present})
    del rows

if review.sha256_file(manifest_path) != manifest_sha or review.canonical_hashes() != canonical:
    raise ValueError("Live manifest or protected inputs changed during cache preparation")
result = {"observed_utc": datetime.now(timezone.utc).isoformat(),
          "state": "validated_raw_cache_prepared", "image_production_complete": False,
          "anatomical_acceptance": False, "prepared_candidates": len(prepared),
          "bytes": sum(row["bytes"] for row in prepared),
          "source_report_sha256": review.sha256_file(source_report),
          "preparer_sha256": review.sha256_file(Path(__file__)),
          "live_manifest_sha256": manifest_sha, "live_manifest_unchanged": True,
          "protected_inputs_unchanged": True, "network_requests": 0, "records": prepared}
(HERE / "unresolved_raw_cache_preparation_20261003.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key != "records"}))
