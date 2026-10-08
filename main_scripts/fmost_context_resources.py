"""Admission budgets and dated failures for sequential native context batches."""
from pathlib import Path
import json
import os
import shutil
import time

import numpy as np
import tifffile

from Visual_toolkit import MAX_HTTP_BLOCK_BYTES, HTTP_ATTEMPTS, _read_highres_cube


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class BoundedCubeSource:
    """One sequential batch; cached pixels remain usable after network limits.

    Response reservations cover all HTTP attempts. Disk admission is conservative
    and never deletes sources; budgets apply to the shared cube cache, while the
    free-space margin also protects other files. HTTP 404 observations expire.
    """

    def __init__(self, out_root, cache_bytes=300 * 1024**3, max_requests=60000,
                 max_seconds=8 * 3600, min_free_bytes=50 * 1024**3,
                 failure_ttl_seconds=6 * 3600, retry_missing=False):
        if any(not np.isfinite(v) or v <= 0 for v in
               (cache_bytes, max_requests, max_seconds, min_free_bytes, failure_ttl_seconds)):
            raise ValueError("Resource and failure-observation bounds must be positive")
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.cache_bytes = int(cache_bytes)
        self.max_requests = int(max_requests)
        self.max_seconds = float(max_seconds)
        self.min_free_bytes = int(min_free_bytes)
        self.failure_ttl_seconds = float(failure_ttl_seconds)
        self.retry_missing = bool(retry_missing)
        self.started = time.monotonic()
        self.requests_upper_bound = 0
        self.cached_hits = 0
        self.failure_hits = 0
        self.admission_stops = {}
        self.present_bytes = sum(p.stat().st_size for p in
                                 (self.out_root / "cache" / "cubes").rglob("*.tif"))
        self.failure_path = self.out_root / "manifest" / "regional_source_failures.json"
        self.failures = json.loads(self.failure_path.read_text(encoding="utf-8")) if self.failure_path.is_file() else {}

    def runtime_expired(self):
        return time.monotonic() - self.started >= self.max_seconds

    def snapshot(self):
        return {"cache_limit_bytes": self.cache_bytes, "present_cache_bytes": self.present_bytes,
                "request_limit": self.max_requests, "request_attempts_upper_bound": self.requests_upper_bound,
                "cached_hits": self.cached_hits, "reused_404_observations": self.failure_hits,
                "elapsed_seconds": time.monotonic() - self.started, "max_seconds": self.max_seconds,
                "minimum_free_bytes": self.min_free_bytes, "admission_stops": self.admission_stops}

    def _stop(self, toolkit, index, reason):
        self.admission_stops[reason] = self.admission_stops.get(reason, 0) + 1
        toolkit._block_errors[index] = f"Resource admission stopped: {reason}"
        return None

    def __call__(self, toolkit, x, y, z):
        index = (x, y, z)
        path = Path(toolkit.cache_http_dir) / str(z) / f"{x}_{y}_{z}.tif"
        if path.is_file() and path.stat().st_size <= MAX_HTTP_BLOCK_BYTES:
            try:
                data = _read_highres_cube(str(path))
                self.cached_hits += 1
                return data
            except (ValueError, OSError, tifffile.TiffFileError):
                pass  # Preserve invalid cache evidence; replacement needs admission.
        key = f"{toolkit.sample_id}|{x}|{y}|{z}"
        observation = self.failures.get(key)
        if observation and not self.retry_missing:
            age = time.time() - float(observation["observed_unix"])
            if 0 <= age < self.failure_ttl_seconds:
                self.failure_hits += 1
                toolkit._block_errors[index] = (
                    f"HTTP Error 404; dated observation reused, age {age:.1f}s; "
                    "does not establish anatomical absence")
                return None
        if self.runtime_expired():
            return self._stop(toolkit, index, "batch runtime budget")
        if self.requests_upper_bound + HTTP_ATTEMPTS > self.max_requests:
            return self._stop(toolkit, index, "request-attempt budget")
        if self.present_bytes + MAX_HTTP_BLOCK_BYTES > self.cache_bytes:
            return self._stop(toolkit, index, "shared cache-byte budget")
        if shutil.disk_usage(self.out_root).free - MAX_HTTP_BLOCK_BYTES < self.min_free_bytes:
            return self._stop(toolkit, index, "free-space margin")
        old_size = path.stat().st_size if path.is_file() else 0
        self.requests_upper_bound += HTTP_ATTEMPTS
        data = toolkit._download_http_block(x, y, z)
        new_size = path.stat().st_size if path.is_file() else 0
        self.present_bytes += new_size - old_size
        if data is None:
            reason = str(toolkit._block_errors.get(index, "source unavailable"))
            if "HTTP Error 404" in reason:
                self.failures[key] = {"sample_id": str(toolkit.sample_id), "cube_xyz": list(index),
                                      "observed_unix": time.time(), "reason": reason,
                                      "ttl_seconds": self.failure_ttl_seconds}
                atomic_json(self.failure_path, self.failures)
        else:
            if key in self.failures:
                del self.failures[key]
                atomic_json(self.failure_path, self.failures)
        return data


def coverage_mask(report):
    """Acquisition coverage independent of fluorescence intensity, source ZYX."""
    shape_xyz = np.array(report["shape_zyx"])[::-1]
    first = np.array(report["source_first_index_xyz"])
    stride = np.array(report["source_stride_xyz"])
    block = np.array([360, 360, 90])
    mask = np.zeros(report["shape_zyx"], dtype=np.uint8)
    for index in report["loaded"]:
        start = np.array(index) * block
        lo = np.maximum(0, np.ceil((start - first) / stride).astype(int))
        hi = np.minimum(shape_xyz, np.ceil((start + block - first) / stride).astype(int))
        if (hi > lo).all():
            mask[tuple(slice(int(lo[a]), int(hi[a])) for a in (2, 1, 0))] = 1
    return mask
