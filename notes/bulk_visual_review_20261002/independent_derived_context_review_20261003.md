# Independent derived-context review

**Author: Codex independent subagent | Date: 2026-10-03 (Asia/Shanghai)**

This is a provisional review while Cursor/Grok continues editing. Initial
renderer SHA256 was
`5746c70f8f6876594c1547fbacf21b2628e0d42c86709ff572a143b6094a7939`;
the source changed during inspection. All reproductions below ran against
stable snapshot
`159d313c5f74645a0809909901556d33f43c3415ce9e8a67f2f92401171605ac`,
unchanged during both test runs. No network, production code/test edits,
server reads or production-array writes occurred. Pilot writes were confined
to an automatically removed temporary directory with a fake toolkit.

## Confirmed snapshot defects

1. **Reported grid origin differs from sampled source coordinates.**
   `derive_highres_context` uses arbitrary `native-half_field` origin, then
   floor-bins decimated cube pixels onto it. For native XYZ
   `[100.25,100.75,101.2]`, field 20 um, depth 6 um and factor 8, independent
   coordinate-ramp cubes show first output voxel source coordinates
   `[93.6,93.6,99.0]` um. The report/NIfTI origin is
   `[90.25,90.75,98.2]` um: errors `[3.35,2.85,0.8]` um under the existing
   source convention. Use a globally defined source sampling lattice and
   record its actual first sample/spacing, or resample onto the declared grid.
   Explicitly record point subsampling or any block-mean center offset. This
   fixes the new derived-grid contract; it does not recalibrate native sources.
2. **Unavailable center/all-missing data are accepted as derived context.**
   An all-missing callback returns an all-zero array and `derived_partial`.
   For root `[233.4,233.4,100.2]`, with central cube `(0,0,0)` unavailable
   and three XY neighbors loaded, the function returns `derived_partial`,
   canvas maximum 100 and zero at the root's array index. There is no root
   coverage requirement or availability mask. Reject an all-missing result;
   explicitly distinguish center-unavailable context from usable partial
   neighboring coverage. Geometric containment is not evidence of acquired
   root data. Keep missing-tile/plane coverage separate from fluorescence zeros.
3. **Pilot can overwrite an existing copied context.**
   `pilot_derived_context` accepts any manifest identity and saves directly to
   `context.nii.gz`. A valid temporary copied-context NIfTI of shape 2 x 2 x 2,
   with copied-source provenance, was overwritten by a 4 x 4 x 2 derived crop;
   its SHA changed and the function returned normally. Protect existing copied
   arrays before downloads/writes, or use a separate derived artifact path.
   Promotion to the review's context component must be deliberate and preserve
   source provenance and existing evidence.
4. **A malformed cube is treated as fully loaded.**
   A `(4,40,40)` callback array, instead of expected `(90,360,360)`, yields
   `derived_highres` even though 96.25% of the requested output is zero.
   Existing focused tests also use this truncated shape and assert success.
   The real toolkit decoder checks cube shape, but this helper's download
   interface must either enforce that contract or use explicit source bounds.

## Resource and integration observations

The default example `[40200.2,31689.6,19036.2]` requests 324 cubes and creates
a `(10,770,770)` uint16 canvas, 11,858,000 bytes, with one decoded cube of
23,328,000 bytes at a time. This avoids materializing a dense high-resolution
whole-block field. The corresponding decoded source payload totals about
7.56 GB; network/cache bytes depend on TIFF compression and cache reuse.
There is a per-candidate cube-count bound, but no explicit output-byte or
cache-byte bound. With accepted `factor=1`, the helper would allocate
757,434,320 output bytes despite staying within its 400-cube bound; the audit
intercepted this allocation before it happened. Validate dimensions/factor,
cap output bytes, and record bounded requests/cache policy. Factor 8 divides
the 360-pixel cube width, but arbitrary factors also need a global-lattice
definition rather than independently restarting the sampling phase per tile.

At inspection time, integration was still incomplete: the pilot wrote an MIP
and marked MIP plus a NIfTI, without the full context sidecar/trace/stack/coverage
package; table spacing remained 5 um and one verifier check still assumed
`[5,5,3]` rather than actual `[5.2,5.2,3]`. These may be fixed by concurrent
edits and are not final acceptance findings. The inspected CLI exposed only
one-identity `pilot-context`; a successful pilot must be followed by the
remaining regional-context queue or documented tested source/resource limits.
One feasible pilot does not complete the original multi-sample regional scope.

## Minimal helper reproductions

Run with projectome Python and `-B`. This uses zero-stride broadcast cube views
and tiny output fields; it performs no file writes or network calls.

```python
import hashlib, types
from pathlib import Path
import numpy as np

path = Path("D:/projectome_analysis/group_analysis/scripts/visual_review_20261002.py")
payload = path.read_bytes()
m = types.ModuleType("independent_derived_snapshot")
m.__file__ = str(path)
exec(compile(payload, str(path), "exec"), m.__dict__)
print("snapshot", hashlib.sha256(payload).hexdigest())

native = [100.25, 100.75, 101.2]
for axis in range(3):
    dimensions = [1, 1, 1]
    dimensions[2-axis] = [360, 360, 90][axis]
    values = np.arange([360, 360, 90][axis], dtype=np.uint16).reshape(dimensions)
    cube = np.broadcast_to(values, (90, 360, 360))
    canvas, report = m.derive_highres_context(native, lambda *idx: cube,
        field_um=20, depth_um=6, factor=8, max_cubes=10)
    actual = float(canvas[0,0,0]) * [.65, .65, 3][axis]
    print("XYZ"[axis], "actual", actual, "reported", report["origin_xyz_um"][axis])

canvas, report = m.derive_highres_context(native, lambda *idx: None,
    field_um=20, depth_um=6, factor=8, max_cubes=10)
print("all missing", report["status"], canvas.max(), len(report["loaded"]))

native = [233.4, 233.4, 100.2]
cube = np.broadcast_to(np.uint16(100), (90,360,360))
def download(x, y, z):
    return None if (x,y,z) == (0,0,0) else cube
canvas, report = m.derive_highres_context(native, download,
    field_um=20, depth_um=6, factor=8, max_cubes=10)
x, y, z = np.floor((np.array(native)-report["origin_xyz_um"])
                  / report["derived_spacing_xyz_um"]).astype(int)
print("central missing", report["status"], canvas[z,y,x], canvas.max())
print("source after", hashlib.sha256(path.read_bytes()).hexdigest())
```

## Exact copied-context preservation reproduction

The derivation and toolkit are mocked in memory; the original context is a
real valid NIfTI in a disposable temporary directory. No download is permitted.

```python
import json, sys, tempfile
import pandas as pd
import nibabel as nib
from unittest.mock import patch

with tempfile.TemporaryDirectory(prefix="projectome-context-preserve-") as directory:
    root = Path(directory)
    (root / "manifest").mkdir()
    folder = root / "252385/003"
    folder.mkdir(parents=True)
    original = folder / "context.nii.gz"
    nib.save(nib.Nifti1Image(np.full((2,2,2),17,dtype=np.uint16),np.eye(4)),str(original))
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    (folder / "provenance.json").write_text(json.dumps({
        "SampleID":"252385", "NeuronID":"003.swc", "UID":"252385|003.swc",
        "overview_source_status":"copied_ch1", "context_coverage_status":"complete"}))
    m.load_manifest = lambda *_: pd.DataFrame([{
        "SampleID":"252385", "NeuronID":"003.swc", "project_id":"Monkey",
        "image_source_status":"available"}])
    m.fetch_raw_swc = lambda *_: ([(1,1,100.25,100.75,101.2,1,-1)],
                                 "synthetic-sha", "synthetic-own-sample-url", "synthetic")
    report = {"status":"derived_highres", "cube_count":1,
        "derived_spacing_xyz_um":[5.2,5.2,3], "origin_xyz_um":[90.25,90.75,98.2],
        "shape_zyx":[2,4,4], "loaded":[[0,0,0]], "missing":[], "downsample":8}
    m.derive_highres_context = lambda *a, **k: (
        np.full((2,4,4),100,dtype=np.uint16), dict(report))
    m._save_gray = lambda *a, **k: {}
    m._draw_marked = lambda *a, **k: None
    class FakeToolkit:
        def __init__(self, *a, **k): pass
        def _download_http_block(self, *a): raise AssertionError("No download permitted")
        def close(self): pass
    fake = types.ModuleType("Visual_toolkit")
    fake.Visual_toolkit = FakeToolkit
    with patch.dict(sys.modules, {"Visual_toolkit": fake}):
        m.pilot_derived_context("252385", "003.swc", out_root=root)
    print("copied context preserved", before == hashlib.sha256(original.read_bytes()).hexdigest())
```

The preservation result was `False`. Re-run against the completed Cursor
implementation before production acceptance; synthetic tests should then
prove geometry, central-data rejection and copied-array protection explicitly.
