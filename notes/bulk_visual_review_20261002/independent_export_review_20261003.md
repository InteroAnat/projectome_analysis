# Independent export-metadata review

**Author: Codex independent subagent | Date: 2026-10-03 (Asia/Shanghai)**

Six tiny synthetic export cases ran only inside an automatically removed
temporary directory. Production code, tests, arrays and reviewer tables were
untouched. No credential configuration was printed. `Visual_toolkit.py` SHA256
was identical before source inspection, before the isolated run and afterward:
`2828848398a026b0bcdef8ffc02b88d874e08b8f12486c3eb3c00a76eaf08621`.
These findings apply to that stable snapshot; later Cursor edits need a recheck.

## Reproduced remaining defects

1. **The 2D TIFF branch receives 3D NIfTI axis assertions.** Exporting a 4 x 5
   YX array saves a correctly shaped TIFF, but metadata at lines 461-464 says
   `source_array_axis_order=ZYX`, `array_axis_order=XYZ`, and
   `nifti_axis_order=XYZ`. It also has `shape_zyx=[4,5]`. Branch metadata by
   actual persisted format/dimensionality: a 2D grayscale TIFF is YX with
   `shape_yx=[4,5]` and no NIfTI axis assertion. If a 2D image is derived from
   a 3D slab, explicitly describe that reduction and its source depth rather
   than silently describing the TIFF as an XYZ volume.
2. **Acquisition/coverage can describe another crop.** Line 451 selects
   `last_high_res_metadata` only for exact suffix `SomaBlock`; every other
   suffix, including default `Volume`, selects `last_low_res_metadata`.
   Lines 466-472 then copy coverage and sources without validating sample,
   origin, spacing or shape. A high crop exported with default `Volume`,
   matching partial high metadata and stale complete low metadata, attaches
   the stale low acquisition and falsely reports `complete=True`. Even
   `SomaBlock` attaches a stale high record with a different origin/shape.
   Bind acquisition explicitly to the exported data, or validate a candidate
   against sample, origin, spacing and source shape before attachment.
   Unmatched acquisition must remain unknown; a display suffix is not source
   identity. Derived 2D exports additionally need an explicit derivation map.
3. **Supplied root coordinates are not validated.**
   `soma_coords=[1,2]` succeeds and writes `center_xyz_um=[1.0,2.0]` at
   lines 452-460, despite the XYZ field contract. Validate finite XYZ origin,
   positive XYZ spacing, supported input dimensionality and any supplied
   finite XYZ root before writing either the image or sidecar. Do not infer
   an absent root from the crop origin. The current unknown-root behavior
   correctly uses null; preserve that behavior.

These are generic export contracts. The current native review's matching
`SomaBlock`/`Context` calls with explicit roots can produce correct 3D metadata;
this report does not establish that existing valid crop arrays are defective.

## Positive checks

- A 2 x 3 x 4 ZYX source array saves exact XYZ NIfTI voxels with shape 4 x 3 x 2,
  micrometre units, and the requested origin/spacing affine.
- With no acquisition and no supplied root, the 3D sidecar reports
  `acquisition=None`, `complete=None`, and `center_xyz_um=None`.
- A matching partial high acquisition exported as `SomaBlock` propagates
  `complete=False` and the missing-cube list, and records a supplied three-value
  root correctly. Source ZYX and persisted XYZ are correct for this 3D case.
- Exporter `output_file` points to the actual file it writes. Subsequent
  caller renaming is a separate responsibility; the review renderer's
  placement helper addresses that path.

## Focused validation suggestions

Use unequal dimensions and a known voxel pattern to test persisted arrays and
axis/shape fields independently. Test 2D TIFF and 3D NIfTI, default/custom
suffixes, no acquisition, stale low/high state with mismatched geometry, a
matching partial acquisition, omitted roots and malformed/nonfinite roots.
For a stale record, assert unknown acquisition/coverage or an explicit failure;
do not merely assert that the last dictionary is copied. Invalid geometry/root
inputs should fail before image creation. No production rerender is required
to test or repair these metadata contracts.

## Minimal disposable reproduction

Run with the projectome interpreter and `-B`. The immutable in-memory source
snapshot avoids mixing versions during concurrent edits; every output is
confined to `TemporaryDirectory`.

```python
import hashlib, json, os, sys, tempfile, types
from pathlib import Path
import numpy as np
import tifffile

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, "D:/projectome_analysis/main_scripts")
path = Path("D:/projectome_analysis/main_scripts/Visual_toolkit.py")
payload = path.read_bytes()
m = types.ModuleType("independent_export_snapshot")
m.__file__ = str(path)
exec(compile(payload, str(path), "exec"), m.__dict__)
print("source_before_sha256", hashlib.sha256(payload).hexdigest())

with tempfile.TemporaryDirectory(prefix="projectome-export-audit-") as directory:
    def toolkit():
        t = m.Visual_toolkit.__new__(m.Visual_toolkit)
        t.sample_id = "252385"
        t.output_dir = directory
        t.last_high_res_metadata = t.last_low_res_metadata = None
        t.last_export_metadata = None
        return t

    def acquisition(t, origin, spacing, shape, complete, source):
        missing = [] if complete else [[1, 2, 3]]
        return t._acquisition_metadata(source, [101, 201, 303], origin,
            spacing, shape, [[1, 2, 3]], [] if missing else [[1, 2, 3]],
            missing, [source + "-path"])

    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    origin, spacing = [100, 200, 300], [.65, .65, 3]
    t = toolkit()
    out = t.export_data(np.arange(20, dtype=np.uint16).reshape(4, 5),
                        [10, 20, 30], [5, 5, 3], "2d", suffix="WideField")
    d = json.loads(Path(out + ".json").read_text())
    print("2D", tifffile.imread(out).shape, d["array_axis_order"],
          d["source_array_axis_order"], d.get("nifti_axis_order"), d["shape_zyx"])

    t = toolkit()
    t.last_high_res_metadata = acquisition(t, origin, spacing, volume.shape,
                                           False, "actual high partial")
    t.last_low_res_metadata = acquisition(t, [10, 20, 30], [5, 5, 3],
                                          [7, 8, 9], True, "stale low")
    out = t.export_data(volume, origin, spacing, "generic")
    d = json.loads(Path(out + ".json").read_text())
    print("generic acquisition", d["acquisition"]["source"], d["complete"])

    t = toolkit()
    t.last_high_res_metadata = acquisition(t, [900, 900, 900], spacing,
                                           [7, 8, 9], False, "stale high")
    out = t.export_data(volume, origin, spacing, "stale", suffix="SomaBlock")
    d = json.loads(Path(out + ".json").read_text())
    print("mismatched acquisition", d["acquisition"]["origin_xyz_um"],
          d["origin_xyz_um"], d["complete"])

    t = toolkit()
    out = t.export_data(volume, origin, spacing, "invalid-root",
                        suffix="SomaBlock", soma_coords=[1, 2])
    print("malformed root accepted",
          json.loads(Path(out + ".json").read_text())["center_xyz_um"])

print("source_after_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
```

Observed outputs: TIFF `(4,5)` with `XYZ / ZYX / XYZ / [4,5]`; generic export
selected `stale low` and `True`; `SomaBlock` attached origin `[900,900,900]`
to actual origin `[100,200,300]`; malformed root `[1.0,2.0]` was accepted.
