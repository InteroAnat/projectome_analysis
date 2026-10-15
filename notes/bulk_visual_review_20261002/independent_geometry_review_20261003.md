# Independent review of Cursor native-image outputs

**Agent: Codex | Date: 2026-10-03 (Asia/Shanghai)**

This records the independent read-only audit before Cursor's follow-up repairs.
It is software/source correspondence evidence, not anatomical acceptance.

## Verified persisted evidence

- All 301 selected copied-overview identities have provenance. All 2,905 PNGs
  decode. All 299 available raw SWCs have matching SHA256 and root coordinates.
  All 581 crop headers match recorded shape, affine, nominal micrometre units
  and root indices. The manifest remained unchanged during that audit.
- For 252385/003, the cached source cube `63/220_135_63.tif` equals
  `soma.nii.gz` voxel for voxel after the XYZ/ZYX transpose. The unmarked PNG
  equals the contrast-normalized central plane. Header/root differences are
  within the stored floating-point precision.
- Canonical and Henry workbook hashes match the protected baseline. The
  initial correction table has unique UIDs and all six human fields blank.

## Reproduced repairs

- Five `ok` rows contain missing neighbor cubes: 252385/086, 252385/101,
  252527/087, 252384/079 and 252714/071. Preserve valid centers but expose
  partial coverage in every status/count.
- Historical `area_44` is stripped to `AREA` by the portal-specific suffix
  normalizer. Historical/staging labels require their own normalization.
- Regeneration blanks reviewer fields; verification rewrites the table.
  Preserve edits by UID, detect CSV/XLSX conflicts, make verification read-only.
- Manifest rebuilding replaces the canonical baseline. Protect its original
  values and record selection input/configuration provenance.
- XZ/YZ previews use equal voxel aspect although Z spacing is 3 um versus
  0.65 um in XY: Z is compressed by 4.615 times. Use physical extents/aspect.
- Trace images overlay full-crop SWC projection on a single central plane.
  For 252527/048, 1,598 of 1,679 displayed soma edges lie wholly outside the
  displayed 3 um index bin; for 252385/003, 891 of 917; for 252714/002,
  4,323 of 4,409. For 048, projected geometry spans approximately -356 to
  +245 um around the displayed plane. Use a matching narrow-slab MIP and
  clip the trace to that same slab, or show central-plane-only geometry.
- Export sidecars for 581 crops use the origin as top-level soma center,
  retain renamed output filenames and call the persisted NIfTI ZYX rather
  than XYZ. Nested acquisition metadata has the correct center. Five
  sidecars report top-level complete despite incomplete nested acquisitions.
  Repair JSON metadata and the exporter without altering valid raw arrays.

Detailed whole-set read-back counts are in `independent_audit_20261003.json`.
Follow-up implementation instructions are in `CURSOR_FOLLOWUP.md` and
`CURSOR_STAGE2.md`. Missing copied overviews do not establish that own-sample
HTTP soma images are unavailable; inspect the pending-source probe report.

Nominal spacing, acquisition Z origin, anatomical axes, region/layer decisions
and tracing correctness still require independent dataset/human evidence.

## Reproduction details

**Independent subagent: Codex | Added: 2026-10-03 (Asia/Shanghai)**

The following read-only calculations produced the source-array and trace-depth
evidence above. Run with the projectome interpreter and `-B`; do not call
`verify_outputs`, `write_correction_table`, `render_available`, or fetch methods
against the production review directory to reproduce this audit. These numbers
describe the persisted previews before the follow-up repairs; later matching
slab previews must be assessed against their own recorded bounds.

```python
import json, sys
from pathlib import Path
import numpy as np
import nibabel as nib
import tifffile
from PIL import Image

sys.path.insert(0, "D:/projectome_analysis/main_scripts")
from fmost_image_geometry import clip_segment, percentile_normalize
from swc_validation import parse_swc

root = Path("D:/projectome_analysis/group_analysis/visual_review_20261002")
folder = root / "252385" / "003"
p = json.loads((folder / "provenance.json").read_text())
image = nib.load(str(folder / "soma.nii.gz"))
xyz = np.asanyarray(image.dataobj)
cube = tifffile.imread(str(root / "cache/cubes/252385/high_res_http/63/220_135_63.tif"))
assert np.array_equal(cube, xyz.transpose(2, 1, 0))
k = int(np.floor(p["soma_local_xyz"][2]))
display = np.sqrt(percentile_normalize(xyz[:, :, k].T, .5, 99.5))
assert np.array_equal(np.asarray(Image.open(folder / "soma_unmarked.png")),
                      (display * 255).astype(np.uint8))
print("252385|003: exact source-array and central-plane PNG correspondence")

for sample, neuron in [("252527", "048"), ("252385", "003"),
                       ("252384", "002"), ("252714", "002")]:
    p = json.loads((root / sample / neuron / "provenance.json").read_text())
    path = root / "swc_raw" / sample / (neuron + ".swc")
    rows = parse_swc(path.read_bytes().decode("utf-8"), source=str(path))
    by_id = {r[0]: r for r in rows}
    for kind, level in [("context", "low"), ("soma", "high")]:
        origin = np.array(p[kind + "_origin_xyz_um"])
        spacing = np.array(p["declared_spacing_" + level + "_xyz_um"])
        depth, height, width = p[kind + "_shape_zyx"]
        k = int(np.floor(p[kind + "_local_xyz"][2]))
        full = outside = 0
        for node in rows:
            if node[6] == -1:
                continue
            parent = by_id[node[6]]
            a = (np.array(parent[2:5]) - origin) / spacing
            b = (np.array(node[2:5]) - origin) / spacing
            c = clip_segment(a, b, [-.5, -.5, -.5],
                             [width - .5, height - .5, depth - .5])
            if c is None:
                continue
            full += 1
            z0, z1 = sorted([float(c[0][2]), float(c[1][2])])
            outside += z1 < k or z0 > k + 1
        print(p["UID"], kind, "full-slab edges", full,
              "outside displayed floor-index bin", outside)
```

The floor-index interval `[k, k+1]` follows the renderer's selected section
convention and has nominal depth 3 um. This diagnostic does not independently
calibrate the physical section origin, optical sampling support, or Z offset.
It establishes that the original overlay included geometry far beyond the
displayed section under its own stated indexing convention.

Exact pre-repair results:

| UID | Context full/outside | Soma full/outside |
| --- | ---: | ---: |
| 252527\|048.swc | 389 / 308 | 1,679 / 1,598 |
| 252385\|003.swc | 568 / 493 | 917 / 891 |
| 252384\|002.swc | 296 / 227 | 724 / 690 |
| 252714\|002.swc | 666 / 553 | 4,409 / 4,323 |

The four raw SWC byte hashes matched their recorded hashes. Independently
reading the persisted crop headers found zero shape or affine mismatches;
all 581 available sidecars had the incorrect top-level center and stale
output filename, and five incorrectly claimed top-level completeness.
