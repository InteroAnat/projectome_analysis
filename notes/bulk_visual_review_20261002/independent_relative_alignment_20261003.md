# Relative overview/soma image correspondence

**Author: Codex independent subagent | Date: 2026-10-03 (Asia/Shanghai)**

Four paired persisted context/soma crops were compared using only local raw
NIfTI arrays. No server reads, downloads, production writes or calibration
changes were made. This is optional relative-source evidence, not anatomical
acceptance or a completion gate for production.

## Method

NIfTI affines supplied native XYZ origins and spacings: context 5 x 5 x 3 um,
soma approximately 0.65 x 0.65 x 3 um (stored float32 XY spacing). The selected
context plane was `floor(context_local_z)`. Its recorded native Z was mapped
into the soma crop with `(context_native_z - soma_origin_z) / soma_spacing_z`;
all four matched exact integer soma indices under the stored convention.

Each fine XY plane was Gaussian-blurred with sigma `5 / (2 * 0.65)`, about
3.846 fine pixels per axis, then linearly interpolated onto overlapping coarse
pixel centers. A three-sigma fine-image margin excluded boundary-filter effects.
The coarse plane stayed fixed while fine Z offsets -2, -1, 0, +1, +2 were
tested, corresponding to nominal -6, -3, 0, +3, +6 um. Pearson correlation of
raw intensities, Pearson correlation after `log1p`, and Spearman rank correlation
were computed over the same XY ROI. No fitted translation, rotation, axis flip,
intensity correction, or image-registration optimization was applied.

Planes were read one at a time using NIfTI array proxies. The largest fine
plane was 1080 x 1080; no complete 1080 x 1080 x 270 array was materialized.
Compressed-file hashes were streamed in 1 MiB chunks. All eight crop files
retained their size and modification time during the comparison. Libraries:
NumPy 1.26.4, SciPy 1.15.3, nibabel 5.4.2.

## Results

| UID | Coarse ROI XY | Context filename Z | Context/soma local k | Nominal source Z um | Raw r at offset 0 | Log r at offset 0 | Rank r at offset 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 252385\|003.swc | 44 x 44 | 5727 | 5 / 57 | 17181 | 0.9203 | 0.3434 | 0.2976 |
| 252527\|048.swc | 138 x 138 | 6297 | 5 / 177 | 18891 | 0.7052 | 0.2962 | 0.1825 |
| 252384\|008.swc | 44 x 44 | 12468 | 5 / 48 | 37404 | 0.8691 | 0.3345 | 0.2598 |
| 252714\|002.swc | 138 x 137 | 14134 | 5 / 94 | 42402 | 0.9546 | 0.2958 | 0.2657 |

Raw-intensity Pearson correlation across the tested fine Z offsets:

| UID | -2 | -1 | 0 | +1 | +2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 252385\|003.swc | 0.9211 | 0.9410 | 0.9203 | 0.9308 | 0.9371 |
| 252527\|048.swc | 0.6297 | 0.7206 | 0.7052 | 0.6712 | 0.6099 |
| 252384\|008.swc | 0.8824 | 0.8769 | 0.8691 | 0.8238 | 0.8279 |
| 252714\|002.swc | 0.9526 | 0.9587 | 0.9546 | 0.9078 | 0.8955 |

| UID | Best raw offset | Best log offset / r | Best rank offset / r | Coarse nonzero fraction |
| --- | ---: | ---: | ---: | ---: |
| 252385\|003.swc | -1 | -1 / 0.3572 | -1 / 0.3418 | 0.8993 |
| 252527\|048.swc | -1 | -1 / 0.3684 | -1 / 0.2898 | 0.4031 |
| 252384\|008.swc | -2 | -1 / 0.3949 | -1 / 0.3427 | 0.5480 |
| 252714\|002.swc | -1 | -1 / 0.3379 | -1 / 0.3369 | 0.7419 |

The ROIs span native XY pixel-center coordinates:
252385/003 [51490, 31600] to [51705, 31815] um;
252527/048 [17090, 31130] to [17775, 31815] um;
252384/008 [47510, 17560] to [47725, 17775] um;
252714/002 [29960, 14520] to [30645, 15200] um.
All four examples reported no missing context planes or soma cubes.

## Interpretation and limits

Bright image patterns agree substantially under the recorded nominal XY
geometry. Log/rank correlations are much lower, so raw correlation is dominated
by intense structures and does not establish exact diffuse-intensity or
individual-neurite correspondence. All four log/rank comparisons prefer the
preceding fine plane; raw correlation prefers it in three examples and the
second preceding plane in the fourth. This is a relative Z-index question for
independent source-calibration review. Adjacent sections share structures, the
peak is shallow in some examples, and overview resampling/contrast provenance
is incomplete. Do not change coordinate origins or select an offset from these
four peaks alone.

Both crops derive from microscopy data and can share the same coordinate
assumptions or systematic error. This comparison does not establish absolute
physical scale, acquisition origin, anatomical orientation, laterality, atlas
region, layer, soma identity or thin-axon continuity. No significance test was
performed: pixels are spatially dependent, and this is a descriptive source
check on four selected examples rather than an independent biological sample.

## Z source-index audit

The audited loader source hashes were unchanged across these code/header reads:
`Visual_toolkit.py` SHA256
`2828848398a026b0bcdef8ffc02b88d874e08b8f12486c3eb3c00a76eaf08621`,
and `fmost_image_geometry.py` SHA256
`0777af591b1bcee1282bb14ca91036397cf836a64e6def61d78444b0788cf4ce`.
Renderer line numbers can move while Cursor/Grok edits concurrently.

- Low resolution: `_low_res_slice_filename` at loader lines 147-148 encodes
  integer `n` as `<sample>_<n:05d>_CH1_resample.tif`. The helper describes
  filenames as one-based and prohibits indices below 1, but explicitly maps
  filename `n` to native `3*n` um, not `3*(n-1)`. The low loader takes
  `floor(native_z/3)`, requests consecutive `[z_start, z_stop)` indices,
  appends them in that order, and records crop origin `3*z_start` at line 372.
  Array position `j` therefore represents filename `z_start+j` and nominal
  native coordinate `3*(z_start+j)` under this declared convention.
- High resolution: loader line 229 selects cube Z index
  `b=floor((native_z/3)/90)`. Cube filenames are
  `<bx>_<by>_<b>.tif` beneath `cube/<b>/`. Cube array pages remain in their
  decoded order, with page `q=0..89`. The loader copies them unchanged into
  the grid volume and sets origin `(first_cube_z)*90*3` at line 276.
  Page `q` in cube `b` therefore maps to global nominal plane `90*b+q`
  and native `3*(90*b+q)` um. A larger neighborhood adds an integer 90-page
  displacement; it does not change the center page's global mapping.

The matching plane under this code is thus `n=90*b+q`. No internal arithmetic
disagreement between saved affines, filename IDs and copied page positions was
found in the four examples:

| UID | Low filename | Center cube filename | Nominal-match page q | Preferred preceding page q |
| --- | --- | --- | ---: | ---: |
| 252385\|003.swc | 252385_05727_CH1_resample.tif | 220_135_63.tif | 57 | 56 |
| 252527\|048.swc | 252527_06297_CH1_resample.tif | 74_134_69.tif | 87 | 86 |
| 252384\|008.swc | 252384_12468_CH1_resample.tif | 203_75_138.tif | 48 | 47 |
| 252714\|002.swc | 252714_14134_CH1_resample.tif | 129_63_157.tif | 4 | 3 |

The last column describes the common log/rank preference; the raw-intensity
peak for 252384/008 is page 46. Header-only inspection of all four local center
cubes found 90 pages of 360 x 360 pixels. Their `ImageDescription` contains only
`{"shape": [90, 360, 360]}`; X/YResolution are `(1,1)`, ResolutionUnit is
`NONE`, and Software is `tifffile.py`. These tags supply no independent physical
Z origin, absolute section number, or spacing calibration.

A source-generation convention in which one-based overview filename `n`
contains high-resolution global page `n-1` would explain a relative -1-plane
preference. The current code deliberately does not subtract 1, and the helper
already calls that physical origin unverified. Filename numbering alone cannot
prove that this alternative is correct: resampling, acquisition offsets and
shared neighboring structure can also affect these comparisons. This remains
a source-calibration ambiguity, not a confirmed indexing repair. Resolve it
using independent generation metadata or identified native landmarks before
changing either source's physical mapping.

## Partial-overview source-label contract

The low loader preserves an array slot for every requested plane; with explicit
`allow_partial=True`, a missing plane becomes a zero-filled slot. Metadata
`requested` therefore maps array positions, while `loaded` lists only available
source IDs. The concurrent renderer still uses `loaded[z]` for its center label
and passes `loaded` to trace/stack labeling. If a plane before the center is
missing, these labels shift. For example, requested `[10,11,12,13]`, loaded
`[10,12,13]`, missing `11`, and array center `z=2` describe actual plane `12`,
but `loaded[2]` labels it as plane `13`. The locator may then read the wrong
full section, and trace/stack source IDs can be wrong or fall back to local
array indices.

Use `requested` or an explicit full slot-to-source map for all center, trace,
stack and locator identities; retain `loaded`/`missing` separately for coverage.
Reject a metadata/volume-length mismatch and test a missing pre-center plane.
This is a potential partial-data contract failure, not an observed defect in
the copied batch: all 299 successfully saved context records have no recorded
missing planes and none reports `context_complete=False`. The current standard
low loader call also defaults to rejecting partial acquisitions; a future
partial tiled-context path must keep correct source-slot provenance.

## Source fingerprints

Paths are under `group_analysis/visual_review_20261002/<sample>/<neuron>/`.
The following SHA256 hashes identify the unchanged compressed NIfTI files:

| UID | context.nii.gz SHA256 | soma.nii.gz SHA256 |
| --- | --- | --- |
| 252385\|003.swc | `299ed9be8883ea8a43b2f2655079c43c2c6d1d2b679d338cdb75329f2f4c4471` | `4307cce667680646dce41e481b7897d0cd209787c7869fa0a72b9f22ccaa7c5b` |
| 252527\|048.swc | `3e32bce6b9af7b43be5b15061c1ce20f22a8ab0ed882b1654d5216072c8a4036` | `aa93dd40447f82de55756f1d0059e178c4eef8333e2dc5cf451a521f2997553d` |
| 252384\|008.swc | `db1d3e33c830e7cac6b3834aca7eb104f8d87d88f83f91a4895ae0e7e4a398ab` | `1dfcdbea716b6249f8c5ec48704eb481016b80c9b19dbba31a1c6e6fc6ec3328` |
| 252714\|002.swc | `b6c7937f20ce33a3f6f450af0b03d2f0a1dd083fb455e8da3ea6c4cca321b039` | `b9ff973489db82dfb996bbc1f266e7ec35d14e31e46a39e6d6d13269d92d743e` |

Provenance JSON hashes recorded at the end of each comparison:

- 252385/003: `0302259060a5b138edd272c367e217262758fe62be2c63a5ac1eed6c4c4bf3c1`
- 252527/048: `46dabd19b457703d42fc48c257d88b355ee84b0088fe84764fe6f5318029be7a`
- 252384/008: `0803adf4a43bf38d0673ba9c43a4fc535a58cc14a01754c3bccdbfe55bd20c9b`
- 252714/002: `03fba3107d275032cdf6709ae660b1125453a32ecd67fddd27722df6560390a0`

## Numerical reproduction

Run with the projectome Python interpreter and `-B`. This writes no files.

```python
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import spearmanr

root = Path("D:/projectome_analysis/group_analysis/visual_review_20261002")
for sample, neuron in [("252385", "003"), ("252527", "048"),
                       ("252384", "008"), ("252714", "002")]:
    folder = root / sample / neuron
    p = json.loads((folder / "provenance.json").read_text())
    c = nib.load(str(folder / "context.nii.gz"), keep_file_open=True)
    s = nib.load(str(folder / "soma.nii.gz"), keep_file_open=True)
    co, so = c.affine[:3, 3], s.affine[:3, 3]
    cs, ss = np.diag(c.affine)[:3], np.diag(s.affine)[:3]
    cz = int(np.floor(p["context_local_xyz"][2]))
    sz = int(round((co[2] + cs[2] * cz - so[2]) / ss[2]))
    sigma = cs[:2] / (2 * ss[:2])
    margin = np.ceil(3 * sigma).astype(int)
    gx = co[0] + cs[0] * np.arange(c.shape[0])
    gy = co[1] + cs[1] * np.arange(c.shape[1])
    ix = np.flatnonzero((gx >= so[0] + margin[0] * ss[0]) &
                       (gx <= so[0] + (s.shape[0] - 1 - margin[0]) * ss[0]))
    iy = np.flatnonzero((gy >= so[1] + margin[1] * ss[1]) &
                       (gy <= so[1] + (s.shape[1] - 1 - margin[1]) * ss[1]))
    xx, yy = np.meshgrid((gx[ix] - so[0]) / ss[0],
                         (gy[iy] - so[1]) / ss[1], indexing="ij")
    low = np.asarray(c.dataobj[:, :, cz], dtype=float)[np.ix_(ix, iy)]
    for offset in [-2, -1, 0, 1, 2]:
        fine = np.asarray(s.dataobj[:, :, sz + offset], dtype=float)
        smooth = gaussian_filter(fine, sigma=sigma, mode="reflect")
        mapped = map_coordinates(smooth, np.array([xx, yy]), order=1,
                                 mode="nearest", prefilter=False)
        raw = np.corrcoef(low.ravel(), mapped.ravel())[0, 1]
        log = np.corrcoef(np.log1p(low).ravel(), np.log1p(mapped).ravel())[0, 1]
        rank = spearmanr(low.ravel(), mapped.ravel()).statistic
        print(p["UID"], offset, raw, log, rank)
        del fine, smooth, mapped
```
