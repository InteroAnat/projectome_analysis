# fMOST image loader handoff

**Agent: Codex | Date: 2026-10-02 (Asia/Shanghai)**

The user requested handoff to Cursor with Grok 4.7 Ultra. Work stopped after
the focused loader fixes, regression checks, and one cached-image smoke check.
No bulk production run, commit, push, journal update, or note promotion occurred.

## Files changed in this loader task

- `main_scripts/Visual_toolkit.py`: sample-safe low-resolution sources;
  deterministic repository cache/output roots; complete TIFF/PIL page decoding;
  missing-data accounting; exact low-resolution slice counts; native coordinate
  export with correct XYZ spacing and micron units; finite contrast handling;
  SWC segment clipping to the displayed XY field and Z slab.
- `main_scripts/fmost_image_geometry.py`: testable native coordinate arithmetic,
  crop bounds, affine, segment clipping, and contrast normalization.
- `tests/test_fmost_image_loader.py`: 12 focused regression tests.
- `notes/bulk_visual_review_20261002/loader_smoke.json`: persisted real cached-cube
  evidence, including input SHA256 and acquisition geometry.

The old public and subsidiary bulk scripts were read but not edited by this
worker. Other changes from the parent task remain in the shared checkout.

## API for the next renderer

`Visual_toolkit(sample_id, cache_dir=None, low_res_dir=None, low_res_ssh_base=None)`
accepts an explicit own-sample low-resolution directory. Only sample 251637
receives the legacy 251637 SSH fallback automatically; other samples never do.
For other samples, the two existing hyphen/underscore CH1 folder conventions
are checked when no explicit directory is supplied.

`get_high_res_block(center_um, grid_radius=1)` and
`get_low_res_widefield(center_um, width_um, height_um, depth_um)` keep the existing
`(volume_zyx, origin_xyz_um, spacing_xyz_um)` return contract. The high-resolution
method's existing default radius remains 2, so callers should explicitly use 1
for a single cube. Both methods default to rejecting any missing data. Explicit
`allow_partial=True` permits zero-filled missing neighbors, with metadata, but
never a missing central cube/section.

`last_high_res_metadata` and `last_low_res_metadata` expose requested, loaded,
missing, source paths, nominal spacing, origin, shape, completeness, and the
native index convention. Metadata is populated before central/missing-data
errors. Whole-section reads can use `_get_low_res_slice_path(z_index)` and
`Visual_toolkit._read_tiff(path)`. Export returns the saved path and writes a
JSON sidecar; `last_export_metadata` records that export.

## Coordinate contract and acceptance limits

Raw SWC XYZ values are passed through by `SwcLoader`; the existing toolkit and
GUI treat them as native microscopy micrometres. Pixel coordinates are
`(native_xyz_um - origin_xyz_um) / spacing_xyz_um`. Arrays are ZYX, TIFF sections
are YX, and exported NIfTI arrays are XYZ. No Y reflection is introduced.
Nominal high spacing is 0.65/0.65/3 micrometres; nominal low spacing is 5/5/3.
Z section filenames use the repository's one-based convention, with native
Z equal to index times nominal spacing. These spacings and the origin offset
still need independent image/metadata calibration; no anatomical orientation,
hemisphere, cortical layer, or region boundary was verified here.

## Validation

`C:/Users/laika_yan/miniconda3/envs/projectome/python.exe -B -m unittest discover -s tests -p test_fmost_image_loader.py -v`
passed all **12** tests. AST checks and the CRLF-aware Git whitespace check passed.
The tests include exact soma pixel placement, missing-central-data rejection,
explicit partial gaps, image-boundary clipping, sample-safe SSH behavior,
malformed HTTP payload rejection before caching, multipage fallback/file-handle
closure, micron NIfTI affine readback, and Z-slab segment clipping.

A read-only public-method smoke used existing cached real image
`resource/cubes/252385/high_res_http/63/220_135_63.tif` at an artificial cube-centre
coordinate. Result: ZYX **90 × 360 × 360**, uint16, range **0–61098**, nonzero
fraction **0.7201908436**. This validates decoding and coordinate arithmetic for
that cached cube; it does not establish soma correspondence or fresh server
availability. No new remote image fetch was requested.

## Outstanding work for Cursor

- Independently review these changes and run the focused/full applicable tests.
- Verify raw-SWC-to-image correspondence, sample-specific spacing, and Z origin
  from real soma landmarks and acquisition metadata before anatomical claims.
- Finish the parent's review-cohort selection, bounded raw-SWC fetching, native
  soma/context/whole-section panels, per-neuron provenance, and review gallery.
- Confirm every new sample's own low-resolution directory; missing data should
  remain an explicit failure or annotated partial observation.
- Production bulk generation and manual Henry-like cortical/region corrections
  remain outstanding. The legacy ION raw fetcher's unbounded retry loop remains
  outside these loader changes.
