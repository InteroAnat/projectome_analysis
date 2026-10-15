# Cursor handoff: fMOST review and bulk native-image correction sources

Agent: Codex | Date: 2026-10-02 (Asia/Shanghai)

## User-authorized objective

Finish the code review, including methods literature using Zotero, then use
the refined code to produce bulk visuals for new monkeys. Prioritize insula,
PrCO, unknown and potentially insular regions, especially monkeys with many
insula neurons. These visual sources should support a Henry-style human
correction table. The user has explicitly requested handing this job to
Cursor using Grok 4.7 Ultra.

The user clarified: **only the shown resampled datasets have been copied**.
Produce the complete available-source review set for **252384, 252385, 252527
and 252714**, and retain a source-pending candidate queue for the other new
monkeys. The copied 251637 dataset is a reference, not a new monkey.

## Checkout and preservation

- Workspace: `D:/projectome_analysis`.
- Current branch: `codex/fmost-methods-review-20261002`.
- Base HEAD: `78008cbaab71bf737dc7129c72e5fe284c3ba40f`.
- All changes are local and uncommitted. No commit, push, PR or publication is
  authorized. Preserve unrelated dirty files, untracked datasets, agent
  journals and tracked bytecode; do not reset, clean or stage the whole tree.
- Use command-scoped `git -c safe.directory=D:/projectome_analysis` on this host.
- Work directly in this checkout to retain the uncommitted refined code.
  A new clean worktree will not contain these changes.
- Do not modify source images, canonical cohort files or human correction
  tables. Keep new outputs separate. Do not expose credentials or private
  library notes, or perform unrelated profile synchronization.

## Completed review evidence

- `docs/project_note.md`: current cohort, scientific limitations and clustering
  decisions; new agent entries must retain author/date stamps.
- `notes/repo_review_20261002.md`: detailed repository/data audit.
- `notes/fmost_methods_research_20261002.md`: 29 annotated primary/official
  references, eight saved Zotero PDF method verifications.
- `notes/fmost_review_20261002.bib`: eight selected Zotero records, no private
  notes/tags/attachment paths.
- `notes/clustering_review_20261002/README.md`, diagnostics and
  `validation_summary.json`: six real-data public CLI runs, 33 passing focused
  tests, verified assignments/input/source hashes, unchanged cohort workbooks.
- SWC validator compatibility: 358 existing files and 4,882,379 nodes passed;
  this structural audit is not tracing/anatomical acceptance.
- `notes/bulk_visual_review_20261002/visual_methods.md`: focused native image,
  coordinate/channel and anatomical-review requirements, verified against five
  relevant saved Zotero PDFs. Reuse this work, supplement only material gaps.

## Refined code and current loader changes

Previously completed: `main_scripts/clustering_validation.py`,
`main_scripts/fnt_dist_clustering.py`, `main_scripts/swc_validation.py`,
`main_scripts/neuro_tracer.py`, `main_scripts/region_analysis/population.py`,
and their tests. Clustering now checks exact identities/complete pair coverage,
uses explicit representations and guided mode, and exports stability/cut
diagnostics. There is still no scientifically validated k.

The current image-loader work is in `main_scripts/Visual_toolkit.py` and
`main_scripts/fmost_image_geometry.py`. Read
`notes/bulk_visual_review_20261002/loader_handoff.md` and the final test output
before relying on it. The existing public tuples remain `(volume_zyx,
origin_xyz_um, spacing_xyz_um)`.

New loader interfaces: constructor `low_res_dir=` and `low_res_ssh_base=`;
`get_high_res_block(..., allow_partial=False)`;
`get_low_res_widefield(..., allow_partial=False)`;
`last_high_res_metadata` / `last_low_res_metadata`. They record requested,
loaded/missing image coverage and provenance. Central missing data always
raises. Partial neighbor filling is explicit and must remain visible.

The previous defects included silent zero-filled missing images, a legacy SSH
fallback hardwired to 251637, cwd-dependent caches, incorrect native NIfTI
affine/units, off-by-one depth and SWC overlays extending outside the displayed
Z slab. Verify the current fixes independently before rendering in bulk.

## Server sources

`X:` maps to `\\10.102.8.200\microscopy_data`.
The source root is
`\\10.102.8.200\microscopy_data\fMOST\5micron_datasets`.

| Sample | Relative directory |
|---|---|
| 251637 reference | `251637-CH1_resample/resample_5um` |
| 252384 | `252384_CH1_resample/resample_5um` |
| 252385 | `252385-CH1_resample/resample_5um` |
| 252527 | `252527_CH1_resample/resample_5um` |
| 252714 | `252714-CH1_resample/resample_5um` |

Planes are single-page uint16 YX TIFFs with LZW compression, named
`{sample}_{z:05d}_CH1_resample.tif`. The toolkit declares **5 x 5 x 3 um** XYZ
spacing, not 5 um isotropic. TIFF headers have no useful calibration
description; treat spacing/channel information as declared until supported
by specimen metadata. Do not infer anatomical laterality from acquisition
axes. No CH2/PI source has been verified in this copied root.

High-resolution cubes use the existing HTTP source declared in the toolkit,
nominal `[0.65,0.65,3]` um spacing and `[360,360,90]` XYZ block dimensions.
Verify availability and correspondence on real smoke cases before batching.
Do not silently substitute another sample's image source.

## Candidate selection and completeness

- Live bounded portal snapshots for all nine new samples are saved in
  `notes/bulk_visual_review_20261002/portal_snapshot/`, with timestamp and SHA256
  manifest. Use these to identify omissions from old step1 workbooks.
- Latest step1 sources: `group_analysis/step1_results/{sample}_*_region_analysis/
  tables/{sample}_results_*.xlsx`, `Summary` sheet.
- Candidate/refinement evidence:
  `group_analysis/staging_20260926/recovery/all_refined_neurons.csv` and
  `group_analysis/staging_20260926/combined/multi_monkey_INS_combined_harmonized.xlsx`.
- Coordinate screening reference:
  `group_analysis/staging_20260926/reference/251637_subregion_bboxes_folded.csv`.
  This is an NMT spatial candidacy test, not anatomical membership.
- Include atlas insula leaves, PrCO, unknown/unmapped labels, staging keepers
  including G, relevant adjacent labels (e.g. Ri/operculum/claustrum), and
  coordinate-screened possible-insula cases. Preserve all selection reasons
  and original labels. Do not simply run the canonical `--potential-ins` mode:
  it reads the older 353-row table and omits new staging samples.
- Historical INS/PrCO/Unknown + staging union has 324 candidate identities;
  copied four-sample subset has **154** before expanding adjacent labels or
  live-list omissions: 252385=69, 252714=31, 252527=30, 252384=24. These are
  baseline counts, not final completion targets. Review the inventory note
  for current counts and exact source identity reconciliation.
- Live lists expose historical gaps: 252384=100 vs99 processed;
  252383=140 vs127; 251730=764 vs79. Do not treat metadata tracing_count as
  current list length. Preserve explicitly unprocessed/unresolved rows.
- `selectNeurons.region` may be blank without meaning atlas-Unknown. Preserve
  step1 atlas labels and distinguish absence of portal metadata. Portal soma
  XYZ matches NMT physical coordinates, **not raw microscopy coordinates**.
- `getSampleInfo` can return multiple entries (250432/251730); select exact
  `fMOST_id`, not blindly `info[0]`, before taking project_id.
- Raw native coordinates must come from validated **raw SWC roots**, fetched
  with bounded retries or verified own-sample raw caches. The raw route is
  `http://10.10.31.31/swc/newswc/{project_id}/{sample}/swc_raw/{NeuronID}`.
  Validate with `parse_swc`; never crop native TIFFs using NMT soma coordinates.

## Remaining implementation and production work

1. Independently review the loader fixes and finish meaningful regression
   tests. Avoid the unbounded retry methods in legacy IONData.
2. Build/reconcile an explicit identity manifest for all available-sample
   candidates, rank batches by insula yield, retain reasons/atlas/refined label
   provenance, and save the other-sample queue with source-pending status.
3. Run real-image smoke cases per available sample and inspect the outputs:
   soma in the expected image bounds, raw SWC/image agreement, nonzero real
   central planes/cubes, correct source sample, field-of-view/depth units,
   missing coverage, slab-clipped overlays and usable contrast.
4. Render the whole available cohort with resumable outputs. Provide native
   section/context and soma detail; preserve a clean unmarked view alongside
   marked/trace views. Add orthogonal/stack evidence where useful. At 5 um,
   context alone does not establish fine axon joins or cytoarchitecture.
5. Produce a browsable gallery/index and a correction-ready table with stable
   SampleID/NeuronID/UID, image links, source files/slices, native coordinates,
   declared resolution, SWC/image provenance, coverage/error status and blank
   human region/layer/reviewer/date/notes fields. Keep existing Henry and
   canonical tables intact. Do not automatically accept atlas or inferred
   labels as human corrections.
6. Verify the persisted output set against the complete candidate manifest,
   inspect representative images across every sample/category plus failures,
   and record all unresolved cases. Update the living project note with an
   author/date stamp and exact produced/pending counts.

Suggested isolated output root:
`D:/projectome_analysis/group_analysis/visual_review_20261002`.
Existing historical server visuals under `fMOST/visual/` can be evidence but
must not be called freshly generated by this refined code.

## Environment and verification

Python: `C:/Users/laika_yan/miniconda3/envs/projectome/python.exe`.
It has numpy/pandas/scipy/scikit-learn/nibabel/tifffile/matplotlib and unittest;
do not install another environment unnecessarily. Use `-B` to avoid bytecode.

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B -m unittest discover -s tests -v
git -c safe.directory=D:/projectome_analysis -c core.whitespace=cr-at-eol diff --check
```

Do not equate passing software tests or pretty plots with anatomical/tracing
acceptance. Preserve cohort hashes; do not rerun the default harmonizer, which
would replace the current 353-row harmonized table from its 306-row raw input.

## Ownership at handoff

Codex has stopped expanding implementation scope on user request. Bulk image
production has **not yet been run** with the refined loader. Cursor is to
continue from the saved evidence through the full available-source deliverable;
the code-review/literature phase is largely complete, production and independent
verification remain outstanding.
