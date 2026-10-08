# ARM-only source groups, 2026-10-09

These new manifests retain exactly the same 436 main and 26 preview neurons, source bytes, animal identities and evidence metadata. Only the primary `Subregion` grouping changes: it is now a direct fresh lookup of the stored SWC root in the pinned NMT v2.1 symmetric **ARM level 6** volume. No other atlas or inferred hierarchy parent is used for source classification. Existing inputs, maps, figures and canonical tables remain unchanged.

`main/`, `preview/` and `combined/` each contain `projection_manifest.csv` and a matching `label_map.csv`. Combined contains all 462 unique neurons across eight established animals; `SourceCohort` preserves the original main/preview membership. All original non-grouping fields are retained exactly, including source labels, human review, candidate flags, anatomical status and registration uncertainty. `EvidenceSourceGroup` preserves the original grouping. These fields are provenance, not primary region names in this variant.

The exact common display schema is:

`Subregion, ARMLevel, ARMIndex, ARMAbbreviation, ARMFullName, Hemisphere, DisplayLabel, AtlasSHA256, AtlasKeySHA256, AtlasPath, AtlasKeyPath, SourceARMStatus`.

`Subregion` is a safe stable ID such as `ARM6_542_R`. `ARMFullName` is the verbatim official key Full_Name, including its C/S and hemisphere prefix. `DisplayLabel` is exactly that full name plus `(Left)`, `(Right)` or `(Unknown)`. For example, `CR_lateral_agranular_insula_area (Right)` replaces an evidence-stratum display name. The original ARM key spelling is preserved, including `CR_precentral_operular_area`; this is not a new terminology correction.

Fresh lookup yields 410 mapped roots and 52 background roots in 17 hemisphere-specific groups. The main set has 410 mapped/26 background; all 26 preview roots are background under the retained current policy. Background has ARMIndex0, empty abbreviation, full name `Atlas background`, and status `zero_unassigned`. Its mask hemisphere is retained. The preparer separately supports outside roots as ARMIndex−1, empty abbreviation, full name `Outside reference`, hemisphere Unknown and status `out_of_FOV`; none occurs in this actual selection. Missing/unmapped positive key entries and level conflicts fail before output creation. Background is not silently promoted to insula or classified as white matter.

The full names describe current software lookup, not independent anatomical acceptance. Henry's coarse visual annotations, atlas-root results, portal labels and candidate evidence can disagree and remain separate. The full names include gustatory cortex, claustrum, precentral operular area and secondary somatosensory cortex where the actual ARM lookup assigns them; they are not renamed INS.

## Coordinate and source evidence

The ARM file is `atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz`, SHA-256 `8f7f1dec9fe6ccd1b6c8fce45e3b2cbc7d542ce4a4256500ee562e7e25ddbac7`. The official ARM key SHA-256 is `be343eacb2418f2493f740f4039adf068c110a3cb71465645cedadcf2866c972`. Reference and LR-plane hashes, geometry and independently label-validated hemisphere semantics are in `preparation_provenance.json`. The reference is the existing symmetric population NMT MRI, not an animal's native fMOST brain.

Every selected SWC was freshly hash-checked and its full seven-column topology validated. Each exact root was read again independently with NumPy. Index-encoded XYZ is divided by its declared scale (250 µm/index for these sources); it is not literal NMT world-mm. Primary source lookup retains `np.rint(index)` with ties to even. Registration/export origin remains unverified.

`independent_readback_v2.json` passed exact selection, source hashes, all full key names/indices, hemisphere, all twelve label-map fields and original metadata preservation for 462 roots. On the same pinned ARM, published `ceil(index)−1` changes 400 root voxels and 60 labels. Half-open `floor(index+0.5)` changes one voxel but zero labels. There are two exact half-integer roots: 251637/459 (rint and half-open both54 on X) and 252790/115 (X72 versus73; both background). The observed zero label differences is specific to these 462 roots; the rounding policies remain distinct. Alternatives are recorded as sensitivity only and never change primary grouping.

Thirteen focused preparation tests pass. They cover exact channel identities, stale hashes, selection overlap, metadata preservation, index-vs-world coordinates, missing positive labels, geometry/level conflicts, unresolved roots, half ties, explicit policy and no-clobber behavior. These are software checks, not anatomical acceptance.

## New map commands for ROOT

The following commands are supplied for a new variant; they have not been executed by this preparer. Set `$variant` to `main`, `preview` or `combined`. Every map, review and figure destination must be new. Do not rewrite provenance of previous maps to current producer hashes.

```powershell
$python = 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe'
$base = 'group_analysis/evolution_20261008'
$variant = 'combined'
$source = "$base/projection_inputs/arm_labels_20261009/$variant"
$nmt = 'atlas/NMT_v2.1_sym/NMT_v2.1_sym'
$reference = "$nmt/NMT_v2.1_sym_SS.nii.gz"
$brain = "$nmt/NMT_v2.1_sym_brainmask.nii.gz"
$endpoints = "$base/endpoint_maps/arm_labels_20261009_$variant"
$axons = "$base/projection_maps/arm_labels_20261009_$variant"

& $python -X utf8 -B 'group_analysis/scripts/build_endpoint_maps.py' @(
    '--manifest', "$source/projection_manifest.csv", '--reference', $reference,
    '--brain-mask', $brain, '--atlas-path', "$nmt/ARM_in_NMT_v2.1_sym.nii.gz",
    '--atlas-key', 'atlas/ARM_key_all.txt',
    '--hemisphere-mask', "$nmt/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz",
    '--output', $endpoints
)
if ($LASTEXITCODE -ne 0) { throw 'ARM endpoint build failed' }
& $python -X utf8 -B 'group_analysis/scripts/review_endpoint_run.py' @(
    '--run', $endpoints, '--output', "${endpoints}_review"
)
if ($LASTEXITCODE -ne 0) { throw 'ARM endpoint review failed' }
& $python -X utf8 -B 'group_analysis/scripts/build_projection_maps.py' @(
    '--manifest', "$source/projection_manifest.csv", '--reference', $reference,
    '--brain-mask', $brain, '--output', $axons
)
if ($LASTEXITCODE -ne 0) { throw 'ARM axon-length build failed' }
& $python -X utf8 -B 'group_analysis/scripts/review_projection_run.py' @(
    '--run', $axons, '--output', "${axons}_review"
)
if ($LASTEXITCODE -ne 0) { throw 'ARM axon-length review failed' }

& $python -X utf8 -B 'group_analysis/scripts/render_projection_slices.py' @(
    '--run', $endpoints, '--readback', "${endpoints}_review/endpoint_run_readback.json",
    '--label-map', "$source/label_map.csv", '--metric', 'endpoint-density',
    '--scope', 'group', '--cut-policy', 'fixed', '--slice-voxels', '56', '200', '87',
    '--output', "$base/figures/arm_labels_20261009_${variant}_endpoint_groups"
)
if ($LASTEXITCODE -ne 0) { throw 'ARM figure rendering failed' }
```

For axon-labelled length figures, use `$axons`, `${axons}_review/projection_run_readback.json` and `--metric axon-density` with the same matching label map and another new figure destination. Existing reference coordinates and numerical kernels remain unchanged. Full-label plot layout must be checked on the newly rendered variant. Source grouping is ARM-only; biological terminal review and statistical inference remain unperformed here.
