# ARM-only NMT map labels: independent scope and resource check

2026-10-09. Read-only resource/code review for the user's requested ARM-only map variant. Historical maps and their declared source groups remain historical; this note does not reassign neurons, change coordinates, or establish anatomical acceptance.

## Exact resource binding

Use the existing **symmetric NMT v2.1, 0.25 mm** reference and its accompanying ARM. The displayed grayscale background is the skull-stripped population MRI, not an individual monkey's native image and not an atlas label image. An ARM contour/label overlay, if requested, is a separate categorical layer sampled from this same grid. The brain mask currently sets the grayscale intensity window and accounts for coverage; it does not trim endpoint or trajectory measurements.

All paths below are relative to `D:/projectome_analysis` and were freshly hashed. They match the bindings in `group_analysis/evolution_20261008/endpoint_maps/multimonkey_coarse_20261009/run_provenance.json`.

| Role | File | SHA-256 |
|---|---|---|
| MRI background/reference | `atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz` | `9e37a94c4b9e5865aabb9fd3b51dcc3b2cb16f3daed39c967e68acf16cf92bee` |
| Only parcellation image | `atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz` | `8f7f1dec9fe6ccd1b6c8fce45e3b2cbc7d542ce4a4256500ee562e7e25ddbac7` |
| Only label key | `atlas/ARM_key_all.txt` | `be343eacb2418f2493f740f4039adf068c110a3cb71465645cedadcf2866c972` |
| Laterality check | `atlas/NMT_v2.1_sym/NMT_v2.1_sym/supplemental_masks/NMT_v2.1_sym_LR_plane.nii.gz` | `29aab4477ffcf2db2a190b5572c92bbf224cc144628d9e89367eb1645aab8aea` |
| Coverage/window mask | `atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_brainmask.nii.gz` | `c763cf9a54362eba55fc8d4a08d93d3487f56dc1e9fd821b87c2a0748320738d` |

The MRI/masks have spatial shape `(256,312,200)`; ARM is **actually** `(256,312,200,1,6)`, int16. Each file codes spatial units as mm and qform/sform codes as 5. Spatial affines agree: diagonal `(0.25,0.25,0.25)` mm and translation `(-31.875,-27.75,-8)` mm. The voxel volume is 0.015625 template mm³. The NIfTI descriptions are blank: filenames plus the hashed package documentation establish the declared resource version, rather than an embedded acquisition date or source registration history.

`atlas/NMT_v2.1_sym/readme.md` explicitly defines ARM as the fusion of CHARM and SARM in this distribution. Thus ARM-only retains its cortical **and** subcortical labels; it does not mean cortex-only. No separate CHARM/SARM image, hierarchy CSV, D99 image or alternate atlas is required for these assignments. The package asks ARM users to cite both [Jung et al., 2021, NeuroImage 235:117997](https://doi.org/10.1016/j.neuroimage.2021.117997) and [Hartig et al., 2021, NeuroImage 235:117996](https://doi.org/10.1016/j.neuroimage.2021.117996). The [official AFNI installation documentation](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/programs/alpha/%40Install_NMT_sphx.html) also distinguishes template versions and symmetric/asymmetric variants. These references describe the template/atlas; they do not validate a particular neuron's transform.

## Exact label and grouping contract

The pinned key has 1,124 rows and unique `Index`, `Abbreviation` and `Full_Name` values. All abbreviations and full names preserve matching `CL_`, `CR_`, `SL_` or `SR_` prefixes (237 cortical labels per side, 325 subcortical labels per side). Preserve both domain and side: stripping these prefixes can merge unrelated regions. Use the integer sampled from the **chosen actual ARM volume** to look up one exact key row. Do not assign a parent by text matching or infer a parent from `First_Level`/`Last_Level`; those fields describe level availability, not parent identifiers.

For a finest-level source grouping, declare level 6 explicitly and bind each group by `(atlas_sha256, level=6, index)`. For another requested scale, create a separately declared level-specific variant. Do not combine levels into an overlapping feature partition. Observed nonzero label counts by actual volume are 18, 64, 142, 250, 512 and 684 at levels 1–6. All observed indices have key rows. All 684 level-6 labels satisfy their key availability ranges.

Keep the following separate:

| Field | Meaning/example |
|---|---|
| `Subregion` or equivalent safe group ID | `ARM_L6_idx0229`; stable filesystem/group key only |
| `ARMLevel`, `ARMIndex` | `6`, `229`; the actual sampled volume and scalar label |
| `ARMAbbreviation` | Exact `CL_Ig` |
| `ARMFullName` | Exact `CL_granular_insula` |
| Display text | `CL granular insula` with `ARM level 6 · index 229` below it |
| Source selection/provenance | Original human/portal/candidate evidence, retained independently |
| Assignment and review status | Mapped/unassigned/outside/conflict plus unresolved coordinate/registration/anatomy review |

The display change above only replaces underscores with spaces. Preserve literal key spelling in the source field: e.g. index 49 is `CL_precentral_operular_area`, including the supplied spelling. A caption can define CL/CR as left/right cortical and SL/SR as left/right subcortical; never drop domain and side from identity. Other examples are 228 `CL_agranular_and_dysgranular_insula`, 729 `CR_granular_insula`, and 123 `CL_rostral_inferior_parietal_lobule_area_7b_(PFG/PF)`. Full names contain slashes and parentheses, reach 52 characters, and cannot safely substitute for manifest IDs or filenames. Wrap display text at spaces and enlarge panels as needed; do not truncate the sole region identification. Retain exact full name and index in the figure provenance/legend even when a compact title is needed.

New ARM-only source grouping must use the sampled soma label under an explicit coordinate policy, not `HumanINS`, `AtlasINS`, `G`, `Candidate`, `OriginCandidate` or `NearINS` as atlas names. Those old values remain provenance about selection and historical strata. Human coarse-INS evidence can remain valid as human evidence while a sampled ARM label disagrees; do not overwrite either. Never relabel PrCO, parainsula, white matter or unassigned voxels to INS merely to keep a neuron in an INS display group. Unassigned/out-of-FOV/conflict cases need an explicit ledger; a software status is not an invented atlas full name. Whether those cases receive a separate non-anatomical QC map or remain unavailable must be stated, rather than silently dropping them.

## Independently observed metadata conflict

The actual level-4 and level-5 volumes each contain 1,641 voxels of index 45 (`CL_AON/TTv`, `CL_anterior_olfactory_cortex`) and 1,641 voxels of index 545 (the right-sided equivalent), while both key rows say `First_Level=6, Last_Level=6`. This is a pinned image/key availability contradiction, **not** an absent key row and not evidence of incorrect neuron anatomy. Existing `EndpointAtlas.lookup` reports `key_level_conflict` and preserves the observed index and names. Keep that behavior and record the conflict; do not silently edit the atlas/key or pretend that lower-level lookup passed. This conflict does not affect the level-6 key availability check. A repaired upstream label table would require an explicit separately versioned resource binding.

## Existing implementation interfaces and scientific limits

`build_projection_maps.py` and `build_endpoint_maps.py` group by `(AnimalID,Subregion)` and then average contributing animal maps equally within that declared source group. Their `checked_manifest` only accepts safe alphanumeric/underscore/hyphen group IDs. Therefore a new source-label adapter plus a bound display-name mapping is appropriate; replacing `Subregion` with literal full names will fail for genuine ARM rows. Endpoint target labels already come directly from all six actual ARM volumes in `main_scripts/endpoint_atlas.py`; no inferred hierarchy is needed.

`render_projection_slices.py` currently displays `entry['Subregion']` directly and records it in figure provenance. It needs an explicit, hash-bound full-name lookup for the new ARM IDs; the display layer must not silently translate the old historical strata into ARM names. Its title-overlap check should remain. The current map renderer provides grayscale NMT MRI plus continuous map color, **not a labeled ARM boundary overlay**. Source-title replacement and optional ARM parcel outlines are distinct changes. Any parcel outlines must use categorical nearest/voxel lookup at the declared single level, preserve the MRI grid, and identify that level in the legend. Dense labels should go in a companion keyed legend/table instead of covering the anatomy.

For a future source lookup intended to match endpoint/trajectory voxel allocation exactly, use the kernels' convention: zero-based reference voxel centre index `i` owns `[i-0.5,i+0.5)` and the reference FOV is half-open. `np.rint` differs at half-integer ties and must not be claimed generically equivalent. The actual prepared source manifest explicitly retains the pre-existing `current_rint_zero_center` primary policy; the cohort-specific reconciliation below documents its result without changing that policy. World-mm input must go through the inverse coded reference affine; index-encoded input must preserve its declared per-axis scale. Validate masks against this grid and preserve label-side/mask-side conflict separately. The existing finest-label contingency assigns LR mask value 1 to R and 2 to L; no inference from numeric mask value alone or undocumented axis flips is justified.

The current metrics remain descriptive: candidate graph-leaf endpoint count/density or neuron occupancy, and axon-labelled segment length/density. Changing the atlas text does not turn graph leaves into verified terminal arbors, boutons or synapses; it does not change denominators, resolve tracing completeness, validate source compartments, establish native-to-NMT registration, or support inferential t/p maps. Preserve the endpoint conditional denominator and the distinction between unassessed neurons and observed zero in-FOV candidates. Missing animal/source groups remain missing rather than zero-imputed.

The parsed Gou Methods specify NMT v2.0 and deposited code loading CHARM/SARM, whereas these local assets are explicitly pinned NMT v2.1 ARM. The local Methods/code review documents a possible half-voxel origin difference and incomplete cached-export lineage. Identical array dimensions, a full ARM name, or an in-bounds soma cannot settle that provenance. See `group_analysis/evolution_20261008/references/fmost_full_methods_20261009.md` (SHA-256 `2e9d7ff0bd8a93713d80ecd5d031c77d41303065bae4e508c5fff26aa5bb9b38`). Keep registration/origin acceptance unresolved and retain sensitivity checks rather than selecting a coordinate convention by anatomical fit.

## Actual 462-neuron manifest reconciliation

After the resource review, the separately prepared delivery at `group_analysis/evolution_20261008/projection_inputs/arm_labels_20261009/combined/` became available. The independent `test_arm_actual_manifest_20261009.py` does not import the preparer, production graph parser or production atlas lookup. It freshly hashes all selected SWC bytes, scans every SWC parent column for its unique root, reads raw root ID/XYZ, samples the pinned ARM level-6 volume and LR mask, resolves the exact key row, and checks all saved primary/alternative source-label fields plus the group/full-name/display-name binding. It also verifies the hashes of every preparation input and output, exact union of the original 436 main neurons and 26 disjoint preview neurons, unchanged source/animal/evidence fields, and per-row atlas/key/reference bindings.

**All three actual-delivery tests passed** (exit 0; 7.140 seconds). The immutable combined manifest SHA-256 is `6bffc5ba0380db0ea0caddf85c188af7b73c7409a274aa53e588cba0272ccdd8`; its label-map SHA-256 is `e7ada68fc3c3f67147811efcde3708b541d7de38f843884eab3cfc75c0bc793b`; preparation provenance SHA-256 is `981857a17995d32aa5f5f697b19c6431e359ff57ab73aa10002e129ee37bdede`. The 462 unique neurons retain eight established animals. There are 410 mapped source somata and 52 explicitly `zero_unassigned` somata (26 main, 26 preview), distributed among 17 saved source groups. `ARM6_0_L/R` and displayed `Atlas background (Left/Right)` are explicit non-parcel QC groups, not named regions from the key or evidence of INS/white-matter membership. The actual saved mapped-group ID convention is `ARM6_<index>_<side>`; it is an equally lossless safe-ID variant of the example proposed above.

| Comparison against retained primary `np.rint(XYZ/250)` | Different source labels | Different source voxels | Different hemispheres |
|---|---:|---:|---:|
| Half-open zero-centre cell lookup, independently implemented by boundary search | 0/462 | 1/462 | 0/462 |
| Published edge-origin sensitivity, `ceil(XYZ/250)-1` | 60/462 | 400/462 | 0/462 |

The one rint/half-open voxel difference is `252790|115.swc`: source root XYZ `(18125,38005.25,28608.875)` µm gives X index **72.5**. Ties-to-even selects voxel `(72,152,114)`; half-open cell ownership selects `(73,152,114)`. Both cells have ARM index 0, so the saved source label and side are unchanged. This actual-cohort label agreement does **not** establish generic rounding equivalence or guarantee agreement for future somata, target endpoints, masks, or other atlas levels. The regression includes exact half-integer counterexamples to preserve that distinction.

The 60 published-origin label differences are sensitivity observations, not corrected labels or accepted registration. Primary source grouping remains explicitly `current_rint_zero_center`; endpoint/trajectory kernels retain their own half-open convention. No automatic coordinate shift, cohort promotion or anatomical reassignment was made. Full ARM names and successful readback establish internal correspondence only; origin and biological registration remain unresolved. The saved independent receipt is `arm_actual462_resource_bound_readback_20261009.json` beside this note. An earlier check receipt, `arm_actual462_independent_readback_20261009.json`, is preserved; the resource-bound receipt additionally asserts each row's reference/atlas/key hash and path against the delivery bindings.

## Validation and checked-source snapshot

The companion `test_arm_label_contract_20261009.py` independently reads the TSV and NIfTI without importing production modules. It checks exact resource hashes/grid, unique labels and safe index IDs, all actual level-6 image labels, and the precise lower-level conflicts above. **All four tests passed** in the installed projectome Python on 2026-10-09 (exit 0; 2.212 seconds). This is resource/software validation only. Run with the installed projectome Python and `-B` to avoid writing bytecode.

Checked source SHA-256 values at this review:

| File | SHA-256 |
|---|---|
| `atlas/NMT_v2.1_sym/readme.md` | `2d95671d798e11fb01609603cde44efd037b747e0c7f16260e422b09e1f083c7` |
| `atlas/NMT_v2.1_sym/NMT_changelog.txt` | `6fca79adb1f24710c156e7d833b8de5a67d44fbed9b1af8d5a7cb1ef1f32eb7e` |
| `atlas/NMT_v2.1_sym/SARM_README.txt` | `b59084a98dd972fae7ac8bd8c11ef65e220261dd94deec4aa5f1fb4aa1ba6858` |
| `group_analysis/scripts/build_projection_maps.py` | `c758d2daa427276436e07c1745daad00574e7e392a6f71516f179653ae623145` |
| `group_analysis/scripts/build_endpoint_maps.py` | `5b87411ae4e32f0f51d4cb7ff3ad6799ce27311aeae268e3709ccba29d27d795` |
| `main_scripts/endpoint_atlas.py` | `fee7d46004b9599079b1c50371a8e276847c33e6de0e0876cc579b80d5ea7b97` |
| `main_scripts/terminal_sites.py` | `27ffb878e40fe2dfe10c36fcf30302843869b07b779b321fd361210f4aec9c2f` |
| `group_analysis/scripts/render_projection_slices.py` | `d9a5da58a4f8c22f93920641d8680c1bbcaf350cc0f55ad8fe45474130c6a862` |

Before any publication/export of repository source, perform a secret scan and resolve the inherited credential issue reported by the root reviewer. No credential value or secret-bearing source is copied into this note or tests.
