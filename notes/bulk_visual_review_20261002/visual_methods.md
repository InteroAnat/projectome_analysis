# Native fMOST evidence for soma and insular boundary review

Agent: Codex  
Date: 2026-10-02 (Asia/Shanghai)  
Scope: focused supplement to the completed 29-reference review. This specifies evidence requirements for correction-ready image sources; it does not assign anatomical labels or approve neurons.

## Evidence actually checked

Relevant saved Zotero PDF text was read through the read-only local API. Coverage is the API's indexed-page count, not a claim that every page or image was inspected. The method sections named below were checked. No library modifications were made.

| Primary source | Local item and PDF | Coverage | Directly verified method relevant to visual review |
|---|---|---|---|
| Gou et al. 2025, Cell; [10.1016/j.cell.2025.06.005](https://www.sciencedirect.com/science/article/pii/S0092867425006397) | [JWB4MYBC](zotero://select/library/items/JWB4MYBC), PDF 8BNNG4KV | 42/42 | STAR Methods: fMOST imaging; collaborative proofreading/QC; block stitching and atlas registration |
| Zhou et al. 2022, Science Bulletin; [10.1016/j.scib.2021.08.003](https://doi.org/10.1016/j.scib.2021.08.003) | [RF4JIFTM](zotero://select/library/items/RF4JIFTM), PDF UFYL9BGY | 12/12 | Image preprocessing and acquisition pipeline: microscopy channels, distinct sampling modes, native MRI registration |
| Gong et al. 2016, Nature Communications; [10.1038/ncomms12142](https://www.nature.com/articles/ncomms12142) | [WPGCKTP9](zotero://select/library/items/WPGCKTP9), PDF FRB2V5NH | 12/12 | Same-specimen fluorescent neurons and nuclear counterstain; co-localization and cytoarchitectonic landmarks |
| Krockenberger et al. 2023, Journal of Comparative Neurology; [10.1002/cne.25571](https://onlinelibrary.wiley.com/doi/10.1002/cne.25571) | [2NZNENNT](zotero://select/library/items/2NZNENNT), PDF DZAU4B4N | 25/25 | Methods 2.2: serial-section landmarks, focal-stack object identification and independent architectonic review |
| Qiu et al. 2024, Science; [10.1126/science.adj9198](https://www.science.org/doi/10.1126/science.adj9198) | [BNB9KHB7](zotero://select/library/items/BNB9KHB7), supplement ZMGEI8V8 | 38/38 | Supplementary Methods pp. 3-4: image suitability, independent tracing/merge, cross-cell lint and specimen-specific registration |

The interpretations and proposed crop sizes below are project recommendations, not numerical acceptance thresholds prescribed by these papers. No new reference count is implied; these five publications are already in the broader review.

## What the primary methods require us to distinguish

Gou's macaque acquisition uses 0.65 × 0.65 × 3 µm voxels, green GFP and red tdTomato neuronal labels, followed by coronal stripe correction/stitching. Proofreading compares skeletons with images, particularly near somata, branch points and close branches; complex errors remain annotated until expert review. Block-spanning axons are joined only with uniquely plausible anatomical and neurite correspondence. Its affine/B-spline/U-Net registration is an additional operation, not a unit conversion. These facts support retaining original image evidence, unresolved annotations and transform provenance.

Channel purpose cannot be inferred from color. Zhou's macaque method uses GFP for circuits and PI for nuclear cytoarchitecture; its cytoarchitectural and axon imaging modes have different sampling. Gong's mouse method demonstrates co-localized neuronal and nuclear signals in one specimen. Neither establishes that this project's available red channel is PI. Obtain the dataset's acquisition/channel metadata before treating overview fluorescence as a cellular architecture stain. Preparation stability reported for another specimen does not validate local geometry.

Krockenberger's insular review draws the gray/white edge, layer 4 and claustrum; stained object identity is checked through a focal stack. At least two examiners independently assess Nissl/myelin architecture without adjacent tract-label information, then maps are aligned. Its Discussion retains uncertainty about fine stripes. This supports contextual, independent boundary assessment while permitting an unresolved result.

Qiu's mouse workflow excludes unsuitable images, combines independent tracers, checks false branches/misconnections and checks overlap between cells. Its PI-based, specimen-specific atlas registration and mouse QC thresholds are examples, not accepted numerical tolerances for macaques. A valid SWC graph or visually attractive overview cannot establish correct neuron identity or a complete axon.

## Current project inputs and the coordinate gap

The coordinating agent reports an overview catalog with datasets named 251637, 252384, 252385, 252527 and 252714; the user confirmed that the available copies are the overview datasets, and work is proceeding on four newly copied monkey datasets with missing sources queued separately. Availability does not establish acquisition-resolution or architecture-bearing channels.

Narrowly inspected main_scripts/Visual_toolkit.py:61,284-295,574-586 declares overview spacing [5, 5, 3] µm in XYZ and passes the SWC root XYZ as center_um. Its depth index is int(z/3), not z/5. The "5micron" folder name therefore must not become an assumption of isotropic 5 µm voxels. These are code declarations; TIFF calibration metadata is reported mostly absent. They do not verify each source SWC's frame, image origin, axis order or dataset identity.

The inspected analysis loader elsewhere declares SWCs to be NMT physical micrometres and divides by 250 for NMT voxel indices. These conventions must not be silently combined. Dividing NMT coordinates by the declared [5, 5, 3] spacing does not create native fMOST coordinates. The root of an unverified tree is a candidate location, not a confirmed image soma.

R_analysis/tables/somainfo_Henry_2026.04.03.xlsx and neuron_tables_new/251637_INS_HE_inferred.xlsx are reported review/reference inputs. Keep their labels, inferred status and source row identities distinct from new human decisions. Existing INS/PrCO/Unknown selection may omit neighboring borderline candidates; preserve explicit selection criteria and adjacent-label coverage. Dataset folder names alone do not establish monkey, block, injection or hemisphere identity.

## Per-neuron evidence package

1. **Source identity.** Record the composite specimen/dataset/block/neuron identity, source SWC path and hash, exact TIFF or volume identifier, channel, dimensions, bit depth, acquisition versus overview spacing, downsampling lineage and file hash or version. Resolve duplicate neuron IDs across specimens. A missing or inaccessible source is a missing-evidence status, not a negative anatomical finding.
2. **Coordinate ledger.** Retain the original coordinate and its declared frame/units; native candidate voxel coordinates; origin, orientation and axis permutation; indexing/rounding convention; slice filenames; and any transform source/target, direction, order and version/hash. Record a separately confirmed native soma position. For block coordinates, include block origin and documented stitching transform.
3. **Unmarked image pair.** Save the source crop without labels and a corresponding marked copy with the candidate point/crosshair or traced skeleton. Keep raw pixel values available; persist display window, contrast mapping, any background subtraction and interpolation. A rendered overlay alone is insufficient evidence.
4. **Soma detail and depth.** A practical starting detail field is 0.25-0.5 mm wide around the candidate, expanding when needed to include proximal neurites and distinguish nearby somata. Show the central native plane plus a navigable stack or neighboring planes. Identify a soma from image morphology and continuous proximal branches, not one bright pixel or a maximum projection. Record stack thickness and focal/slice positions.
5. **Regional context.** Start with a 4 mm field, expanding to contain the relevant cortical ribbon, pial and gray/white boundaries, sulcal banks, neighboring opercular/PrCO context and claustrum where visible. Include an overview locating the crop within the full native section. A crop lacking the landmarks needed for its particular boundary question must be enlarged or marked insufficient.
6. **Optional axon evidence.** Inspect suspicious branch points, crossings, weak endpoints and block joins at the source's acquisition resolution with linked 3D cubes and neighboring slices. A 0.15-0.30 mm cube is a configurable starting point, expanding along the relevant branches. The 5 µm overview is suitable for context screening; it cannot be the sole evidence for thin-axon continuity, boutons or tracing completeness. A depth projection can support navigation but can merge independent neurites and must not resolve connectivity on its own.

All crop dimensions above are proposed physical fields of view. Derive pixel extents only from verified image spacing. Keep crop bounds, actual retained extent and clipping/missing-slice flags. Do not replace absent tissue with apparently valid black data without marking it.

## Coordinate and anatomical acceptance

An in-bounds point passes only an array check. Native soma identity additionally needs a visible matching object and local branch context, with no unresolved competing cell assignment. Across representative specimens/blocks, verify the coordinate convention against independently located image features, including local landmarks near the target; record residuals in native voxels and calibrated physical units when available. No universal residual tolerance or atlas-to-image transform is supplied by this note. Failed axes, origin, specimen or block correspondence requires correction before label review.

Label native panels with native axis names until physical orientation is independently established. Absolute hemisphere, soma-relative laterality and reflected display conventions remain separate. Atlas labels/contours may be supplied as secondary guides only with a verified, specimen-specific mapping and uncertainty. They do not establish the image boundary that is being evaluated.

For anatomical correction, obtain sufficient tissue context and an appropriate architecture-bearing channel or matched histology. GFP/tdTomato signal and gross background may support soma localization or coarse boundaries but cannot automatically demonstrate a fine cytoarchitectonic stripe. Adjacent-section evidence requires documented section order, separation and alignment; it is not automatically pixel-coincident with the neuron channel.

A reviewer should record identity, coordinate validity, image suitability and anatomical confidence separately. Suitable decision fields are blank corrected region/subregion, confirmed native soma, reviewer/date, supporting asset and slice IDs, reason, unresolved issue and further evidence required. Keep supplied atlas, Henry-reference and inferred labels visible as provenance, without pre-filling the human correction. An optional independent assessment can hide those labels initially to reduce anchoring.

Review outcomes should allow confirmed, corrected and unresolved/missing-evidence states. A scientific correction requires the reviewer's anatomical rationale and the supporting images; generating panels or passing software tests does not make that decision. Fine-boundary cases merit a second independent review or explicit disagreement record. Promote cohort membership only through the project's separate acceptance process, retaining the original record and correction lineage.

## Delivery checks and limits

A correction-ready package must reopen its assets, show scale/plane/stack extent, locate detail crops in native regional context and link each review row to the correct source identity. Inspect representative and borderline cases for empty crops, saturation, wrong channel/plane and mistaken cross-specimen joins. These checks verify the evidence package, not the underlying scientific label.

The outstanding dependencies are per-source coordinate/channel/calibration provenance and sufficient image detail for each review question; missing sources remain queued. This supplement provides no fitted atlas-to-image transform, image-based neuron confirmations, anatomical boundary decisions or human acceptance. It complements the existing software and clustering review without replacing native tracing or architectonic evidence.

## Terminal batch provenance — 2026-10-03

Subsequent regional driver invocations assign a unique run ID and save their full
terminal ledger under `manifest/regional_runs/<run_id>.json`, in addition to the
current-state ledger. Each terminal record retains selected identities, outcomes,
source failures, resource accounting, manifest hash, and UTC start/end times.
The record is persisted before gallery refresh, so a refresh failure or later
sample cannot erase the earlier run's outcomes. The already-running 252790 worker
loaded the preceding version; capture its terminal ledger separately after it
ends. The independent incremental readback tool saves such a terminal observation
when it sees one. Test evidence and the old/new source distinction are recorded
in `terminal_ledger_validation_20261003.json`.

## Independent rendered-plane consistency — 2026-10-03

The incremental readback additionally compares every unmarked PNG pixel with
the saved XYZ native volume's documented central plane. It derives the depth
index from raw-root position, origin, and spacing, checks the persisted source
plane ID, then reconstructs the stated 0.5–99.5 percentile contrast and gamma
0.5 display without calling the renderer or its coordinate/normalization helpers.
The audit permits one 8-bit gray level for numerical rounding. At 11:06 local
time, all 44 finished fields match exactly: 25,433,510 pixels, maximum error zero.
Per-identity `derived_panel_readback_*` records retain plane IDs, quantiles,
dimensions, pixel counts, and stable file hashes. This checks artifact consistency
with the nominal source convention; it does not verify physical calibration,
soma identity, anatomical suitability, or quantitative fluorescence preservation.
