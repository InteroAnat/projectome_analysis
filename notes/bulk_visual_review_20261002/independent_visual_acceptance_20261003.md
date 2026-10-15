# Independent scientific acceptance checklist for native CH1 review

Agent: Codex  
Date: 2026-10-03 (Asia/Shanghai)  
Scope: read-only assessment of saved methods and deliverable documentation. Only this report was written. Cursor stage 2 production was not duplicated; no code, crops, review tables, cohort workbooks or Zotero records were changed.

## What the previews can support

CH1 is not established as a cytoarchitecture channel. Gou's macaque method uses green GFP and red tdTomato for neuron labeling, whereas Zhou's macaque and Gong's mouse methods describe nuclear counterstaining for architecture. Their channel meanings cannot be assigned to this dataset from a filename or color. [Gou 2025](https://doi.org/10.1016/j.cell.2025.06.005), [Zhou 2022](https://doi.org/10.1016/j.scib.2021.08.003), [Gong 2016](https://doi.org/10.1038/ncomms12142).

| Review question | CH1 evidence can support | Additional acceptance requirement |
|---|---|---|
| Is this the intended soma? | Candidate cell body, proximal neurites and local identity correspondence in a native stack | Unique specimen/neuron identity; source/root correspondence; no competing labeled cell; human confirmation across neighboring planes |
| Where is it in the cortical tissue? | Local position and gross context when tissue edges and sulcal landmarks are actually visible | Sufficient context, verified axes/coordinate convention and recognizable native anatomy; cropped detail alone is insufficient |
| What region/subregion is it in? | Evidence for a proposed gross region; targeted inspection of border cases | Human anatomical rationale with appropriate native tissue or matched histology/reference evidence; atlas/source labels remain annotations until reviewed |
| What cortical layer is it in? | Soma position and possibly cortical depth when surfaces are identifiable | Layer architecture visible in an appropriate tissue channel or registered histology; depth, sparse fluorescent cell density and projection type alone do not establish histological layer |
| Is its axon correctly reconstructed? | Local visible correspondence where source detail resolves neurites | Acquisition-resolution, depth-matched review of branches/crossings/endpoints/joins; overview or projected trace alone cannot establish connectivity or completeness |

The Henry-style precedent is independent architectonic assessment: Krockenberger's Methods 2.2 uses gray/white boundaries, layer 4 and claustrum, focal-stack object identification, and at least two examiners reviewing architecture before consulting adjacent tract labeling. A published parcel map helps form hypotheses; it does not prove a boundary in this specimen. [Krockenberger 2023](https://doi.org/10.1002/cne.25571).

## Supplied overview versus derived tiled context

The copied overview has nominal XYZ spacing **[5,5,3] µm**, not isotropic 5 µm. Its resampling history and independent calibration remain unresolved. A derived context is assembled from own-sample high-resolution cubes with nominal spacing **[0.65,0.65,3] µm**, restricted to a recorded thin Z slab and downsampled in XY. It is a different image product, not a recovered copy of the supplied overview or a new architecture stain.

The stage 2 proposal is approximately 4 mm XY and 30 µm Z. Report actual retained dimensions, source planes/tiles, half-open Z bounds, reduction method, output spacing and missing coverage. With ten 3 µm sections, voxel-support depth is 30 µm while the first-to-last plane-center span is 27 µm; retain plane count as well as depth. Integer XY reduction by 7 or 8 would give nominal 4.55 or 5.2 µm respectively, so neither should be labeled 5 µm without an explicit resampling step.

A thin slab may provide useful regional context while missing structures outside its depth. A MIP merges depth and may create apparent contacts; an average or central plane has different signal characteristics. Retain unmarked central planes and clearly label projections. Trace geometry must be clipped to the exact displayed plane/slab. Do not infer axon joins, cell identity or absence of a landmark from a projection or missing tiles. The saved independent geometry audit already identified mismatched full-crop traces over central-plane backgrounds; stage 2 repairs require their own verification. [Saved geometry audit](independent_geometry_review_20261003.md), [stage 2 scope](CURSOR_STAGE2.md).

## Apply these checks per neuron before scientific acceptance

- [ ] **Identity and frame:** composite sample/neuron UID, own-sample raw SWC hash, actual source image IDs and channel agree. Native root coordinates are kept distinct from portal/NMT coordinates. No atlas-to-native mapping is assumed.
- [ ] **Geometry:** spacing, origin, XYZ/ZYX ordering, slice convention and physical aspect are explicit; the candidate matches a visible soma and proximal branches in neighboring native planes. In-bounds arithmetic and a valid NIfTI header are software evidence only.
- [ ] **Coverage and display:** central soma evidence and necessary regional landmarks are present; clipping, missing neighbors/tiles and absent context remain visible. Unmarked images, scales, plane/slab IDs and contrast settings are available. Black gap-fill is not negative tissue evidence.
- [ ] **Anatomical decision:** the reviewer states the region/subregion rationale and supporting tissue/reference assets. Leave fine subregion or layer unresolved when architecture is insufficient. An inferred label is not independent anatomical confirmation.
- [ ] **Independent review:** fine-boundary or disputed cases obtain a second review or explicit disagreement record. Do not use projection labels to manufacture independent architecture confirmation.
- [ ] **Decision lineage:** record reviewer/date, corrected region/subregion/layer, confidence or unresolved reason and asset IDs. Preserve original atlas/refined/Henry/inferred labels separately. Cohort promotion follows the project's separate acceptance process.
- [ ] **Tracing claim:** any claim about major branches, crossing identity or block joins has matching acquisition-resolution evidence and unresolved annotations. Gou/Qiu support image-based proofreading and independent tracing checks; their numerical mouse QC thresholds are not macaque acceptance rules. [Gou 2025](https://doi.org/10.1016/j.cell.2025.06.005), [Qiu 2024](https://doi.org/10.1126/science.adj9198).

## Documentation corrections or clarifications needed

1. **docs/project_note.md:153-155:** the 746/301/445 and 277/5/17/2 counts are the saved pre-stage-2 snapshot. Preserve those dated entries, then append component-specific counts after production/readback. "Source-pending" must distinguish missing copied overview from unavailable own-sample soma or derived context. "Complete" means configured image coverage, not scientific review.
2. **docs/project_note.md:155:** "reviewer columns survive a table rebuild" was qualified by independently reproduced duplicate-UID and orphaned-annotation cases. Do not generalize annotation preservation until the stage 2 repair is independently checked; the saved phase-1 findings were snapshot-specific, not proof of current final-code failure. [Phase-1 audit](independent_phase1_review_20261003.md).
3. **visual_methods.md, current-input section:** four copied new-monkey overviews remain a valid inventory fact, but are not the limit of accessible native evidence. The saved probe tested own-sample raw-SWC/cube routes in all nine samples; one successful probe per sample does not establish availability for every candidate. Separate copied-overview, HTTP soma and derived-context status.
4. **visual_methods.md, anatomical-acceptance section:** explicitly extend its fine-stripe warning to **histological layer**. A CH1-only cortical-depth estimate must be named depth, not a validated layer. Its proposed crop sizes are already correctly described as recommendations rather than paper thresholds.
5. **Both notes:** identify any thin-Z tiled output as **derived native high-resolution context**, with actual spacing/depth and coverage. Do not call it supplied 5 µm overview, full-volume anatomical coverage, PI evidence or a verified atlas registration.

## Provenance and limits

Reused locally verified PDFs: Gou **8BNNG4KV (42/42 indexed pages)**; Zhou **UFYL9BGY (12/12)**; Gong **FRB2V5NH (12/12)**; Krockenberger **DZAU4B4N (25/25)**; Qiu supplement **ZMGEI8V8 (38/38)**. Exact item links and inspected method sections are in [visual_methods.md](visual_methods.md). No repeat broad literature search was performed. The broader review remains 29 references with eight selected BibTeX records.

This report read the saved Cursor result/stage 2 scope and representative 252385/003 and 252527/048 provenance. It did not independently inspect new stage 2 output, fit transforms or assign anatomical labels. Reviewed document SHA256 prefixes: project note **dfa2a5345708**; visual methods **448ada714ed7**; stage 2 scope **f9659f74d5e5**; saved Cursor result **5fb5d34f2560**. These identify the read snapshot while production is active. No scientific acceptance is inferred from file counts or passing tests.
