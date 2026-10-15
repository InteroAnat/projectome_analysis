# Visualization methods and style reconciliation

Agent: Codex. Date: 2026-10-04. Branch: `codex/fmost-methods-review-20261002`.

The intended native review remains a high-resolution soma view and a wider context with highlighted neuron traces. Original inputs and existing bulk outputs have not been replaced during the region/laterality audit.

### User-requested native-style previews (2026-10-08)

The user explicitly prefers the original hi-res/low-res appearance and raised apparent out-of-brain placement in the locator. The added public-plot orientation subtitle was removed; other inherited backend geometry/source fixes were retained. `main_scripts/render_cached_toolkit_pair.py` renders saved crops through the original public soma and wide-field MIP methods, without downloads or overwriting existing bulk outputs. All six output PNGs for 252384/003, 252527/058 and 252383/001 were actually inspected. The gallery is `group_analysis/visual_review_native_style_20261008/index.html`; these are three previews, not a full 832-case regeneration.

The optional derived context belongs to the batch workflow, not a new GUI control. It assembles a wider local field from the same sample's high-resolution tiles when copied 5-micron sections are unavailable. Derived outputs are labelled and stored separately from genuine LowRes views. For 252383/001, missing tiles remain conspicuous in the original-style green MIP; changing presentation does not repair source gaps. A local field cannot supply a whole-section locator.

For 252384/003, native soma XY divided by nominal 5 um spacing yields the same point as the original toolkit. No new XY shift was found. Single-section 12464 is low-signal near that point, whereas the ten-section MIP (12459-12468) shows fluorescence beneath it. The original-style local MIP also shows fluorescent processes under the trace overlay. This is evidence for local image correspondence and a projection/source-content contribution; it is not anatomical registration or a verified tissue outline. Raw-to-overview shared origin/axes remain unverified, and no coordinates were corrected. The earlier source-gap explanation is insufficient to establish correct placement.

The supplementary max-pooled whole-section comparison is `notes/visual_toolkit_native_style_20261008/sample-252384_neuron-003_desc-sectionVsMIP_alignmentQC.png`. Its display differs intentionally from the primary native toolkit plots. All original source crops and SWCs remain unchanged. Seventeen image-loader and six export/provenance tests pass. Exact pre-edit snapshots preserve inherited review edits and support scoped rollback under `notes/visual_toolkit_native_style_20261008/rollback/`.

**Individual-section correction to that interpretation:** ten reread TIFFs match their prior whole-file hashes. At shared display scale, 12461-12464 have pronounced displaced strips/seams, whereas adjacent 12459-12460 and 12465-12468 are more continuous. The defect is in the source overview pixels before plotting; its upstream cause is not established. MIP fluorescence beneath the marker is insufficient evidence of valid mapping because projection can conceal these defects. The 12464-based locator for 252384/003 is unsuitable for anatomical review. Acquisition completeness means all requested files loaded and does not establish valid stitching. The exact source-section QC and hashes are preserved in `notes/visual_toolkit_native_style_20261008/individual_section_readback.json`; the ten-panel diagnostic was actually inspected. No coordinate or rescue decision was made.

The GUI source has no changes relative to branch HEAD. Its shared `Visual_toolkit.py` backend has substantial changes: crop/axis geometry, NIfTI spacing and axis signs, volume-aware trace clipping, source validation, partial/missing-data behavior and provenance. These can affect numerical exports and displayed content. The GUI being unchanged does not make its behavior identical to the earlier backend. Public plotting interfaces and their principal styling remain present.

| Output | Rendering method | Interpretation |
| --- | --- | --- |
| Public toolkit soma plot | `plot_soma_block`: grayscale MIP, cyan soma box/text | Original public soma style, current corrected geometry |
| Public toolkit wide-field plot | `plot_widefield_context`: green signal, red clipped SWC edges, white soma marker and scale bar | Original public wide-field style, current corrected geometry |
| October bulk `soma_trace.png` / `context_trace.png` | Custom batch panels in `visual_review_20261002.py`, using toolkit acquisition/geometry | New review derivatives; do not describe them as identical GUI/public-toolkit renders |
| Atlas laterality sheets | Separate NMT/mask/root/endpoint renderer | Supplementary atlas-coordinate checks |
| Native locator contact sheets | Existing full-section locator images arranged into labeled pages | Supplementary native relative-side checks; boxes mark crops, not individual soma boundaries |

`toolkit_style_reference/` contains a concrete pair for **252385/050**, rendered through the existing public toolkit methods using saved own-source NIfTI crops and its raw SWC. Both images were actually viewed. Their provenance records source/output/toolkit hashes. The public methods project the supplied crop depth, whereas custom bulk panels can use a narrower local slab; these projections are not pixel-equivalent.

The different black-and-white atlas/contact-sheet figures are checking figures. They do not replace the high-resolution and wide-field neuron review package. Native contact sheets inspected 299 available locators in four copied samples; the other five samples do not have full-section locators in the available copied source set. Visual relative-side agreement does not establish absolute anatomical orientation or fine region boundaries.

Keep public-style neuron views and supplementary checking figures explicitly identified in future reports. Accepted images, annotations and cohort workbooks remain unchanged. Region changes have scoped rollback snapshots and a reverse-apply checked patch; the tracked toolkit changes remain reviewable in Git separately from this region patch.
