# ARM projection maps in NMT space

This is the primary display set after the user's request for full ARM names and fewer redundant outputs. Source regions come from direct level-6 lookup in the pinned `ARM_in_NMT_v2.1_sym.nii.gz`, using its exact official key. No separate atlas or manual fine-parcel crosswalk supplies these names. `CL`/`CR` identify left/right cortical labels; `SL`/`SR` identify left/right subcortical labels. Figure text replaces underscores with spaces; metadata preserves the exact key spelling.

## Selection and anatomical evidence

The source-verified ledger contains 462 neurons across eight registered animals. **410 have a positive ARM source label**, including 301 in ARM insula parcels and 109 in named neighboring regions: 67 precentral opercular, 32 gustatory, nine claustrum and one secondary somatosensory. These are direct atlas locations under the declared coordinate convention, not 410 accepted INS neurons.

**52 have an unassigned ARM soma location.** Their [case ledger](../classification/background_neuron_audit_20261009/README.md) retains 13 Henry-reviewed coarse INS cases from sample 251637 and 39 spatial candidates. Human review, original atlas metadata and coordinate candidacy are separate fields. An unassigned location does not cancel human evidence or establish native white matter. Current segmentation and alternative-origin results are reported separately.

The existing 436-source preparation contains all 410 positive locations plus 26 unassigned cases. Its immutable `main/` manifest and run name are retained for reproducibility. Another 26 unassigned sources remain in the 52-case ledger; no additional preview-map run is needed. These historical preparation partitions are not new biological cohorts.

## Primary figures

One set of 12 sheets shows the 15 named ARM source parcels. Each metric uses the same three MRI/map planes at XYZ voxels **56, 200, 87**. Endpoint abundance and occupancy answer different questions; axon length is the complementary trajectory measure.

| Measure | Sheets | Interpretation |
|---|---|---|
| Candidate axon-end density | [1](main/figures/endpoint-density_group/space-NMTv2p1_desc-candidateEndpointDensity_matchedSlices_page-01.png), [2](main/figures/endpoint-density_group/space-NMTv2p1_desc-candidateEndpointDensity_matchedSlices_page-02.png), [3](main/figures/endpoint-density_group/space-NMTv2p1_desc-candidateEndpointDensity_matchedSlices_page-03.png), [4](main/figures/endpoint-density_group/space-NMTv2p1_desc-candidateEndpointDensity_matchedSlices_page-04.png) | Graph ends/mm³ per eligible neuron. |
| Candidate axon-end occupancy | [1](main/figures/endpoint-occupancy_group/space-NMTv2p1_desc-candidateEndpointOccupancy_matchedSlices_page-01.png), [2](main/figures/endpoint-occupancy_group/space-NMTv2p1_desc-candidateEndpointOccupancy_matchedSlices_page-02.png), [3](main/figures/endpoint-occupancy_group/space-NMTv2p1_desc-candidateEndpointOccupancy_matchedSlices_page-03.png), [4](main/figures/endpoint-occupancy_group/space-NMTv2p1_desc-candidateEndpointOccupancy_matchedSlices_page-04.png) | Fraction of eligible neurons with at least one candidate end in a voxel. |
| Axon-labelled length density | [1](main/figures/axon-density_group/space-NMTv2p1_desc-axonLengthDensity_matchedSlices_page-01.png), [2](main/figures/axon-density_group/space-NMTv2p1_desc-axonLengthDensity_matchedSlices_page-02.png), [3](main/figures/axon-density_group/space-NMTv2p1_desc-axonLengthDensity_matchedSlices_page-03.png), [4](main/figures/axon-density_group/space-NMTv2p1_desc-axonLengthDensity_matchedSlices_page-04.png) | Template-space mm/mm³ per selected neuron; includes passage and arbor segments. |

The background is the symmetric NMT v2.1 population T1-weighted MRI at 0.25 mm, with a shared 1st–99.5th percentile intensity window inside its brain mask. MRI and overlays use matching slices, without MIP. Saved maps are unsmoothed physical values; density colors use `log10(1 + density)` and occupancy uses 0–1. Display cuts do not establish target coverage or biological absence.

Endpoint maps require at least one non-root type-2 full-graph leaf anywhere, including outside the reference field of view. Of the 410 named-location sources, **378 qualify**; the other 32 remain unassessed for this proxy. The corresponding numbers for direct ARM insula are 272/301. Axon maps use every selected source. Each animal/source-region map averages its own neurons; the group map gives contributing animals equal weight. Missing animal/region sampling is not zero-filled.

## NIfTI maps and hierarchy tables

The numerical data are compressed NIfTI (`.nii.gz`), with the exact reference grid and coded millimetre transform. Endpoint count/density/occupancy maps are in `main/endpoints/{animal_maps,group_maps}`; axon length/density maps are in `main/axons/{animal_maps,group_maps}`. The independent [endpoint](main/endpoints_review/endpoint_run_readback.json) and [axon](main/axons_review/projection_run_readback.json) receipts list their paths and hashes and check complete saved volumes. Source and registration provenance are retained in each `run_provenance.json`.

The [multi-hierarchy table delivery](../../../notes/region_analysis_review_20261009/hierarchy_tables_20261009/README.md) exports all six actual ARM target volumes, with explicit identity, units, denominators and unresolved outcomes. Each hierarchy is a separate representation; ancestors and descendants must not be summed into one anatomical budget. New map-based length/endpoint measures remain distinct from legacy proximal-edge, graph-leaf-gated regional lengths. Original legacy workbooks are preserved.

Earlier selection-stratum displays, repeated wording variants and animal/unassigned-location panels remain archived derivatives. The current figure index does not present them as additional cohorts or accepted anatomical regions. Scientific volumes and archived panels remain local with published hashes; the 12 primary sheets, matrices, summaries, provenance and code are published.

## Reproduction and limits

Use a fresh output root in the existing projectome Python environment:

```powershell
python -X utf8 -B notes/region_analysis_review_20261009/run_arm_mapping.py --output group_analysis/evolution_20261008/arm_repeat_01
```

The default builds and independently reviews one primary map set, then renders the three group measures. Per-animal numerical maps are retained for validation; animal figure sheets and unassigned-location QC sheets are optional. The historical preparation manifests retain source hashes and the exact ARM labels/evidence fields.

The [parsed primary Methods](../references/scientific_basis_20261009.md) support sample-to-NMT registration and separately transformed SWCs, while the actual export origin and individual transform receipts remain unresolved. Alternative-origin sensitivity changes 60/462 source labels. Graph leaves are candidate reconstruction ends, not verified terminal arbors, boutons or synapses. Fine anatomy, terminal/truncation review and CM032/CM033 registration/contrast acceptance remain required. No t-map, new anatomical acceptance or projection–stimulation association is claimed.
