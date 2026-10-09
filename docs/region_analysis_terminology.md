# Reading region analyses and projection maps

These definitions separate what the pipeline measures from anatomical interpretation. Existing script filenames, column keys and metric IDs remain stable for compatibility. New help text, reports and figures use the plain-language terms below.

The SWC compartment code **2 means axon**, as defined in the [NeuroMorpho.Org SWC format documentation](https://www.neuromorpho.org/myfaq.jsp?id=qr3). It describes a reconstructed part of a neuron, not a neuronal cell type. Use **axon** in readable labels. A **connected axon section within an ARM region** means original axon-labelled points joined by original links inside the sampled region; it can include passing shafts and branching fields. Keep numeric compartment codes in technical definitions/provenance, and retain unresolved/custom labels as supplied.

## What is measured

| Display term | Stored name or workflow | Meaning and limit |
|---|---|---|
| Total reconstruction length | `Total_Length` in legacy region tables | Sum of computed reconstruction edges, including stored non-axon compartments. This differs from the retained regional subset. |
| Retained reconstruction length | `Region_projection_length`, `Projection_Length_*` | Legacy whole-edge length assigned by the proximal node, rounded per edge and retained only for target regions containing a graph endpoint. Use the declared source unit; missing units remain unresolved. It is not verified terminal-arbor length. |
| Log-scaled retained length | `Projection_Strength_*` | `log10(1 + retained length)`, calculated after anatomical aggregation. This is a descriptive scale, not a count of terminals, synapses or response amplitude. |
| Share of log-scaled retained length | R `prop`, `mean_prop`, normalized profiles | A feature's log-scaled value divided by the sum of selected features. It is not the fraction of raw axonal length. Overlapping L3 ancestors and L6 descendants do not form an exclusive anatomical partition. |
| Endpoint-target regions | Legacy `Terminal_Regions`, `Terminal_Count` | Distinct region labels reached by the stored graph endpoints, including non-axon endpoints in this legacy method. The count is of region labels, not individual endpoints or reviewed biological terminal sites. |
| Candidate axon ends | New endpoint maps | Non-root axon-labelled points with no children in the complete stored graph. Breaks, incomplete tracing and true endings can all produce this graph feature. |
| Axon-labelled segment length | New axon maps | Physical length of segments selected by the declared SWC axon-label rule, rasterized in the reference grid. It includes passage and arbor segments; compartment and registration review remain necessary. |
| Reconstructed axon end-branch length | End-branch supplement | Original axon ending to the first full-graph branch point, root or non-axon parent, with the final transition edge included and flagged. Length is allocated along the chain in template mm. It is a graph proxy; unfinished shafts can qualify and reviewed terminal fields are not established. |
| Reviewed terminal arbor or field | Anatomical review, not the graph proxy | Target-local terminal morphology supported by source images and a recorded review. Graph leaves alone cannot establish this. |

Do not convert a log-scaled value into millimetres by multiplying it by voxel size. Unit conversion, when justified by coordinate provenance, applies to the underlying lengths before logarithmic transformation.

## Three different laterality questions

| Question | Formula | Range and interpretation |
|---|---|---|
| What fraction of known-side retained length is contralateral to this neuron? | `Contra / (Ipsi + Contra)`; legacy `Laterality_Index` | 0–1: 0 means entirely ipsilateral; 1 entirely contralateral. A zero denominator is undefined. Unknown-side length is kept separately. |
| Is this neuron's length biased toward contra or ipsi? | `(Contra - Ipsi) / (Contra + Ipsi)`; length-based `Ibias` | −1 to +1: negative means ipsilateral, positive contralateral. This is relative to the source hemisphere, not absolute left/right. |
| Do left-source or right-source groups have a larger target profile? | Inspect each R metric's declared group formula, such as `(L - R) / (L + R)` | This compares source groups. Its sign cannot be read as individual-neuron ipsi/contra dominance. Log-strength, raw-length and prevalence versions are different quantities. |

The producer's 0–1 `Laterality_Index` must not be read using a signed −1–1 legend. Some historical notebooks redefine a similarly named variable; their formula and source version must be checked before reuse. The [methods audit](../notes/region_analysis_review_20261009/methods_audit/method_contract_and_findings_20261009.md) records these distinctions.

## Reading the spatial maps

An **eligible neuron** in an endpoint map has at least one candidate axon end anywhere in its stored graph, including outside the reference field of view. Neurons without such an end remain unassessed for this proxy.

- **Endpoint count:** candidate axon ends per eligible neuron in each voxel.
- **Endpoint density:** that count divided by reference voxel volume, in ends/mm³ per eligible neuron.
- **Endpoint occupancy:** the fraction of eligible neurons with at least one candidate axon end in that voxel. Multiple ends from the same neuron count once. This is not innervation probability.
- **Axon-length density:** axon-labelled segment length per selected neuron, divided by reference voxel volume, in mm/mm³ per selected neuron.

Each animal/source-group map first averages its own neurons. A group map then averages the contributing animals equally. An animal without a sampled source group contributes no map, rather than an assumed zero. Always read the neuron and animal counts beside the map.

The background is the pinned symmetric population T1-weighted NMT MRI. MRI and overlay use the same slice. A maximum-intensity projection (MIP) selects the brightest value through depth; it can obscure depth-specific anatomy. Current matched-slice views use actual planes. Density colors may show `log10(1 + density)` for readability; the saved maps retain their physical values. These descriptive maps contain no t statistic or significance threshold.

## Identity and uncertainty

`SampleID` identifies a source dataset; `AnimalID` requires an explicit animal registry. A bare `NeuronID` can recur across subjects. Combined tables use `NeuronUID = SampleID::NeuronID`.

Keep **human annotation**, **atlas label**, **coordinate-based candidate** and **unresolved** as separate evidence channels. A candidate near INS has not acquired anatomical acceptance. An unknown atlas label does not establish white matter, and an unavailable measurement is not an observed zero. Cortical/subcortical homonyms retain their domain: for example, `C_Pi` is cortical parainsula and `S_Pi` is pineal gland.

Map headings use the official ARM source parcel and side. Historical `HumanINS`, `Candidate` and `G` strings remain original evidence/selection fields, not anatomical labels. Henry's recorded review covers sample 251637 only. The 52 unassigned locations retain 13 reviewed cases and 39 candidates in case QC.

Projection matrices retain **ARM levels 1–6 as separate sheets**; overlapping levels cannot be summed. New axon matrices measure template-space axon-labelled edge length. Endpoint matrices count candidate graph ends or per-neuron presence. The [end-branch supplement](../notes/region_analysis_review_20261009/axon_end_branches_20261009/README.md) retains the same ledger and six hierarchy levels, conditioning its means on end-eligible neurons. These measures preserve the separate legacy retained-length definition. Spatial data are NIfTI (`.nii.gz`); PNGs are displays. Summed voxel occupancy is not regional neuron frequency.

Clustering uses an exclusive target set and retains total extent and outside/unassigned coverage as QC. Stable partitions are not accepted biological types. Signed MSTIM fitted contrasts, SPM T statistics, reconstructed lengths and candidate endpoints remain different measurements even when atlas labels match.

## Which script to use

| Task | Entry point | Result |
|---|---|---|
| Analyze one subject's reconstructions | `main_scripts/step1.run_region_analysis.py` | Region tables, source provenance, QC and optional plots. |
| Review coordinate candidates | `group_analysis/scripts/03_scan_insula_recovery_candidates.py`, then `04_refine_soma_region_by_coords.py` | A candidate queue under an explicit geometry policy; anatomical review remains separate. |
| Apply the documented label harmonization | `group_analysis/scripts/06_harmonize_atlas_to_manual.py` | A new derivative workbook; empirical/manual mapping assumptions remain recorded. |
| Export a side split from recoverable absolute lengths | `main_scripts/region_analysis/laterality_projection_analysis.py` | Ipsi, contra and separate unknown-side output with exact identities and source units. Old split-only workbooks cannot recover absolute anatomy. |
| Build descriptive spatial maps | `group_analysis/scripts/build_endpoint_maps.py`, `build_projection_maps.py` | Candidate-end and complementary axon-labelled length maps from explicit manifests. |
| Draw matching MRI/map slices | `group_analysis/scripts/render_projection_slices.py` | A new figure variant from existing maps; source maps remain unchanged. |
| Export all ARM target hierarchies | `group_analysis/scripts/export_arm_projection_tables.py` | One exact-identity workbook with L1–L6 matrices, denominators, coverage and preserved evidence. |
| Cluster projection profiles | `group_analysis/scripts/cluster_arm_projection_profiles.py` | Separate exploratory axon/end profiles with stability, animal and human-reviewed sensitivity checks. |
| Compare MSTIM regional summaries | `group_analysis/scripts/summarize_mstim_arm.py` | Signed response and additive projection summaries on explicit common coverage, with unresolved registration. |
| Inspect R group analyses | `group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd` and companion `.R` | Exploratory profiles. Exact cohort, target namespace, units and animal-level design must be reconciled before a new inferential run. |

Use each Python entry point's `--help` for its actual arguments. The historical single-file `main_scripts/region_analysis.py` is retained as a reference; the active pipeline imports the `region_analysis/` package.

The terminology follows the [parsed primary Methods](../group_analysis/evolution_20261008/references/scientific_basis_20261009.md), which distinguish segmented arbors, total neurite length and binary arbor presence. The local legacy endpoint-region gate is an adaptation, not reproduction of a published terminal-arbor detector.
