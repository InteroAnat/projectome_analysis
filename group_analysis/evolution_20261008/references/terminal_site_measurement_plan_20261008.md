# Terminal-site priority: measurement proposal and current display provenance

Date: 2026-10-08. Status: methods proposal; no terminal maps have been implemented or scientifically accepted by this note. Scope: preserve existing maps and source bytes; prioritize target-local terminal evidence, retain whole-axon extent as a secondary descriptor. Read-only evidence review plus this new note only. No Zotero attachments or indexed full text were retrieved.

## What the displayed maps actually show

`projection_maps/curated_reference/run_provenance.json` pins the background to `atlas/NMT_v2.1_sym/NMT_v2.1_sym/NMT_v2.1_sym_SS.nii.gz` (SHA256 `9e37a94c4b9e5865aabb9fd3b51dcc3b2cb16f3daed39c967e68acf16cf92bee`). This is the skull-stripped symmetric NMT v2.1 template, on a 256 x 312 x 200 grid at 0.25 mm isotropic resolution; it is a template background rather than an individual fMOST specimen image.

`review_projection_run.py:render_sheets` displays grayscale maximum-intensity projections of that template along voxel axes 2, 1 and 0, transposed with `origin=lower`. A white contour is the corresponding brain-mask projection. Colored overlays are `log10(1 + maximum mean axon-length density along the same axis)` with a common 99.5th-percentile upper display cap across the animal/subregion panels. These projections collapse depth and can make separate structures appear superimposed; voxel-axis labels are not independent anatomical orientation validation. Orthogonal slices or 3-D review should accompany terminal localization.

The underlying measure is the exact voxel-wise allocation of all edges whose child SWC compartment is type 2, averaged per selected neuron within animal/subregion. It includes axon trunks and arbors. The brain mask records coverage and does not remove in-FOV edges from the delivered maps. Values describe template-space length density (mm/mm3/neuron), with no endpoint requirement, bouton detector, synapse detector or t test. Template-space lengths are not native specimen lengths after nonrigid warping.

## Proposed operational definitions

| Output | Operational definition | Interpretation limit |
|---|---|---|
| Candidate axon endpoint | Non-root type-2 node with zero children in the complete validated SWC graph. Determine leaves before compartment filtering or ROI clipping. Keep stable source node IDs. | A reconstruction end, potentially caused by signal loss, tissue loss or a tracing break. It is not automatically a biological termination. |
| Reviewed axon endpoint | Candidate with source-image/reconstruction-history evidence supporting a true end, with reviewer, image reference and decision recorded. | An anatomical endpoint; it does not enumerate all boutons or prove a synapse. |
| Reviewed target-local terminal arbor | A source-traceable connected axonal subtree or separately documented local component occupying a target, with supported terminal branches and explicit boundary from the entering trunk. Record length, branch points, tips and segmentation rule. | Mesoscopic terminal territory. An ROI containing one tip does not make every axon edge in that ROI part of the terminal arbor. |
| Putative bouton/varicosity | A separately detected and reviewed local swelling in calibrated fluorescence images, including en passant sites and end-associated sites. | Cannot be recovered simply by counting SWC nodes, tips or placeholder radii; morphology alone is not a verified synapse. |
| Verified synaptic site | Site established by suitable ultrastructural or independently validated synaptic evidence, with method specified. | No such site-level evidence was established for the current cached INS SWCs. |
| Secondary axon extent | Existing all-axon length/density maps, or a future explicitly segmented non-arbor/trunk measure. | Current all-axon maps include arbors; call them all-axon extent, not exclusively passage fibers. |

The first computational terminal representation can be an explicitly provisional candidate-endpoint map. The biological primary analysis should favor reviewed target-local arbor occupancy, with endpoint counts as a complementary descriptor. Do not impose a universal branch-point minimum that would silently reject genuine unbranched endings. Calibrate arbor segmentation and target criteria on this macaque dataset, review disagreement cases, and report threshold sensitivity. Published mouse target thresholds are not automatic macaque INS acceptance criteria.

## Normalization and summaries

For each animal and accepted INS source subregion, report the fraction of eligible reconstructed neurons with a reviewed terminal arbor in each target parcel. This limits domination by one exceptionally elaborate cell. State the animal, injection and neuron denominators, and distinguish zero evidence with adequate target coverage from unknown coverage. The fraction describes the reconstructed sample, not every neuron in the insula.

Alongside occupancy, retain reviewed endpoint counts per neuron per voxel or parcel, endpoint density per template mm3 per neuron, and reviewed arbor length per neuron. Counts survive a one-to-one coordinate transform, while density and length depend on the coordinate space and deformation. Report within-animal means before equal-animal descriptive means; absent source subregions and unreviewed targets must not become zeros. Keep local/source-region arbors separately identifiable rather than silently excluding them.

Optional relative endpoint allocation uses each neuron's eligible reviewed endpoint total as denominator and must retain out-of-FOV, outside-mask, unassigned and unresolved totals. A fraction conditional on in-brain assigned endpoints is a different measure and requires that explicit denominator. Neurons with no eligible endpoints have undefined endpoint fractions, not zero allocation. Absolute counts and occupancy should remain available because within-neuron normalization removes total arbor complexity.

## Required exclusion and QC gates

1. Validate full graph topology, IDs, parents, finiteness and compartment labels. Exclude roots and type-filter/ROI-induced artificial leaves. Flag coincident duplicate tips, zero-length terminal edges, implausible joins and ambiguous compartment transitions without repairing source bytes automatically.
2. Classify tips/arbors as supported, truncated/broken, or unresolved using original signal and proofreading history. Record tissue/block boundaries, weak fluorescence, source-volume coverage and join uncertainty. Missing evidence is not a negative target. Boundary proximity is a QC flag, not sufficient proof of truncation.
3. Freeze source cohort, animal/injection identity, accepted soma/subregion/hemisphere labels and completeness rules before comparing targets. The 251637 coordinate rule remains a candidacy rule; it does not accept anatomy or cohort membership.
4. Bind each derivative to exact SWC and reference hashes, coordinate encoding, index scale, transform lineage and reviewer evidence. The current SWC-to-NMT contract is a repository declaration, not an accepted native-image registration. Inspect target boundaries and landmark correspondence before assigning biological names or comparing to stimulation.
5. Keep unknown/missing coverage, outside-FOV and outside-mask observations visible. A template mask is not specimen-specific completeness evidence. Terminal localization should be inspected in slices/3-D and source images, not accepted from maximum-intensity projections alone.
6. Use animals as replication units for animal-level inference. Neurons, endpoints and voxels are nested observations, not independent animals. Current maps remain descriptive. Crossmodal comparisons require accepted source/site matching, CM032/CM033 GLM contrast/effect definitions, transform QC and common valid coverage. Signed stimulation BOLD effects/t statistics measure a functional network response, not direct monosynaptic anatomical strength; correspondence needs spatial-autocorrelation-aware nulls and sensitivity to target parcellation and registration.

## Primary-source support and access limits

- **Gou et al. (2025), Cell**, DOI [10.1016/j.cell.2025.06.005](https://www.sciencedirect.com/science/article/pii/S0092867425006397), selected Zotero key `JWB4MYBC`: macaque single-neuron fMOST projectomes distinguish axon targeting and target-local patchy terminal arborization. This is the closest library precedent for arbor-centered analysis. The publicly indexed primary article supports that distinction; exact STAR Methods segmentation thresholds were not freshly recoverable here and are not asserted as a ready-to-copy algorithm.
- **Yan et al. (2022), eLife**, DOI [10.7554/eLife.72534](https://elifesciences.org/articles/72534), selected Zotero key `RXUGG8AT`: population excitatory axonal tracing compared with diffusion tractography. This study uses serial two-photon tomography, not the present single-cell fMOST reconstruction procedure. Its axon-density comparison is not validation of endpoint, bouton or synapse counts.
- **Liu et al. (2024), Nature Communications 15, 10269**, DOI [10.1038/s41467-024-54745-6](https://www.nature.com/articles/s41467-024-54745-6), primary full article also [PMC11599929](https://pmc.ncbi.nlm.nih.gov/articles/PMC11599929/): whole-brain morphometry separately analyzes arbors and detected axonal varicosities, including en passant and terminal-associated forms. Predicted varicosities indicate potential synaptic locations; the authors explicitly leave biological validation outside the resource study's scope. This does not establish varicosity detection in our SWCs.
- **Winnubst et al. (2019), Cell**, DOI [10.1016/j.cell.2019.07.042](https://pmc.ncbi.nlm.nih.gov/articles/PMC6754285/): the MouseLight reconstruction pipeline explicitly does not detect synapses and uses axon length as a connectivity surrogate. Its approximate constant-synapse-density assumption is not a measured property of these macaque INS data.
- **Drawitsch et al. (2018), eLife 7:e38976**, DOI [10.7554/eLife.38976](https://pmc.ncbi.nlm.nih.gov/articles/PMC6158011/): correlated fluorescence/EM testing finds that light-microscopic swelling size alone does not unequivocally identify synapses. This is direct validation evidence for keeping endpoint, putative bouton and verified synapse labels separate.

Selected Zotero bibliographic provenance remains in `selected_library_evidence.json` and `selected_zotero_references.bib`; additional web-verified papers above are not represented as confirmed library items. Open-web primary-source search was used where direct publisher pages returned access challenges. No new terminal measurements, source-image review or anatomical acceptance occurred.

## Inspected implementation snapshot

- `main_scripts/projection_maps.py`: SHA256 `e47a54bfe5e8159c8e4a4f70d687b845a0810f073189a26f60c5945a65aa90da`.
- `group_analysis/scripts/review_projection_run.py`: SHA256 `e0c9ea928aa1bc78b28ac15e84c24ec21630ad888ee992501b92e0046ea9600d`.
- `selected_library_evidence.json`: SHA256 `55b4a9ccac43398866b3fd3f26321dc8203179df0ee01f6736888f5544ff8e65`.

The legacy `neuro_tracer.py:_mark_branch_terminals` marks graph leaves without an explicit type-2/root gate. Its terminal label should not be treated as the reviewed axon-endpoint contract proposed above.
