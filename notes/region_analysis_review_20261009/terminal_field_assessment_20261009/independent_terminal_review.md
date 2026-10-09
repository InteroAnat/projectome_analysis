# Independent terminal-field pilot review — 2026-10-09

**PASS within software, source-pixel and selected saved-display scope. Anatomical terminal-field, bouton and synaptic acceptance remain unestablished.** No source files, producers or cohorts were modified by this review. The five cases are purposive examples, not a prevalence sample.

The machine-readable [independent receipt](independent_terminal_review.json) preserves the initial half-open/display review and adds current-policy readback. Current evidence is [current-rint generation](native_cube_pilot_current_rint/generation_provenance.json), [unprojected planes and reconstructed passage](optical_plane_checks_current_rint/optical_plane_provenance.json), and the [readable display correction](review_panels/display_provenance.json). All 44 current generation/optical bindings and 14 display bindings match actual saved files. The display receipt also transitively binds the passage TIFF/native reconstruction through the optical and generation receipts. No biological acceptance follows from these hashes.

## Checked definitions and implementations

- **Original graph leaves:** [graph_information](assess_native_terminal_fields_current_rint.py:78) uses the full original parent graph across all compartments. A candidate is a non-root type-2 node with no original children. A region boundary or compartment cut does not manufacture a candidate. [target_components](assess_native_terminal_fields_current_rint.py:96) computes local connected type-2 components but retains the full-graph leaf definition. The independently checked passage component has nodes 7082–7128, entry 7081→7082, exit 7128→7129, and zero original leaves or local branch nodes. This establishes reconstructed passage, not absence of en passant boutons.
- **Native/NMT pairing:** original ID, type and parent assignments match for each independently checked pair; exact topology pairing is required before transferring labels. Correspondence of node identities does not prove a correct spatial transform. Five native SWCs, five atlas-coordinate SWCs, actual ARM level-6 voxels and TIFF source bytes were read independently, without importing the producer helper for real-data arithmetic.
- **Cube axes and sampling:** source TIFFs are uint16 `(Z,Y,X)=(90,360,360)`; display data transpose to `(X,Y,Z)`. Independent source-pixel probes agree. Nominal XYZ sampling is `(0.65,0.65,3)` µm; block membership uses half-open XYZ blocks. [draw_image_views](assess_native_terminal_fields_current_rint.py:128) now places pixel centres at `origin + index × spacing`, matching the declared nominal affine. The initial deterministic half-pixel shift (0.325 µm in XY and 1.5 µm in Z) is corrected in the fresh current variant; original artifacts remain preserved. Optical resolution, calibrated image/SWC origin and physical anatomical orientation remain unverified.
- **Atlas policy:** [atlas_labels](assess_native_terminal_fields_current_rint.py:88) explicitly uses `np.rint(atlas_xyz/250)`, including ties to even. The original `floor(x+0.5)` variant is retained as sensitivity evidence. Across these five full SWCs, the two policies change 11 node labels, no full-graph candidate-leaf labels, and none of the five selected components. This is a bounded sensitivity result, not proof of coordinate-origin validity or equivalence everywhere.
- **Lengths:** reported values sum native Euclidean parent-child edges whose two nodes lie in the selected component. They are **nominal native component length in µm**. They are neither calibrated optical arbor measurements nor template-space map lengths nor the legacy all-compartment retained-length metric.
- **Projection versus optical section:** [optical-plane extraction](inspect_native_optical_planes_current_rint.py:23) selects actual unprojected cached XY planes around the leaf's nearest nominal Z index. The repeated ring marks one leaf's XY position, not a distinct ending on each plane. Orthogonal panels produced by [MIP rendering](assess_native_terminal_fields_current_rint.py:128) are explicitly maxima over the omitted dimension. Their brightness cannot establish a bouton or synapse.

## Independent real-data readback

All exported component-node sets, branch-node sets and full-graph leaf identities agree with independent graph traversal. Native lengths agree within `2.11e-10` nominal µm; selected components are unchanged from the preserved first variant.

| NeuronUID | Declared ARM6 index | Component nodes | Original type-2 leaves | Nominal native length (µm) | Node labels differing between rounding policies |
|---|---:|---:|---:|---:|---:|
| 252790::014.swc | 41 | 6689 | 71 | 59704.682547 | 4 |
| 252790::032.swc | 229 | 61 | 1 | 479.410163 | 1 |
| 252383::018.swc | 729 | 5649 | 45 | 54304.705567 | 0 |
| 252383::121.swc | 1556 | 21 | 1 | 168.604366 | 0 |
| 252384::047.swc | 1004 | 38 | 1 | 257.576851 | 6 |

The selected leaf in 252384::047.swc lies only 3.879 nominal µm from one cube face. Adjacent source coverage and continuation review are required before interpreting apparent cessation; graph termination alone cannot distinguish a biological ending, missed trace or truncated coverage.

I actually inspected the corrected [252790::032 optical planes](review_panels/252790_032_ending_optical_planes.png) and [passage MIPs](review_panels/252790_032_passage_image_and_swc.png). UID, Z/plane and internal-passage labels are readable. The planes show bright curved fluorescence near the candidate XY mark, mainly in planes 9–13; this supports local source-pixel correspondence only. The passage panel marks internal node 7105 and explicitly avoids an ending/no-bouton claim. Earlier clipped/overlapping titles are superseded by these display-only panels.

## Coverage and validation limits

The exact ordered 462-neuron ledger is preserved. Native reconstructions are available for 201 neurons, with 201 ID/type/parent matches. Their 48,574 full-graph candidate leaves include 9,209 with an existing cached cube (18.96%). Cached availability is not complete-image or expert-review coverage. Five purposive examples were generated; this independent review inspected two final PNGs, not every source plane or every candidate. No biologically accepted terminal-field denominator is available; its count remains zero.

Six focused tests pass (0 failures/errors, 0.570 s): original full-graph leaves across region/compartment cuts, ID/type/parent pairing, cube boundaries/ZYX axes, the two explicit atlas-rounding policies, and current pixel-centre positions/internal-passage labels.

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B -m unittest discover -s 'notes/region_analysis_review_20261009/terminal_field_assessment_20261009' -p 'test_native_terminal_assessment.py' -v
```

Before any biological claim, the remaining work is a declared expert review protocol for endings versus passage, adequate adjacent-plane/cube continuation evidence and truncation status, calibrated image/SWC geometry and trustworthy native-to-atlas correspondence. Bouton/synapse claims require evidence beyond reconstructed leaves and these nominally sampled fluorescence panels. None of those unresolved dependencies invalidates the bounded software/source-pixel checks reported here.
