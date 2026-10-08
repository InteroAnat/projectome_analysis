# Staged descriptive-map wording, 2026-10-09

`terminology.patch` stages three script and two figure-guide edits. Production files, map provenance, NIfTI maps and PNG figures were not changed. `static_validation.json` binds original/staged hashes and confirms all 344 protected originals remained unchanged. The staged Python files compile; their ASTs match production after removing only module descriptions, CLI help, and the explicitly approved display strings/title. Metric IDs, columns, argument names/defaults/choices, output paths, formulas, checks and numerical computations remain identical. No producer or renderer was executed.

| Display term | Exact interpretation |
|---|---|
| Axon-labelled segment length | Length of edges selected by the existing SWC axon-type rule, distributed in the NMT reference grid. Includes passing segments and branches. Compartment labels and anatomical registration remain unverified. |
| Candidate axon end | A non-root axon-labelled node with no children in the complete stored graph. It is not an image-reviewed terminal arbor, synapse or demonstrated termination. |
| Endpoint-eligible neuron | A neuron with at least one candidate axon end anywhere in its complete graph, including outside the reference field of view. The machine field `n_computable` retains its existing name. |
| Within-animal axon length/density | Sum over selected neurons in that animal and declared source group, divided by the number selected; density also divides by reference voxel volume. |
| Within-animal candidate end density | Count of candidate ends in a voxel, divided by endpoint-eligible neurons in that animal/source group and by reference voxel volume. |
| Occupancy | Fraction of endpoint-eligible neurons with at least one candidate end in a voxel. Several ends from one neuron count once. This denominator includes eligible neurons whose ends lie entirely outside the reference. |
| Across-animal average | Equal average of the contributing animal maps for the same declared source group. Counts pooled across animals are not the denominator. Absent animal/source groups are not filled with zeros. |

Plot labels use “candidate axon ends” and “eligible neurons,” with a short explicit eligibility caption. Length labels identify axon-labelled segments. Color labels retain the existing display transformation: `log10(1 + density)` for density and linear 0–1 for occupancy. The changes make no projection-strength, significance or t-map claim. Source-group identifiers remain unchanged, because they carry human/atlas/candidate evidence strata rather than interchangeable fine anatomical parcels.

The two staged guide additions link to the root-owned `docs/region_analysis_terminology.md`; that note did not yet exist at static validation. The parent is writing it. Previous producer/renderer hashes are historical receipts of the saved delivery: do not rewrite them to current code hashes after applying this wording patch.

Applying the renderer wording changes would change text in a future figure variant. The existing figure bytes remain preserved. New caption/title layout requires validation when the parent renders that separate variant; this static pass does not claim visual layout acceptance. Staged code copies retain their repository-relative layout for review, but are not standalone runnable copies because ROOT/import resolution belongs to the production location.
