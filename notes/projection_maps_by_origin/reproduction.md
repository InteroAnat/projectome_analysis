# Reproduce source-origin maps and count reconciliation

Run from `D:/projectome_analysis` using the existing projectome Python environment. Requires numpy, pandas, nibabel, matplotlib and openpyxl. Exact environment versions, input hashes, code hashes, parameters and image geometry are recorded in the producer receipts. Use a fresh destination; producers reject existing directories. Preserve the original 462-neuron ledger, two original complete-graph endpoint runs, their hashed leaf records, prior independent inspection/display receipt, NMT MRI, ARM labels/key and source SWCs.

## Procedure

1. Review the dated July tracker CSVs, older 306/353 Summary sheets, September staging Summary, registry and current 462 ledger. The count reconciler checks exact UID uniqueness, all retained/added identities and the eight unregistered September entries. It distinguishes dated tracker counts from the local older identity workbook's uncertain historical byte date.
2. Run the source-parcel producer. It checks source hashes, unique soma/root metadata, current `rint(XYZ/250)` lookup, direct official ARM6 assignment and endpoint-run membership/eligibility. It writes raw selected soma counts and projection-density NIfTIs with unchanged established projection values. Locator cuts maximize per-parcel soma counts; target cuts remain X=56, Y=200, Z=87. The new titles specify neuron origin.
3. Run the evidence-view producer. It joins both complete-graph leaf ledgers by exact sample/filename, independently checks their hashed source lineage and known graph census, and assigns the unchanged 462 selection to six exhaustive evidence categories. Endpoint density is `(sum candidate ends / eligible neurons)` per animal/category, then the equal mean over contributing animals, divided by 0.015625 mm³. Neurons without original candidate ends are endpoint-ineligible rather than biological zeros. Ends outside the image count for eligibility but not for image voxels. No missing animal is added as zero.
4. Run focused regressions, then the independent verifier. The verifier imports neither producer and rereads all original SWCs. It derives source somata and original non-root type-2 leaves with no children in the full graph; target voxelization uses right-sided half-open face lookup. It reconstructs every new endpoint map, checks raw soma counts, numerical equality to established named maps, NIfTI shape/affine/coded transforms/mm units, source membership, category counts and displayed slice data. Passing software tests does not establish anatomical membership or terminal-field acceptance.
5. Inspect all nine source cards and six evidence figures, plus the per-monkey comparison. Confirm full ARM names, source-versus-target panel headings, visible marker contrast, complete colourbar labels and actual source-specific locator cuts. Preserve a dated hash-bound review receipt for a new run.
6. Use exact source identities from `named_ARM_origins/neuron_origin_membership.csv` with the existing six-level `arm_projection_hierarchy_tables.xlsx` and `per_neuron_regional_measures.csv`. Preserve target hierarchy, status and measure-specific denominators; do not sum the same axon across hierarchy levels or treat source categories as independent animals. Existing whole-axon and end-branch maps remain separate measures.

## PowerShell commands

```powershell
$originPython = 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe'
$originRun = 'group_analysis/evolution_20261008/soma_origin_maps_reproduction_20261010'
& $originPython -X utf8 -B notes/projection_maps_by_origin/reconcile_neuron_counts.py --output "$originRun/count_reconciliation"
& $originPython -X utf8 -B notes/projection_maps_by_origin/investigate_source_additions.py --output "$originRun/source_investigation"
& $originPython -X utf8 -B group_analysis/scripts/map_projections_by_soma_origin.py --output "$originRun/named_ARM_origins"
& $originPython -X utf8 -B group_analysis/scripts/map_insula_origin_evidence.py --output "$originRun/evidence_categories"
& $originPython -X utf8 -B -m unittest discover -s notes/projection_maps_by_origin -p test_origin_membership.py -v
& $originPython -X utf8 -B notes/projection_maps_by_origin/verify_origin_maps.py --output-root $originRun --receipt "$originRun/independent_readback.json"
& $originPython -X utf8 -B notes/projection_maps_by_origin/summarize_display_scale.py --output-root $originRun --output "$originRun/scale_display_qc.csv"
```

Stop on a nonzero process exit. These commands reuse independently checked prior endpoint leaves; the verifier reconstructs their endpoints directly from SWCs. Rebuilding the original numerical projection/whole-axon runs, if necessary, follows the existing [full reproduction procedure](../projection_map_review_round2/reproduction.md). Local source access is required; no missing transform is synthesized. Hash changes require renewed validation rather than changing a receipt to match new bytes.
