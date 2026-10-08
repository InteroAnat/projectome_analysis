# Publication log — 2026-10-09

User authorization: “Ensure all updates are logged and pushed.” Work is scoped to `codex/insula-pipeline-evolution-20261008`, based on `78008cbaab71bf737dc7129c72e5fe284c3ba40f`; the default branch is `master` and is not the push target.

## Scope

Publish reviewed pipeline sources and their necessary tested baseline dependencies, focused regressions, complete dated inventory/QC, citation/methods records, corrected diagnostic derivatives, one primary ARM figure set, six-level matrices, compact clustering/MSTIM outputs, reproduction commands and reports. Native-context and FNT dependencies receive a separate reviewable baseline commit. Preserve unrelated journals/configuration, scratch work, original data and canonical inputs. Redundant historical/animal/QC figure variants remain archived locally with hashes; their provenance remains published.

The subsequent user correction requires full ARM names in the NMT maps. Earlier HumanINS/G/Candidate text identified custom selection groups, not ARM parcels. New variants regroup the same sources by their direct ARM level-6 lookup. Full names are joined through the pinned ARM key, with cortical/subcortical identity and hemisphere retained. Human/candidate decisions remain source-attributed metadata. Background/outside cases remain unresolved. No separate atlas or manual parcel crosswalk supplies the new map regions. Previous maps remain historical derivatives.

Scientific NIfTI/SWC inputs and maps remain local under the existing ignore rules. Large per-node endpoint ledgers also remain local because they exceed practical Git artifact limits; their hashes, region/QC summaries and commands are published. The publication manifests list these local artifacts explicitly. Their absence from Git is not a claim that the binary maps were uploaded.

An inherited embedded SSH password was removed from the source selected for publication. Optional SSH now reads `PROJECTOME_SSH_PASSWORD` from the environment. Its value is not logged or added to a new artifact. The IONData import path also resolves from the checkout for reproducibility. Existing credentials in previously published history, if still active, require rotation by the account owner; this task does not rewrite repository history.

## Validation and commits

Final software, source/derivative preservation, ARM-label identity, map review and display receipts are linked from the [audit report](README.md). Syntax/unit checks remain separate from anatomical acceptance. File/hash manifests and credential scan outcomes are prepared before staging. Commit and live remote verification records are appended here after each completed push; no pending push is called complete.

Completed local commits: `c6bea20287c7d53e498fa813927c2177d2efba0f` (audited reconstruction/native-context dependencies), `a3fa188` (region identity, units, exports and official ARM labels), `d95b9be` (six-level tables), `3671700` (projection-profile clustering), `534344d` (coverage-aware MSTIM summaries) and `adf64fe` (one primary ARM display set by default). These were not yet remotely published when recorded. Commit authorship uses the GitHub noreply address.

The required primary ARM stages passed before optional rendering was deliberately stopped at the user's request to reduce redundant outputs. The scope-reduction receipt preserves that distinction. The single 462-neuron hierarchy export reconciles 154 saved NIfTIs and retains 429 eligible endpoint rows plus 33 NA rows. Primary clustering reveals hemisphere-driven axon separation and unstable endpoint partitions; provisional CM032 integration reproduces the contrast exactly but supplies no accepted registration or biological association. Scientific limitations remain in the linked reports.

The final live suite passed **409 tests**, with zero failures, errors or skips. Independent saved-table/profile/map checks are separate receipts. Primary axon clusters closely follow source hemisphere; relative-side sensitivity changes the partition and weakens stability, and endpoint cuts remain unbalanced. No cell types are accepted. Hash-bound artifacts will be staged with command-scoped `core.autocrlf=false` and checked against exact index bytes, so Windows line-ending conversion cannot invalidate their recorded hashes.

The [final delivery check](final_delivery_receipt_20261009.json) passed: 4,973 source bindings over 4,955 paths, five corrected diagnostic workbooks, 245 historical and 320 current ARM NIfTI files, 12 primary ARM sheets, and the saved hierarchy/clustering/MSTIM independent receipts. Source hashes and complete earlier numerical checks agree; original producer receipts are preserved. No source data were written and no anatomical acceptance is claimed.

Status: verified delivery; exact-byte staging and remote publication in progress.
