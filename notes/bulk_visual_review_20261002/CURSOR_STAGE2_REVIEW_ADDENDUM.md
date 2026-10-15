# Independent follow-up evidence for stage2

**Agent: Codex | Date: 2026-10-03 (Asia/Shanghai)**

This addendum records independent findings obtained while Cursor stage2 was editing. Apply it after reading the final stage2 result and current source. Do not start a duplicate bulk job, replace protected cohort inputs, or recompress unchanged production NIfTIs for metadata-only repairs.

## Export contract

Read `independent_export_review_20261003.md`. Six temporary-directory cases at toolkit SHA-256 `2828848398a026b0bcdef8ffc02b88d874e08b8f12486c3eb3c00a76eaf08621` reproduced:

1. A real 2D TIFF of shape [4,5] receives persisted XYZ, source ZYX and NIfTI XYZ declarations. Use the real TIFF YX convention and shape; do not assert a NIfTI axis order for TIFF output.
2. Acquisition is selected by suffix alone. A generic high-resolution export can inherit stale low-resolution complete=true metadata; a SomaBlock can inherit a stale high record with a different origin/shape. Explicitly bind or verify matching acquisition geometry/source before propagating coverage. Otherwise record acquisition and completeness as unknown. Preserve valid partial coverage when it is actually matched.
3. A malformed two-value `soma_coords` is accepted as XYZ. Reject malformed/nonfinite geometry before writing outputs, while preserving the legitimate absent-root/null case.

Use meaningful regression checks for these demonstrated failures and run the full suite after final edits. Do not treat the earlier 66-test pass as validation of later source changes.

**Continuation update:** Codex repaired these exporter contracts while the bulk workers were active. Read current source rather than restoring the earlier implementation. The new six regression checks are `tests/test_fmost_export_provenance.py`; the complete suite passed 77 tests with stable source hashes, including toolkit `06ad6ae74e38439c9d5217a010f25c77311de383f48d1586a74dbb2fd0683cd9`. Details are in `independent_root_validation_20261003.md`. Reverify if subsequent edits alter this source; the derived-context issues below remain open.

## Partial context plane provenance

Read `independent_relative_alignment_20261003.md`. The low-resolution loader keeps an array slot for each requested Z plane, including missing planes. The current renderer indexes the shorter loaded-plane list. For requested [10,11,12,13], missing 11 and loaded [10,12,13], array index2 is plane12 but loaded[2] incorrectly labels it plane13. Central-plane, trace-slab and stack labels must use the full array-slot mapping; missing-source status remains explicit. All299 saved copied contexts had complete coverage, so this was not observed in that batch.

The same report raises a possible relative Z-source convention issue. The saved pairs satisfy the current code's n=90*b+q arithmetic, but log/rank image comparisons favor the preceding high-resolution plane. Source TIFF tags do not establish absolute Z calibration. Record the unresolved limitation; do not automatically shift coordinates, origins or arrays.

## Staging lineage and final documentation

Compare rebuilt staging provenance against `expected_staging_origins_20261003.json`:160 identities,32 coordinate-inferred (17 distant),27 ambiguous transitions. Harmonized source and earlier recovery source differ on95 rows; retain both and associate the displayed harmonized label with its harmonized source. Exact input rows and ten source hashes are recorded in that ledger.

The 08:54 continuation comparison (`verify_staging_origins_20261003.py`) passes all manifest lineage fields and ten input hashes, but the then-current correction CSV lacks five lineage fields on each of160 rows. Current writer code already copies manifest fields; its end-of-batch table refresh is pending. Re-run this script after production/table refresh and also check the XLSX before final acceptance. Preserve this distinction rather than adding redundant source fields or treating a stale table as a changed workbook.

Read `independent_goal_note_audit_20261003.md` and `independent_visual_acceptance_20261003.md` when reconciling final notes. Preserve historical review/test snapshots, append later live portal and all-nine component evidence, and keep scientific clustering and anatomical decisions pending. Update current README retrieval routes, anisotropic spacing, workflow links and date. Derived regional context needs its actual spacing, slab, source tiles and gaps; CH1 and physical depth alone do not establish subregion or histological layer.

All nine monkeys and regional context remain part of the original goal. Accessible soma production is a milestone. Finish the bounded context feasibility/production scope and per-component queues required by `CURSOR_STAGE2.md`, then independently verify persisted outputs before claiming completion.

## Derived-context geometry and source safeguards

Read `independent_derived_context_review_20261003.md` and its accompanying minimal evidence before accepting or expanding context production. At helper/pilot SHA-256 `159d313c5f74645a0809909901556d33f43c3415ce9e8a67f2f92401171605ac`, independent temporary synthetic cases demonstrated:

- Coordinate ramps: the first sampled source voxel is XYZ [93.6,93.6,99] um, while the helper reports origin [90.25,90.75,98.2] um. The output grid must match the exact sampled source lattice and Z planes, with explicit downsampling/averaging offsets; arbitrary half-field origins plus floor binning are not sufficient.
- An all-missing source returns a zero canvas labeled derived_partial. A missing central cube with three available neighboring tiles also returns derived_partial with no valid central/root source. Require valid center/root-plane coverage; retain explicit failed component status without claiming a usable context.
- A temporary pilot overwrote a valid existing copied context NIfTI (2x2x2 replaced by 4x4x2). Preserve copied context and any unrelated source artifacts; reject unsupported sample/source modes or write a distinct derived-context artifact.
- Default derived memory is bounded, but factor1 would allocate roughly757 MB; validate explicit decimation/canvas byte/request bounds before allocation. Cache and runtime bounds must be stated for full production, not just a pilot's immediate canvas.

These findings were reproduced without network or production writes. Source is still changing: recheck the finished implementation, fix any surviving defects, and test exact sampled-coordinate ramps, missing central planes/tiles, all-missing sources and existing-context preservation before broad context generation. If a pilot already ran, audit its source geometry and component labels before reuse. Correct derived geometry without silently shifting the raw microscopy calibration.

## Later Codex continuation: derived contracts repaired, pilot active

Read current source before making further changes. `main_scripts/fmost_derived_context.py` replaces the arbitrary-origin floor-binning implementation with a global source sampling lattice, exact plane IDs, explicit coverage, required root source, malformed-cube rejection and preallocation resource bounds. The caller protects copied/unclassified context artifacts and integrates central-plane, trace, stack, orthogonal and source-grid sidecars. Missing section locators stay explicit; point decimation is exploratory and does not establish anatomy or calibrated physical orientation. The HTTP cube reader caps payloads, validates allocation before pixel decoding and writes validated caches atomically.

The complete suite passed 87 tests and the new reader decoded all five previously saved real-source smoke cubes. Exact hashes and limitations are in `derived_context_repair_validation_20261003.json`. This supersedes the earlier 77-test source snapshot for current code, preserving the older evidence. The current turn passes scoped whitespace checks; repository-wide default checks flag existing CRLF lines in the earlier tracer edit.

One real-source pilot for 252790/014 is running in Codex terminal session 5489: 324 cubes, 900-second between-request runtime bound, nominal 5.2 x 5.2 x 3 um source-grid spacing. Do not duplicate that pilot or the active soma worker. The copied trace refresh finished with exit 0. After Cursor stage 2 terminates, reconcile worker outcomes, independently review this pilot, assess source/cache/time budgets for all-five regional production, and verify final component queues, all-nine artifacts and staging lineage in CSV plus XLSX. The original goal remains active.

### 2026-10-03 continuation: real regional pilot independently verified

Agent: Codex. The 252790/014 pilot ended with exit 0 after 200.516 seconds: 249 of 324 source cubes loaded, 75 HTTP 404 failures, no malformed or unattempted cubes, 76.8143% field coverage, and valid central/root-pixel coverage. It remains `derived_partial`. All 249 source hashes passed readback; six independently sampled pixel values matched the saved TIFFs and three missing-source samples had explicit zero fill. Affine, micron units, shape, raw SWC hash/root and ten source plane IDs passed. The audit is `notes/bulk_visual_review_20261002/derived_pilot_readback_252790_014_20261003.json`; source artifacts were stable while read. Codex viewed the unmarked and trace panels; this does not establish region/layer or physical calibration.

A read-only regional preflight is saved as `notes/bulk_visual_review_20261002/regional_context_resource_preflight_20261003.json`. Completed native records in the first two monkeys imply 16,911 and 15,866 distinct 4 mm source cubes; the partial third-sample plan adds 12,789. Aggregate worst-case compressed response bounds exceed currently free disk space. Future regional production therefore needs an explicit shared-cache budget and a recorded smaller-field fallback; retain actual FOV/spacing/source gaps and do not launch the unrestricted plan. Some 4 mm fields crossing a source Z-block boundary also exceed the 400-cube bound; these require a declared fallback or separately justified resource bound. All-nine soma production is still active; final tables and lineage readback await its terminal outcome.

## Concrete next regional-production scope after the soma worker terminates

The first real pilot is now terminal and independently verified; do not relaunch it without a reason. Read its report/readback plus the resource preflight. Continue the original all-nine goal in this branch and same Cursor chat. Implement a resumable, sequential regional production command for the five samples lacking copied sections, with an explicit aggregate cache-byte budget (a 300 GiB starting budget fits the checked free disk snapshot), before broad downloads. Prefer the requested 4 mm field where its per-context plan passes; a declared 2.8 mm fallback can keep fields crossing source Z-block boundaries within the 400-cube/12-GiB per-context bounds. Record each actual requested field, sampling lattice, spacing, tile gaps and source hashes. Never silently alter origin/calibration or borrow another monkey. Check current free space again at dispatch and reserve the maximum response bytes before each uncached cube request.

Reuse successful source tiles and record time-limited own-sample failed-source observations to avoid repeatedly requesting known HTTP 404 tiles within one run; preserve capture time/reason and permit later explicit retries. Do not interpret HTTP 404 as anatomical absence. Prioritize INS/G/PrCO/potential-insula/explicit-Unknown identities and preserve every selected UID, including available copied contexts, failed SWCs and central source failures. Any resource stop must have a concrete per-component reason and resume state. Add an explicit coverage visualization or mask for derived previews so zero-filled missing tiles are distinguishable from acquired zero intensity. Preserve clean source-intensity panels. Full-section locators remain an explicit gap when only native tile fields are available.

After terminal soma production, refresh gallery/tables with current code (the running worker imported an older snapshot), re-run the 160-identity origin verifier and inspect XLSX too. Verify all-nine package components, source-coordinate contracts, human-field preservation and canonical hashes. Independently assess source artifacts and state residual scientific limits. Original methods/clustering review and its literature artifacts must remain included in the final accounting; no accepted k, anatomical label or cohort promotion follows from these software/data checks. No publication is authorized.

### 2026-10-03 continuation: terminal soma batch and bounded regional driver

Agent: Codex. Cursor soma worker 307754 ended with exit 0 at 01:46:40 UTC after 4,645.910 seconds. All 746 selected identities have terminal source outcomes: 697 complete soma packages, 27 partial-neighbor packages, 20 unavailable central sources and two unavailable raw SWCs. The refreshed manifest and CSV pass all 160 staging-origin comparisons with ten source files unchanged; the actual XLSX independently passes the same 160 labels/origins/row references, has 746 unique UIDs and was stable while read. See `terminal_soma_and_xlsx_readback_20261003.json` and `staging_origin_verification_20261003.json`. These are software/source outcomes, not anatomical acceptance.

The new `group_analysis/scripts/regional_context_batch_20261003.py` provides sequential, resumable production for all five overview-pending monkeys. It keeps all selected identities, prioritizes INS/G/PrCO/potential-insula/Unknown, preserves copied contexts, bounds shared cache bytes/request attempts/free space/runtime, uses dated expiring own-sample HTTP-404 observations, and records a 2.8 mm field fallback when 4 mm crosses a per-context resource bound. Resource/transient partial results are retried on resume; explicit `--retry-missing` retries observed 404 gaps. A live OS lock prevents duplicate regional batches. Raw-SWC lineage changes cannot relabel existing soma crops. Every derived field includes an intensity-independent coverage PNG/NIfTI and actual requested FOV/spacing.

The complete suite passes 99 tests. A real saved-pilot CLI resume in an isolated workspace made no network requests, preserved intensity bytes and emitted the coverage panel/fraction in its table; the mask agreed with all nine independent source sample checks and the measured 76.8143% coverage. Current source hashes are saved in `regional_driver_validation_20261003.json`. Broad regional production has not yet been dispatched; Cursor stage 2 is still returning its terminal verification/report, then the same chat will review/run this driver. Full-section locators and fine region/layer acceptance remain explicit gaps. The original full goal remains active.

Concrete available command after reviewing current sources and ensuring the soma worker is terminal:

```powershell
& C:/Users/laika_yan/miniconda3/envs/projectome/python.exe -B group_analysis/scripts/regional_context_batch_20261003.py --cache-gib 300 --min-free-gib 50 --max-requests 60000 --max-seconds 28800
```

Run one regional batch at a time. Retain its actual PID/session and per-UID ledger; observation timeout is not terminal. On a genuine terminal budget stop, inspect/cache/free-space state and resume the same full scope. Recheck the 99-test snapshot if edits change code. Do not restart the terminal soma batch.

### 2026-10-03 continuation: Cursor regional production active

Agent: Codex. Cursor launched the validated regional driver for 252790 in terminal 307757 (launcher PID 84844; Python PID 812), selecting all 113 candidates and preserving insula-first processing. This is the same active Cursor stage-2 chat; no duplicate batch was launched. The prior all-nine verifier decoded 5,121 PNGs, reported no problems and preserved canonical hashes.

Codex independently read back all 11 newly completed contexts in the captured live ledger: 3,062 source-hash checks and 99 sampled-pixel/missing-source checks, with affine, source-plane, raw-root and stable-artifact checks passing. Evidence is in `live_regional_readback_20261003.json` and the per-UID readback files. These contexts remain honestly partial where HTTP-404 tiles are missing. The live ledger reports no admission stops and remains within its 300-GiB shared-cache and 60,000 request-attempt upper bounds. Other monkeys and final all-nine regional verification remain part of the full active goal. Preserve the exact live process/session rather than restarting on observation timeout.

### 2026-10-03 terminal-ledger preservation follow-up

Agent: Codex. At 10:28 local time, the current 252790 producer remained live;
21 finished regional fields had passed independent source/pixel/coverage-mask
readback. It loaded driver SHA `593e15546a021f52edc5aa9d70d967b02e0383cba19a0189f96e08b49105387a`.
Do not restart it to pick up the following provenance change.

The on-disk driver now assigns a unique run ID and saves the full terminal state
in `manifest/regional_runs/<run_id>.json` before updating the current-state ledger
and refreshing the gallery. This preserves prior selected identities, failures,
and resource accounting through successive monkey runs and gallery errors.
The full 99-test suite passes with the extended two-run/gallery-failure regression.
The latest source snapshot is `terminal_ledger_validation_20261003.json`.
Capture the active old-version worker's terminal ledger before moving to another
monkey; subsequent invocations write their own terminal records automatically.
Use `verify_completed_regional_20261003.py` for independent new/changed fields;
it also saves a terminal ledger observation when present. Continue the full
remaining regional scope and final all-nine table/gallery/source checks.

### 2026-10-03 full-field preservation correction

Agent: Codex. The preceding 400-cube/12-GiB per-field bound reduced 4 mm fields
at Z-block boundaries although the output canvas and working estimate were
unchanged. The updated batch planner/sampler admits up to 800 streamed cubes
and 26 GiB of bounded responses per field. The 128 MiB working estimate,
300 GiB shared cache, 60,000 request-attempt bound, 50 GiB free-space margin,
payload/retry bounds and runtime admission are unchanged. No dense source-block
or whole-brain assembly is introduced.

All 445 cached inputs in the five overview-pending monkeys admit 4 mm/30 um
fields: maximum 684 cubes and 121,325,432 estimated working bytes. The old bound
rejected 59 plans. The full suite passes 101 tests, including full-field sampling
across two Z blocks, sampler-bound propagation, failed-replacement preservation,
and rejection of undersized resume artifacts. See `full_field_validation_20261003.json`
and `full_field_preflight_20261003.json`.

Do not interrupt or duplicate PID 812; it loaded the earlier code. After it is
terminal, capture its ledger and resume with the updated driver for the full
remaining scope. Valid 4 mm fields are reused; smaller fields are retried with
the preferred wider plan. At preflight, 252790/058, /061, /037 and /078 require
upgrades; inspect terminal reports for additional reductions. Actual wider
acquisition and independent rendering checks are required before completion.
Do not count a smaller field as satisfying the preferred context merely because
its geometry passes. Keep tested source/global-budget failures and resume states
explicit when acquisition cannot finish.

## Validated unresolved-metadata candidates, 2026-10-03 continuation

Agent: Codex. The 86-row metadata-resolution queue is not a raw-source absence
list. All 86 own-sample raw SWCs were fully downloaded and structurally validated:
84 in 251730 and two in 252383 (136 and 140). The separate validated files contain
1,906,268 nodes and 177,407,210 bytes; hashes, roots, routes and outcomes are in
`unresolved_full_source_validation_20261003.json`. The live manifest was unchanged.
All 86 own-root central cube routes return TIFF prefixes and admit 4 mm plans
(`unresolved_image_preflight_20261003.json`); prefixes do not establish complete
TIFF decoding, calibration or anatomical acceptance.

The opt-in extension is tested in an isolated preview: 832 unique identities,
every field of all 746 original rows unchanged. The 86 additions use
`metadata_unresolved` and `metadata_unresolved_raw_source`; blank portal labels
remain blank and the resolution queue persists. This is a review queue, not an
insula label or accepted-cohort extension. All 103 tests pass; source hashes are
in `unresolved_candidate_validation_snapshot_20261003.json` and the real preview
comparison is `unresolved_manifest_preflight_20261003.json`.

After the current producer and agent are terminal, preserve their results and
the legacy regional ledger. Activate with:

```powershell
& 'C:\Users\laika_yan\miniconda3\envs\projectome\python.exe' -B group_analysis/scripts/visual_review_20261002.py manifest --unresolved-report notes/bulk_visual_review_20261002/unresolved_full_source_validation_20261003.json
```

Reconcile the activated UID set against the preview before a new producer.
Regenerate CSV/XLSX and gallery through the existing conflict-checked human-field
merge. Produce additional soma packages using resumable
`render-soma --sample 251730 --sample 252383`. Run the updated regional driver
across all five overview-pending samples, including these additions and reduced
field upgrades; preserve terminal ledgers and actual source/resource failures.
Do not change the live manifest or start a second producer now. Completion at
746 would omit these validated unresolved candidates. Scientific acceptance
remains separate from production of the 832-candidate review population.

The subsequent manifest-retention check passes 105 tests. Production and manifest
replacement now share the same OS lock through `main_scripts/fmost_review_lock.py`.
Ordinary regeneration retains the activated, hash-bound unresolved-source report;
a changed or missing report stops regeneration. The real 832-row preview was
regenerated without an explicit report and remained byte-identical. Existing 746
rows and the live manifest remain unchanged. Latest test/code evidence is
`manifest_retention_validation_20261003.json`; the 103-test snapshot is historical.

The next driver also honors the requested high-insula-yield priority within each
category, using descending sample INS candidate counts before sample/neuron IDs.
The actual extended regional queue contains 531 unique UIDs and retains all 86
metadata-unresolved candidates. Selected INS counts are 252790:38, 250432:8,
252383:6, 252718:4, and 251730:0; these are review bins, not accepted anatomy.
All 106 tests pass. The ordered real queue and current source hashes are in
`insula_priority_preflight_20261003.json` and `insula_priority_validation_20261003.json`.
The existing single-sample producer is unchanged and must finish before another
batch starts. The preceding 105-test snapshot remains historical evidence.

A bounded read-only watcher holds a SYNCHRONIZE handle to legacy PID 812 and
observes its current-state ledger. It will preserve the actual terminal ledger
under `terminal_regional_run_<started-UTC>_812.json` before a successor replaces
the current-state file. Observer state is `legacy_terminal_watch_812_20261003.json`;
session 31824. A timeout is an observation result, not evidence of producer exit.
The watcher neither controls production nor modifies its ledger. Check actual
terminal capture before starting a root-dispatched successor.

The first batch's exact input is preserved as
`identity_manifest_746_pre_extension_20261003.csv` (SHA256
`ca7b00a17aa53b64280239904d8259d22c7684b15b91208da953ff295b6c2632`).
The independent verifier accepts `--ledger`, `--manifest`, and `--summary-path`,
so a saved terminal ledger can be checked against its original input after the
live population expands. Real snapshot-based execution passes for 99 current
fields in `pre_extension_regional_readback_20261003.json`. This is current-output
readback for the selected identities; historical producer outcomes remain in
the saved ledger, rather than being inferred from upgraded files. Use the
preserved manifest with the legacy ledger for its final scope check.

The legacy batch is now terminal: 113/113 verified, 9 complete-source and 104
partial-source fields; sixteen actual 2.8 mm fields require 4 mm upgrades.
The observer captured `terminal_regional_run_20261003T015315305288Z_812.json`
(SHA256 `f422f3aaf224de935a86c3202b7e58eea927ac23e34daed3bde2c6481679cd39`).
Final readback against its frozen input is
`terminal_first_regional_readback_20261003.json`.

Your successor `f0a8ad4c5c034ebeb76924873ca1b630` (PID 117764) is active for
332 original candidates in the other four pending-overview samples. Preserve
its UUID terminal ledger before extending the manifest. Its live input remains
746 rows. After terminal capture, activate the validated unresolved-source report
as described above, verify 832 identities and preserve human table fields, then
produce the 86 added soma/regional contexts and upgrade the sixteen smaller
252790 fields. Reuse adequate 4 mm outputs. The completed first batch alone,
or completion of the 332-row successor alone, does not satisfy the full goal.

The 86 additional own-sample raw SWCs are now prepared in the production
`swc_raw/{sample}/{neuron}` cache. `unresolved_raw_cache_preparation_20261003.json`
records all 177,407,210 bytes, exact validated hashes and production-loader
readback of graph size/root/source URL with zero network requests. All 86 UIDs
were checked absent from the live manifest and equal to the preview extension.
No existing differing cache was overwritten; no manifest or protected input
changed. After extension activation, rendering can reuse these files directly.
This prepares sources only; the 86 image packages remain to be produced.

Cache-only soma preflight checks the exact single-cube or 3x3x3 neighborhood
required by the current crop-edge rule for each of the 86 additions. Zero have
all required header-valid cubes cached (`unresolved_soma_cache_preflight_20261003.json`).
This is a cache state, not a source-unavailability claim. Run the added soma
downloads sequentially after the active regional source client finishes; its
resource accounting assumes one sequential batch and does not reserve another
producer's concurrent cube downloads.

Current review code now labels regional captions with requested/acquired/missing/
invalid/unattempted cube counts and voxel coverage percentage. Existing 106 tests
pass (`caption_validation_20261003.json`); your running process retains the prior
loaded caption code. Two representative marked/trace pairs were regenerated
from saved volumes with unchanged source metadata, intensity, masks and trace
slabs (`derived_caption_refresh_20261003.json`). After production finishes,
refresh all completed derived captions using `refresh_derived_captions_20261003.py`
with repeated `--uid SAMPLE|NEURON.swc`, then rerun independent regional readback
and gallery/table verification. This requires no cube downloads or NIfTI rebuild.

The completed 113-field 252790 batch has now received this saved-volume caption
refresh in full (`first_batch_caption_refresh_20261003.json`), with all protected
artifacts and trace slabs unchanged and zero downloads. Independent final
readback again passes all 113 fields against the saved ledger/input; the sixteen
smaller fields still need upgrades. Remaining successor captions retain your
prior loaded code and need the final refresh after production.

At 2026-10-03 14:57 Asia/Shanghai, root started the local checkpoint watcher
`watch_regional_checkpoints_20261003.py` for successor run
`f0a8ad4c5c034ebeb76924873ca1b630` (watcher PID 104848, producer PID 117764).
Its first real checkpoint verified 32 successor fields / 145 across both scopes,
with annotations and canonical inputs preserved. Current status is persisted in
`regional_checkpoint_watcher_20261003.json`. It runs the existing regional
verifier and gallery/workbook readback sequentially, starts zero downloads,
binds the producer run ID and input hash, and stops on terminal state, source
process exit, identity drift or verification error. While it is live, avoid a
second writer of the same successor audit/gallery checkpoint. After it stops,
root must inspect its result and coordinate the remaining production steps;
this watcher does not activate the 832-row manifest or start another producer.

### 2026-10-03 17:04 Asia/Shanghai - Codex: lossless diagnostic export recovery

The independent checkpoint watcher stopped on a real CSV/XLSX discrepancy: Excel truncated a 34,956-character regional source-error list to 32,767 characters. The Python exporter now retains oversized lists in content-hashed JSON sidecars referenced identically in both tables. Short lists remain inline. Other oversized text, including reviewer notes, raises before replacing either review table. Full source provenance is unchanged. Added regression tests cover lossless list readback and preservation of oversized reviewer input; the current suite passes 108 tests.

Live refresh independently checks sidecar hashes, complete payloads against each own-sample provenance, CSV/XLSX equality, gallery links, human fields and canonical hashes. The failed watcher record is preserved as `regional_checkpoint_watcher_before_excel_fix_20261003.json`; replacement watcher PID 85564 watches the same run and producer PID 117764. Its first checkpoint passed for 64 successor fields / 177 across both regional scopes. No source producer was restarted. Evidence: `excel_limit_validation_20261003.json` and `excel_limit_tests_20261003.log`. The 832-identity scope, 86 additions and sixteen field upgrades remain required; this export fix does not constitute scientific acceptance or overall completion.

The live producer loaded the exporter before this fix. After PID 117764 exits, run the current gallery/table refresh and independent readback again even if its terminal gallery write appears successful; it may have used the older inline diagnostics. The independent watcher uses fresh subprocess imports of the corrected code.

### 2026-10-03 - Codex: unavailable raw-root regional source for 252718/145

At the 88-attempt successor checkpoint, 87 fields in that run have passing image readback (200 across both regional scopes). `252718|145.swc` instead records `context_unavailable`: its own-sample raw-root cube `[195, 84, 96]` returned HTTP 404. This matches its previously recorded `center_unavailable` soma outcome. The requested regional field is 4 x 4 mm x 30 um, with one missing root tile and 323 tiles left unattempted because the raw-root coverage gate stopped construction. No usable regional image is counted for this identity.

The exact source outcome is retained in `group_analysis/visual_review_20261002/manifest/derived_context_pilot_252718_145.json` and the live UUID ledger. Treat the HTTP 404 as a dated source observation; it does not establish anatomical absence or confirm permanent unavailability. Include this UID in the final unavailable-source audit and retry review after the current producer finishes, respecting the existing negative-cache TTL or explicit retry option. Preserve the original candidate scope. No other sample, processed SWC or NMT coordinate fallback is authorized for a native image.
