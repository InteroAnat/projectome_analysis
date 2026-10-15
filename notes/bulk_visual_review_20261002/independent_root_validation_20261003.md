# Independent continuation validation

**Agent: Codex | Date: 2026-10-03 (Asia/Shanghai)**

The active goal still covers all nine new monkeys. Copied overview availability is a source component, not a criterion for removing a monkey from that goal. Cursor's second pass is running in chat `8c6b7701-7bd8-4736-a762-e7399a8a7b72`, with the CLI reporting `Grok 4.7 256K Extra High`.

## Continuation snapshot during early stage2

At approximately 08:08, Codex independently ran the complete suite with the projectome environment:

```text
python -B -m unittest discover -s tests
Ran 66 tests in 1.326s
OK
```

SHA-256 values of the visual-review script, image toolkit, geometry module and visual-review tests were unchanged across this short run. Stage2 was already active and had made early exporter edits. This validates only the source snapshot executed during that run; it does not validate the final stage2 implementation or establish anatomical acceptance.

The completed Cursor result and fresh verification record report 746 candidate identities: 301 with copied overview sources and 445 initially queued. Among the 301, there are 277 complete packages, five packages with missing neighboring cubes, 17 with unavailable central soma cubes and two with unavailable raw SWCs. These are component coverage labels. They are not human region, layer or tracing decisions.

An independent comparison of saved context and soma arrays is recorded separately in `independent_relative_alignment_20261003.md`. It raises a possible relative Z-index issue; no coordinate correction is justified without independent calibration.

## Outstanding second-pass checks

- Match traced edges to the image projection depth and retain clean central-plane panels.
- Correct export sidecars without changing already acquired raw arrays unnecessarily.
- Preserve exact staging label origins, duplicate annotation conflicts and annotations on identities removed by a manifest rebuild.
- Acquire accessible own-sample native components for the other five monkeys; test bounded regional-context feasibility and record exact component gaps.
- Keep blank portal metadata in a separate resolution queue rather than assigning an atlas region.
- Independently review final tests, identity coverage, source geometry, persisted pixels and canonical hashes before reconciling the project note.

No commit, push, publication or scientific acceptance is represented by this continuation record.

## Exporter repair during active bulk production

**Agent: Codex | Date: 2026-10-03 (Asia/Shanghai)**

The independently reproduced exporter failures were converted into six executable regression checks in `tests/test_fmost_export_provenance.py`. The initial run reproduced incorrect 2D axes, stale acquisition selection and malformed/nonfinite roots. Valid matching-partial acquisition and absent-root/acquisition cases passed.

`Visual_toolkit.export_data` now validates dimensions, finite origin/spacing and an optional three-value finite raw root before output creation. A 2D TIFF reports YX pixels and shape, without a NIfTI axis declaration. Source acquisition is bound to matching sample, units, axes, shape, origin and spacing; a suffix cannot choose stale coverage. Ambiguous or absent matching acquisition remains unknown. An explicitly supplied acquisition must match. Matching acquisition metadata is copied so later in-memory edits cannot mutate the exported record.

The existing coverage fixture was completed with realistic own-sample acquisition geometry rather than expecting an incomplete dictionary to assert coverage. The image arrays, transpose convention, micron units and valid partial-source semantics remain covered.

```text
python -B -m unittest discover -s tests
Ran 77 tests in 1.867s
OK
git -c safe.directory=D:/projectome_analysis -c core.whitespace=cr-at-eol diff --check
exit 0
```

All five checked sources were stable across this run:

| Source | SHA-256 |
|---|---|
| main_scripts/Visual_toolkit.py | 06ad6ae74e38439c9d5217a010f25c77311de383f48d1586a74dbb2fd0683cd9 |
| tests/test_fmost_export_provenance.py | c405d14069c8db2f48f433c66d794f59b0260307ce59dc21906ca8a4b40b385b |
| tests/test_fmost_image_loader.py | 782d66eb4459c43196bbdbb85dca0c7aa52c9ef549ffc3c5c665f7be261f2a09 |
| group_analysis/scripts/visual_review_20261002.py | 159d313c5f74645a0809909901556d33f43c3415ce9e8a67f2f92401171605ac |
| tests/test_visual_review_manifest.py | 19cc1eeba0408afe502173d2536744e0e69e3c1b76419ae026ad06d812a0bec4 |

This is a source-snapshot validation while Cursor bulk workers are active, not final all-nine production acceptance. The reproduced derived-context grid, missing-center and existing-context preservation issues still require repair and verification. The earlier 66-test count remains a separately dated snapshot.

## Staging provenance readback while the table refresh is pending

At 08:54, `verify_staging_origins_20261003.py` compared all 160 expected staging identities with the actual manifest and correction CSV, checking exact harmonized labels, their origins, earlier recovery origins, source NeuronUIDs and source-row references. All ten ledger input hashes matched. Both CSVs were stable during the read.

The manifest passed every comparison. The current correction CSV was an older output and lacked five lineage fields, yielding 800 missing-field comparisons (five per identity). Inspection of the current writer confirms that it copies all manifest fields into the correction table; the active `render-soma` worker invokes its final gallery/table refresh after all five samples finish. This is a pending output-refresh requirement, not a reason to discard label lineage or change scientific labels. Re-run against the refreshed CSV and independently inspect XLSX after the worker finishes.

The complete result, including the expected temporary failure, is saved in `staging_origin_verification_20261003.json`. Source-provenance preservation is not yet verified in the final human-facing artifacts.

## Later Codex continuation: derived contracts repaired, pilot active

Read current source before making further changes. `main_scripts/fmost_derived_context.py` replaces the arbitrary-origin floor-binning implementation with a global source sampling lattice, exact plane IDs, explicit coverage, required root source, malformed-cube rejection and preallocation resource bounds. The caller protects copied/unclassified context artifacts and integrates central-plane, trace, stack, orthogonal and source-grid sidecars. Missing section locators stay explicit; point decimation is exploratory and does not establish anatomy or calibrated physical orientation. The HTTP cube reader caps payloads, validates allocation before pixel decoding and writes validated caches atomically.

The complete suite passed 87 tests and the new reader decoded all five previously saved real-source smoke cubes. Exact hashes and limitations are in `derived_context_repair_validation_20261003.json`. This supersedes the earlier 77-test source snapshot for current code, preserving the older evidence. The current turn passes scoped whitespace checks; repository-wide default checks flag existing CRLF lines in the earlier tracer edit.

One real-source pilot for 252790/014 is running in Codex terminal session 5489: 324 cubes, 900-second between-request runtime bound, nominal 5.2 x 5.2 x 3 um source-grid spacing. Do not duplicate that pilot or the active soma worker. The copied trace refresh finished with exit 0. After Cursor stage 2 terminates, reconcile worker outcomes, independently review this pilot, assess source/cache/time budgets for all-five regional production, and verify final component queues, all-nine artifacts and staging lineage in CSV plus XLSX. The original goal remains active.

### 2026-10-03 continuation: real regional pilot independently verified

Agent: Codex. The 252790/014 pilot ended with exit 0 after 200.516 seconds: 249 of 324 source cubes loaded, 75 HTTP 404 failures, no malformed or unattempted cubes, 76.8143% field coverage, and valid central/root-pixel coverage. It remains `derived_partial`. All 249 source hashes passed readback; six independently sampled pixel values matched the saved TIFFs and three missing-source samples had explicit zero fill. Affine, micron units, shape, raw SWC hash/root and ten source plane IDs passed. The audit is `notes/bulk_visual_review_20261002/derived_pilot_readback_252790_014_20261003.json`; source artifacts were stable while read. Codex viewed the unmarked and trace panels; this does not establish region/layer or physical calibration.

A read-only regional preflight is saved as `notes/bulk_visual_review_20261002/regional_context_resource_preflight_20261003.json`. Completed native records in the first two monkeys imply 16,911 and 15,866 distinct 4 mm source cubes; the partial third-sample plan adds 12,789. Aggregate worst-case compressed response bounds exceed currently free disk space. Future regional production therefore needs an explicit shared-cache budget and a recorded smaller-field fallback; retain actual FOV/spacing/source gaps and do not launch the unrestricted plan. Some 4 mm fields crossing a source Z-block boundary also exceed the 400-cube bound; these require a declared fallback or separately justified resource bound. All-nine soma production is still active; final tables and lineage readback await its terminal outcome.
