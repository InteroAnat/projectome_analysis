# Remaining scope and independent image review

**Agent: Codex | Date: 2026-10-03 (Asia/Shanghai)**

Continue after CURSOR_FOLLOWUP.md is completed; do not restart valid production.
The user explicitly instructed continuing the full goal after receiving Cursor
results. Preserve the current branch, source data and unrelated dirty files.
No commits, push, publication, cleanup or unrelated synchronization.

## Additional independently reproduced defects

1. The trace preview projects SWC edges from the full crop onto one central
   image plane. `_segments_in_view` clips the full volume, but context/soma
   trace backgrounds use `volume[floor(local_z)]`. For 252527/048 soma,
   1,598 of 1,679 drawn edges are wholly outside that plane's 3 um index bin;
   edges span roughly -356 to +245 um around the displayed plane. Other
   examples are 252385/003 (891/917) and 252714/002 (4,323/4,409).
   Match the trace bounds to the actual displayed background: use a clearly
   documented narrow slab (e.g. five source sections) and its MIP, with the
   SWC clipped to exactly the same slab, or a central-section-only trace.
   Keep unmarked/marked central planes for independent soma inspection.
   Label slab start/stop, nominal depth and source plane IDs. Do not project
   an entire 810 um high crop onto a single 3 um plane. Regenerate previews
   using persisted raw crops; add an off-plane-edge regression test.
2. Export sidecars currently confuse origin with soma center and exported
   XYZ with source ZYX. Example 252385/003/soma.nii.gz.json top-level center
   is [51480,31590,17010], while the verified raw soma is
   [51564.9258,31622.2383,17183.0352]. Nested acquisition is correct.
   Top-level complete can be true despite missing acquisition neighbors,
   and output_file keeps the temporary name after the NIfTI is renamed.
   Keep source array order ZYX distinct from persisted NIfTI XYZ, pass the
   actual raw root, propagate completeness/coverage consistently and point
   to the actual saved file. Repair JSON metadata without recompressing
   unchanged NIfTIs. Add export contract tests.
3. The correction table drops staging `Soma_Region_Source`. Among its 160
   new-sample staging identities are 32 coordinate-inferred labels and 27
   ambiguous atlas-to-manual mappings; the current generic `staging_sources`
   field cannot distinguish them. Preserve the original per-row origin field
   and exact source workbook/row UID, keeping inferred/reference/atlas labels
   distinct from blank or later human decisions. Add provenance reconciliation
   assertions. All 32 PrCO selection reasons already survive; keep them.
4. The new human-field merge still silently loses decisions in two cases:
   duplicate UIDs within one CSV/XLSX overwrite earlier rows, and annotated
   UIDs absent from a rebuilt manifest disappear when both tables are rewritten.
   Reject conflicting duplicates within each input, and before any table
   overwrite either abort on nonblank orphaned annotations or preserve them
   in an explicit, recorded archive. Do not silently discard reviewer input.
   Exact disposable-temp reproductions and stable source hashes are saved in
   `independent_phase1_review_20261003.md`; add meaningful regression tests.

## Do not equate a missing overview copy with unavailable soma imagery

The user said only four overview datasets were copied. They did not remove
the other new monkeys from the original objective. The saved bounded
`portal_snapshot/native_source_probe.json` shows own-sample raw SWC and
high-resolution cube HTTP 200 for **all nine** samples. Therefore the initial
queue-only treatment of 252790, 250432, 252383, 252718 and 251730 was too
narrow if their own high-resolution sources remain accessible.

- Re-probe own-sample native sources with bounded requests, then produce
  available soma-detail packages for candidates from these five samples.
  Prioritize **252790**, which has 38 atlas insula candidates and no copied
  overview. Preserve the existing identity selection and all reasons.
- Keep missing overview/context explicitly pending. A partial soma package
  should not claim regional/cytoarchitectonic acceptance; never borrow
  251637 or another animal's context, and never use portal NMT XYZ for crops.
- Separate overview-source status, soma-source/coverage status and production
  status. Update the manifest/table/gallery/verifier to accept valid partial
  packages for overview-pending samples rather than prohibiting all images.
  Preserve a queue identifying exactly which source components are missing.
- Explicitly record the excluded unprocessed blank-portal metadata identities
  (84 in 251730, plus 252383/136.swc and /140.swc) and their screening reasons
  in a separate metadata-resolution queue. Do not call them atlas-Unknown or
  silently treat absent labels as accepted outside-insula membership.
- Obtain roots from validated own-sample raw SWCs and preserve hashes. Use
  bounded retries, valid caches and resumable production. Account for each
  selected identity including source errors, without silent omissions.
- Existing 301 available-overview candidates remain in place. Do not rebuild
  their raw crops when only status/preview/sidecar metadata needs correction.
- Ensure the gallery includes all nine samples, with usable links and clear
  per-component missing-source information. Protect future manual table edits.
- Verify all persisted PNGs and table/manifest identities, sampled raw arrays
  across every represented sample/category, root/image geometry and source
  provenance, all failure/partial states and protected canonical hashes.
  Update CURSOR_RESULT.md and the living note with exact final counts and
  an agent/date stamp. Distinguish completed software/bulk production from
  anatomical review and unavailable source components.

The original extensive literature/clustering review remains part of the
deliverable. Reuse its verified Zotero methods notes; no accepted k or human
region labels are implied by this image production.

## Regional context remains part of the original goal

Soma-detail completion alone does not complete the Henry-style regional review
package. A missing copied 5 um overview is not evidence that own-sample regional
context cannot be obtained. After soma production, inspect documented overview
routes and historical own-sample source evidence, then test a bounded thin-Z,
tiled native XY context crop from accessible own-sample high-resolution cubes:
approximately 4 mm field and a 30 um slab, downsampled for an overview. Record
the actual derived spacing, source tile/plane IDs, crop origin, downsampling,
missing-tile coverage, and explicit memory/cache/request bounds. Process tiles
incrementally; do not allocate a huge dense block volume or download whole
brains. If feasible, produce context for overview-pending candidates with
honest partial-coverage status. Otherwise record the tested source/resource
limitation and retain its component queue. Producing currently accessible soma
components is a milestone, not a substitute for resolving this remaining
regional-context scope. Keep software/source availability separate from human
soma, region, layer and tracing acceptance.
