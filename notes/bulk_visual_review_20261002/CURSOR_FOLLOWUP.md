# Independent review follow-up

**Agent: Codex | Date: 2026-10-03 (Asia/Shanghai)**

Continue the existing Cursor/Grok handoff on the current branch. The first
batch exists; preserve its image crops and all source/canonical files. This
follow-up comes from independent read-back, not from a request to restart the
whole project. The original handoff restrictions still apply, especially no
publication, commits, resets, cleanup or unrelated journal/profile sync.

## Reproduced defects requiring correction

1. `label_reasons` calls `portal_base_label` for every source. The latter is
   appropriate for portal `area_44_76`, but strips the region number from
   historical step1 `area_44`. Reproduction:
   `step1_reasons('area_44', set()) == []`. Normalize historical/staging labels
   without stripping their anatomical number; strip only the portal atlas
   index when the source format has it. Test actual source conventions and
   reconcile the manifest so historical-only adjacent candidates cannot drop.
2. Five rows marked `ok` have explicitly missing neighboring high-resolution
   cubes: 252385|086.swc, 252385|101.swc, 252527|087.swc,
   252384|079.swc, 252714|071.swc. They must have an explicit partial-coverage
   status in the provenance, correction table, gallery and completion counts.
   Keep the valid center crop and raw arrays; do not download or rerender the
   complete cohort solely to relabel these five rows. Metadata `complete` and
   `missing` must affect the renderer's production status. Test this contract.
3. `write_correction_table` always blanks human fields and overwrites both CSV
   and XLSX. `verify_outputs` calls it. This destroys annotations on a later
   run. Preserve manual fields by UID across resume/table/gallery rebuilds,
   permit reviewers to annotate either table format with a clear safe merge,
   and fail before writing if existing nonblank CSV/XLSX values conflict.
   A read-only verifier must never rewrite editable review tables. New rows
   start blank; preserve existing human answers, and do not make nonblank
   reviewer input a verification failure. Test regeneration and conflicts.
4. `build_manifest` overwrites `canonical_hashes.json` on every run, allowing
   a changed canonical/Henry file to replace the protection baseline. Keep
   the initial baseline, check it before mutations, and fail on mismatch.
   Record source-input and configuration hashes for reproducible selection.
5. `_save_ortho` displays XZ/YZ with voxel aspect 1 despite anisotropic
   spacing. For 252527/048, the 1080 x 1080 x 270 XYZ crop is nominally
   702 x 702 x 810 um, but XZ appears four times shorter vertically than
   horizontally. Render orthogonal axes with the corresponding physical
   extent/aspect, source plane/root markers and usable scale information.
   Regenerate only these panels from existing NIfTIs, not server downloads
   or unchanged crops. Add a meaningful anisotropic geometry regression.

## Continue production and verification

- Existing result: 746 identities, 301 available, 445 source-pending;
  282 ok, 17 partial, 2 raw-SWC errors before coverage-status correction.
- A read-only audit expanding the folded 99% boxes by 8 voxels (2 mm)
  added zero identities in this current set. Document the intended screening
  padding explicitly; do not arbitrarily broaden selection.
- Reconcile counts after normalization/status changes. Render newly added
  available candidates only, resume valid existing outputs, and retain
  unavailable source cases with precise reasons. Retry d001/d002 SWCs with
  bounded requests and inspect verified own-sample caches/other documented
  native routes if needed. Never use NMT coordinates as a native fallback.
- Reopen image files, source sidecars and NIfTIs; verify complete candidate
  accounting, UID matches, sample source mapping, root coordinates, coverage
  status, file availability and gallery links. Verify all PNG files decode;
  retain stratified raw-array checks across sample/category plus failures.
- Preserve original project-note attribution. Append a Cursor/Grok author and
  Asia/Shanghai date stamp with corrected counts and limitations; update
  CURSOR_RESULT.md to the actual final state. Explain partial neighbor gaps
  separately from missing center cubes and missing SWCs.
- Run meaningful focused regression tests and the full existing unittest
  suite with the projectome Python environment. No installation is needed.

No human anatomical acceptance is requested. The table remains a human
review source; CH1 and nominal acquisition axes do not verify cytoarchitecture
or laterality.
