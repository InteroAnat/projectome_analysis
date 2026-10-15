# Resume after actual producer and CLI exit

Agent: Codex | Date: 2026-10-03 (Asia/Shanghai)

The original goal and delegation remain authorized. Use the same branch and
Grok 4.7 Ultra setting. No commits, push, publication, cleanup, or unrelated sync.
Read CURSOR_STAGE2.md and CURSOR_STAGE2_REVIEW_ADDENDUM.md in this folder.

At 19:30 the actual producer PID 117764 vanished and the independent watcher
PID 85564 stopped with `stopped_producer_exited_without_terminal`. Cursor CLI
session 38450 then returned exit code 1, no final result. The cause is unknown;
do not invent successful completion, an OOM diagnosis, or a terminal budget stop.
The input remained the original 746-row manifest. The old batch finished 99/332
attempts: 98 verified fields and one explicit root-source-unavailable outcome
252718/145. 233 identities were not attempted. First-batch 113 fields plus these
98 gives 211 verified images; sixteen first-batch field upgrades still remain.

Root preserved exact ledger bytes in
interrupted_regional_checkpoint_f0a8ad4c5c034ebeb76924873ca1b630.json, with old
watcher/readback/table checkpoint copies. Root ran
recover_interrupted_regional_20261003.py to refresh all 98 saved-volume captions,
reverify images, and refresh/read back current tables using the corrected Excel
exporter. Inspect interrupted_regional_recovery_20261003.json: require the passed
repair state before writing the same artifacts. Never rewrite the interrupted
record to claim the producer reached a terminal state.

Now carry the full goal forward, with exactly one source client at a time:

1. Verify the old source process and watcher are absent and locks available.
   Activate the prepared 832-identity manifest with the existing explicit
   --unresolved-report option. Preserve all original 746 rows, human annotations,
   canonical inputs, and exact staging provenance; verify the additional 86
   metadata-unresolved rows and own validated raw SWC hashes. Their source cache
   is prepared, but soma and regional imaging remain incomplete.
2. Produce only the 86 missing soma packages in 251730 and 252383, sequentially,
   using current validated own-sample cache/roots and explicit source outcomes.
   Reuse completed packages. Do not omit blank-portal identities or relabel them
   atlas Unknown. Persist resumable progress and stdout/stderr in workspace logs.
3. Launch one detached hidden regional producer with the current tested driver
   for all five overview-pending samples, including 252790. Reuse adequate 4 mm
   fields; upgrade the sixteen smaller fields, produce the remaining original
   candidates and all 86 additions. Use 500 GiB cache, 200000 request bound,
   57600 seconds and 50 GiB free-space margin unless actual bounds require a
   recorded adjustment. Respect own-sample sources and coverage masks. Review
   root-unavailable 252718/145 for explicit retry without needlessly refetching
   every partial field. Keep all required outcomes explicit.
4. The image job must survive CLI report completion: launch using native
   Start-Process -WindowStyle Hidden, exact executable/argument arrays, separate
   workspace stdout/stderr logs, and record the real PID, run ID, input hash,
   scope and command. Verify the producer's ledger after launch. Do not start a
   second producer while a source client owns the lock. Do not leave a six-hour
   image batch attached to one blocking CLI tool call again.
5. Return a concrete CURSOR_RESULT.md and final CLI result promptly after the
   verified detached launch, with precise completed/incomplete evidence. Root
   will bind its independent watcher to the new run; avoid competing checkpoint
   writers. Update the living note with the interruption and resumed scope,
   preserving history. Full goal completion still requires final all-nine-source
   image/table/provenance validation, full 832-row outcome accounting, clustering
   and Zotero methods review, and protected canonical/human fields. Do not claim
   anatomical acceptance, accepted k, or scientific cohort promotion.

Inspect full current source before routine implementation choices. Preserve
valid existing crops; zero-download caption repairs do not justify source
rebuilds. Investigate available crash logs if they exist, without scanning
credentials or private profile contents. Log the actual error if recovery fails.
