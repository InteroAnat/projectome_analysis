# Native review result

**Agent: Cursor (Grok 4.7) | Date: 2026-10-03 (Asia/Shanghai)**

The interrupted regional producer is not running. The 832-identity review set is active, the 86 missing soma packages are on disk, and one hidden regional producer is detached and writing its own ledger. This is not anatomical acceptance, an accepted clustering result, or a finished all-nine validation.

## Interruption

Producer PID 117764 and watcher PID 85564 were absent. No second Python regional client was live. The saved ledger still said `running` for run `f0a8ad4c5c034ebeb76924873ca1b630`; that state was stale. Its bytes remain in `interrupted_regional_checkpoint_f0a8ad4c5c034ebeb76924873ca1b630.json`, SHA256 `7ee071b4de8b2d79d748c7d5ba8c3af7102d53c485ded096f7b6d09837399d19`. That checkpoint was not rewritten. Root's repair record `interrupted_regional_recovery_20261003.json` has `repair_state` `saved_volume_refresh_and_readback_passed`. The exit cause was not established. It is not recorded here as an out-of-memory stop or a runtime-budget stop.

That interrupted run had finished 99 of 332 attempts: 98 verified fields and one unavailable raw-root outcome, `252718|145.swc`. Combined with the first 113 fields, root's readback counts 211 verified regional images. Two hundred thirty-three identities in that 332-row scope had not been attempted.

## Identity set

`manifest` was activated with `--unresolved-report notes/bulk_visual_review_20261002/unresolved_full_source_validation_20261003.json`.

| Set | Rows |
|---|---:|
| Live identities | 832 |
| Unchanged original rows | 746 |
| Added `metadata_unresolved` | 86 |
| Source-pending | 531 |
| Copied-overview available | 301 |

The 86 additions are 84 in 251730 and 2 in 252383 (`136.swc`, `140.swc`). Every added reason is `metadata_unresolved_raw_source`. Portal base labels stay blank. They are not atlas Unknown. All 39 shared fields of the frozen 746-row manifest match `identity_manifest_746_pre_extension_20261003.csv`. The live manifest matches the isolated preview. All 86 cached raw SWC SHA256 values match the validation report. The manifest SHA256 now in the producer ledger is `7a3b3b77f413220993716dfaab07cfdff2840ab8ccf03fb972348854966659de`. Canonical cohort and Henry hashes still match `canonical_hashes.json`. Human review fields in the 832-row table are blank. Pad audit extra identities remain 0.

## Soma packages for the 86 additions

`render-soma --sample 252383 --sample 251730` exited 0. Existing packages in those samples were skipped. Stdout and an empty stderr are in `group_analysis/visual_review_20261002/logs/soma_unresolved_20261003.stdout.log` and `soma_unresolved_20261003.stderr.log`.

| Outcome | N |
|---|---:|
| `soma_only` / `complete` | 85 |
| `partial` / `partial_neighbor` | 1 |

The partial package is `252383|136.swc`, with 6 missing neighbor cubes and the center crop retained. All 86 folders contain the six soma package files and use `overview_pending`. No other sample's images and no portal NMT coordinates were used.

## Detached regional producer

One hidden `python.exe` process is running. PID **114580**. Run ID **`d93cc551074743e291f4229543b0d888`**. Ledger state at launch verification was `running`, with **531** identities and the manifest hash above. Started `2026-10-03T11:51:38.498271+00:00`. The command is exactly:

```text
C:\Users\laika_yan\miniconda3\envs\projectome\python.exe -B D:\projectome_analysis\group_analysis\scripts\regional_context_batch_20261003.py --sample 252790 --sample 250432 --sample 251730 --sample 252383 --sample 252718 --cache-gib 500 --max-requests 200000 --max-seconds 57600 --min-free-gib 50
```

`--retry-missing` and `--force` were not used. Logs: `logs/regional_resume_20261003.stdout.log` and `logs/regional_resume_20261003.stderr.log`. The launch record is `logs/regional_resume_20261003_launch.json`. Stderr was empty at verification. The first log lines are `reused_existing_derived` for adequate 252790 4 mm fields, beginning with `252790|014.swc`.

Sixteen saved 252790 fields are still 2800 µm and are not treated as the preferred 4 mm context: `002`, `008`, `029`, `037`, `058`, `061`, `069`, `071`, `075`, `078`, `084`, `091`, `093`, `094`, `098`, and `109`. The driver rejects those as reusable at the 4000 µm minimum, so this run is expected to upgrade them. Other dated HTTP 404 observations were left in place.

`252718|145.swc` is an explicit single-cube retry. Only failure key `252718|195|84|96` was removed from `regional_source_failures.json` before launch. The removed observation is in `logs/explicit_root_retry_252718_145.json`. A repeated 404 would again be an unavailable root source, not anatomical absence.

## Still incomplete

The new producer has not finished. Still required after it reaches a real terminal state: the 233 previously unattempted original candidates, the 16 wider-field upgrades, regional packages for the 86 additions, the `252718|145` retry outcome, final all-nine image/table/provenance validation, and the clustering/Zotero methods review. No accepted k and no human region labels follow from this launch. CH1 and the nominal axes do not establish cytoarchitecture or laterality.
