# Per-monkey insula inventory from the legacy overview

Agent: Codex | Processing date: 2026-10-09 (Asia/Shanghai)

This eight-monkey view uses the original **Monkey Data** overview workbook and the existing `group_analysis.data_progress.table.build_progress_table` function, then reconciles their exact animal/fMOST identities with the saved inventory and selected ledger. The original overview, manifest and combined workbook are preserved. No legacy cleanup runner, network request, image download or anatomical relabelling is performed.

- [Per-monkey table](per_monkey_insula_inventory.csv): injection/source claims, reconstruction counts, separate INS evidence, selected-neuron counts, dataset status and review priority.
- [Readable overview](per_monkey_insula_overview.png): all eight monkeys in the established overview order; actually inspected for readable headers, rows and notes.
- [Legacy reconciliation](legacy_reconciliation.csv): 15 missing or differing source fields, retained rather than automatically corrected.
- [Sources, hashes and definitions](provenance.json) and [independent readback](independent_readback.json).

**Counts have different meanings.** Atlas-labelled INS is recomputed from nonexcluded snapshot identities. Henry visual INS counts distinct coarse annotations in the selected ledger; absent Henry evidence stays blank, not an accepted zero. Spatial candidates follow the original 251637 screen, include atlas-labelled cases and are lower bounds where coordinates are incomplete. The separate non-atlas candidate column exposes that overlap. Legacy review claims and current legacy combined-workbook rows remain historical bookkeeping, not interchangeable with confirmed anatomy or the selected 462-neuron exploration ledger.

The legacy overview claims 353 reviewed INS neurons; the present combined workbook contains 306 rows under the eight exact fMOST IDs. Monkey 331 has 47 claimed versus 36 present combined rows; 900 and 945 have claims of 13 and 23 but no rows in that current workbook. Legacy reconstruction claims of zero for 797/252790 and 631/252714 disagree with the saved lists of 123 and 144 neurons. These differences are documented by source and scope; no missing neuron is manufactured and no table is silently substituted.

Henry's source worksheet has **261 INS annotation rows but 260 unique neurons**, all for 251637/936. Neuron 114 appears twice with `L-IDD5` and `L-IDM`; coarse INS evidence agrees, while that fine subregion conflict remains unresolved. The independent reader directly checks this source workbook and exact unique identities rather than summing annotation rows.

Five nominal 5 µm CH1 copies are verified in the saved scoped inventory: 936, 945, 948, 331 and 631. For 797, 605 and 900 no copy is verified at that root; availability elsewhere is **unknown**. This is not evidence that the datasets do not exist. The manifest's two “Yes” entries do not override the five dated verified copies. Nominal 5 µm does not imply isotropic sampling or optical resolution. Evidence dates and scope are retained in the CSV.

Review priority puts animals without a verified local copy first, then documented Henry count, atlas INS count and spatial lower bound. The first three are **797, 605, 900**. Overlapping evidence counts are never added. This view does not merge channel variants or historical aliases. The other 41 sample/channel identities remain in the [complete 49-identity parent inventory](../live_inventory_20261008/README.md), with no invented animal mapping.

Eight focused regression tests cover stale zeros, missing step1 evidence, exact identity conflicts, duplicate neurons, candidate overlap and missing manual evidence. Independent source readback checks 72 count cells, all eight source identities, dataset-status/priority rules, the unchanged 462-neuron selected ledger and 32 hash bindings. The existing 437-test scientific baseline remains unchanged; this addition does not rerun imaging or clustering and cannot confer anatomical acceptance.

Reproduce from the repository root with a fresh destination:

```powershell
python -B group_analysis/scripts/build_monkey_insula_inventory.py --output <fresh-output-directory>
```

`verify_overview.py` imports no producer code; its saved receipt preserves the direct source checks and eight-test log. The producer records actual legacy step1 paths, source/output hashes and processing time separately from source evidence dates. It refuses existing output directories. Original data, source workbooks, map values, selected cohorts and deferred CM032/CM033 work remain unchanged.
