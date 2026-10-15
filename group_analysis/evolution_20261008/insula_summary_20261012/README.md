# Per-animal insula summary, 2026-10-12

Recount of every neuron in the review manifest, then one row per animal. This is not anatomical acceptance and it does not add anyone to the 462-neuron ledger.

Outputs:

- [per_animal_insula_summary.csv](per_animal_insula_summary.csv)
- [per_animal_insula_summary.xlsx](per_animal_insula_summary.xlsx) — `Completed` cells are green `#C6EFCE`, `Pending` cells are blue `#B8CCE4`
- [recount_250432_crosscheck.json](recount_250432_crosscheck.json)

The 2026-10-09 overview files were not rewritten.

## Progress Details row for 250432

The tab is **Progress Details**, id `BB08J2`, in workbook `macaque_fMOST`. It was open in view-only mode. The grid was readable. Spreadsheet row 141:

`NO.` = `20`, `注射动物_ID` = `747`, `fMOST_ID` = `250432`, first injection cell `L-CLA-1`, tracing status `completed`.

No cell on that tab contains `250432ch2`, `ION2508004`, or `ION2508005`. `250432ch2` stays a channel of sample 250432 with the animal unresolved. It is not counted inside animal 747. Portal aliases `ION2508004` and `ION2508005` are sample IDs, not animal names.

Local `monkey_experiment_data_summary.xlsx` has only a `Monkey Data` sheet, and `notes/insula_projectome_progress_tracker.xlsx` has `insula_progress` and `legend`. Neither sheet contains `747` or `250432`.

## Rules

Same rules as the extra-dataset screen:

- ARM insula: `rint(soma_um / 250)` on `ARM_in_NMT_v2.1_sym` slice `[..., 0, 5]`. Label 0 is unassigned, not white matter. A saved SWC-root label is used when it exists; otherwise the portal soma coordinate is used.
- Folded spatial screen: `abs(Soma_NII_X - 128)`, Henry/reference 0.005–0.995 bounds times 0.25 mm, inclusive 2 mm pad.
- Missing coordinates stay unassessed. They are not zeros.
- Henry coarse INS is the existing selected-ledger flag, and only for 251637 / animal 936. Other animals are blank, not zero.
- Endpoint-eligible is copied from `per_monkey_counts_history_and_evidence.csv` when that sample already has a value. It was not recomputed. 747 has no such value.
- Channel rows are not extra monkeys. A `ch2` sample is attached to a named animal only when Progress Details says so. This tab never does.

`monkey_data_status` is filled only from the Monkey Data overview (`Completed` or `Pending` on the eight). Progress Details tracing text is a separate column and is not recoloured, including the lowercase `completed` on 747.

## Counts

Manifest neurons recounted: 8746. Stored bbox hits matched the recomputed folded screen on every neuron (0 mismatches).

| Animal | fMOST | Screened | With coordinates | Missing | ARM insula | ARM only | Spatial only | Both | Henry | Endpoint, already computed | Monkey Data status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 936 | 251637 | 574 | 562 | 12 | 212 | 0 | 49 | 212 | 260 | 228 | Completed |
| 605 | 252383 | 140 | 140 | 0 | 6 | 6 | 0 | 0 |  | 21 | Pending |
| 900 | 252718 | 190 | 190 | 0 | 4 | 3 | 0 | 1 |  | 14 | Pending |
| 945 | 252527 | 99 | 99 | 0 | 14 | 0 | 11 | 14 |  | 25 | Pending |
| 797 | 252790 | 123 | 123 | 0 | 38 | 5 | 7 | 33 |  | 52 | Pending |
| 948 | 252384 | 100 | 100 | 0 | 5 | 0 | 12 | 5 |  | 18 | Completed |
| 331 | 252385 | 128 | 128 | 0 | 18 | 2 | 32 | 16 |  | 55 | Completed |
| 631 | 252714 | 144 | 144 | 0 | 4 | 4 | 0 | 0 |  | 16 | Pending |
| 747 | 250432 | 234 | 234 | 0 | 8 | 6 | 21 | 2 |  |  |  |
| 786 | 242521 | 355 | 197 | 158 | 0 | 0 | 2 | 0 |  |  |  |

Other rows with a hit, animal unresolved because Progress Details does not name the channel:

| Channel sample | Of sample | Screened | ARM insula | Spatial only | Both |
| --- | --- | --- | --- | --- | --- |
| 233001ch2 | 233001 | 25 | 1 | 0 | 0 |
| 242521ch2 | 242521 | 107 | 0 | 56 | 0 |
| 250432ch2 | 250432 | 73 | 0 | 0 | 0 |

Animal 97 is the Progress Details name for 233001. The ARM hit is on `233001ch2`, which the tab does not assign to 97, so that neuron stays on the channel row.

The prior 250432 screen is unchanged: 234 screened, 8 ARM insula, 21 spatial-only, 2 both, union 29. The recomputed union is the same 29 neuron IDs.

Folded-screen pass (`both` + `spatial_only`) matches the 2026-10-09 spatial lower bound for 936 (261), 945 (25), and 948 (17). It does not match that older column for 605 (0 vs 6), 900 (1 vs 4), 797 (40 vs 45), 331 (48 vs 50), or 631 (0 vs 4). ARM counts match the 2026-10-09 atlas counts for all eight. The older spatial column is the saved candidate flag. This table keeps the recomputed folded screen and does not overwrite that older file.

## Animals beyond the eight

The sheet names 32 animals outside the eight. They are in the CSV, including names with no fMOST id (`49`, `65`, `935`, `狗狗/enterprise(rhesus)`, `688`, `795`, `874`, `532`, `681`). Those rows have an empty screened count because the manifest has no neurons for them. Animal 138 keeps the sheet cell `221720  221747` as one animal, not two channels.

## Not run

No new portal or SWC download. No new endpoint calculation. Henry was not re-annotated. No map, no t-map, and no edit to the 462 ledger.
