# Sheet identity on the projectome overview

Progress Details (tab `BB08J2`) of the Tencent workbook `macaque_fMOST` was read on 2026-10-12 12:53 Asia/Shanghai. The grid was visible in view-only mode. A login button was on the page and was not used.

`build_progress_table` is unchanged. `extend_overview_with_sheet` in `group_analysis/project_overview/sheet_overview.py` keeps the eight legacy rows and appends animals this sheet names. The run is `group_analysis/scripts/build_sheet_overview.py`. The 2026-10-09 overview files were not rewritten.

## 250432

Animal-name cell, quoted exactly: `747`

fMOST cell: `250432`

This sheet has no `ION2508004`, `ION2508005`, `2508004`, or `250432ch2`. Those strings were not used as animal names, and `250432ch2` was not attached to 747.

## Sheet columns kept

| Sheet header | Overview column | Why |
| --- | --- | --- |
| 注射动物_ID | `sheet_animal_name` | Animal name, quoted from the cell |
| fMOST_ID | `sheet_fmost_id` | Sample id used to join the eight |
| 注射脑区 | `sheet_injection_sites` | Lab injection regions, one cell per animal |
| 追踪进度(Neurite tracing) | `sheet_tracing_status` | Tracing progress the legacy overview does not carry |

Repeated identical site labels on one animal were stored once. A line-broken `L-Hippo-1` cell on animal 746 was stored once. Animal 138's fMOST cell is `221720  221747` (two spaces). It was not split into two samples, because the sheet does not call them channels.

Animal 935's injection cell is a death note, not a region, so `sheet_injection_sites` is empty. The name `935` is still listed. Empty fMOST cells stay empty.

## Sheet columns skipped

Sex, birth date, injection date, MRI date, virus, dilution, volume, wait time, perfusion date, ethics id, imaging site, notes, imaging progress, imaging-copy status, traced-neuron count, registration status, database-import status, and import count. They do not decide which animal belongs in the overview, and the eight already have their neuron counts from the legacy inventory. `NO.` is only the sheet's row index. Trailing columns past 入库数量 were empty. No channel column was added: this sheet never labels a channel variant.

## Legacy columns

Every column from `legacy_overview_20261009/per_monkey_insula_inventory.csv` is still on the eight rows, with the same values. Added columns are `sheet_animal_name`, `sheet_fmost_id`, `sheet_fmost_cell_kind`, `sheet_injection_sites`, `sheet_tracing_status`, `in_legacy_eight`, and `neuron_screen_status`.

Animals outside the eight have blank count cells and `neuron_screen_status` = `not_screened`. No neuron screen was run. Nothing here is anatomical acceptance.

## Animals outside the eight

Name cell, then fMOST cell.

- `49` (no fMOST id)
- `83` / `221059`
- `65` (no fMOST id)
- `138` / `221720  221747`
- `549` / `231623`
- `917 (2023.02.03)` / `232717`
- `97` / `233001`
- `670` / `234162`
- `462` / `234197`
- `401 (2023.04.28)` / `240025`
- `23` / `240901`
- `493` / `240941`
- `654` / `240984`
- `758` / `241349`
- `846` / `241334`
- `712` / `242161`
- `732` / `242443`
- `786` / `242521`
- `951` / `242693`
- `747` / `250432`
- `布布` / `250433`
- `茸茸` / `250442`
- `935` (no fMOST id)
- `529` / `251440`
- `842` / `251636`
- `狗狗/enterprise(rhesus)` (no fMOST id)
- `746` / `251730`
- `688` (no fMOST id)
- `795` (no fMOST id)
- `874` (no fMOST id)
- `532` (no fMOST id)
- `681` (no fMOST id)

The eight are on the sheet as bare numbers and match the legacy pairs: 331/252385, 605/252383, 631/252714, 797/252790, 900/252718, 936/251637, 945/252527, 948/252384.

## Output

- `per_monkey_overview_with_sheet.csv` — eight legacy rows plus the 32 animals above
- `sheet_animals.csv` — one row per sheet animal
- `progress_details_BB08J2.tsv` — cell extract of this tab
- `provenance.json`

Other workbook tabs were visible and not loaded: Progress Overview, 统计表, 苏州测试用fMOST注射信息, 工作表1.

Reproduce from the repository root:

```powershell
python -B group_analysis/scripts/build_sheet_overview.py
```
