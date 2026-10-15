# Extra-dataset insula screen (2026-10-12)

This folder is an additive candidate screen. It does not accept anatomy, add neurons to the published 462 ledger, or make a between-animal map or a t-map.

Neuron table: [extra_insula_candidates_20261012.csv](extra_insula_candidates_20261012.csv).

Per-dataset counts: [extra_dataset_summary_20261012.csv](extra_dataset_summary_20261012.csv).

Source hashes actually checked are in [provenance.json](provenance.json).

## Monkey ID for dataset 250432

No source explicitly binds fMOST `250432` or portal alias `ION2508004` to a monkey ID. The animal stays **unknown**. Channel sibling `250432ch2` is alias `ION2508005`, also unbound. Those aliases are sample IDs, not monkey IDs.

Binding that does exist, and what it does not say:

| Source | Field | What it binds |
| --- | --- | --- |
| `group_analysis/evolution_20261008/inventory/live_inventory_20261008/live_snapshots/20261008T152413Z/catalog_sample_info.json` | `fMOST_id`, `sample_id` | `250432` to `ION2508004`; `250432ch2` to `ION2508005`. No monkey field. Injection region empty. |
| `notes/bulk_visual_review_20261002/portal_snapshot/250432_sample_info.json` | same fields | Same two records. |
| `group_analysis/evolution_20261008/inventory/sample_inventory.csv` and the live copy | `tracker_monkey_id`, `historical_audit_monkey_id` | Both blank for `250432` and `250432ch2`. |
| `group_analysis/docs/dataset_status_manifest.csv` | `animal`, `fmost_id` | Eight monkeys only. No `250432` row. |
| `group_analysis/docs/monkey_experiment_data_summary.xlsx` | sheet `Monkey Data`, columns `Monkey ID` and `fMOST ID` | Same eight pairs. No `250432`. |
| `group_analysis/docs/data_progress_table_20260701_1416.csv`, `data_progress_table_20260713_1649.csv`, `data_progress_table_20260727_1620.csv` | `monkey_id`, `fmost_id` | No `250432` row. |
| `group_analysis/portal_audit_20260926/portal_macaque_samples.csv` | `monkey_id_tracker` | Blank for `250432` and `250432ch2`. |
| Review manifest `registry_animal` | `all_neuron_review_manifest_distance_corrected.csv` | Blank on all 234 `250432` rows; status `missing_registry_animal_mapping`. |
| Local SWC headers | `group_analysis/visual_review_20261002/swc_raw/250432/076.swc`, `077.swc`, `078.swc`; `neuron-vis/resource/swc/250432/001.swc`, `002.swc` | Tracer comment or no comment. No animal ID. |
| Portal neuron list | `notes/bulk_visual_review_20261002/portal_snapshot/250432_neurons.json` | `comment` and `type` empty on all 234. `sampleid` is `250432`. |
| Step1 report | `group_analysis/step1_results/250432_20260927_151627_region_analysis/reports/analysis_summary.txt` and workbook sheet `Summary` | `SampleID` `250432` only. |
| `D:\multimodal_fmri\registry\subjects.tsv` | `fmost_id` | No `250432`. cm032 and cm033 are still `TBD`. |
| Cited 5 µm root `\\10.102.8.200\microscopy_data\fMOST\5micron_datasets` and `X:\fMOST\5micron_datasets` | directory names | `250432` is not a folder there. Present names: `251637-CH1_resample`, `252384_CH1_resample`, `252385-CH1_resample`, `252527_CH1_resample`, `252714-CH1_resample`. |
| `D:\projectome_analysis\projectome.sqlite` | schema | File length 0. No tables. |
| OneNote notebook `my PhD`, section `fMost`, page `General` | page body | The only text is `lab`. |

Two other sample IDs carry historical monkey numbers, and those numbers conflict with the current registry. They are not an ID for 250432.

- `252334`: `historical_audit_monkey_id` `948` in both sample inventories. The identity note says that historical tracker identity was absent/unverified and that the current tracker maps this monkey to a different fMOST ID. `dataset_status_manifest.csv` binds animal `948` to fMOST `252384`.
- `252985`: `historical_audit_monkey_id` `331`, with the same note. The manifest binds animal `331` to fMOST `252385`.

Both alias rows have no neuron list in this screen. They are not counted as extra monkeys.

## Who was screened

The eight registered samples used for between-animal maps were left out: 331/252385, 605/252383, 631/252714, 797/252790, 900/252718, 936/251637, 945/252527, 948/252384.

Outside that table, the neuron manifest has 39 sample IDs and 7,248 neurons. No additional sample has a verified monkey ID. Channel suffixes `ch2` are the same unresolved dataset, not another animal. `250442L` and `250442R` stay separate datasets because nothing here binds them to one animal.

- 24 datasets with no verified monkey ID.
- 15 `ch2` channel variants of some of those datasets.
- 2 tracker-only alias rows (`252334`, `252985`) with no neurons.

Denominators for the 7,248 neurons: 6,923 have portal soma coordinates; 325 do not. Missing coordinates are unassessed, not zeros. They are all in 242521 (158), 260208 (61), 260209 (41), 260212 (62), and 260213 (3).

The parent sample inventory lists 242521 as 197 reconstructions and does not contain 260208, 260212, or 260213. The live inventory lists 242521 as 355, of which 197 have coordinates and 158 do not, and it does contain those three 2602 samples. This screen follows the neuron manifest (355, 61, 62, and 3 rows). Both inventory counts are retained in the summary CSV.

## Counts

Rules, unchanged: direct ARM label at `rint(soma_um / 250)` on `ARM_in_NMT_v2.1_sym` slice `[..., 0, 5]`; label 0 is unassigned atlas, not white matter. Folded spatial screen uses `abs(Soma_NII_X - 128)`, the Henry/reference 0.005–0.995 bounds, 0.25 mm/index, and an inclusive 2 mm pad. Recomputing that screen matched the stored bbox hits on all 6,923 coordinated neurons.

Where a local SWC-root lookup already existed, that label is the ARM call. Where it did not, the ARM call is the portal soma coordinate. The two agreed on insula membership wherever both existed (no conflicts). Portal region strings were not used as the ARM call.

| Sample | Monkey | Screened | With coordinates | Missing coordinates | ARM insula | Spatial pass | Both | ARM only | Spatial only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250432 | unknown | 234 | 234 | 0 | 8 | 23 | 2 | 6 | 21 |
| 250432ch2 | unknown (channel of 250432) | 73 | 73 | 0 | 0 | 0 | 0 | 0 | 0 |
| 233001ch2 | unknown (channel of 233001) | 25 | 25 | 0 | 1 | 0 | 0 | 1 | 0 |
| 242521 | unknown | 355 | 197 | 158 | 0 | 2 | 0 | 0 | 2 |
| 242521ch2 | unknown (channel of 242521) | 107 | 107 | 0 | 0 | 56 | 0 | 0 | 56 |
| All other extra datasets | unknown | 6454 | 6287 | 167 | 0 | 0 | 0 | 0 | 0 |
| **Extra total** | | **7248** | **6923** | **325** | **9** | **81** | **2** | **7** | **79** |

The other 6,454 neurons are in the summary CSV. 167 of them have no coordinates (the four 2602 samples). The rest were assessed and are neither ARM insula nor inside the folded screen.

`233001ch2` / `qroot146.swc` is ARM abbreviation `CL_Iam/Iapm`, full name `CL_medial_agranular_insular_region`, index 39, from the portal soma file `.../233001ch2_soma.json`. No local SWC root was available, and it does not pass the folded screen.

`242521` and `242521ch2` contribute only folded-screen hits. They are one unknown animal plus its channel variant, not two monkeys.

## 250432 cross-check against the prior 8+21=29

The prior union file `dataset250432_selected29_candidates.csv` has the same 29 neuron IDs as this screen's union of SWC-root ARM insula or folded-screen pass. That union is not replaced.

Split of those 29, which the earlier note did not separate this way:

- ARM insula and not in the folded box: 6. `116, 117, 123, 126, 128, 131`. Index 729, `CR_Ig`, full name `CR_granular_insula`.
- Both: 2. `184, 188`. Index 228, `CL_Ia/Id`, full name `CL_agranular_and_dysgranular_insula`.
- Folded screen only: 21.

6 + 2 + 21 = 29. The prior "21 other" count matches the folded-screen-only set. The six granular roots are in the 29 because the SWC root is named insula, not because they pass the folded box. All eight SWC-root labels agree with a fresh portal-coordinate lookup. `250432ch2` adds no further candidate.

## What this does not do

No new anatomical acceptance. No injection-site acceptance. Structure tags on 250432 (M1, F5, 3a/b, claustrum, and other subcortical labels) stay metadata, not injection sites. No between-animal mean, no t-map, and no edit to the 462 ledger, canonical workbooks, original SWCs, or published map NIfTIs.

## Not run

- No new portal download and no new SWC download. SWC-root labels for the 430 local reconstructions, including all 234 of 250432, are the labels already stored on the review manifest. Portal coordinates were looked up fresh for every extra neuron that has them.
- SQLite MCP `user-sqlite-projectome` failed tool discovery. The local `projectome.sqlite` file is empty, so there was no schema to query.
- OneNote page bodies were not searched beyond `fMost` / `General`. A whole-notebook content search was not available from the page-title filter.
- `250442L` and `250442R` were not merged.
