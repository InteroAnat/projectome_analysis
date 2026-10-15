# Native fMOST source inventory and candidate scope

**Agent: Codex | Date: 2026-10-02 (Asia/Shanghai)**

Read-only checks were performed against the actual server share and bounded ION metadata requests. No raw image, SWC or server folder was modified. The user's latest clarification confirms that only the five listed resampled datasets have been copied; the immediate production scope is the four available new monkeys, while the remaining samples require source delivery or a separately specified HTTP-only review.

## Server mapping and available image series

`X:` is connected to `\\10.102.8.200\microscopy_data`. Both the mapped path and UNC `\\10.102.8.200\microscopy_data\fMOST\5micron_datasets` were accessible. The following are the complete immediate dataset directories at this root. Counts come from a shallow `*.tif` inventory; only the first and last TIFF headers were read.

| Sample | Actual relative path under the 5micron root | Slice TIFFs | Filename range | First/last page shape Y×X |
|---|---|---:|---|---|
| 251637 reference | `251637-CH1_resample/resample_5um` | 20,744 | `251637_00001_CH1_resample.tif`–`251637_20744_CH1_resample.tif` | 10,120 × 11,711 |
| 252384 | `252384_CH1_resample/resample_5um` | 19,996 | `252384_00001_CH1_resample.tif`–`252384_19996_CH1_resample.tif` | 10,653 × 16,266 |
| 252385 | `252385-CH1_resample/resample_5um` | 20,788 | `252385_00001_CH1_resample.tif`–`252385_20788_CH1_resample.tif` | 11,451 × 17,941 |
| 252527 | `252527_CH1_resample/resample_5um` | 19,276 | `252527_00001_CH1_resample.tif`–`252527_19276_CH1_resample.tif` | 9,854 × 12,351 |
| 252714 | `252714-CH1_resample/resample_5um` | 20,036 | `252714_00001_CH1_resample.tif`–`252714_20036_CH1_resample.tif` | 10,653 × 15,601 |

All inspected endpoints are one-page, `uint16`, YX, LZW-compressed, MINISBLACK TIFFs. Their ImageDescription is empty. The TIFF X/Y resolution tags are `(4294967295,1300716909)` with unit INCH (about 3.302 pixels/inch), which do **not** establish microscopy calibration. The existing toolkit declares low-resolution XYZ spacing `[5,5,3]` micrometres and high-resolution spacing `[0.65,0.65,3]` micrometres, with HTTP cube dimensions `[360,360,90]` in XYZ. These are software declarations; no independent acquisition/resampling calibration JSON or log was found in the dataset folder or its immediate slice entries. Z spacing is not 5 micrometres in that declaration.

Scoped adjacent checks covered the immediate `X:/fMOST` folders, the two 251637 reference folders, and immediate non-TIFF files in each listed dataset. Only `Thumbs.db` appeared as non-TIFF sidecars for the resampled series. No CH2, PI, or separately identified tissue-stain series was verified in this scope. CH1 should not be presented as confirmed PI/cytoarchitecture evidence. This is a scoped inventory, not a recursive search of the server.

252790, 252718, 250432, 252383 and 251730 have no copied 5micron series at the confirmed root. Missing sources must remain explicit pending rows; an atlas render is not a substitute for native image evidence.

## Current candidate denominators

A fresh snapshot at approximately 22:45 Asia/Shanghai queried `getSampleInfo`, `selectNeurons` and `getSoma` for all nine non-reference samples. All 27 requests returned HTTP 200. Saved UTF-8 JSON and request timestamps/hashes are in [portal_snapshot/manifest.json](portal_snapshot/manifest.json). These are current metadata snapshots, not fresh image or full tracing acceptance.

Labels in `selectNeurons.region` have a trailing atlas index such as `Ial_42`; counts below strip only that numeric suffix. Core INS follows `insula_label_set.py`; PrCO and missing/unknown labels are separate. G, retroinsula, claustrum and ProM are review candidates, not automatically accepted insula. A broader opercular screen can retain SII, Tpt, area_7op, area_44, F4 and F5 for coordinate/visual review without promoting them.

| Sample | Live neurons / latest step1 | Live INS | PrCO | Unknown | Core total | G | Ri | Cl | ProM | Core + G/Ri/Cl/ProM | Copied 5micron |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 252385 | 128 / 128 | 18 | 21 | 19 | 58 | 11 | 0 | 2 | 0 | 71 | Yes |
| 252790 | 123 / 123 | 38 | 0 | 23 | 61 | 0 | 0 | 4 | 0 | 65 | Pending |
| 252718 | 190 / 190 | 4 | 0 | 36 | 40 | 9 | 0 | 3 | 0 | 52 | Pending |
| 252527 | 99 / 99 | 14 | 11 | 5 | 30 | 0 | 0 | 0 | 0 | 30 | Yes |
| 252714 | 144 / 144 | 4 | 0 | 15 | 19 | 12 | 0 | 1 | 0 | 32 | Yes |
| 250432 | 234 / 234 | 8 | 0 | 19 | 27 | 0 | 0 | 41 | 0 | 68 | Pending |
| 252383 | 140 / 127 | 6 | 0 | 31 | 37 | 0 | 0 | 0 | 0 | 37 | Pending |
| 252384 | 100 / 99 | 5 | 0 | 19 | 24 | 0 | 0 | 7 | 0 | 31 | Yes |
| 251730 | 764 / 79 | 0 | 0 | 84 | 84 | 0 | 0 | 0 | 0 | 84 | Pending |
| **Total** | **1,922 / 1,223** | **97** | **32** | **251** | **380** | **32** | **0** | **58** | **0** | **470** | Four new samples |

The current core + G screen totals 412, including 154 across the four copied new samples. Adding all Cl review rows gives 164 across those four. For priority within available samples, 252385 has the greatest staged potential-insula yield (50), followed by 252527 (25), 252714 (16), and 252384 (5). 252790 has 38 atlas-INS somata and remains a high-priority source-pending sample.

The old latest-step1 INS/PrCO/Unknown selection plus staging keepers totals 324 unique sample/neuron identities, adding 32 G neurons that the bulk INS vocabulary otherwise excludes. This is a historical subset: current portal records add 699 identities absent from latest step1 (685 in 251730, 13 in 252383 and one in 252384). The refreshed core + G screen adds 88 candidates compared with that old 324-member union. See [step1_identity_delta.json](portal_snapshot/step1_identity_delta.json) for exact names, source paths and source hashes. Missing step1 identities must not be silently excluded from a current full-scope manifest.

## API and native HTTP source checks

Current neuron records contain `name`, `sampleid`, `region`, `type`, `exclude`, and `comment`. Soma records contain `sampleid`, `name`, `somax`, `somay`, `somaz`, `type`, `exclude`, and `comment`. In each sample the neuron and soma identity sets match, with zero duplicates. `getSoma` coordinates match persisted NMT-space physical soma values; they must not be reused as native fMOST crop coordinates.

All exact sample metadata records declare project `Monkey`. Prefix matching in `getSampleInfo` also returns `250432ch2` for 250432 and `251730ch2` for 251730. Filter `fMOST_id` exactly; do not rely on an arbitrary first returned record. Metadata `tracing_cell_number` can lag the actual list (251730 metadata 79 versus 764 listed neurons).

One atlas-INS neuron per sample (an unknown neuron for 251730) was tested using raw-SWC HEAD, a 4,096-byte HTTP Range prefix, and a soma-cube HEAD derived using the toolkit's declared calibration. All nine raw HEAD requests returned 200, all nine prefix reads returned 206, and all nine cube HEAD requests returned 200 with `image/tiff`. No full SWC or cube was downloaded by this inventory. This establishes a tested source route for one neuron per sample, not availability or correspondence for every selected neuron.

Exact tested URLs, native raw-root coordinates, response lengths and status are saved in [native_source_probe.json](portal_snapshot/native_source_probe.json). The raw SWC route is `http://10.10.31.31/swc/newswc/Monkey/{sample}/swc_raw/{neuron}` and the cube route is `http://bap.cebsit.ac.cn/monkeydata/{sample}/cube/{z}/{x}_{y}_{z}.tif`. Read native coordinates from the raw SWC; the portal soma is a distinct coordinate frame.

## Workflow and identity risks

- `main_scripts/step3.1.bulk_visual_data.py` and its subsidiary copy are fixed to 251637 inputs. They fetch native raw SWC roots, produce high-resolution soma blocks and 8 × 8 mm low-resolution context crops, and optionally export NIfTI data. Their configurations are not an automatic current multi-monkey selection.
- `Visual_toolkit.py` currently maps the copied share only for 251637, 252385 and 252527. Although 252384 and 252714 series exist, those two require explicit mapping before using the toolkit; the legacy SSH fallback is rooted in 251637 and missing slices can become zero arrays.
- `run_bulk_visual_sample.py --potential-ins` prefers canonical harmonized 353, rather than staging 420, and therefore has no potential rows for newly staged 252790/250432/252714. It does not represent the full current portal screen above. The manifest should explicitly union live portal, latest step1 and selected staging candidate identities while preserving every source label.
- Dataset filenames, neuron IDs and sample IDs require exact handling: underscore versus hyphen differs by sample; use composite sample + neuron identity; preserve channel suffixes; distinguish human monkey IDs from fMOST IDs. Keep current portal labels, refined/staging labels and native image evidence in separate fields.
- Folded-X padded bbox inclusion, atlas-neighbor labels, G/Cl screening and unknown somata are prioritization flags only. Do not call them anatomical membership, laterality confirmation, layer labels or VEN identity without image/registration evidence and human review.
- Crops/HTTP cubes require bounds, missing-source and nonzero-pixel checks with native root provenance. Declare the software calibration and channel explicitly. Do not silently replace absent native context with atlas plots, treat a CH1 fluorescence projection as cytoarchitecture, or equate the presence of a raw SWC with complete reconstruction.
