# ION portal audit — macaque insula / PrCO (2026-09-26)

## Method

- Sample catalog: cached `getSampleInfo` (`C:\Users\laika_yan\AppData\Local\Temp\ion_sampleinfo.json`), 1125 samples.
- **Macaque filter:** `project_id == 'Monkey'` OR `spicies` contains monkey/macaque/猕猴/恒河/rhesus/macaca. Distinct `spicies` values seen: `{'mouse': 1057, 'monkey': 2, 'None': 66}`. Macaque n=44 (mostly `spicies=None` with `project_id=Monkey`).
- Per-sample neuron lists: `selectNeurons?id=` via `requests.get` (timeout=30s, max retries=3, sleep=0.2s). **Did not** call `IONData` methods (infinite retry).
- Insula vocabulary: `insula_label_set.build_insula_label_set()` (19 labels). Portal regions stripped of trailing `_<atlas_id>` then prefix-normalized.
- Current cohort IDs: union of `combined/multi_monkey_INS_combined_harmonized.xlsx` Summary (n=353 insula neurons) + `recovery/*_INS_HE_coord_inferred.xlsx` + `recovery/all_refined_neurons.csv`.
- PrCO rescue: getSoma Phys coords → NII = Phys/250.0; inside 251637 subregion 99% bbox + 2.0 mm (`reference/251637_subregion_bboxes.csv`).
- Cross-check: `neuronsBySoma?region=` for ARM/portal insula labels + PrCO.

## Endpoint reachability

- Logged calls: 83 (ok=83, fail=0).
- See `endpoint_http_log.csv` for per-call HTTP status.
- All audited API calls returned HTTP 200 (within retry budget).

## Tracker ID inconsistency

| Monkey | Tracker fMOST | Tracker injection claim | Portal injection_region (abbrev) |
|---|---|---|---|
| 936 | 251637 | vaIC / IDFP / cingulate (multi) | `Ial_42 ;PrCO_49 ;Iai_41 ;Ia/Id_228 ;;M1_79 ;F4_85 ;PMdc_82 ;area_24c_10 ;preSMA_88 ;area_3a/b_93 ;Ig_229 ;` |
| 605 | 252383 | IDFA-L/R; IDFP-L/R | `F4_85 ;;Ig_229 ;area_7op_124 ;SII_95 ;M1_79 ;PMdc_82 ;Tpt_200 ;area_7a_122 ;areas_1-2_94 ;area_44_76 ;F5_86 ;` |
| 900 | 252718 | IDFA; IDFP | `Hi_312 ;S_313 ;;area_44_76 ;G_48 ;SII_95 ;Tpt_200 ;CM_214 ;MST_119 ;area_7a_122 ;MPul_446 ;Ig_229 ;AI_223 ;f_315 ;F5_86 ;CL_208 ;` |
| 945 | 252527 | vaIC; IDFM | `M1_79 ;;PMdc_82 ;areas_1-2_94 ;area_3a/b_93 ;Iai_41 ;PrCO_49 ;F4_85 ;SII_95 ;Ial_42 ;Iapl_43 ;` |
| 797 | 252790 | vaIC; IDFM; IDFP | `(empty injection_region + injection_structure)` |
| 948 | 252334 | vaIC; IDFM | `**MISSING from sampleInfo**` |
| 331 | 252985 | vaIC; IDFM; IDFP | `**MISSING from sampleInfo**` |
| 631 | 252714 | IDFA-L/R | `(empty injection_region + injection_structure)` |

- **252383 / monkey 605:** tracker says IDFA/IDFP; portal `injection_structure` is dominated by **F4** (+ Ig, 7op, SII, M1, …). Matches the prior F4-premotor reading; tracker IDFA claim is inconsistent.
- **252334 (948) and 252985 (331):** listed in `notes/projectome_latest_update_progress.md` as Done, but **absent from sampleInfo** (['252334', '252985']).
- **252790 / 252714:** in Monkey project with traced neurons, but both `injection_region` and `injection_structure` empty in sampleInfo (cannot verify planned vaIC/IDFA sites from portal metadata).
- Note: sampleInfo stores atlas injection tags in **`injection_structure`** for macaques; `injection_region` is usually empty.

## Findings — portal vs current

Samples with portal atlas-insula and/or PrCO soma labels (n=10):

| sample | in_current | n_portal | n_used | n_combined | n_insula | n_prco | n_prco_rescuable | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 251637 | 1 | 562 | 260 | 260 | 212 | 35 | 0 | grown_portal_gt_used |
| 252790 | 0 | 123 | 0 | 0 | 38 | 0 | 0 | new_sample_not_in_analysis |
| 252385 | 1 | 128 | 128 | 47 | 18 | 21 | 0 | match |
| 252527 | 1 | 99 | 99 | 23 | 14 | 11 | 0 | match |
| 250432 | 0 | 234 | 0 | 0 | 8 | 0 | 0 | new_sample_not_in_analysis |
| 252383 | 1 | 140 | 127 | 5 | 6 | 0 | 0 | grown_portal_gt_used |
| 252384 | 1 | 100 | 99 | 5 | 5 | 0 | 0 | grown_portal_gt_used |
| 252714 | 0 | 144 | 0 | 0 | 4 | 0 | 0 | new_sample_not_in_analysis |
| 252718 | 1 | 190 | 190 | 13 | 4 | 0 | 0 | match |
| 233001ch2 | 0 | 25 | 0 | 0 | 1 | 0 | 0 | new_sample_not_in_analysis |

### Gap-relevant new / not-in-combined candidates

- **IA/ID** candidates not in combined: n=6 sides={'L': 5, 'R': 1} samples=['250432', '252714']
- **IG** candidates not in combined: n=12 sides={'R': 7, 'L': 5} samples=['250432', '252383', '252790']
- **IAL** candidates not in combined: n=17 sides={'R': 16, 'L': 1} samples=['252790']
- **IAI** candidates not in combined: n=16 sides={'L': 15, 'R': 1} samples=['252790']
- **IAM/IAPM** candidates not in combined: n=1 sides={'L': 1} samples=['233001ch2']
- **PrCO** candidates: n=5, bbox-rescuable (pad=2.0 mm, Phys/250→NII): n=0

**Headline adds (not in combined):** sample **252790** (tracker 797) — 38 portal insula (IAL 1L/16R, IAI 15L/1R, IG 5L); **250432** — 8 (IA/ID 2L + IG 6R); **252714** (tracker 631) — 4 IA/ID (3L/1R). 251637 portal grew to 562 cells but **all** portal insula/PrCO IDs already sit in combined (new IDs are M1/empty/3a–b). 5 residual PrCO (252385/252527) are **outside** the 251637 bbox (rescue not applicable).

### Flagged injection samples (insula/opercular terms)

- 233001ch2 (tracker monkey —): flags=insula_terms; in_current=0; inj=`area_13m_32 ;;Iam/Iapm_39 ;`
- 251637 (tracker monkey 936): flags=insula_terms+opercular_PrCO; in_current=1; inj=`Ial_42 ;PrCO_49 ;Iai_41 ;Ia/Id_228 ;;M1_79 ;F4_85 ;PMdc_82 ;area_24c_10 ;preSMA_`
- 252385 (tracker monkey —): flags=insula_terms+opercular_PrCO; in_current=1; inj=`area_45a_74 ;F5_86 ;area_45b_75 ;area_12l_70 ;;G_48 ;Ia/Id_228 ;Ial_42 ;Cl_304 ;`
- 252527 (tracker monkey 945): flags=opercular_PrCO; in_current=1; inj=`M1_79 ;;PMdc_82 ;areas_1-2_94 ;area_3a/b_93 ;Iai_41 ;PrCO_49 ;F4_85 ;SII_95 ;Ial`

## Blockers / caveats

- Spot-check (2026-09-26): raw SWC `http://10.10.31.31/swc/newswc/.../swc_raw/001.swc` returned **HTTP 200** for 251637, 252383, 252384, 252385, and **252790** (historically some 252383/4/5 raw fetches were HTTP 500 — may be neuron-specific or fixed; not exhaustively re-probed).
- Metadata APIs used here (`selectNeurons`, `getSoma`, `neuronsBySoma`) all returned HTTP 200 (83/83 in `endpoint_http_log.csv`).
- Hemisphere for portal-only neurons without CL_/CR_ prefix is inferred from NII X vs midline 128 when getSoma coords exist.
- `n_used` counts any neuron ID present in recovery+combined tables (full sample scan), while scientific cohort size is `n_combined_insula` (combined Summary currently **353** neurons, L155/R198 — not the older 306 figure).
- Portal does **not** emit IDD5/IDM labels (those are curated/CHARM); dysgranular-like portal tags are mainly `Ia/Id`.

## Output files

- `portal_macaque_samples.csv`
- `portal_vs_current_diff.csv`
- `new_candidate_neurons.csv`
- `portal_neuronsBySoma_insula_PrCO.csv`
- `endpoint_http_log.csv`
- `audit_portal.py` (this script)
- `README.md`
