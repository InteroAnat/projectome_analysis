# Staging 2026-09-26 — Phase 0 fixes + portal ingest

All outputs under this folder. **Canonical** `group_analysis/{combined,recovery,reference}` were not overwritten.

## What changed (code)

1. `scripts/cohort.py` — central `NEW_SAMPLES` (9 IDs incl. 252527/252718/252790/250432/252714), geometry constants, `PROJECTOME_SAMPLES` / `PROJECTOME_PAD_TOL_MM` env; `KEEP_ATLAS_LABELS={'G'}` so gustatory G stays in cohort as G (not mapped to IG).
2. Folded-X rescue geometry (`dx=|X-128|`) + pad in mm (`PAD_TOL_MM` → voxels via 0.25 mm).
3. Ig harmonization: atlas `Ig` → granular `IG` (`atlas_Ig_to_IG_granular_user_rule_20260926`); atlas `G` untouched; only `auto_atlas_insula` rows edited; 251637 remains `curated_251637`.
4. R: drop `pmin` across presence/magnitude BH families; F3A greying for `n_source<10`; deprecate `multi_monkey_lr_analysis.R`.

## Commands run

```powershell
$PY = (conda run -n deb_fmri_pipeline python -c "import sys;print(sys.executable)").Trim()
$env:PYTHONIOENCODING='utf-8'
$ST='D:\projectome_analysis\group_analysis\staging_20260926'
$env:PROJECTOME_REFERENCE_OUT="$ST\reference"
# 01 folded reference
& $PY group_analysis\scripts\01_build_insula_coord_reference.py
# step1 ingest (multimodal_ins_win — seaborn missing in deb_fmri_pipeline)
& $PY_MM group_analysis\scripts\run_step1_one_sample.py 252790
& $PY_MM group_analysis\scripts\run_step1_one_sample.py 250432
& $PY_MM group_analysis\scripts\run_step1_one_sample.py 252714
$env:PROJECTOME_REFERENCE_DIR="$ST\reference"
$env:PROJECTOME_PAD_TOL_MM='0.5'; $env:PROJECTOME_RECOVERY_OUT="$ST\recovery_pad0p5"
& $PY group_analysis\scripts\04_refine_soma_region_by_coords.py
$env:PROJECTOME_PAD_TOL_MM='2.0'; $env:PROJECTOME_RECOVERY_OUT="$ST\recovery_pad2p0"
& $PY group_analysis\scripts\04_refine_soma_region_by_coords.py
$env:PROJECTOME_RECOVERY_DIR="$ST\recovery"; $env:PROJECTOME_COMBINED_OUT="$ST\combined"
& $PY group_analysis\scripts\05a_build_combined_table.py
$env:PROJECTOME_COMBINED_IN="$ST\combined\multi_monkey_INS_combined.xlsx"
$env:PROJECTOME_HARMONIZED_OUT="$ST\combined\multi_monkey_INS_combined_harmonized.xlsx"
& $PY group_analysis\scripts\06_harmonize_atlas_to_manual.py
```

### Not run (list only)

```powershell
# Global FNT rebuild (heavy) — do NOT run here:
& $PY group_analysis\scripts\05c_run_global_fnt.py
# (point inputs at staging harmonized / updated cohort after freeze)
```

## Membership vs canonical harmonized (n=353)

- Staging harmonized n = **420**
- Added neurons: **67** (see `qc_membership_added.csv`)
- Removed neurons: **0** (see `qc_membership_removed.csv`)

## Balanced-stratum L:R (before = canonical, after = staging)

| Stratum | Can L:R (n) | Stg L:R (n) |
|---------|-------------|-------------|
| IDD5 | 38:36 (74) | 33:30 (63) |
| IDM | 47:32 (79) | 52:33 (85) |
| IDD5+IDM | 85:68 (153) | 85:63 (148) |
| IAL | 32:126 (158) | 33:147 (180) |
| IG | 0:0 (0) | 10:12 (22) |
| G | 18:2 (20) | 25:7 (32) |

## Rescue rebuild (folded X; vs canonical `recovery/all_refined_neurons.csv`)

Note: canonical `all_refined_neurons.csv` lacked 252527/252718 (those lived only as per-sample xlsx / combined).

```
 SampleID  n_old  n_pad0p5  n_pad2p0  gained_pad0p5  lost_pad0p5  gained_pad2p0  lost_pad2p0  rescued_old  rescued_pad0p5  rescued_pad2p0
   250432      0         8         8              8            0              8            0            0               0               0
   252383      5         5         5              0            0              0            0            0               0               0
   252384      5         5         5              0            0              0            0            0               0               0
   252385     36        47        50             11            0             14            0           18              18              21
   252527      0        23        25             23            0             25            0            0               9              11
   252714      0        16        16             16            0             16            0            0               0               0
   252718      0        13        13             13            0             13            0            0               0               0
   252790      0        38        38             38            0             38            0            0               0               0
```

- Pad 0.5 mm = legacy effective pad (2 vox). Pad 2.0 mm = documented intent (8 vox).
- Primary staging recovery used for 05a/06 = **pad 2.0 mm** (`recovery/` mirrored from `recovery_pad2p0/`).

## Ingest yields vs portal audit expected

| Sample | Observed atlas-leaf×side (combined keepers) | Expected (audit) | Match? |
|--------|---------------------------------------------|------------------|--------|
| 252790 | Iai_L=15, Iai_R=1, Ial_L=1, Ial_R=16, Ig_L=5 | IAI_L=15, IAI_R=1, IAL_L=1, IAL_R=16, IG_L=5, IG_R=0 | OK |
| 250432 | Ia/Id_L=2, Ig_R=6 | IA/ID_L=2, IA/ID_R=0, IG_L=0, IG_R=6 | OK |
| 252714 | G_L=7, G_R=5, Ia/Id_L=3, Ia/Id_R=1 | IA/ID_L=3, IA/ID_R=1 | DEV: extra G_R=5 (not in audit insula list); extra G_L=7 (not in audit insula list) |

- Ig→IG source-tag rows in staging harmonized: **22**
- Staging `Soma_Region_Refined==G` (gustatory, unmapped): **32**

## Mapping_Rule (staging)

```
atlas_leaf manual_dominant confidence                          source                                                                                                  notes
       Ial             IAL       high empirical_251637_crosstab_n=260                                                          100% concordance in 251637 empirical crosstab
      PrCO             IAL       high empirical_251637_crosstab_n=260                                                          100% concordance in 251637 empirical crosstab
        Ig              IG    user_ig  user_rule_20260926_granular_IG USER RULE 2026-09-26: atlas Ig → granular IG (overrides empirical Ig→IDD5 crosstab); atlas G untouched
       Iai            IAPM  ambiguous empirical_251637_crosstab_n=260                                                  split assignment in 251637; verify by coord/visual QC
     Ia/Id             IDM  ambiguous empirical_251637_crosstab_n=260                                                  split assignment in 251637; verify by coord/visual QC
 Unknown_0             IDM  ambiguous empirical_251637_crosstab_n=260                                                  split assignment in 251637; verify by coord/visual QC
```

## Crosstabs

- `qc_crosstab_canonical.csv` — SampleID × Soma_Region_Refined × Side (canonical)
- `qc_crosstab_staging.csv` — same for staging harmonized

## Env notes

- `deb_fmri_pipeline` lacked `seaborn` → step1 used `multimodal_ins_win`.
- ION fetch via existing `region_analysis` / IONData path (step1); new code elsewhere follows bounded-retry pattern.

## Revert

Branch `fix/phase0-20260926`, based on tag `pre-phase0-20260927`. Nothing merged into `test/liu-ven-fmost-bridge`; nothing pushed.

| SHA | Concern | Files | Undo |
|---|---|---|---|
| `3a58cbc` | Central cohort list (`cohort.py`) | `scripts/cohort.py`, `02`, `02a`, `03`, `03c`, `05b`, `bulk_visual_dryrun`, `bulk_visual_multi_monkey` | `git revert 3a58cbc` (revert later commits first; they import `cohort`) |
| `a15da4b` | Folded-X rescue boxes, pad in mm (default 2.0 mm), staging env overrides | `01`, `03b`, `04`, `05a` | `git revert a15da4b` |
| `4afec97` | Atlas `Ig` -> granular `IG` | `06_harmonize_atlas_to_manual.py` | `git revert 4afec97` |
| `d27c855` | BH ranking within family (no cross-family `pmin`), F3 n<10 greying | `R_analysis/v2_combined_primary_pipeline.R`, `combined_lr_primary_analysis.R` | `git revert d27c855` |
| `b452602` | Legacy `multi_monkey_lr_analysis.R` stops unless `PROJECTOME_ALLOW_LEGACY_LR=1` | `R_analysis/multi_monkey_lr_analysis.R` | `git revert b452602` |

Abandon everything: `git switch test/liu-ven-fmost-bridge` (optionally `git branch -D fix/phase0-20260926`).

Data (untracked, never committed):

```powershell
Remove-Item -Recurse group_analysis\staging_20260926
Remove-Item -Recurse group_analysis\step1_results\252790_20260927_151228_region_analysis
Remove-Item -Recurse group_analysis\step1_results\250432_20260927_151627_region_analysis
Remove-Item -Recurse group_analysis\step1_results\252714_20260927_152050_region_analysis
```

Canonical `combined/`, `recovery/`, `reference/`, `step1_results/` were not modified (hash-checked 2026-09-27). Pre-change copy: `D:\projectome_backups\pre_phase0_20260927\`.

Behaviour change to note: running `04` with defaults now uses 2.0 mm padding (previously an effective 0.5 mm); set `PROJECTOME_PAD_TOL_MM=0.5` to reproduce the old rescue set. 251637 curated labels keep their 24 atlas-Ig somata as IDD5, so atlas-Ig means IDD5 in 251637 but IG in other monkeys; flag this in methods.
