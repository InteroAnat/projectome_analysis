# Insula single-neuron projectome — analysis plan v3

**Date:** 2026-09-26  
**Supersedes:** `notes/whole_insula_lr_continuation_plan_v2.md`  
**Anchor review:** `notes/LR_insula_analysis_review.md` (§3.2, §10, §11A–C)  
**Poster inspirations (2026 Brain Mapping Forum):**  
1. Xu/Shi (ION/CEBSIT) — VTA DA subtypes: projection-defined taxonomy ↔ molecular markers ↔ divergent function.  
2. Hou/Kennedy/Knoblauch (Inserm/SBRI) — visual-space → structure/molecular: log(FLN), exponential distance rule (EDR), continuous spatial gradients, Stereo-seq layer/cell-type composition.  

**Cohort follow-up identity:** this work is a **Gou et al. 2025** macaque PFC projectome follow-up (methods + Table_S2 schema apply where flagged), not a generic methods cite.

---

## 0. Data snapshot (verify before any v3 claim)

| Item | Status on disk (2026-09-26) | Notes |
|---|---|---|
| Canonical workbook | `group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx` | Sheets include Summary, L6/L3 ipsi/contra, Mapping_Rule, Henry_Layer_Provenance |
| **v2 freeze (published analyses)** | **n = 306** (121 L : 185 R) | Samples: 251637 + 252383/252384/252385 only — see v2 README |
| **Current harmonized Summary** | **n = 353** | SampleID counts: 251637=260, 252385=47, **252527=23**, **252718=13**, 252383=5, 252384=5 |
| Recovery files present | `group_analysis/recovery/252527_*.xlsx`, `252718_*.xlsx` | **Already ingested** into harmonized Summary; do not treat as “pending first ingest” |
| Henry layers | 260/353 (`Layer_Source=henry_20260403`); 93 NA | Henry only covers 251637; **do not equate Henry L5 with Liu L5b** |
| Granular / “G” | `Soma_Region_Refined=G`: n=20 (18 L : 2 R) from 252385+252718 | Atlas granular; reconcile with Ig / IG labels before inference |
| IAL imbalance | Harmonized: IAL 32 L : 126 R | Still imbalanced; L-IAL gap remains primary acquisition priority |
| IDD5+IDM | IDD5 38:36; IDM 47:32 | Remains primary laterality stratum (recompute after Phase 0 freeze) |
| VEN candidates (251637) | 007, 026, 028, 036, 055, 056 | Liu 2026 bridge: branch `test/liu-ven-fmost-bridge` / `external/ven_validation/` |
| FNT matrix | `group_analysis/fnt/multi_monkey_INS_dist.txt` | Built for prior 306; **must rebuild** after cohort freeze |
| Portal audit (parallel) | `group_analysis/portal_audit_20260926/` | Ingest any NEW portal hits via Phase 0; do not assume counts |
| Cross-modal | `D:/multimodal_fmri/registry/subjects.tsv` | cm032/cm033 AIC opto/mstim ↔ fMOST IDs still TBD |
| Injection plan | `notes/projectome_latest_update_progress.md` | VEN AIV/vaIC ×14; FIDA/FID/FIDP ×6 each |

**v3 rule:** freeze a dated cohort snapshot under `group_analysis/combined/snapshots/` before primary stats; never mix “306-era” FNT with 353+ Summary without rebuild.

---

## 1. Goal and primary questions

**Goal.** Deliver a publishable macaque whole-insula single-neuron projectome analysis that (i) locks a refreshed multi-monkey cohort with audited labels, (ii) retains v2 laterality claims only where strata remain powered, and (iii) adds projection-defined subtype + spatial-embedding analyses inspired by the Xu and Hou posters, with explicit power gates.

### Primary questions (ranked by feasibility with current on-disk n)

| Rank | ID | Hypothesis | Feasibility now | Gate |
|:---:|---|---|---|---|
| 1 | **Q1** | Insula neurons form stable **projection-defined subtypes**; subtypes map onto soma gradient / Henry layer / (where available) VEN morphology — Xu-style taxonomy without claiming molecular one-to-one. | **High** (n≥300 projection vectors) | Consensus cluster stability (Jaccard ≥0.6 across bootstrap); subtype n≥10 for enrichment tests |
| 2 | **Q2** | **Soma position** along AP / DV / granular→agranular axes predicts target profile as a **continuous map** (Hou eccentricity analogue); discrete subregion is a coarse binning of the same axis. | **High** | Gradient regression + PERMANOVA residual after continuous covariates; BH within axis family |
| 3 | **Q3** | A large fraction of projection strength is explained by **soma–target Euclidean distance (EDR / spatial embedding)**; residuals mark “specific” connectivity (cf. Markov/Kennedy FLN). | **Medium–High** | Fit EDR on targets with n_projecting≥10; compare residual ranks to literature FLN where available |
| 4 | **Q4** | In **IDD5+IDM balanced** stratum, L>R for dysgranular→Ig and L>R for LPal/VPal/Cl (Craig left-approach / Hoy RPE / EPIC) survive cohort refresh + LOSO. | **Medium** (powered in 251637; refresh may shift n) | Inferential only if both sides n≥30 in stratum; else descriptive |
| 5 | **Q5** | Projection-defined AIC/VEN-related subtypes’ target maps **overlap** left-AIC opto/mstim fMRI activation maps (cm032/cm033). | **Low–Medium** | Requires registry ID link + available GLM maps; spatial Dice/overlap only — no causal claim |

Secondary (queued, not primary): hurdle decomposition of L>R Ig signal; brainstem RVLM/DMV/NTS segmentation for Oppenheimer; Stereo-seq cell-type composition proxy by subregion/layer (Chen et al. 2023).

---

## 2. Phase 0 — Data refresh / QC (mandatory before v3 primary)

| Step | Analysis | Method | Inputs (verified paths) | Outputs (`…/combined_primary_v3/…`) | Stats / family | Min n / power gate | Poster tag | Effort |
|---|---|---|---|---|---|---|---|---|
| **0.1** | Portal ingest audit | Compare ION portal sampleInfo + neuronsBySoma labels vs harmonized SampleID set; list NEW samples / NEW neurons (esp. L-IAL, dysgranular, Ig/G). Parallel worker owns live audit; this plan only defines acceptance. | `group_analysis/portal_audit_20260926/`; `group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx`; `group_analysis/recovery/all_refined_neurons.csv`; tracker IDs in `notes/projectome_latest_update_progress.md` | `phase0/portal_delta_candidates.csv`, `phase0/acceptance_log.md` | Descriptive counts only | Accept a neuron only if criteria below pass | — | 1–2 d |
| **0.2** | ID reconciliation | Canonical `NeuronUID` / `NeuronID` (`{int:03d}.swc`); reconcile 306-era vs 353-era rows; flag 252527/252718 provenance (`Soma_Region_Source`) | Harmonized Summary + `group_analysis/scripts/05a_build_combined_table.py` | `phase0/cohort_freeze_manifest.csv`, `phase0/id_collision_report.csv` | — | 0 collisions allowed | — | 0.5 d |
| **0.3** | PrCO rescue re-validation | Replace sole reliance on 251637 bbox+2 mm with **dual rule**: (A) atlas PrCO→insula mapping agreement rate vs flatmap location inside insula mask; (B) distance-to-bbox vs padded/distant flags. Drop or quarantine `*_distant` / ambiguous if fail | `group_analysis/reference/251637_subregion_bboxes.csv`; `group_analysis/recovery/prco_distance_to_bbox.csv`; `group_analysis/recovery/opercular_missed_audit.csv`; flatmaps under `group_analysis/R_analysis/outputs/figures/flatmap/` | `phase0/prco_rescue_revalidation.csv`, quarantine list | Agreement %; Cohen’s κ atlas vs flatmap if both labels exist | Keep rescued neuron only if **≥1 of**: flatmap-in-insula OR strict bbox (pad≤2 mm) OR high-confidence atlas rule; **reject** padded_distant unless flatmap confirms | — | 1–2 d |
| **0.3a** | Fix rescue geometry (blocks 0.3) | Code review verified: Soma_NII voxel = 0.25 mm, so `PAD_TOL = 2.0` was 0.5 mm (docs say 2 mm); 251637 bboxes pool L+R so X spans the midline (IAL 75–195 vox). Rebuild bboxes on R→L folded X (midline 128 vox), express pad in mm (`pad_vox = mm / 0.25`), re-run `04_refine_soma_region_by_coords.py` and `03b_relaxed_scan.py`, diff rescued sets vs current | `group_analysis/scripts/01_build_insula_coord_reference.py`, `04_refine_soma_region_by_coords.py`, `03b_relaxed_scan.py` | `phase0/rescue_rebuild_diff.csv` | — | Report gained/lost neurons per sample before freezing | — | 0.5 d |
| **0.3b** | Stats code fixes (block Phase A) | Drop `pmin(p_pres_BH, p_mag_BH)` ranking (report families separately); retire `multi_monkey_lr_analysis.R` family reading non-harmonized xlsx; add n_source<10 greying to F3. See `notes/code_review_20260926.md` | `group_analysis/R_analysis/v2_combined_primary_pipeline.R`, `combined_lr_primary_analysis.R` | — | BH within stratum × family only | — | — | 0.5 d |
| **0.4** | Label harmonization | Map `G`↔Ig/IG; freeze Mapping_Rule sheet; Henry merge coverage assert for 251637 | `group_analysis/scripts/06_harmonize_atlas_to_manual.py`, `06a_merge_henry_layers.py` | Updated snapshot xlsx + `phase0/label_vocab.csv` | — | Henry merge n=260 for 251637 or hard fail | — | 0.5–1 d |
| **0.5** | Rebuild FNT + projection matrices | Re-run FNT join for frozen cohort; L6 ipsi strength matrix | `group_analysis/fnt/`; `main_scripts/step2_fnt-dist_pipeline.py` (upstream) | `phase0/fnt_rebuild_qc.csv`; refreshed `multi_monkey_INS_dist.txt` | Within vs cross-monkey FNT distribution overlap check | Matrix dim = frozen N | Hou (distance) | 1–2 d |
| **0.6** | Power table | Per-stratum L:R and per-target projecting counts | Frozen Summary | `phase0/power_table_per_stratum.csv` | — | Mark strata: inferential / descriptive / blocked | — | 0.5 d |

### Phase 0 acceptance criteria (new samples / neurons)

Include a portal neuron in the v3 freeze **only if all** hold:

1. Soma atlas label ∈ insula vocabulary **or** PrCO passes **0.3** dual rule.  
2. Complete L6 ipsilateral projection row (no empty strength vector).  
3. Soma_NII coords in NMT frame; side assignable.  
4. Sample on injection tracker **or** documented incidental yield with injection site recorded.  
5. Prefer fills that reduce gaps: **L-IAL**, bilateral dysgranular, **Ig/G**, VEN-rich AIV/vaIC.

**Do not** claim planned injection totals (14+6+6+6) as available n.

### Phase 0.1 result — ION portal audit (2026-09-26)

Source: `group_analysis/portal_audit_20260926/` (`portal_vs_current_diff.csv`, `new_candidate_neurons.csv`). All metadata endpoints HTTP 200; raw SWC spot-checks 200.

| Sample (tracker) | Status | New insula-atlas somata (L / R) | Harmonized target (current `06_harmonize` rule) |
|---|---|---|---|
| **252790** (797, vaIC-L/R, IDFM-R, IDFP-L) | not in analysis, 123 neurons | Iai 15 / 1; Ial 1 / 16; Ig 5 / 0 | Iai → IAPM (ambiguous); Ial → IAL; Ig → **IDD5** |
| **250432** (not on tracker) | not in analysis, 234 neurons | Ia/Id 2 / 0; Ig 0 / 6 | Ia/Id → IDM (ambiguous); Ig → **IDD5** |
| **252714** (631, IDFA-L/R) | not in analysis, 144 neurons | Ia/Id 3 / 1 | → IDM (ambiguous) |
| 252383 (605) | +13 portal neurons | Ig 0 / 1 | → IDD5 |
| 233001ch2 | not in analysis | Iam/Iapm 1 / 0 | IAPM |
| 251637 | portal 562 vs 260 used | 0 new insula/PrCO IDs (growth = M1 / unlabeled / 3a-b) | — |
| 252385, 252527 | match | 5 residual PrCO, all outside bbox | not rescuable |

Up to **+57** candidates: +17 L IAL/IAPM-like (mostly 252790 Iai), +11 IDD5-mapped "Ig", +6 IDM-mapped Ia/Id.

**Label caveat (decide before ingest):** `06_harmonize_atlas_to_manual.py:78` maps atlas `Ig` → IDD5 ("improvised; verify in widefield"). Under that rule new "Ig" somata enlarge the *dysgranular* stratum, not granular Ig — which contradicts treating them as a granular gap fill. Resolve in 0.4 (flatmap / widefield check) before counting them toward either stratum.

**Tracker inconsistencies:** 252383 portal injection = F4-dominated (not IDFA/IDFP as in `projectome_latest_update_progress.md`); 252334 (948) and 252985 (331), marked Done, are absent from portal sampleInfo; 252790 / 252714 have neurons but empty injection metadata. Portal PrCO re-check must be re-run after 0.3a (folded-X bboxes).

---

## 3. Analysis phases (v3)

**Root outputs:** `group_analysis/R_analysis/outputs/combined_primary_v3/`  
**Gou 2025 methods apply** where tagged **[Gou]** (vote_connections heatmaps, Mann–Whitney/Fisher, Ibias definition, Table_S2 panels, Ward+barjoseph order, thr=3 / n_source≥10 display rules — see `notes/gou_stats_methods_index.md`).

### Phase A — Carry-forward primary (refresh v2 endpoints on frozen cohort)

| ID | Analysis | Method | Inputs | Outputs | Stats + BH family | Min n / gate | Poster | Effort |
|---|---|---|---|---|---|---|---|---|
| A1 | Context / provenance | Counts by Sample×Region×Side×Source | Frozen harmonized | `stats/00_context_*.csv` | — | — | — | 0.5 d |
| A2 | Subregion × Gou panels | Mean prop + frac projecting; L+R heatmap + L vs R secondary | Summary + `R_analysis/scripts/data_output/gou_function_table/area_function_category_full.csv` | `stats/01_*`, `figures/F1_*` | Mann–Whitney / Fisher; **BH within (stratum × Gou-panel family)** **[Gou]** | Cell grey if n_source<10; test if n≥10/side when laterality | Xu (matrix layout) | 1 d |
| A3 | Subregion × key targets | Same for Ig, Cl, LPal, VPal, caudal OFC, autonomic proxies | L6 strength sheets | `stats/02_*`, `figures/F2_*` | BH within (stratum × key-target family) | Same | Xu | 1 d |
| A4 | Intra-insula heatmap | **[Gou]** `vote_connections` / Ward+barjoseph, k=2, thr=3 | L6 insula–insula columns | `stats/03_*`, `figures/F3_*` | Descriptive + edge filter | Edge ≥3 neurons | Xu | 1 d |
| A5 | Interoceptive gradient regression | `lm` slope vs soma AP/ML per side; L–R contrast **[Gou]** `linear_reg_scatter` idiom | Soma_Phys + target props | `stats/04_*`, `figures/F4_*` | BH within gradient-endpoint family | n≥30/side for L–R slope contrast | **Hou** (continuous map) | 1 d |
| A6 | Per-neuron Ibias | Contra/(ipsi+contra) length; group MWU **[Gou]** | Length sheets | `stats/05_*`, `figures/F5_*` | MWU; note sparse contra (~2%) | Report applicability NA if contra n tiny | — | 0.5 d |
| A7 | Asymmetric receivers + hierarchy sensitivity | Presence + magnitude; target-level | L6 | `stats/06_*`, `07_*`, `figures/F6–F7_*` | BH within (stratum × receiver family) | Balanced strata only for L/R claims | — | 1–2 d |
| A8 | Mantel FNT ↔ Bray-prop | Spearman Mantel; SQ4 logic (`notes/sq4_sq5_distance_digest.md`) | Rebuilt dist + projection Bray | `stats/08_mantel_*.csv` | Permutation p | N = freeze | Hou / Gou SQ | 0.5 d |
| A9 | LOSO | Drop-one-SampleID on headline endpoints | Same | `stats/09_loso_*.csv`, `figures/F8_*` | Direction consistency | ≥2 monkeys in stratum | — | 1 d |
| A10 | Flatmap overlays | Julia base + Python strip | `group_analysis/julia_scripts/`; `scripts/13_flatmap_context_strip.py` | `flatmap_overlays/F9–F10_*` | Visual QC | — | Hou (spatial) | 1 d |
| A11 | Stat registry | Every manuscript number → CSV row | All above | `spec/stat_registry.csv` | — | — | — | 0.5 d |

**Primary laterality stratum (recompute after freeze):** `IDD5_plus_IDM_balanced`. IAL = descriptive unless both sides reach n≥30.

### Phase B — Projection-defined subtype taxonomy **[Xu]**

| ID | Analysis | Method | Inputs | Outputs | Stats + family | Min n / gate | Effort |
|---|---|---|---|---|---|---|---|
| B1 | Feature matrix | Row-normalize L6 ipsi strength (Bray-prop space); optional log1p | Frozen L6 strength | `subtypes/features_L6_prop.csv` | — | Neurons with Total_Length>0 | 0.5 d |
| B2 | Consensus clustering | Hierarchical / PAM on Bray; 1000 bootstrap; consensus matrix; choose k by PAC / silhouette | B1 | `subtypes/consensus_k*.csv`, `figures/S1_consensus.png` | Stability Jaccard vs resampled labels | Keep subtypes with n≥10 for enrichment; unstable k → report continuum only | 2–3 d |
| B3 | FNT + projection joint embedding | Concatenate z-scored FNT-neighbor summary + projection PCs; UMAP/PCA; optional LDA for known region labels | FNT dist + B1 | `subtypes/joint_embedding.csv`, `figures/S2_umap.png` | PERMANOVA subtype~embedding | Same n gate | 2 d |
| B4 | Map subtypes → covariates | Enrichment vs Henry layer, Soma_Region, Neuron_Type, VEN candidate flag, AP/DV bins | Henry columns; VEN ID list; Liu bridge morphometrics if ready | `stats/10_subtype_enrichment.csv`, `figures/F11_*` | Fisher / χ²; **BH within covariate family** | Henry tests = 251637 only; VEN n=6 → descriptive | 1–2 d |
| B5 | Subtype × target heatmap | Xu-style neuron×target matrix ordered by subtype | B1–B2 | `figures/F12_subtype_projection_matrix.png` | — | — | 1 d |

**Claim boundary:** projection-defined subtypes **yes**; molecular one-to-one (**Xu DAT/Tacr3 style**) **no** without snRNA-seq / Stereo-seq join.

### Phase C — Spatial embedding / EDR / stream-like splits **[Hou]**

| ID | Analysis | Method | Inputs | Outputs | Stats + family | Min n / gate | Effort |
|---|---|---|---|---|---|---|---|
| C1 | Soma–target distance | Euclidean (or geodesic proxy) from soma NMT xyz to target ROI centroids | Soma_NII; atlas centroids (NMT/ARM) | `spatial/soma_target_distance.csv` | — | Targets with centroid available | 1–2 d |
| C2 | EDR fit | `log(strength) ~ −distance` per neuron or per source bin; pool by subtype/region | C1 + strength | `stats/11_edr_fits.csv`, `figures/F13_*` | Mixed model or robust regression; **BH within source-bin family** | ≥10 targets with nonzero strength per fit unit | 2 d |
| C3 | Specificity residuals | Observed − EDR-predicted strength; rank residual hubs | C2 | `stats/12_edr_residuals.csv` | Compare residual ranks to Markov/Kennedy FLN for overlapping areas (literature table, not re-inject) | Descriptive if FLN coverage sparse | 1–2 d |
| C4 | Continuous gradient map | Multivariate regression / redundancy analysis: projection PCs ~ AP + DV + granular index | Soma coords + architectonic rank | `stats/13_gradient_RDA.csv`, `figures/F14_*` | Permutation tests; BH within PC family | n≥100 | 2 d |
| C5 | Limbic/visceromotor vs somatomotor/cognitive split | Assign Gou Table_S2 targets to two “streams”; dorsal/ventral-stream analogue | Gou function CSV | `stats/14_stream_bias.csv` | MWU / Cliff’s δ; BH within stream-index family | Balanced stratum for L/R stream bias | 1 d |

### Phase D — Morphology / VEN / layer (bridge)

| ID | Analysis | Method | Inputs | Outputs | Stats | Gate | Effort |
|---|---|---|---|---|---|---|---|
| D1 | Henry layer × projection | Layer as factor in PERMANOVA / subtype enrichment | Henry Cortical_Layer (251637) | `stats/15_layer_projection.csv` | BH within layer×panel | Henry n=260; layers 3/5/6 only | 1 d |
| D2 | VEN candidate morphometrics | Liu 2026 bridge features vs matched non-VEN ITi in same region | `external/ven_validation/`; IDs 007,026,028,036,055,056 | `stats/16_ven_bridge.csv` | Descriptive + exact tests | n=6 → **no** population inference | 2–3 d |
| D3 | Layer naming note | Document Henry L5 ≠ Liu L5b | Methods paragraph | `spec/layer_nomenclature.md` | — | — | 0.5 d |

### Phase E — Molecular proxy (Stereo-seq) **[Hou]**

| ID | Analysis | Method | Inputs | Outputs | Stats | Gate | Effort |
|---|---|---|---|---|---|---|---|
| E1 | Insula cell-type composition by area/layer | Re-analyze public Chen et al. 2023 macaque Stereo-seq for insula parcels/layers (esp. L4-analogue vs agranular) | External Stereo-seq deposit (path TBD when downloaded) | `molecular/stereo_insula_composition.csv` | Descriptive proportions; LDA optional | Requires data access; **no** claim that fMOST neurons are those cell types | 1–2 wk |
| E2 | Soft correspondence | Correlate Stereo-seq area profiles with mean projectome profiles of matching subregions | E1 + A2 | `stats/17_stereo_projectome_corr.csv` | Spearman; BH across areas | ≥5 areas with both modalities | after E1 |

### Phase F — Functional link to AIC opto-fMRI **[Xu function arm]**

| ID | Analysis | Method | Inputs | Outputs | Stats | Gate | Effort |
|---|---|---|---|---|---|---|---|
| F1 | Registry resolve | Map cm032/cm033 ↔ fMOST animal when known | `D:/multimodal_fmri/registry/subjects.tsv` | `crossmodal/registry_status.csv` | — | Blocked while fMOST IDs TBD | ongoing |
| F2 | Target overlap | Threshold opto/mstim SPM maps; overlap with high-residual / subtype-preferred targets in NMT | multimodal GLM maps-NMT; projectome residual hubs | `crossmodal/aic_opto_overlap.csv`, figures | Dice / null spin if available | Maps must exist in NMT; report exploratory | 3–5 d when unblocked |

### Phase G — Review-queued mechanistic extras

| ID | Analysis | Method | Inputs | Outputs | Stats | Gate | Effort |
|---|---|---|---|---|---|---|---|
| G1 | Hurdle decomposition | Presence (Fisher) vs magnitude (Wilcoxon \| projecting) for dysgranular→Ig L>R | Balanced stratum | `stats/18_hurdle_Ig.csv` | Two families, BH separate | Same as Q4 | 0.5–1 d |
| G2 | Brainstem fine labels | Custom RVLM/DMV/NTS segmentation on NMT | Atlas work | `brainstem/fine_targets_*` | Re-test autonomic L/R | **Blocked** until atlas; IAL still underpowered | weeks |
| G3 | L6 systematic hub refresh | Extend §11B per-target hurdle table on freeze | L6 | `stats/19_L6_hubs.csv` | BH within hub family | n≥10/side | 1–2 d |

---

## 4. What we cannot claim with current n

| Claim | Why blocked |
|---|---|
| Molecular marker ↔ projection subtype one-to-one (Xu DAT/Tacr3/Aldh1a1 style) | No snRNA-seq on these neurons; Stereo-seq is area-level proxy only |
| Population inference on VENs | n=6 candidates; Liu bridge = morphology comparison, not subtype census |
| Oppenheimer cardiac laterality (R→sympathetic RVLM vs L→DMV) | NMT L6 aggregates brainstem; IAL still L-scarce (≈32:126 even after 252527) |
| Agranular IAL L vs R (salience / Menon–Uddin R-AI) | Imbalanced; IAL remains descriptive |
| Ig as primary-interoceptive cortex population stats | Granular still small (G+Ig historically ≤20-ish); descriptive until portal refresh fills |
| Henry L5 = Liu L5b | Different definitions; state explicitly |
| Causal “activation → behavior” for subtypes | No optogenetic activation of fMOST-labeled cells; fMRI overlap is correlative |
| Gou-style per-neuron Ibias as main laterality metric | Contra projections rare (~2%); Ibias secondary only (`notes/hemispheric_asymmetry_methods_comparison.md`) |
| Treating planned injections (14+6+6+6) as analyzed n | Acquisition ≠ reconstructed neurons |
| Using 306-era Mantel/FNT numbers on 353-row Summary without rebuild | Dimension mismatch |

---

## 5. Priority order, dependencies, effort

### Next 2 weeks

1. **Phase 0** complete (portal delta + PrCO dual rule + freeze + FNT rebuild + power table).  
2. **A1–A11** re-run on freeze (v2 carry-forward; manuscript numbers update).  
3. **G1** hurdle on Ig L>R (cheap, high yield).  
4. **B1–B2** consensus clustering scaffold (even if k soft).  
5. **C1–C2** EDR first pass on major targets.

### Later (after freeze stable / more L-IAL or Ig)

- B3–B5 joint embedding + enrichment; C3–C5 residuals/streams; D2 VEN bridge figures; E1–E2 Stereo-seq; F1–F2 when registry filled; G2 brainstem.

### Dependencies

```
Portal audit ──┐
Recovery/PrCO ─┼─► Phase 0 freeze ─► FNT rebuild ─┬─► Phase A (v2 refresh)
Henry merge ───┘                                  ├─► Phase B (needs A feature matrix)
                                                  ├─► Phase C (needs centroids + A)
                                                  └─► G1/G3
Phase B subtypes ─► D1/D2 enrichment
Phase C residuals + B subtypes ─► F2 (needs multimodal maps + registry)
E1 external data ─► E2
G2 atlas ─► Oppenheimer retest (still needs L-IAL)
```

---

## 6. Diff vs plan v2

| Item | Status |
|---|---|
| Harmonized table as input; Mapping_Rule | **Carried** (re-freeze; n may be 353+) |
| Strata: IDD5+IDM primary; IAL descriptive | **Carried** (recompute counts) |
| Gou vote_connections heatmaps; thr=3; n_source≥10 | **Carried** **[Gou]** |
| Gradient regression; Ibias; BH within stratum×family | **Carried** |
| LOSO; Mantel Bray-prop; flatmap overlays; stat registry | **Carried** |
| Output tree under `combined_primary_v2/` | **Changed** → `combined_primary_v3/` |
| Assume cohort = 306 / 4 monkeys | **Changed** → Phase 0 acknowledges **252527+252718 already in xlsx**; freeze after portal |
| PrCO rescue = bbox+2 mm only | **Changed** → dual rule (atlas/flatmap + quarantine distant) |
| Single-bar domains | Already replaced in v2 by heatmaps — **carried** |
| Projection subtype consensus + FNT joint embedding | **New** **[Xu]** (was review §10 #7) |
| EDR / spatial-embedding null + residuals | **New** **[Hou]** |
| Continuous AP/DV/granularity map (beyond discrete subregion) | **New** **[Hou]** (extends v2 gradient) |
| Limbic vs somatomotor “stream” bias | **New** **[Hou]** |
| Stereo-seq composition proxy (Chen 2023) | **New** **[Hou]** |
| AIC opto-fMRI overlap (cm032/033) | **New** **[Xu]** function arm |
| Hurdle Ig; L6 hubs refresh | **New** (from review §10 #6 / §11B) |
| Explicit “cannot claim” + power gates | **New** |
| Data refresh phase with acceptance criteria | **New** |
| VEN / Liu bridge + Henry≠Liu L5b | **New** explicit phase |

---

## 7. Stats conventions (global)

| Rule | Spec |
|---|---|
| Effect sizes | Cliff’s δ (continuous); log-OR + Fisher (presence); Wilcoxon (magnitude) |
| BH | Within **(stratum × analysis-family)** only — never global min across families (v2 fix retained) |
| Display | Grey cell if n_source<10; heatmap edge if n≥3 **[Gou]** |
| Inferential laterality | Only strata with both sides meeting power table gate (default n≥30 for headline L/R) |
| Contra / Ibias | Report; do not lead manuscript with Gou PFC-style Ibias |
| Registry | `spec/stat_registry.csv` mandatory for any quoted number |

---

## 8. Key file index

| Role | Path |
|---|---|
| This plan | `notes/analysis_plan_v3_20260926.md` |
| v2 plan | `notes/whole_insula_lr_continuation_plan_v2.md` |
| Critical review | `notes/LR_insula_analysis_review.md` |
| Gou methods index | `notes/gou_stats_methods_index.md` |
| Gou vs current stats | `notes/gou_vs_current_stats_detailed_comparison.md` |
| Asymmetry methods | `notes/hemispheric_asymmetry_methods_comparison.md` |
| SQ3d / SQ4–5 notes | `notes/sq3d_mixed_method_note.md`, `notes/sq4_sq5_distance_digest.md` |
| Injection / monkey table | `notes/projectome_latest_update_progress.md` |
| Living note | `docs/project_note.md` |
| Harmonized data | `group_analysis/combined/multi_monkey_INS_combined_harmonized.xlsx` |
| Composition helpers | `group_analysis/combined/monkey_neuron_composition_*.csv` |
| Recovery / PrCO audit | `group_analysis/recovery/` |
| FNT | `group_analysis/fnt/multi_monkey_INS_dist.txt` |
| v2 pipeline | `group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd` |
| v2 outputs | `group_analysis/R_analysis/outputs/combined_primary_v2/` |
| v3 outputs (new) | `group_analysis/R_analysis/outputs/combined_primary_v3/` |
| Portal audit | `group_analysis/portal_audit_20260926/` |
| Cross-modal registry | `D:/multimodal_fmri/registry/subjects.tsv` |

---

## 9. One-line execution motto

**Freeze auditable n → refresh v2 claims under power gates → add Xu subtypes + Hou EDR/gradients → never overclaim molecular, VEN, or Oppenheimer.**
