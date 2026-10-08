# Clustering review: historical FNT inputs

**Agent: Codex | Date: 2026-10-02 (Asia/Shanghai)**

These are fresh exploratory diagnostics on persisted historical matrices. No
original assignments, neuron tables, accepted cohort or previous R result was
replaced. The 212/306 matrices do not cover the current harmonized 353 or staging
420 neurons. No subtype count is accepted by this review.

## What the runs establish

All six runs use the revised public `main_scripts/fnt_dist_clustering.py`
entrypoint with average linkage, candidate k=2..20, 100 random subsets without
replacement, fraction 0.8 and seed 42. Raw FNT self scores are recorded before
setting the clustering diagonal to zero. Actual joined-FNT marker order defines
identity. Complete triangular inputs contain 22,578 pairs for n=212 and 46,971
pairs for n=306, including self pairs.

| Run directory | n | Representation | Type penalty | Exported exploratory cut | Silhouette suggestion |
|---|---|---|---|---|---|
| `multi_raw` | 306 | raw FNT | off | 2 | 2 |
| `multi_log1p` | 306 | log1p FNT | off | 2 | 2 |
| `multi_spearman_profile` | 306 | distance-profile Spearman | off | 2 | 2 |
| `multi_type_guided` | 306 | distance-profile Spearman | max distance × 1.5 | explicit 9 | 5 |
| `single_raw` | 212 | raw FNT | off | 2 | 2 |
| `single_spearman_profile` | 212 | distance-profile Spearman | off | explicit 9 | 7 |

The multi runs use the historically matching **raw 306-row** workbook, not the
current harmonized file. The single runs annotate 212 matrix neurons using the
260-row curated reference; the 48 additional annotations are explicitly recorded
and do not enter the fit. Every fitted neuron must have a unique annotation.

## Findings and limits

- Raw multi k=2 splits 3/303 despite silhouette 0.749; mean subset ARI is 0.512
  and its fifth percentile is -0.008. It isolates a very small high-dissimilarity
  group; direct review is needed to determine whether those neurons are biological
  outliers or reflect tracing/measurement differences. It does not establish two
  useful subtypes. Raw single k=2 similarly splits 1/211.
- Unguided profile-Spearman multi k=2 splits 147/159, silhouette 0.742, mean
  subset ARI 0.959. That is useful conditional robustness evidence for a coarse
  distance-profile division. It still requires direct inspection and biological
  validation. The profile representation was computed once using the full cohort.
- At k=9, raw versus profile-Spearman multi ARI is 0.140. Profile versus
  type-guided ARI is 0.166. Changing representation or adding type information
  substantially changes the partition; stability within one choice is insufficient.
- Unguided profile k=9 has three singleton groups. Guided profile k=9 has no
  singletons and mean subset ARI 0.992, but a group of 216/306 and prior type
  information. Neither that high ARI nor label agreement independently validates
  biological types. Its silhouette optimum is k=5, not the explicitly tested 9.
- Removing 251637 leaves only 46 neurons. At k=9, remaining-cohort ARI is 0.253
  for raw and 0.573 for profile-Spearman. These are sensitivity comparisons of the
  remaining neurons, not predictions for the omitted animal.
- Best-match Jaccard is computed against the reference cluster restricted to
  sampled neurons. Reference groups represented by fewer than two neurons are
  unassessed. The diagnostics explicitly count assessed/unassessed groups and
  singleton groups; a high score for the only large group must not imply that a
  singleton is stable.
- Global ARI is heavily influenced by large groups. Read the full clusterwise
  tables, sample composition and size distribution. Stability cannot establish
  discreteness, registration accuracy, complete tracing or molecular/VEN identity.

## Saved artifacts

- [Diagnostic overview](diagnostic_overview.png), also available as a standalone
  [PDF figure](diagnostic_overview.pdf).
- `mode_comparison.csv`: every k for every run; `selected_cut_summary.csv`: cuts
  exported by each public run; `cross_mode_partition_ari.csv`: same-cohort
  partition agreement at k=2,9,20.
- Within each run: `clustering_diagnostics.csv`, `c_index_diagnostics.csv`,
  `candidate_cluster_assignments.csv`, `cluster_assignments.csv`,
  `cluster_stability_k2.csv` through `cluster_stability_k20.csv`,
  `group_holdout_diagnostics.csv`, separate annotated workbook, and
  `clustering_metadata.json`.
- `swc_compatibility.json`: a separate read-only structural audit of 358 existing
  NMT/VEN SWCs (4,882,379 nodes); hashes were not captured in that audit. Passing
  it establishes parser compatibility, not verified anatomical reconstruction.

## Reproduction

From the repository root, using the projectome environment:

```powershell
$taskPython = 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe'
& $taskPython -B main_scripts/fnt_dist_clustering.py `
  --dist-file group_analysis/fnt/multi_monkey_INS_dist.txt `
  --type-file group_analysis/combined/multi_monkey_INS_combined.xlsx `
  --sheet Summary --mode raw --linkage average `
  --max-k 20 --repeats 100 --fraction 0.8 --seed 42 --no-plots `
  --output-dir output/clustering_review/multi_raw
```

For the other multi runs, change mode to `log1p` or `spearman-profile`;
the guided sensitivity also uses `--supervised-penalty --k 9`.
For the single runs, use `--dist-file
main_scripts/processed_neurons/251637/fnt_processed/ins/ins_dist.txt` and
`--type-file neuron_tables_new/251637_INS_HE_inferred.xlsx`; use raw, or
`spearman-profile --k 9`. Use a separate output directory for each setting.
`--joined-fnt` is inferred from `*_dist.txt`, or can be supplied explicitly.

Rebuild compact tables and figures from these saved six run directories with:

```powershell
& $taskPython -B notes/clustering_review_20261002/summarize.py
```

Metadata pins input/source hashes and Python/package versions. These outputs are
reproducible conditional on those persisted inputs. Before current-cohort
analysis, review/freeze membership and rebuild FNT; then assess projection
representation, recomputed profile representations, continuous alternatives,
reconstruction completeness, independent animal support and null models.
