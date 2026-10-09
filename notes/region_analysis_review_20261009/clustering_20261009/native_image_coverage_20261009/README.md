# Native-image evidence coverage of the saved clusters

Agent: Codex | Date: 2026-10-09 (Asia/Shanghai)

This audit joins the unchanged selected-462 cluster ledger to the source-bound native-image availability census. It adds a reconstruction-evidence diagnostic; it does not refit clusters, introduce another cohort, classify terminal arbors, or use cache availability as a tracing-quality score. Exact composite identities, animal IDs, source paths, hashes and official ARM source names must agree before joining.

**The dominant animal and the Henry-only sensitivity have no native SWCs in this designated cache.** All 261 selected neurons from animal 936, including all 260 Henry coarse visual INS neurons, lack a native pair here. The remaining 201 selected neurons across seven animals have native pairs. The 261 missing native measurements remain NA; availability elsewhere is unknown. The existing NMT SWCs and Henry's independent wide-field evidence remain preserved and usable within their original scope.

| Animal | Selected neurons | Native pairs found here | Neurons with a cached end cube | Cached end positions / original ends in available native graphs |
|---|---:|---:|---:|---:|
| 331 | 55 | 55 | 50 | 1,402 / 8,551 |
| 605 | 21 | 21 | 20 | 1,206 / 1,766 |
| 631 | 16 | 16 | 16 | 466 / 11,048 |
| 797 | 52 | 52 | 47 | 5,100 / 18,382 |
| 900 | 14 | 14 | 12 | 206 / 1,515 |
| 936 | 261 | 0 | 0 | NA / NA |
| 945 | 25 | 25 | 25 | 542 / 4,024 |
| 948 | 18 | 18 | 14 | 287 / 3,288 |

Across available native graphs, 9,209 of 48,574 original axon-end positions have a cached cube (18.96%). This pooled fraction weights ends unequally and is not animal-level innervation prevalence. Per-animal fractions range from 4.22% to 68.29%, conditional on finding that animal's native graphs. A cached cube is evidence available for review, not a reviewed ending. The five-case pilot and the [expanded native review](../../terminal_review_expansion_20261009/README.md) record their actual inspected cases separately; the availability column intentionally does not encode review decisions.

The saved primary axon Hellinger k=2 clusters contain 229 and 199 neurons, with native pairs for 91 and 109 respectively. The endpoint k=2 cut contains a 424-neuron group and a singleton; the singleton has a native SWC but no cached cube at its one original axon end. These coverage differences cannot validate either partition biologically. In particular, native-image checks of the smaller seven-animal subset cannot establish reconstruction completeness for animal 936 or independently validate the Henry-only sensitivity. Existing hemisphere, animal, profile-stability and missing-profile findings remain in the [clustering report](../README.md).

The methods follow the [local Gao/Gou/Liu review](../../literature_method_update_20261009/gao_local_methods.md): source-image tracing/completion checks are distinct from graph topology and template-space length. No leaf is promoted to a bouton, synapse or complete terminal field. No p-value or extra independent animal is inferred from neuron counts.

The [compact 462-row ledger](all462_native_image_coverage.csv) preserves missing native counts, source evidence and unchanged k=2 assignments. The [aggregate CSV](native_image_coverage_by_animal_evidence_cluster.csv) reports animal, Henry-evidence and cluster summaries; these are descriptive strata, not newly fitted cohorts. [Provenance](coverage_audit_provenance.json) records exact inputs, code, versions, parameters and output hashes. Four focused regression checks cover unknown-versus-zero measurements, reused bare IDs, changed graph hashes and prohibited zero filling.

Reproduce from the project root into a fresh directory:

```powershell
python -X utf8 -B notes/region_analysis_review_20261009/clustering_20261009/test_native_image_coverage.py
python -X utf8 -B notes/region_analysis_review_20261009/clustering_20261009/audit_native_image_coverage.py --output <fresh-output-directory>
```
