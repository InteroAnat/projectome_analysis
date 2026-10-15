# Repository and method review

**Agent: Codex | Date: 2026-10-02 (Asia/Shanghai)**

## Scope and provenance

Reviewed `D:/projectome_analysis`: living notes, September 26 code review/plan v3,
older L/R reviews, staging and portal audits, saved workbooks, core Python region
analysis, SWC/FNT processing, clustering, R analysis paths and deposited Gou code.
Primary literature and relevant Zotero methods are documented in
[the research note](fmost_methods_research_20261002.md).

Local starting state was `fix/phase0-20260926` at
`78008cbaab71bf737dc7129c72e5fe284c3ba40f`. Created
`codex/fmost-methods-review-20261002` from that state, preserving pre-existing
project-note/journal edits, tracked bytecode changes and extensive untracked
research artifacts. No external messages, synchronization, journal rewriting,
commit, push, merge, dataset promotion or destructive cleanup was performed.

Live `git ls-remote --heads origin` confirmed remote refs `master` at
`f096e8165da18f84e91ea69d6e5cbfd58f248e75`,
`test/liu-ven-fmost-bridge` at `3f1a39493d32cc15badfab5c0c319e89e4a32dc6`,
and `cleaner_projectome_scripts` at `178a123be4a6734101e6159aef1adaad555f3873`.
The new review branch remains local. Remote refs are distinct from local working
files and do not establish current scientific acceptance.

## Persisted cohort reconciliation

| Workbook | n | L:R | Contributing SampleIDs | SHA256 |
|---|---|---|---|---|
| Canonical raw combined | 306 | 121:185 | 4 | `a4ef7d74bd3729ece755a9543c6cfd42c720e294ccd37ec6b03eb12332089df3` |
| Canonical harmonized | 353 | 155:198 | 6 | `dea7146bf74af7f605ab2a8ff8f75c46de3ba9d388ca38f78ba8af5de316b178` |
| Staging raw combined | 420 | 188:232 | 9 | `2d01bf9da07e4a6da46f4401d73b18b196e68de4e21cd5505e1ad7b4f1b547d1` |
| Staging harmonized | 420 | 188:232 | 9 | `29fb507ce89b25c2edaaaf07b3f65e33b9989fd096e5782d25e0789866109229` |

All four workbooks have internally matching unique UID sets/order across eight
projection sheets, no missing/duplicate sheet identities and no negative or
nonfinite projection cells. Cross-workbook membership differs materially.
Staging adds 67 and removes none versus harmonized canonical, but changes 11
existing IDD5 labels to IG. Larger n is not independent anatomical acceptance.

Canonical harmonized counts: 251637=260, 252385=47, 252527=23, 252718=13,
252383=5, 252384=5. Staging additionally includes 250432=8, 252714=16,
252790=38, while 252385 becomes 50 and 252527 becomes 25.

Staging regions L:R: IAL 33:147; IAPM 31:3; IDD5 33:30; IDM 52:33; IG 10:12;
G 25:7; IAPL 1:0; IDV 3:0. The pooled IDD5+IDM stratum is 148 (85:63), but 139
are from 251637 (77:62). The remaining nine come from only three samples, mostly
left-sided. A neuron-level balanced count does not solve animal/site confounding.

17 retained coordinate rescues carry a distant tag (15 padded, two strict).
27 staging mappings are tagged ambiguous. Curated 251637 remains 260, and Henry
layers cover 260; the other 160 staging rows lack those annotations. G 32 and IG 22
remain distinct; 251637 curated atlas-Ig somata retain their historical IDD5 label.

The September 28 saved portal refresh checked 46 samples, reported unchanged counts
and no changed-sample ID entries versus September 26. No live October 2 portal fetch
was performed. The old dataset manifest/overview still describes 252790 as empty
and 252714 as missing, contradicted by saved ingest/portal evidence. Preserve the
human acquisition tracker until animal/site identities and reconstruction yields
are reconciled; acquisition plans are not reconstructed-neuron denominators.

## Workflow map and unresolved defects

Current flow: curated reference/step1 SWCs → region projections → coordinate
recovery → combined workbook → harmonization/Henry annotation → FNT processing →
R laterality/gradients and morphology/projectome clustering → figure/flatmap QC.
The sources of accepted labels, algorithmic rescues and display transforms need
separate provenance throughout that flow.

| Priority | Evidence | Implication and required next action |
|---|---|---|
| Critical before rebuild | Harmonizer defaults to raw306 and replaces harmonized output353; its check only compares to input | Reconcile/freeze source membership and use separate paths. Do not directly rerun defaults. |
| Critical before primary clustering/Mantel | Global FNT306; single legacy FNT212; current harmonized353/staging420 | Rebuild on exact freeze and assert ID equality. R currently intersects/truncates data, so a successful render is insufficient. |
| High: anatomical membership | Rescue accepts `_distant` as keepers; harmonization contains ambiguous assignments | Review local anatomy/registration and preserve exclusion/reviewer reasons. Coordinate proximity is a candidate screen. |
| High: measurement semantics | `neuron_analysis.py`: voxel edge lengths, rounding, whole edge assigned to parent voxel, terminal-region filtering, no axon-only filtering | Specify compartments, units, outside-atlas accounting, edge allocation and native/template distinction before changing/recomputing measurements. |
| High: endpoint semantics | `population.py`: `Terminal_Count` counts unique region entries | Describe endpoint-containing regions, not individual traced terminals or synapses. |
| High: normalized strength | R v2 normalizes log10(length+1), not raw length | Correct “axonal budget” wording or recompute raw-length shares; summing logs also differs from logging summed lengths. |
| High: sampling/inference | Neuron-level OLS/Wilcoxon/Fisher and unrestricted Mantel permutations; dominant251637 | Design supported animal/injection-aware inference, restricted permutation/sensitivity and appropriate scope of claims. |
| High: hemisphere provenance | Laterality fallback threshold derives from observed left/right coordinate medians | Pin trusted atlas/mask/transform evidence; do not treat inferred physical side as independent validation. |
| Medium: network/cache provenance | IONData retains unbounded retries in legacy routes; FNT resumes from caches and joins folder contents | Future bounded client and exact manifest-driven cache/join selection. No network FNT rebuild executed here. |
| Medium: divergent analyses | R clustering retains old penalty/Ward/fixedk9; monkey clustering demo uses scalar/spatial features rather than FNT scores | Identify intended primary entrypoint; compare representations explicitly rather than treating scripts as equivalent replications. |

## Implemented code improvements

`main_scripts/clustering_validation.py`: complete triangular/directed pair
validation, explicit directional max/mean policy, actual joined-marker identities,
raw self-score provenance, matrix validation, permitted general-dissimilarity
linkages and seeded k/separation/stability/group-removal diagnostics.

`main_scripts/fnt_dist_clustering.py`: strict annotation coverage/uniqueness,
explicit legacy stem/composite UID mapping, raw/average/unguided defaults,
raw/log1p/profile modes, opt-in type penalty, bounded requested/realized cuts,
no hidden fixed30, noninteractive CLI and separate outputs. Exported input/source
hashes, versions, all candidate assignments, Jaccard coverage and interpretation
limits make the calculation reviewable. The default silhouette suggestion remains
exploratory; it may isolate outliers and is not a biological acceptance rule.

`main_scripts/swc_validation.py` and `neuro_tracer.py`: reject duplicate/fractional
IDs, bad parents, missing roots, disconnected cycles, nonfinite coordinates/radii
and negative radii before traversal/cache writes. Preserve valid order/custom
types/dendrite-to-axon transitions, replace tracer state only after success,
retain prior graph on failure, fix explicit output directory handling.
`region_analysis/population.py` now reports the specific load failure while
retaining the existing skip contract. Partial sample exclusions still need review.

README now points to the executable CLI instead of a nonexistent run_clustering
API. Existing scientific labels/projection calculations, R hypothesis tests and
prior outputs were not regenerated or silently reinterpreted.

## Verification and limits

- **33 focused unittest tests passed** in the existing projectome environment:
  12 distance/stability helper, eight public clustering integration,13 SWC.
- Six real public CLI runs completed on historical212/306 inputs, each k2..20,
  100 subsets/fraction.8/seed42. Persisted manifests, assignments and diagnostics
  were read back; [results and reproduction](clustering_review_20261002/README.md).
- Real-data evidence exposes very small high-dissimilarity groups, representation sensitivity and
  dominant-animal dependence. It does not justify accepting k9/k30 or newly
  suggested cuts as biological types.
- Independent code review found a legacy marker/stem compatibility issue; fixed
  and regression tested before final real runs. SWC and clustering changes were
  reviewed independently of their implementers.
- Read-only compatibility audit:358 local NMT/VEN SWCs,4,882,379 nodes,0 failures.
  This does not establish tracing completeness, compartment labels or registration.
- AST/whitespace checks and persisted-input hash checks complete the local
  software verification. No full FNT/registration reconstruction, full R analysis,
  biological null benchmark, independent-animal prediction or anatomical review
  was performed.

## Priorities after this branch

1. Resolve anatomical cohort decisions and data identity; freeze membership and
   image-reviewed exclusions with atlas/transform/tracing provenance.
2. Rebuild exact-cohort FNT from a manifest, preserving original laterality and
   recording decimation/reflection/cache choices. Reject stale/missing/extra names.
3. Inspect the actual neurons driving outlier/coarse profile groups. Compare raw
   morphology, recomputed distance-profile and raw-length projection definitions;
   specify whether clusters describe morphology, projection targets or joint data.
4. Evaluate clusterwise/global stability, metric/linkage sensitivity, continuous
   gradients, animal/injection structure and sensible nulls. Predeclare which
   hypotheses use labels and keep guided-enrichment conclusions separate.
5. Re-run accepted R endpoints with supported animal-level scope, dispersion and
   blocked-permutation reasoning. Molecular/VEN/causal interpretation requires
  independent evidence rather than clustering agreement.

## Later evidence and implementation addendum — 2026-10-03

The review above preserves the initial October 2 cutoff. Its statement that no
live portal fetch had occurred applies to that cutoff. Later October 2 work
queried all nine requested monkeys (27 HTTP 200 responses, 1,922 portal neuron
records); portal counts remain source inventory, not accepted cohort membership.
See the [server inventory](bulk_visual_review_20261002/server_inventory.md),
[visual methods](bulk_visual_review_20261002/visual_methods.md), and
[independent acceptance review](bulk_visual_review_20261002/independent_visual_acceptance_20261003.md).

The all-nine soma batch terminated successfully on October 3. Its 746 selected
identities yielded 697 complete crops, 27 crops with missing neighbors, 20
unavailable centers, and two unavailable raw SWCs. These are acquisition outcomes,
not anatomical decisions. Only four monkeys have copied 5-micron overview stacks;
the other five remain in scope and use their own raw SWC roots and native cube
sources. The bulk verification decoded 5,121 PNGs, reported no problems, and
confirmed protected cohort hashes unchanged. The separate terminal CSV/XLSX
readback checked all 160 staging origins against their source rows. See
[terminal readback](bulk_visual_review_20261002/terminal_soma_and_xlsx_readback_20261003.json)
and the [completion audit](bulk_visual_review_20261002/completion_audit_20261003.md).

Code improvements now validate TIFF layout and size before cache admission,
preserve exact source-plane positions through gaps, write nominal micron geometry
with acquisition provenance, preserve human correction fields, and expose missing
tiles through a separate coverage mask. Regional derivation uses a globally
aligned point-decimation grid, requires acquired root coverage, and enforces cache,
request, disk-space, and runtime bounds. Its resume logic distinguishes observed
404 gaps from transient or resource-limited attempts. A dated, hashed source
snapshot passed 99 unit tests; this supersedes the earlier 33-test count only for
the expanded code scope. See [driver validation](bulk_visual_review_20261002/regional_driver_validation_20261003.json).

Broader regional production is still running for 252790, with insula candidates
first. The remaining four monkeys without overview copies also require regional
context or explicit source-failure outcomes. CH1 crops and successful software
checks do not establish PI-based cytoarchitecture, fine parcels, layer identity,
absolute calibration, laterality, or tracing completeness. Clustering remains
exploratory: exact accepted-cohort FNT rebuilding, independent anatomical review,
and animal/injection-aware validation are still prerequisites for interpretation.
