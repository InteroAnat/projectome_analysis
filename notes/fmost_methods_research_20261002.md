# fMOST methods and clustering research for the macaque insula projectome

Agent: Codex  
Date: 2026-10-02 (Asia/Shanghai)  
Scope: primary-source literature review, repository method audit, and scientific acceptance criteria. This is an agent-authored research note; it does not change attribution of earlier notes.

## Evidence and current status

This review connects acquisition, reconstruction, spatial annotation, projection quantification and clustering. Publications and official documentation were checked online and relevant indexed Zotero PDFs were read on the date above. Original script findings describe historical base commit 78008cb; completed branch changes and checks are identified separately. No accepted cluster count or anatomical acceptance is asserted here.

The project concerns multi-monkey macaque insular single-neuron projectomes and an IONData portal workflow. The inspected loader convention declares SWC XYZ in NMT physical micrometres, then divides by reso=250 to obtain voxel indices; Soma_NII fields store indices. That declaration needs supporting export and registration provenance. The SWC suffix establishes a file structure, not the actual coordinate frame. Likewise, midline=128 is a project convention requiring validation against the exact atlas, orientation and indexing. The official NMT distribution supplies multiple fields of view/resolutions and hemisphere masks [6].

Historical canonical membership, newer staging candidates, portal records and anatomically accepted neurons must retain separate statuses. A successful coordinate rescue or an in-bounds soma is evidence of spatial compatibility, not sufficient evidence for cohort promotion or correct local anatomy.

## The immediate clustering problem

At historical base 78008cb, main_scripts/fnt_dist_clustering.py defaulted to Spearman correlations between distance profiles, optionally used log1p distances instead, enabled a type penalty of 1.5 and called SciPy Ward. It calculated C-index across k=2..65 but used fixed k=30. R_analysis/fnt_dist_clustering.r instead fixed k_final=9; main_scripts/fnt_dist_clustering.r permitted an interactive override. These were divergent analysis choices, not independent confirmations of a common taxonomy. This paragraph does not describe the new branch defaults.

Spearman correlation of distance profiles asks which neurons rank their relationships to the current cohort similarly. It does not use the original pairwise morphology dissimilarity directly. Profiles include cohort-specific columns and self-distances, so deleting neurons can change the feature representation. log1p also changes distance geometry. Type penalties incorporate existing biological labels into the grouping: concordance with those labels afterwards cannot independently validate their recovery.

SciPy's official linkage documentation states that Ward, centroid and median require Euclidean distances and that users supplying precomputed distances must establish that condition [23]. Symmetry, nonnegativity and a zero diagonal alone do not establish Euclidean geometry. Rank-profile dissimilarities, nonlinear transformations and category penalties cannot be assumed compatible with Ward. A square distance matrix passed as observations would introduce a different distance-of-profiles analysis again. Input shape and metric semantics need explicit validation.

Gou's STAR Methods, directly checked in Zotero, first assigns projection types manually, adds a constant between different types, applies ward_presquared to FNT-dist morphological dissimilarities and cuts 32 groups [2]. The deposited references/analysis-code_gou_etal_2025/monkeyrec/2-analysis.jl:2284-2317 specifies penalty 9000. This is a guided classification precedent, not an independent optimum. Julia ward_presquared and SciPy ward have different input contracts; numerical equivalence is not established by their names.

The new branch implements average linkage on raw distances, no default type penalty, candidate-k diagnostics, subsample adjusted Rand index (ARI), clusterwise Jaccard overlap and group-held-out comparisons. Average linkage supports general dissimilarities [23]. These clearer defaults do not establish raw FNT distances as the best biological representation or select an accepted k. The coordinating agent reports 33 passing focused tests (12 helper, 8 public clustering, 13 SWC) and six completed real-input CLI diagnostic runs; these are software and exploratory evidence.

Gao's mouse PFC study uses dynamic-programming FNT-dist alignment, Ward clustering and a local C-index minimum at 64 [24]. Qiu's mouse hippocampal study explains that shared routes and soma-dependent geometry impaired distance-based classification; it instead built trajectory templates and classified arborization patterns [25]. These precedents favor explicit representation comparisons. Neither published k nor shared clustering software makes insular groups homologous to PFC or hippocampal types.

## What a credible clustering assessment must establish

First validate identity and alignment: one unambiguous neuron ID per row/column; identical row/column order; valid finite distances; explicit handling of triangular exports; measured asymmetry and documented symmetrization; zero diagonal; no silent replacement of missing values by similarity. Persist original IDs, matrix/input hashes, cohort filters, metric, transform, linkage, label use, requested and realized k, random seed and parameters with every result.

Separate unsupervised discovery from guided reproduction. Use pre-existing neuron types as annotations for the primary unguided analysis. Run penalties only as explicitly named sensitivity/reproduction analyses. Compare raw morphology distances with distance-profile and projection-feature alternatives, since each answers a different biological question. Fixed k=9, 30 and the PFC precedent 32 can be sensitivity cuts without becoming the expected answer.

Consensus clustering summarizes coassignment across resamples [19]. Global ARI can obscure unstable small clusters, so report clusterwise best-match Jaccard, size distributions, dissolution/merger patterns and per-neuron assignment uncertainty [20]. Ties may produce fewer than the requested maximum clusters; report realized groups. Singletons and tiny groups require explicit undefined-score policies, not fabricated stability. Preserve the complete diagnostic curves and plausible k range instead of elevating one optimum to a cell-type count.

A submatrix of a fixed full-cohort distance matrix assesses conditional sensitivity to removal. It does not repeat raw reconstruction, registration, feature learning or distance estimation. For Spearman profiles, recomputing correlations after subsetting is a different perturbation from subsetting already computed correlations; state which operation was performed. Neither is an independent new-animal test.

Group-held-out reclustering of remaining monkeys followed by comparison with the full fit tests robustness to group composition. It is not prediction strength unless training-defined assignments are applied to held-out observations and their pairwise memberships are compared with an independently fitted held-out partition [21]. Data-dependent normalization, feature selection and tuning must also be fitted on training groups. Related neurons from one monkey or injection cannot be treated as independent animal replicates.

Stability is a necessary diagnostic for a useful partition, not sufficient evidence for distinct biological types [20,22]. Continuous topographic gradients, algorithmic rigidity and acquisition batches can yield stable cuts. Ask whether groups add interpretable information beyond soma position, animal, injection, axon completeness and total length. Use per-monkey summaries and leave-one-monkey-out sensitivity; test continuous representations and suitable null models before claiming discreteness. A classifier's agreement with labels derived from the same data is not external validation.

## Acquisition and reconstruction implications

MOST and fMOST made continuous large-volume neurite imaging feasible [8,9], but platforms, embedding chemistry, fluorescent labels, channels and sampling differ between studies. Record actual specimen preparation, channel purpose, anisotropic acquisition resolution, sectioning intervals, missing blocks, preprocessing and reconstruction software. Acquisition resolution, overview downsampling and final NMT resolution are different quantities [3,5].

Graph validation should detect duplicate or noninteger IDs, missing parents, self-links, cycles and root reachability before measuring edges. Parent lookup must use identifiers, not row position. Preserve originals and produce explicit issue reports. Do not silently repair questionable joins or infer completeness from a single connected tree. Missing branches, wrong-neuron connections and tangled crossings require raw-image assessment and reviewer provenance [4,5,11].

Type transitions need biological interpretation. VEN axons may originate from basal dendrites [1]. A dendrite-to-axon connection is therefore not automatically a tracing error. Likewise, radius placeholders can invalidate surface/volume estimates without invalidating coordinate-based skeleton length. Official consumers differ in ordering and standardization conventions; local validation must distinguish representation constraints from biological acceptance.

## Registration, hemispheres and projection measures

Persist native reconstruction, NMT reconstruction and display-normalized copies separately. Each needs specimen ID, source/target atlas identity, units, axis order, affine/index convention, transform direction/order and registration software/version. Validate transformed landmarks and local insular boundaries alongside global overlap. Neither image similarity nor an atlas label proves a neuron's soma lies within a small cytoarchitectonic subregion [10,14].

Gou uses landmark-initialized affine plus B-spline nonrigid registration and U-Net conversion of fluorescence background to MRI-like images, including simultaneous registration/stitching of multiple blocks [2]. A simple scale-to-NMT assumption therefore needs its own transformation lineage. Fine arbor patches were analyzed within specimens because cross-specimen overlap is compromised by biological variation and registration error.

Use verified hemisphere masks or a validated midsagittal plane where available. Retain absolute anatomical laterality, soma-side-relative ipsilateral/contralateral labels and visualization reflections as distinct fields. Official NMT variants differ; shape alone cannot establish physical laterality [6,7].

Projection extent should derive from actual parent-child edges with explicit compartment rules and units. Node counts change with tracing density and resampling; they should remain sampling descriptors. Report regional axon length, branch/arbor distribution and endpoints separately. A fiber passing through an ROI and a reconstruction endpoint at weak signal do not establish a synaptic target. Choose target thresholds using this dataset's detection limits and sensitivity analyses, not automatic transfer of mouse thresholds [12].

Published measures differ: Gou's Figure 2A heatmap uses logarithmic regional axon length; its co-innervation analysis uses binary arbor presence [2]. Qiu requires a primary target to have at least 1 mm arbor length, one terminal and one branch point [25]. Gehrlach's mouse insula map measures thresholded fluorescent pixels, excludes obvious passing bundles and normalizes regional output [29]. Length, arbor presence, fluorescence density and node count are different observables; comparison requires matching their definitions.

Nonrigid normalization changes distances [13]. A metric computed from declared NMT-micrometre SWCs should be called NMT template-space axon length. Native physical length requires calibrated specimen-space coordinates. Region assignment can use atlas mappings while edge extent remains native, if matching correspondences and transforms are available. Be explicit about crossing-edge allocation, outside-atlas portions and unknown compartments; do not discard these silently.

## Boundaries on scientific interpretation

Gou's macaque PFC projectomes provide a close comparator for sparsity, collaborative reconstruction, NMT integration and refined terminal arborization [2]. Their PFC subtype definitions and fixed cut are hypotheses for insula comparisons, not accepted insular classes. Mouse studies provide transferable QC and analysis principles; mouse cell markers, landmarks, thresholds, sampling proportions and trained registration/classification models require macaque-specific validation [12,14-18].

Liu's 2026 AIC atlas establishes multimodal VEN evidence and dendritic axon origins, but explicitly leaves VEN-L/VEN-S long-range projections unresolved [1]. Whole-brain axon clusters alone cannot establish those identities or their physiological functions. Linking this projectome to molecular/local morphology evidence is an important future test rather than a completed result.

Primate insular anatomy also supplies independent hypotheses: Chikama relates striatal projection territories to insular cytoarchitecture [28]; Krockenberger compares tracer patches with adjacent architectonic sections and supports finer dysgranular stripes [27]. Evaluate soma/subregion and projection motifs against such evidence. A coarse MRI atlas label or a stable dendrogram cut cannot directly validate a fine stripe.

Regional occupancy, axon arbors and potential contacts support mesoscopic projection descriptions. They are not verified synapse counts or functional connection strengths [15,16]. Cohort projection fractions describe reconstructed sampled neurons; missing labels, partial tracing and unequal injection coverage limit population and absence claims. Report animal/injection denominators and completeness without equating a larger neuron count with independent replication.

## Acceptance gates for the branch

1. **Software:** malformed matrices fail explicitly; ID alignment and metric semantics are tested; tie/constant/small-cluster cases are handled; seeded diagnostics reproduce; manifests match saved outputs. Passing these tests verifies implementation.
2. **Input/provenance:** accepted cohort and matrix IDs match; source SWCs, exports, transformation lineage, label use and omissions are traceable.
3. **Anatomy/tracing:** local soma/subregion and hemisphere checks use trusted evidence; unresolved endpoints/joins and tissue loss remain visible; reviewer decisions are recorded.
4. **Analysis:** inspect multiple k values, clusterwise/global stability, metric/linkage sensitivity, group imbalance, batch dependence and continuous alternatives. Re-estimate upstream representations when making broader pipeline-robustness claims.
5. **Biological:** replicate useful motifs across animals and independent measurements. Strong cell-type or VEN claims require cross-modality evidence; held-out predictive claims require a specified train/test assignment protocol.

Implementation and focused software checks are complete as reported above. The real-input audit found strong representation dependence: the historical multi-monkey matrix (306 entries) at k=2 gave raw-distance sizes 3/303, silhouette 0.749 and subsample ARI 0.512; Spearman-profile sizes 147/159, silhouette 0.742 and ARI 0.959; log1p sizes 1/305. All used average linkage, k=2..20, 100 subsets of 80%, seed 42. These are exploratory diagnostics, not a selected taxonomy. No scientific gate is marked passed by this review.

## Zotero full-text verification and provenance

Relevant library records were searched read-only, then their PDF attachments' indexed text was inspected. Indexed-page coverage below is the local API's report; it does not guarantee flawless equations/OCR or imply every page was examined. No private notes or unrelated records were copied. Publication identities were cross-checked with DOI/publisher records. Item links open Zotero's local library.

| Paper | Zotero item | PDF attachment | Indexed coverage | Verified methodological scope |
|---|---|---|---|---|
| Gou 2025 [2] | [JWB4MYBC](zotero://select/library/items/JWB4MYBC) | 8BNNG4KV | 42/42 | STAR Methods e3-e6: proofreading, block joins, registration, guided clustering; Figure 2A length heatmap |
| Liu 2026 [1] | [DFH77L3E](zotero://select/library/items/DFH77L3E) | JUAS5RLT | 47/47 | Morphoelectric/VEN evidence and Discussion's unresolved projection targets |
| Gao 2022 [24] | [L5TALCAJ](zotero://select/library/items/L5TALCAJ) | IQ5H7F89 | 39/39 | Clustering of single-neuron projectome; C-index and regional length measures |
| Qiu 2024 [25] | [BNB9KHB7](zotero://select/library/items/BNB9KHB7) | ZMGEI8V8 | 38/38 | Supplementary Methods pp. 3-5: tracing QC, registration, template representation and target definition |
| Yan 2022 [26] | [RXUGG8AT](zotero://select/library/items/RXUGG8AT) | 7974HUDI | 28/28 | STP imaging, segmentation/tractography comparison and limitations |
| Krockenberger 2023 [27] | [2NZNENNT](zotero://select/library/items/2NZNENNT) | DZAU4B4N | 25/25 | Adjacent-section architectural comparison and indirect tracer analysis |
| Chikama 1997 [28] | [UA24ZU9V](zotero://select/library/items/UA24ZU9V) | VPPUUD5U | 20/20 | Cytoarchitecture and striatal projection territory associations |
| Gehrlach 2020 [29] | [YGWU4TT3](zotero://select/library/items/YGWU4TT3) | M6UIMT5F | 25/25 | Output pixel quantification, bundle exclusion and normalization |

The focused bibliography notes/fmost_review_20261002.bib contains these eight selected publications. Private notes, tags, abstracts and attachment filesystem paths were omitted. Liu's volume/pages were refreshed from the publisher because the local record retained early-online pagination; the Zotero library was not modified. Duplicate Gao records were not counted as independent sources. Gou's alternate PDF UDDF94HW has no indexed full text; the verified content came from 8BNNG4KV. A title search for Gapr found no match, so its evidence remains the primary online paper [4].

## Annotated primary bibliography

Summaries are deliberately short; code recommendations above are methodological inferences. DOI links identify publications, with official/author access links where useful.

1. **Liu, R.-F., Huang, M., Shen, Y. et al. (2026).** *An atlas of primate insular cortex reveals a signal-processing strategy in von Economo neurons*. Nature Cell Biology 28, 1742-1757. [DOI 10.1038/s41556-026-02009-4](https://www.nature.com/articles/s41556-026-02009-4). Direct AIC multimodal evidence; actual VEN subtype long-range targets remain unknown.
2. **Gou, L., Wang, Y., Gao, L. et al. (2025).** *Single-neuron projectomes of macaque prefrontal cortex reveal refined axon targeting and arborization*. Cell 188, 3806-3822.e24. [DOI 10.1016/j.cell.2025.06.005](https://www.sciencedirect.com/science/article/pii/S0092867425006397). 2,231 PFC projectomes, seven monkeys, 32 guided projectome subtypes; deposited code must inform exact reproduction.
3. **Zhou, C., Yang, X., Wu, S. et al. (2022).** *Continuous subcellular resolution three-dimensional imaging on intact macaque brain*. Science Bulletin 67, 85-96. [DOI 10.1016/j.scib.2021.08.003](https://www.sciencedirect.com/science/article/pii/S2095927321005533). Primate PNAGA preparation and continuous axon-resolution imaging; different acquisition modes must remain distinguishable.
4. **Gou, L., Wang, Y., Gao, L. et al. (2024).** *Gapr for large-scale collaborative single-neuron reconstruction*. Nature Methods 21, 1926-1935. [DOI 10.1038/s41592-024-02345-z](https://www.nature.com/articles/s41592-024-02345-z). Collaborative proofreading and deposited reconstruction histories distinguish exported skeletons from review evidence.
5. **Ren, B., Shi, X., Zhao, B., Xu, C. and Wang, X. (2025).** *Reconstruction of Single-Neuron Projectomes in Mice*. Bio-protocol 15(9), e5300. [DOI 10.21769/BioProtoc.5300](https://bio-protocol.org/en/bpdetail?id=5300&type=0). Primary protocol covers image suitability, independent tracing, conflict QC and landmark-based transformation.
6. **Jung, B., Taylor, P. A., Seidlitz, J. et al. (2021).** *A comprehensive macaque fMRI pipeline and hierarchical atlas*. NeuroImage 235, 117997. [DOI 10.1016/j.neuroimage.2021.117997](https://pubmed.ncbi.nlm.nih.gov/33789138/). [Official NMT documentation](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/macaque_tempatl/template_nmtv2.html) confirms variants/resolutions and hemisphere masks.
7. **Seidlitz, J., Sponheim, C., Glen, D. et al. (2018).** *A population MRI brain template and analysis tools for the macaque*. NeuroImage 170, 121-131. [DOI 10.1016/j.neuroimage.2017.04.063](https://pmc.ncbi.nlm.nih.gov/articles/PMC5660669/). Population template provenance; an average reference is not individual anatomical ground truth.
8. **Li, A., Gong, H., Zhang, B. et al. (2010).** *Micro-optical sectioning tomography to obtain a high-resolution atlas of the mouse brain*. Science 330, 1404-1408. [DOI 10.1126/science.1191776](https://pubmed.ncbi.nlm.nih.gov/21051596/). Foundational Golgi-stained MOST demonstration; technology history rather than validation of these macaque reconstructions.
9. **Gong, H., Zeng, S., Yan, C. et al. (2013).** *Continuously tracing brain-wide long-distance axonal projections in mice at a one-micron voxel resolution*. NeuroImage 74, 87-98. [DOI 10.1016/j.neuroimage.2013.02.005](https://pubmed.ncbi.nlm.nih.gov/23416252/). Early fluorescence pathway tracing; acquisition continuity does not guarantee per-neuron reconstruction completeness.
10. **Gong, H., Xu, D., Yuan, J. et al. (2016).** *High-throughput dual-colour precision imaging for brain-wide connectome with cytoarchitectonic landmarks at the cellular level*. Nature Communications 7, 12142. [DOI 10.1038/ncomms12142](https://www.nature.com/articles/ncomms12142). BPS/WVT same-specimen cytoarchitectonic landmarks support precise localization.
11. **Winnubst, J., Bas, E., Ferreira, T. A. et al. (2019).** *Reconstruction of 1,000 Projection Neurons Reveals New Cell Types and Organization of Long-Range Connectivity in the Mouse Brain*. Cell 179, 268-281.e13. [DOI 10.1016/j.cell.2019.07.042](https://pmc.ncbi.nlm.nih.gov/articles/PMC6754285/). MouseLight comparator quantifies annotator accuracy using axonal length and consensus.
12. **Peng, H., Xie, P., Liu, L. et al. (2021).** *Morphological diversity of single neurons in molecularly defined cell types*. Nature 598, 174-181. [DOI 10.1038/s41586-021-03941-1](https://www.nature.com/articles/s41586-021-03941-1). 1,741 morphologies; axon-length projection measures and >1-mm mouse target threshold are methodological precedent.
13. **Wang, Q., Ding, S.-L., Li, Y. et al. (2020).** *The Allen Mouse Brain Common Coordinate Framework: A 3D Reference Atlas*. Cell 181, 936-953.e20. [DOI 10.1016/j.cell.2020.04.007](https://pmc.ncbi.nlm.nih.gov/articles/PMC8152789/). Explicitly cautions that nonrigid normalization changes distances; physical distance/volume measurements belong in specimen space.
14. **Qu, L., Li, Y., Xie, P. et al. (2022).** *Cross-modal coherent registration of whole mouse brains*. Nature Methods 19, 111-118. [DOI 10.1038/s41592-021-01334-w](https://www.nature.com/articles/s41592-021-01334-w). mBrainAligner addresses cross-modal artifacts/anatomy; mouse-trained tools need primate validation.
15. **Liu, Y., Jiang, S., Li, Y. et al. (2024).** *Neuronal diversity and stereotypy at multiple scales through whole brain morphometry*. Nature Communications. [DOI 10.1038/s41467-024-54745-6](https://www.nature.com/articles/s41467-024-54745-6). Multiscale morphometry and potential synaptic varicosities support richer descriptors without establishing synapses.
16. **Liu, L., Yun, Z., Manubens-Gil, L. et al. (2025).** *Connectivity of single neurons classifies cell subtypes in mouse brains*. Nature Methods 22, 861-873. [DOI 10.1038/s41592-025-02621-6](https://www.nature.com/articles/s41592-025-02621-6). Potential axon-dendrite connectivity; completeness, registration precision and uneven sampling are acknowledged limitations.
17. **Tustison, N. J., Chen, M., Kronman, F. N. et al. (2025).** *The ANTsX ecosystem for mapping the mouse brain*. Nature Communications 16, 11548. [DOI 10.1038/s41467-025-66741-5](https://www.nature.com/articles/s41467-025-66741-5). Open modular registration workflows offer benchmarks, not an automatically adopted macaque replacement.
18. **Sorensen, S. A., Gouwens, N. W., Wang, Y. et al. (2026).** *Connecting single-cell transcriptomes to projectomes in the mouse visual cortex*. Nature. [DOI 10.1038/s41586-026-10424-8](https://www.nature.com/articles/s41586-026-10424-8). Current integrated Patch-seq/whole-morphology precedent; cross-species cell assignments require separate evidence.
19. **Monti, S., Tamayo, P., Mesirov, J. and Golub, T. (2003).** *Consensus Clustering: A Resampling-Based Method for Class Discovery and Visualization of Gene Expression Microarray Data*. Machine Learning 52, 91-118. [DOI 10.1023/A:1023949509487](https://link.springer.com/article/10.1023/A:1023949509487). Resampled coassignment visualization/assessment motivates consensus diagnostics without making consensus a biological proof.
20. **Hennig, C. (2007).** *Cluster-wise assessment of cluster stability*. Computational Statistics & Data Analysis 52, 258-271. [DOI 10.1016/j.csda.2006.11.025](https://www.sciencedirect.com/science/article/pii/S0167947306004622). Best-match Jaccard distinguishes unstable clusters; stable results can also reflect algorithmic inflexibility.
21. **Tibshirani, R. and Walther, G. (2005).** *Cluster Validation by Prediction Strength*. Journal of Computational and Graphical Statistics 14, 511-528. [DOI 10.1198/106186005X59243](https://doi.org/10.1198/106186005X59243); [author manuscript](https://gwalther.su.domains/predictionstrength.pdf). Training-to-test pairwise membership prediction differs from deleting groups and reclustering the remainder.
22. **von Luxburg, U. (2010).** *Clustering Stability: An Overview*. Foundations and Trends in Machine Learning 2(3), 235-274. [DOI 10.1561/2200000008](https://doi.org/10.1561/2200000008); [author preprint](https://arxiv.org/abs/1007.1075). Reviews theoretical limits of stability-based selection; journal volume carries a 2009 label, while author/copyright publication records use 2010.
23. **SciPy official documentation.** [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html). Ward's Euclidean restriction and average-linkage formula are explicit; document the installed SciPy version for reproducibility.
24. **Gao, L., Liu, S., Gou, L. et al. (2022).** *Single-neuron projectome of mouse prefrontal cortex*. Nature Neuroscience 25, 515-529. [DOI 10.1038/s41593-022-01041-5](https://www.nature.com/articles/s41593-022-01041-5). FNT-dist dynamic-programming morphology alignment; 64 subtypes at a local C-index minimum, not an insula preset.
25. **Qiu, S., Hu, Y., Huang, Y. et al. (2024).** *Whole-brain spatial organization of hippocampal single-neuron projectomes*. Science 383, eadj9198. [DOI 10.1126/science.adj9198](https://www.science.org/doi/10.1126/science.adj9198). Morphology-distance limitations motivated trajectory templates and arborization-pattern representation; mouse registration/QC and thresholds are context-specific.
26. **Yan, M., Yu, W., Lv, Q. et al. (2022).** *Mapping brain-wide excitatory projectome of primate prefrontal cortex at submicron resolution and comparison with diffusion tractography*. eLife 11, e72534. [DOI 10.7554/eLife.72534](https://elifesciences.org/articles/72534). Macaque viral/STP bulk projection comparator; differences from tractography caution against treating modalities as interchangeable connectivity evidence.
27. **Krockenberger, M. S., Saleh-Mattesich, T. O. and Evrard, H. C. (2023).** *Cytoarchitectonic and connection stripes in the dysgranular insular cortex in the macaque monkey*. Journal of Comparative Neurology 531, 2019-2043. [DOI 10.1002/cne.25571](https://onlinelibrary.wiley.com/doi/10.1002/cne.25571). Independent architectonic/tracer correspondence is a more appropriate test of fine parcels than cluster stability alone.
28. **Chikama, M., McFarland, N. R., Amaral, D. G. and Haber, S. N. (1997).** *Insular Cortical Projections to Functional Regions of the Striatum Correlate with Cortical Cytoarchitectonic Organization in the Primate*. Journal of Neuroscience 17, 9686-9705. [DOI 10.1523/JNEUROSCI.17-24-09686.1997](https://www.jneurosci.org/content/17/24/9686). Primate anterograde/retrograde tracing supports territory-dependent insular projections; this is regional evidence, not a single-neuron taxonomy.
29. **Gehrlach, D. A., Weiand, C., Gaitanos, T. N. et al. (2020).** *A whole-brain connectivity map of mouse insular cortex*. eLife 9, e55585. [DOI 10.7554/eLife.55585](https://elifesciences.org/articles/55585). Population input/output mapping provides an insula comparator whose fluorescence measures and species boundaries remain explicit.

Additional official format resources: [SWC specification](https://swc-specification.readthedocs.io/en/latest/swc.html), [AllenSDK validator source](https://allensdk.readthedocs.io/en/latest/_modules/allensdk/morphology/validate_swc.html), [NeuroMorpho standardization](https://www.neuromorpho.org/StdSwc1.21.jsp), and [MorphIO specification](https://morphio.readthedocs.io/en/v3.4.0/specification.html). Their policies differ; none establishes image-level correctness by parsing alone.

Access caveat: publisher access was intermittent, but relevant local Zotero full texts resolved the central Gou and comparator methods. Exact penalty 9000 was verified in deposited source. Remaining reconstruction-specific exclusions, transform quality and this project's source export lineage require direct dataset evidence. This is a focused extensive review, not an exhaustive fMOST survey or scientific acceptance of the branch.

## Implementation evidence addendum — 2026-10-03

The processed IONData/NMT SWC convention discussed above must be distinguished
from the later microscopy workflow. That workflow explicitly requests each
monkey's `/swc_raw/` reconstruction and uses its root in the nominal native image
grid. A `.swc` extension alone does not establish coordinate frame. Portal NMT
coordinates and processed reconstructions cannot substitute for a missing raw
root when placing a microscopy crop. Source origin, orientation, and physical
calibration remain nominal until independently established.

The [visual methods](bulk_visual_review_20261002/visual_methods.md) document
source-grid geometry, root coverage, bounded downloads, missing-tile masks,
point decimation, and correction-table provenance. The
[server inventory](bulk_visual_review_20261002/server_inventory.md) distinguishes
the four copied overview stacks from the five monkeys requiring derived context.
The [acceptance review](bulk_visual_review_20261002/independent_visual_acceptance_20261003.md)
explains why CH1 candidate localization cannot by itself support the finer
architectonic assignments described in references 27 and 28. No independently
verified PI/tissue channel or full-section locator has been established for these
derived fields. Missing source pixels are acquisition gaps, not anatomical absence.

The original six clustering runs remain exploratory comparisons on their recorded
fixed matrices. Finite-subset or guided-label stability does not establish
independent-animal prediction or a biological taxonomy. Current image generation
does not change the accepted cohort, accept a cluster count, or resolve distant,
ambiguous, or blank portal labels. The 99-test expanded software snapshot and
independent source-pixel readbacks verify implementation properties only; see
the [completion audit](bulk_visual_review_20261002/completion_audit_20261003.md)
for completed evidence and outstanding work.
