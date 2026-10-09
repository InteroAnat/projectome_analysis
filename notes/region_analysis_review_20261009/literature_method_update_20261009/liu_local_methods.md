# Local Liu Methods review — 2026-10-09

Two actual local PDFs were read: Yufeng Liu (2024), for arbor and image-based varicosity methods, and Rui-Feng Liu (2026), for primate insular tissue/cell classification. The exact Sang Liu (2024) PDF was not located within the bounded local search. These are three different papers; no method is transferred between them by surname.

All page numbers below are **physical PDF pages, counted from 1**. This is a literature-input review, not anatomical acceptance or validation of a projectome classifier. PDFs and extracted full text remain outside the repository. Exact paths, hashes, anchors and read status are in [liu_local_methods.json](liu_local_methods.json).

## Yufeng Liu et al. (2024)

*Neuronal diversity and stereotypy at multiple scales through whole brain morphometry*. Nature Communications 15, 10269. [DOI](https://doi.org/10.1038/s41467-024-54745-6).

The 23-page PDF already existed in the prior review's temporary cache; its DOI, page count and exact SHA256 were reverified. Full Methods pp16–20 and Discussion p15 were read. No matching Zotero parent/attachment was located. This is a **mouse CCFv3 morphometry study**, not a macaque ARM/NMT atlas or a substitute for the current ARM-only region labels.

- **Arbors, p19:** the authors define an arbor as a densely packed subtree. Axons are subdivided using spectral clustering of an undirected graph of original tree nodes, with pairwise weights exp(−node distance). For comparisons they use the dominant automatic arbor number among neurons in the same region by majority vote. Branch number, volume and maximal node density characterize each arbor; maximal density counts axonal nodes within 20 µm of a node. Proximal/distal classification uses whether the maximal-density node lies more than 750 µm from the soma. Features are min–max normalized. The printed Methods do not provide a complete universal spectral-clustering parameter specification; no missing bandwidth or per-neuron cluster-selection rule is invented here.
- **Primary tract, p19:** start from the longest axonal path and iteratively remove shorter branches from its terminal side towards the soma. Grouped tracts are sampled at 200 positions; PCA cross-sectional radii use the 75th percentile. This describes a tract motif, rather than a validated passing-axon versus terminal-arbor classifier.
- **Varicosities, pp19–20:** image enhancement precedes intensity and radius profiles along 20 µm axonal fragments. Coincident peaks generate candidates. Heuristics require a radius 1.5 times that of surrounding axonal nodes and image intensity above 120 on an 8-bit scale. Candidates closer than five highest-resolution voxels are deduplicated (reported as roughly 1–2 µm). Two independent experts annotated 235 image blocks (approximately 59 × 59 × 256 µm³), yielding 1,450 reference varicosities for algorithm evaluation. These are image-derived candidates, not SWC leaf counts.
- **Limits and implementation:** Discussion p15 explicitly leaves biological validation of predicted varicosities beyond scope. Code availability p20 links the Vaa3D BoutonDetection implementation; that code was not audited in this local-PDF review. The 75-dimensional cross-scale morphology analysis on p20 and its correlation-based comparisons are separate from the current regional projection-profile/Hellinger analysis.

For this project, the arbor definition can motivate a separately validated candidate segmentation. The varicosity algorithm requires native calibrated images and image-derived radius/intensity. Template-space SWC radius, leaf topology or whole-axon occupancy alone cannot reproduce it. The 20 µm/750 µm and image-voxel thresholds must not be silently applied to deformed NMT coordinates, changed sampling density or uncalibrated fluorescence. A detected varicosity is not a confirmed synapse; an axon end is not a reviewed terminal arbor.

## Sang Liu et al. (2024)

*Single-neuron analysis of axon arbors reveals distinct presynaptic organizations between feedforward and feedback projections*. Sang Liu, Le Gao, Jiu Chen and Jun Yan. Cell Reports 43(1), 113590. [DOI](https://doi.org/10.1016/j.celrep.2023.113590).

**Full Methods were not read.** Focused Zotero title/DOI queries, including child/full-text search, did not identify the target record. The only `113590` hit was a reference in the Gou paper attachment, not the Sang paper itself. A bounded first-page title/DOI check of all 832 PDFs in the configured `C:/Users/laika_yan/Zotero/storage` found no target and no unreadable PDFs. Targeted filename listings in repository references, that storage root and Downloads also found no target. This does not establish absence elsewhere on disk. The search stopped there; an exact user-supplied path can close this dependency.

The earlier review recorded abstract-level access only. No passing-axon/arbor classifier architecture, training labels, numerical thresholds, performance or implementation details from this paper are claimed or used. Yufeng Liu (2024) and Rui-Feng Liu (2026) do not fill this Methods gap.

## Rui-Feng Liu et al. (2026)

*An atlas of primate insular cortex reveals a signal-processing strategy in von Economo neurons*. Nature Cell Biology. [DOI](https://doi.org/10.1038/s41556-026-02009-4). Zotero item `DFH77L3E`, actual PDF attachment `JUAS5RLT` (47 pages); hash and DOI reverified. Relevant Methods pp17–21 and Discussion p14 were read directly. The supplement was not newly reviewed here.

Cell annotation combines marker genes and cross-dataset homology, with some laminar annotations checked by RNAscope (p18). Biocytin-filled neurons are manually reconstructed with Neurolucida (p21). The soma defines a native slice origin and the pia defines the vertical direction. Anatomical annotation uses Nissl staining, cell density/size and layer markers; a manually drawn pia-to-white-matter path through the soma provides normalized depth. Forty morphology features and 20 × 20 × 300 µm dendritic-density bins describe reconstructed cells. These native tissue measurements are not a whole-brain NMT layer lookup or a universal VEN classification threshold.

Simultaneous recordings from up to eight neurons test local unitary postsynaptic responses (p19). Crucially, Discussion p14 states that subcerebral projection targets were not directly identified; molecular resemblance of VEN classes to L5 ET or L5/6 CT does not establish their actual long-range projections. This paper supplies no whole-brain passing-axon, terminal-arbor or bouton detector.

For the current data, Henry's exact wide-field visual annotations remain human evidence for coarse insular membership. Fluorescence context, soma shape and ARM position alone do not supply Nissl/laminar/molecular proof of a fine parcel, layer or VEN type. Fine-label uncertainty, registration provenance, candidate status and the provisional status of individual reviewed cases remain separate. No labels, cohorts, maps or canonical inputs were changed by this review.

Prior source reviews: [fMOST evidence](../../../group_analysis/evolution_20261008/references/fmost_full_methods_evidence_20261009.json) and [primate insula Methods](../../../group_analysis/evolution_20261008/references/insula_full_methods_20261009.md).
