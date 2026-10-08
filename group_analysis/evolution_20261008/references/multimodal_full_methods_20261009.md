# NMT, insula fMRI and multimodal projectome: original Methods review

Date: 2026-10-09, Asia/Shanghai. Sources: the user's local Zotero library, read only. This review reads original Methods, rather than relying on titles or abstracts. Full paper text and rendered source pages remain in temporary storage; the repository contains paraphrases, source identifiers, hashes and section anchors only. [Machine-readable evidence](multimodal_full_methods_20261009.json) records the exact attachments and citation metadata.

## NMT and atlas construction

**Jung et al. (2021), A comprehensive macaque fMRI pipeline and hierarchical atlas. NeuroImage 235:117997. DOI [10.1016/j.neuroimage.2021.117997](https://doi.org/10.1016/j.neuroimage.2021.117997).** Zotero `T5SFIMV4`, attachment `FYH7NL8F`, accepted manuscript, 50/50 pages indexed. The duplicate record `FQFN5FEP` was identified but not counted as another study.

Methods sections 2.1–2.2.3 establish the background as a T1-weighted population template from 31 rhesus macaques, with a final 0.25 mm grid. Symmetric construction combines mirrored scans and explicitly makes the template symmetric. The stereotaxic origin lies on the midsagittal/interaural intersection. The midline lies on a voxel edge; origin and index conventions therefore matter. This is a population reference, not the raw fMOST brain or an individual MSTIM subject.

Sections 2.3.1–2.3.2 describe D99-to-NMT nonlinear warping, cortical segmentation refinement and a six-level CHARM hierarchy. Some small finest-level D99 parcels are deliberately combined. This supports explicit hierarchy/scale reporting; it does not establish a one-to-one crosswalk to the finer Evrard insula subdivisions or validate an individual's warp.

**Hartig et al. (2021), The Subcortical Atlas of the Rhesus Macaque (SARM) for neuroimaging. NeuroImage 235:117996. DOI [10.1016/j.neuroimage.2021.117996](https://doi.org/10.1016/j.neuroimage.2021.117996).** Zotero `8AV5I9VP`, attachment `C9M3SMAE`, 23/23 pages indexed.

Methods 2.1.1–2.1.4 describe manual subcortical delineation on one ex vivo rhesus scan (G12; 0.15 x 0.15 x 1 mm), guided by histological references, nonlinear transfer to NMT, and further automatic/manual refinement. Section 2.2 groups 210 primary ROIs into six scales. Discussion 4.2–4.4 cautions about alignment, resolution and heterogeneous composites; functional localization is not universal parcel ground truth. This supports named hierarchy-based target sets and coarse summaries when fine localization is unreliable. A numeric label interval is not an anatomical definition of thalamus.

The local NMT v2.1 release notes separately document SARM's +1000 offset, right-hemisphere +500 offsets, changed hippocampal subdivisions and combined ARM. Its README describes ARM as the fusion of that release's CHARM/SARM. This does not make an older v2.0 label table numerically interchangeable. Pin atlas, key, level, laterality and release together even when reference MRI pixels match.

## Insula functional measurements

**Charbonneau et al. (2024), Intrinsic functional and structural network organization in the macaque insula. Imaging Neuroscience 2:imag-2-00261. DOI [10.1162/imag_a_00261](https://doi.org/10.1162/imag_a_00261).** Zotero `SQ2SJEIX`, PDF `JKMYTA8J`. Its text index returned 404; the existing 25-page PDF was extracted read only, and Methods pages 4–5 were rendered and visually checked.

Methods 2.1–2.3 and Figure 2 describe 19 male rhesus macaques, 31 left-insula seeds and 2 mm-diameter spheres. The replicated anterior seed analysis uses F99/112RM conventions; the gradient route uses AFNI processing and other surface displays use NMT. These distinct routes correct an overly broad description of the whole paper as F99-only. Connectivity is BOLD time-series correlation, with autocorrelation adjustment and subject-level random effects. Section 2.3.1 contrasts dorsal-minus-ventral connectivity within each animal, then uses 10,000 permutations, TFCE and multiplicity correction. This supports animal-level contrasts when replication permits them. Neither MRI-derived clusters nor their thresholds can be transferred as terminal-arbor or INS parcel acceptance rules.

**Sypré, Durand and Nelissen (2023), Functional characterization of macaque insula using task-based and resting-state fMRI. NeuroImage 276:120217. DOI [10.1016/j.neuroimage.2023.120217](https://doi.org/10.1016/j.neuroimage.2023.120217).** Zotero `NDA9X5DQ`, attachment `I249FQTR`, 13/13 pages indexed. Full Methods are now available; the earlier metadata-only access limitation is superseded.

Sections 2.1 and 2.5–2.9 distinguish eight resting-state animals from two-animal task samples. Most acquisitions use iron contrast; vestibular measurements use BOLD. Images undergo rigid/nonrigid JIP alignment to the individual M12 template, not directly to our NMT. Twenty-two seeds per hemisphere have 1 mm radii. Resting correlations and task GLM contrasts are different measurements; task maps use fixed effects with n=2 and the display threshold p<0.001 uncorrected. This supplies functional hypotheses and explicit modality/contrast cautions, not animal-population inference, direct anatomical output or a parcel crosswalk to our ARM.

## Acquisition precedent and calibration

**Zhou et al. (2022), Continuous subcellular resolution three-dimensional imaging on intact macaque brain. Science Bulletin 67:85–96. DOI [10.1016/j.scib.2021.08.003](https://doi.org/10.1016/j.scib.2021.08.003).** Zotero `RF4JIFTM`, attachment `UFYL9BGY`, 12/12 pages indexed. This is an acquisition precedent cited by Gou 2025, rather than proof that every current dataset used identical parameters.

Methods 2.14–2.15 distinguish the 10x acquisition (0.65 x 0.65 x 3 micrometer voxel sampling) from the 20x acquisition (0.32 x 0.32 x 10 micrometers). Optical resolution was assessed separately using fluorescent beads. Green-channel GFP and red-channel PI cytoarchitecture support complementary tracing/anatomy observations. Stripe correction/stitching precedes TIFF storage; PI images were registered to presurgical individual MRI using BrainsMapi, not directly to our NMT. Results 3.5 distinguish axon passage through structures from target-local branches and image-resolved giant terminals. This supports source-image continuity and morphological review, not a generic SWC-tip detector. The dataset folder label “5 micrometer” cannot establish isotropic spacing, native calibration, optical resolution or resampling correctness. Those require each current dataset's own metadata and provenance.

## Multimodal anatomical comparison

**Yan, Yu et al. (2022), Mapping brain-wide excitatory projectome of primate prefrontal cortex at submicron resolution and comparison with diffusion tractography. eLife 11:e72534. DOI [10.7554/eLife.72534](https://doi.org/10.7554/eLife.72534).** Zotero `RXUGG8AT`, attachment `7974HUDI`, 28/28 pages indexed. The citation has 19 authors; Zotero also lists three editors, which are not authors.

Methods “Fluorescence image preprocessing,” “STP Image Processing” (printed pp18–19), and “Probabilistic tractography” (pp20–21) use serial two-photon tomography, not fMOST single-neuron tips. Supervised segmentation distinguishes GFP-positive fibers/varicosities from lipofuscin; held-out labeled images assess segmentation. Red-channel anatomy, GFP and injection volumes are mapped to a cynomolgus template with ANTs SyN. Regional GFP counts/volume and density quantify bulk labeling. Same-space 500 micrometer grids support thresholded Dice and pixel correlation comparisons with dMRI. These are useful common-space/coverage precedents; bulk fluorescence, skeleton length and endpoint count remain different observables. Their cortex/subcortex parcellations are not our NMT ARM.

**Yang et al. (2025), Multimodal Correspondence between Optogenetic fMRI, Electrophysiology, and Anatomical Maps of the Secondary Somatosensory Cortex in Nonhuman Primates. Journal of Neuroscience 45(21):e2375242025. DOI [10.1523/JNEUROSCI.2375-24.2025](https://doi.org/10.1523/JNEUROSCI.2375-24.2025).** Zotero `VY9HAKFP`, attachment `YMUVA6N8`, 12/12 pages indexed.

Materials and Methods (printed pp2–3) use four squirrel monkeys/five hemispheres, electrophysiologically localized S2, optogenetic stimulation, virus-free controls, 9.4T BOLD and postmortem immunohistology. The Methods explicitly name **two** histology animals (SM5411/SM6599); the earlier evidence JSON's “one subject” statement is corrected here. Run-level maps, repeated-session reproducibility and Figure 6 anatomical correspondence address different questions. The study supports modality-specific corroboration and neurovascular caution. It does not provide NMT-space macaque INS/MSTIM pairing or justify equating BOLD amplitude, axon count and monosynaptic strength. Its selected peak-voxel summaries are not an independent validation design for our proposed atlas-wide comparison.

## Decisions for this pipeline

1. Preserve the existing maps as descriptive all-axon template-space length. Their software readback does not establish native lengths, registration or terminal fields.
2. Use the verified fMOST/arbor definition in the [fMOST review](fmost_full_methods_20261009.md) for the primary biological question. Candidate SWC endpoints remain reconstruction QC and a separately labelled exploratory proxy.
3. Resolve SWC export origin, exact atlas version and sample-to-NMT transform lineage before accepting soma or terminal labels. Compare coordinate policies explicitly; do not silently shift points to improve agreement.
4. Keep manual INS labels, atlas lookup and coordinate-screen candidates separate. Cytoarchitectonic parcels and MRI hierarchy scales require an explicit crosswalk with unresolved cases retained.
5. Begin any eligible MSTIM comparison with prespecified regional summaries: reviewed arbor presence/length and complementary all-axon extent, paired with signed stimulation effects on common valid coverage. Record animal, injection/site, hemisphere, atlas level, sampling denominator, acquisition contrast and effective resolution.
6. Preserve zeros, unassessed reconstructions and absent coverage as distinct states. Current endpoint-proxy averages condition on neurons with at least one qualifying reconstructed type-2 leaf; that denominator is an implementation choice, not a paper-validated estimate of biological innervation prevalence.
7. Current CM032 registration acceptance and CM033 usable NMT/model provenance remain unresolved in the [input audit](../crossmodal/input_readiness.json). No correlation, t-map or spatial-null result is warranted merely by a common template grid. Independent animal replication and a valid spatial/multiplicity design are separate requirements.

## Remaining dependencies

Original fMOST registration/export receipts and biological terminal/truncation review are still needed. The paper's algorithm does not prove that every IONDATA export used that exact implementation. Source soma-label assignment inside the IONDATA server remains undocumented by the local client. Full Methods improve the scientific basis; they do not independently accept this cohort, a fine INS parcel, a neuron subtype, or a CM032/CM033 alignment.
