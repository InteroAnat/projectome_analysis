# ION pyswcloader testing ground

Evaluation of [pyswcloader](https://github.com/txgxxx/pyswcloader) 1.1.1 (PyPI, author `gaots@ion.ac.cn`) on the same question already used for Liu et al. 2026. The ground reads SWC files and writes a JSON receipt. It does not change reconstructions, atlases, or cohort tables.

## What each tool is for

| Tool | Where it comes from | What this project already uses it for |
|---|---|---|
| **NeuroM 4.0.5 + MorphIO** | Blue Brain. Liu et al. import it in [`RFLiu2021/AIC_proj` `morpho/preprocessMorphData.ipynb`](https://github.com/RFLiu2021/AIC_proj/blob/main/morpho/preprocessMorphData.ipynb). | The VEN bridge in `external/ven_validation/`. Axon removed, dendrites kept inside an 800 µm soma ball, then NeuroM features. |
| **pyswcloader 1.1.1** | ION / Digital Brain. Methods paper: Jiao et al., *Nat Neurosci* 2025, hypothalamic peptidergic projectomes in male mice. | Not used in the bridge. This ground is the first local check. |

Liu's notebook also imports local helpers `batchExtractMorphFeatures`, `anaASCFileFuncs`, and `morphClassifier`. Those files are not in the public repository tree, so they are not executed here. The handoff note that mentions L-Measure describes an earlier plan; the bridge that ran used NeuroM.

pyswcloader's bundled atlas is the Allen mouse CCF. Its default box is 13.2 × 8.0 × 11.4 mm, and Z is treated as the mouse medial-lateral axis. Macaque NMT/ARM labels are not in the package.

## Operator

From the repo root, with the `projectome` environment:

```powershell
$py = "$env:USERPROFILE\miniconda3\envs\projectome\python.exe"
& $py -m pip install --target work/pyswcloader_ground/vendor --no-deps pyswcloader==1.1.1 treelib rdp
& $py tools/pyswcloader_ground.py --dry-run --synthetic
& $py tools/pyswcloader_ground.py --synthetic --output work/pyswcloader_ground/receipts/synthetic.json
& $py tools/pyswcloader_ground.py --exemplars --output work/pyswcloader_ground/receipts/exemplars.json
& $py -m unittest tests.test_pyswcloader_ground
```

`--exemplars` is the bounded Liu-bridge pair: `processed_neurons/251637/007.swc` and `external/liu2026/swc/VENL/unSM1114.swc` (the first VEN-L file if that name is absent). NeuroM and MorphIO stay in the `projectome` environment. The wheel is unpacked only under `vendor/`, which is gitignored.

### Parameters

| Name | Flag | Default | Effect | Valid values |
|---|---|---|---|---|
| mode | `--mode` | `compare` | `inspect` skips NeuroM; `compare` runs both; `preprocess-risk` reports the Allen box only; `distance` scores two SWCs | those four names |
| case | `--case` | unset | one SWC | existing file; choose exactly one of `--case`, `--synthetic`, `--exemplars` |
| synthetic | `--synthetic` | off | built-in bipolar neuron: basal 600 µm, apical 400 µm, axon 5000 µm | flag |
| exemplars | `--exemplars` | off | 251637/007 plus one Liu VEN-L SWC | both files present |
| other | `--other` | unset | second SWC for `distance` | existing file |
| r_local | `--r-local` | `800` | soma ball passed to the existing NeuroM extractor, in µm | greater than 0 and at most 5000 |
| flip | `--flip` / `--no-flip` | `--flip` | mouse Z flip inside the package preprocess used for the risk report | flag |
| output | `--output` | `work/pyswcloader_ground/receipts/<mode>.json` | receipt path | inside `receipts/` |
| dry run | `--dry-run` | off | print resolved settings, cases, and output path | flag; writes nothing |

Invalid radius, a missing file, or an output path outside `receipts/` stops with a short message. `distance` does not call `morphology_matrix`, so it does not spawn a process pool or write `scores_record.txt`.

## Plan of the tests

1. Confirm the pinned wheel loads and that its ontology is the Allen mouse tree.
2. Compare the reader with this repo's `parse_swc` on spacing, comments, scientific notation, and exact integer ids.
3. Measure `total_length` against an id-keyed cable sum. The package adds an edge only when both endpoints are outside types 3 and 4.
4. Put a native-scale soma through the default preprocess and record clipping and the Z flip.
5. Run projection, ipsilateral length, and terminal counts on a tiny label volume with known geometry.
6. Score the morphology distance and a precomputed hierarchical clustering on four small neurons.
7. Run the existing `extract_features_neurom` (800 µm, axon removed) on the bipolar control, on the same neuron translated to the 251637/007 soma, and on the two exemplars.

## What the checks show

- On the bipolar control, pyswcloader `total_length` is **5000 µm** (the axon). NeuroM dendritic length inside 800 µm is **1000 µm**, with basal max path 600 µm and apical max path 400 µm. Those are the two sides of the Liu comparison: axon cable versus local dendritic shape.
- Exemplars in `receipts/exemplars.json`, NeuroM 4.0.5, 800 µm crop. **251637/007**: soma `[47541.4, 54080.1, 20557.8]` µm; pyswcloader length **93812.27 µm**, equal to the axon cable; full basal cable 16068.98 µm with no type-4 tag; NeuroM after polarity inference reports dendritic length **14811.85 µm** (basal max path 884.74 µm, apical max path 879.16 µm). Default preprocess moves the first node to `[13199, 7999, −9157.8]`. **Liu VEN-L unSM1114**: soma near the origin; pyswcloader length **878.12 µm** (slice axon); full basal/apical cable 7700.22 / 6461.16 µm; NeuroM dendritic length **13122.83 µm**, basal max path **939.91 µm**, apical max path **900.46 µm**, expert type-4 labels present. The Allen box does not move this file. The path lengths match the existing bridge note for unSM1114.
- Translating that neuron to the 007 soma leaves the NeuroM lengths unchanged. The package preprocess moves the first node to **[13199, 7999, −9157.8] µm**.
- 251637/007 is in native microscopy coordinates. Default preprocess flips Z and clips X and Y. Liu VEN-L `unSM1133` is slice-local, near the origin, so the Allen box does not move it. The length rule still sums the non-dendrite edges.
- A parent id that is not the dataframe row index can make `total_length` return 0 for a 5 µm axon. Unsorted files fail in `swc_tree` because the parent node is not created yet. `downsample` calls `DataFrame.append`, which pandas 2.3 removed.
- Projection length splits a region-crossing axon 50/50 by the labels of the two endpoints. After the Z flip, a negative coordinate indexes the array from the end, so an out-of-box node can receive a real region id.
- Ipsilateral projection length follows the soma's side of the annotation Z midline. Ipsilateral terminal counts keep nodes with Z below that midline. The package comment says somas were mirrored to the left hemisphere.
- Saved neuron names are split on `/`. On Windows the row label becomes the path without its extension.
- The point-cloud distance is 0 for an identical file and larger for a larger shift. It uses raw coordinate units, so a slice-local Liu neuron and a native fMOST neuron are not in one frame. The VEN bridge compares features after the crop for that reason.
- Hierarchical clustering of a precomputed distance matrix can separate two shifted pairs. `cluster` draws a dendrogram and calls `plt.show`.

## Adoption boundary

NeuroM remains the tool for the Liu dendritic comparison. pyswcloader 1.1.1 can be revisited for axon length inside a label volume only after the SWC coordinates are already in that volume's frame. This evaluation does not make that frame, and it does not accept any anatomical label.
