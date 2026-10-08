# Coarse INS review and map source inventory

The priority CSV is a retrieval queue, not corrected anatomy. `reviewed_INS` uses exact Henry visual annotations or hash-bound case decisions. Fine labels remain original and provisional where applicable. Adjacent numeric IDs are retrieval cues only.

The [distance-priority v2 supplement](distance_priority_v2_20261009/README.md) supersedes the original nearest-anchor distances and priority queue: it excludes self/alternative-origin anchors and contains 559 entries. Original hash-bound delivery files remain preserved. The [mapping report](../../reports/evolution_status_20261009.md) records 436 main and 26 disjoint additional candidate neurons mapped under the declared unresolved coordinate-origin convention.

Official segmentation classes come from the NIfTI embedded AFNI table: 1 CSF, 2 GM, 3 scGM, 4 WM, 5 BV. Atlas background is not WM. Both current zero-center and literal published edge-origin policies are retained; the export convention and original transforms remain unverified.

Map sources require valid connected rooted seven-column graphs and exact numeric equality of every cached copy after sorting node IDs. Selection is deterministic only after equality, preserving hashes and all source paths in graph readback. No image/network downloads, relabeling, canonical writes or anatomical acceptance. See provenance.json for exact counts and input hashes.
