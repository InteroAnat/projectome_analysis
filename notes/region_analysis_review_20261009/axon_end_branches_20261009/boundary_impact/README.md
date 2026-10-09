# Whole-axon boundary-fix impact on the selected 462 neurons

**No numerical allocation impact was found in the complete current dataset.** The corrected half-open voxel rule remains a valid software repair for the demonstrated floating-point boundary case, but these existing whole-axon maps require no numerical repair from that change.

The [source-bound audit](all462_boundary_impact.json) compares the pre-fix `HEAD` implementation, using `floor(point+0.5)`, against the corrected fractional comparison. It checks every original edge selected by the child axon label on the exact combined 462-neuron manifest, declared 250-µm index scales and pinned NMT reference. Source byte hashes are checked before graph parsing. The original graph, coordinate declaration and selection are unchanged.

| Exhaustive check | Result |
|---|---:|
| Selected neurons | 462 |
| Original child-axon edges | 7,828,761 |
| Zero-length selected edges | 0 |
| Fast same-voxel edges, old and corrected | 7,385,577 each |
| Edges with changed endpoint bins or fast-path allocation | 0 |
| Remaining slow-path edges | 443,184 |
| Clipped positive midpoint intervals checked | 878,039 |
| Intervals with different old/corrected voxel indices | 0 |
| Edges or neurons with changed positive voxel allocation | 0 |

This is a voxel-allocation comparison, **not merely a length-conservation check**. It first compares both endpoint bin rules and their fast-path predicates for all selected edges. For the remaining positive edges it reproduces the unchanged FOV clipping and voxel-face parameter intervals, then compares the old and corrected midpoint indices exactly. Both assignment paths and their accumulation order are unchanged on these sources. Dense old/new neuron maps are only needed if a positive allocation difference is found; none were needed or generated. This does not claim equality for other SWCs, references or hypothetical coordinates, nor validate biological anatomy.

The [scanner regression tests](test_boundary_impact_scanner.py) pass against both actual rasterizer implementations on 128 seeded random segments and seven adversarial segments. They also detect the known synthetic failure: a segment parallel to a voxel face at `nextafter(0.5,-inf)` was allocated to the upper voxel by the old rule and correctly stays in the lower voxel after the fix. The repair therefore addresses a real boundary defect even though it has zero effect on this particular source set.

[Per-neuron results](per_neuron_boundary_impact.csv) retain exact UID, source hash, edge/fast-path/midpoint counts and zero observed changes. [Source bindings](source_bindings.json) enumerate all 462 original SWCs. The audit pins the old Git commit and code bytes, current code bytes, manifest and reference hashes. AST comparisons verify that the two numerical functions differ only by the expected rounding substitutions. The separately added `save_map` description argument is outside this numerical comparison and is never called. The old source snapshot is retained solely for reproduction.

Run the scanner tests from the repository root:

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -X utf8 -B -m unittest discover -s notes/region_analysis_review_20261009/axon_end_branches_20261009/boundary_impact -p test_boundary_impact_scanner.py -v
```

The original audit script refuses an existing receipt destination and uses the then-current Git `HEAD`; its receipt identifies the exact inspected old version. Any future rerun should use a separate directory and explicitly compare the recorded source versions rather than silently replacing this historical result. No scientific sources, maps, old receipts or cohorts were modified by this audit.
