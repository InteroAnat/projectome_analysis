# Flatmap Sub-Plan (Format Replication First)

## Scope
- Keep the big transfer plan unchanged.
- Focus this sub-plan on reproducing the **visual flatmap format** seen in Gou et al. 2025 before full biological integration.
- Use synthetic/mock inputs first, then replace with real projectome data.

## Reference Anchors
- Paper: `references/Gou et al. - 2025 - Single-neuron projectomes of macaque prefrontal cortex reveal refined axon targeting and arborizatio.pdf`
- Code style references:
  - `references/analysis-code_gou_etal_2025/monkeyrec/zz2fig/src/2-plot.jl`
  - `references/analysis-code_gou_etal_2025/monkeyrec/0-meshes.jl`
  - `references/analysis-code_gou_etal_2025/monkeyrec/2-meshes.jl`

## Immediate Objectives
1. Build a Julia mock flatmap renderer that matches the visual grammar:
   - black boundary contours
   - filled subregions with transparent colors
   - soma scatters overlaid by type
   - equal aspect ratio and fixed map limits
2. Generate demo outputs (`.png` and `.svg`) for quick style review.
3. Use the mock output to lock style choices before connecting real coordinates.

## Style Targets From Gou Pipeline
- Region masks rendered as contour/contourf style overlays (`plot_flatmap_base!` pattern).
- Soma overlays as small points on top (`plot_flatmap_soma!` pattern).
- Stable canvas coordinates in normalized 2D flatmap space (roughly `[-1, 1]` frame).
- Annotation strategy compatible with later label placement.

## Risks To Control Early
- Coordinate orientation confusion (voxel/index/world/RAS mismatch).
- Hemisphere mirroring inconsistencies.
- Marker density and alpha hiding subregion boundaries.

## Next Integration Step (after mock approval)
- Replace synthetic boundaries with project-derived mesh/flatmap coordinates from `step_by_step_insula_flatmap.py` outputs.
- Preserve current R laterality pipeline; only add a visualization branch.

## Current Execution Status
- Added Julia prototype: `gou_julia_scripts/mock_flatmap_gou_style.jl`.
- Julia environment install succeeded, but rendering failed under local Julia `1.12.x` due Makie/Cairo precompile incompatibility with this project's older dependency resolution.
- Added working Python fallback prototype: `main_scripts/mock_flatmap_gou_style.py`.
- Generated mock outputs:
  - `figures_charts/mock_flatmap_gou_style/mock_flatmap_gou_style_py.png`
  - `figures_charts/mock_flatmap_gou_style/mock_flatmap_gou_style_py.svg`
