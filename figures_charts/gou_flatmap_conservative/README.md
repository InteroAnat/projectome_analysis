# Gou Flatmap Outputs (Current)

## Purpose
This folder stores the current validated flatmap outputs generated from `gou_julia_scripts/gou_flatmap_minimal.jl` and companion plotting scripts.

## Current Structure
- `cache/`
  - `depth_volume.jld2` (shared across regions)
  - `flatmap_leftfrontal_n30000.jld2`
  - `flatmap_leftinsula_n30000.jld2`
- Root files
  - `flatmap_base_leftfrontal.png/.svg`
  - `flatmap_soma_leftfrontal.png/.svg`
- `insula/`
  - `insula_base.png`
  - `insula_soma_LR_combined.png/.svg`
  - `insula_soma_LR_split.png/.svg`
  - `insula_soma_type.png`
  - `insula_LR_mirror_flatmaps.png/.svg`

## Data Source
Current insula soma table:
- `neuron_tables/251637_INS_HE_inferred.xlsx`
- Counts: `L=105`, `R=155`, total `260`

## Notes
- Insula flatmap uses CHARM level-3 mask (`224=floor_of_ls`, `37=caudal_OFC`).
- Subregion labels (`Ia/Id`, `Ig`, `Pi`, `Ri`, `Iam`, `lat Ia`, `G`, `PrCO`) are atlas-driven centroid labels.
- `insula_LR_mirror_flatmaps.*` is a presentation layout:
  - viewer-left panel: Right insula
  - viewer-right panel: Left insula
  - both panels inward-facing (anterior poles toward center).
- If atlas/surface source changes, delete `cache/depth_volume.jld2` and region flatmap caches before rerun.
