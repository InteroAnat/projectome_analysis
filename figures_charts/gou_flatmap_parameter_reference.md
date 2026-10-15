# Flatmap Pipeline — Parameter Reference

Two scripts in `gou_julia_scripts/`:

| Script | Role |
|--------|-------|
| `gou_flatmap_minimal.jl` | Core pipeline (builds flatmaps + individual-panel figures) |
| `insula_lr_mirror_flatmap.jl` | Presentation layer (LR mirrored/inward-facing layout) |

Both reuse the same JLD2 caches. **Only edit `insula_lr_mirror_flatmap.jl` for presentation; only edit `gou_flatmap_minimal.jl` to add a new brain region.**

---

## gou_flatmap_minimal.jl

### Constants (lines 12-15)

```julia
const LOCAL_NMT_ROOT = raw"D:\projectome_analysis\atlas\NMT_v2.0_sym"
const SURF_ROOT      = joinpath(LOCAL_NMT_ROOT, "NMT_v2.0_sym_surfaces")
const SOMA_TABLE     = raw"D:\projectome_analysis\neuron_tables\251637_INS_HE_inferred.xlsx"
const OUT_ROOT       = raw"D:\projectome_analysis\figures_charts\gou_flatmap_conservative"
const CACHE_DIR      = joinpath(OUT_ROOT, "cache")
```

| Constant | When to change |
|----------|---------------|
| `SOMA_TABLE` | New neuron dataset |
| `LOCAL_NMT_ROOT` / `SURF_ROOT` | Different atlas |
| `OUT_ROOT` | Different output folder |

---

### Add a New Brain Region — `build_flatmap()` (lines 142-181)

Add one block to the `cfg` NamedTuple inside `build_flatmap()`:

```julia
leftparietal = (;
    clipfn = (obj1) -> begin
        nmt = NmtDat.load_images()
        # ① CHARM level + index that defines the region
        mask = nmt.charm[:,:,:,LEVEL] .== INDEX
        charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res),
            CartesianIndices(nmt.seg)[mask])
        tree = NearestNeighbors.KDTree(charmpos)
        dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
        dist1 = dijkstra_vertex(obj, findmax(dists)[2])
        # ② Geodesic cutoff — how much surrounding cortex to include
        (v) -> dists[v] < 500 || dist1[v][1] > 25000
    end,
    # ③ Flattening seed orientation
    rightfn = p -> p[1],   # 1 = anterior-posterior axis
                             # 2 = medial-lateral axis
),
```

| Parameter | How to find | What it controls |
|----------|-------------|-----------------|
| **CHARM level + index** | See `atlas/NMT_v2.0_sym/tables_CHARM/CHARM_key_all.txt` | Which atlas voxels define the region |
| **Dijkstra threshold** (`dist1[v][1] > N`) | Start 25000; larger = more surrounding cortex | How much mesh is kept around region |
| **`rightfn`** (`p[1]` or `p[2]`) | Try both; pick whichever gives better flattening | Anterior-posterior vs medial-lateral seed for flattening |

---

## insula_lr_mirror_flatmap.jl

All parameters are in the plotting section of `run_insula_lr_mirror()`.

### 1. Region tag (line 23)
```julia
tag = :leftinsula   # must match a tag defined in build_flatmap()
```

### 2. Output subfolder (line 26)
```julia
out_dir = joinpath(OUT_ROOT, "insula")   # → figures_charts/gou_flatmap_conservative/insula/
```

### 3. CHARM subregion highlights (line 56)
```julia
highlights = get_insula_highlights()
```
Define in `gou_flatmap_minimal.jl`. Path format: build from `unique()` of abbreviation indices at all 6 CHARM levels for a voxel.
```
Example — granular insula voxel:
  L1=224(floor_of_ls), L2=225(Ins/Pi), L3=227(Ins), L4=228(Ia/Id)
  Path: "nmt|charm|Temporal|floor_of_ls|Ins/Pi|Ins|Ia/Id"

Styles:
  "path" => :gray80       # filled gray area
  "path" => 0.75           # medium contour line
  "path"                   # thin contour (default 0.5)
```

### 4. Centroid labels (line 56)
```julia
add_insula_labels!(ax, objflat, objphy)
```
Define in `gou_flatmap_minimal.jl`. For each label: look up CHARM index at each physical vertex → mask matching vertices → compute centroid in UV space → `Makie.text!()`.
```julia
label_defs = [
    # (charm_level, charm_index, display_label, fontsize, font_style)
    (6, 228, "Ia/Id",  9, :italic),
    (6, 229, "Ig",     9, :italic),
    (5, 226, "Pi",     9, :italic),
    (4, 230, "Ri",    10, :bold),
    # ...
]
```

### 5. Axis zoom padding (line 69)
```julia
limsL = insula_axis_limits(objflatL, objphy; pad=0.20)
limsR = insula_axis_limits(objflatR, objphy; pad=0.20)
```
Controls how tight the zoom is around the annotated region. `pad=0.20` = 20% margin.

### 6. Marker styles (lines 81-82, 91-92)
```julia
color    = :steelblue   # Left  / :firebrick  # Right
marker   = :circle      # Left  / :utriangle  # Right
markersize = 7
strokecolor = :black
strokewidth = 0.4
```

---

## Running

```bash
# Core insula pipeline (individual panels)
julia +1.11.5 --project="D:\projectome_analysis\references\analysis-code_gou_etal_2025\monkeyrec" -e 'include("D:\projectome_analysis\gou_julia_scripts\gou_flatmap_minimal.jl"); run_insula_pipeline()'

# LR mirrored presentation figure
julia +1.11.5 --project="D:\projectome_analysis\references\analysis-code_gou_etal_2025\monkeyrec" "D:\projectome_analysis\gou_julia_scripts\insula_lr_mirror_flatmap.jl"

# PFC frontal flatmap
julia +1.11.5 --project="D:\projectome_analysis\references\analysis-code_gou_etal_2025\monkeyrec" -e 'include("D:\projectome_analysis\gou_julia_scripts\gou_flatmap_minimal.jl"); run_pipeline(tag=:leftfrontal)'
```

---

## Cache Files

| File | Size | Reusable across regions? |
|------|------|--------------------------|
| `cache/depth_volume.jld2` | ~2 GB | **Yes** — one per atlas |
| `cache/flatmap_{tag}_n{niter}.jld2` | ~20 MB | **No** — per region |

Delete a cache file to force recomputation of that stage.
