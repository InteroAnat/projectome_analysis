# Configurable NMT flatmap / depth wrapper (Gou `monkeyrec` pipeline)

This folder is a **small Julia project** that reproduces the heavy **depth model → depth volume → cortical flatmap → XYZ→UVW** sequence from the Gou et al. `monkeyrec` code, without hardcoded `/data/dat/...` paths. Volumes and surfaces are read from **`config.toml`**, and Gou’s mesh geometry (`refine_mesh`, `clip_mesh`, `flatten_mesh`, `dijkstra_vertex`, `sample_mesh`) is **reused via `include`** of `references/analysis-code_gou_etal_2025/monkeyrec/0-meshes.jl`.

## Prerequisites

- **Julia 1.9+** on **Windows** or Linux/WSL (GPU optional but strongly recommended for the depth model).
- **NMT v2.0_sym**-consistent NIfTI stack + cortical `.obj` surfaces (same ordering / resolution assumptions as upstream).
- **Display** for **GLMakie**: `flatten_mesh` and depth training call `display(fig)`. On Windows this is usually your normal desktop session; on headless Linux use WSLg, an X server, or an offscreen Makie path if you add one later.

## Paths in `config.toml`

Use **absolute paths**. On Windows, **forward slashes** in TOML are simplest (e.g. `D:/projectome_analysis/atlas/...`). Change the drive letter and folder name if your repo is not on `D:`.

If you use **WSL/Linux** instead, use POSIX paths there (e.g. `/mnt/d/projectome_analysis/...`) in the same keys.

## Setup (Windows PowerShell)

```powershell
cd D:\projectome_analysis\tools\nmt_flatmap_pipeline
Copy-Item config.example.toml config.toml
# Edit config.toml: set D:/... paths (or your drive)
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Check GPU from Julia (optional):

```julia
using CUDA
CUDA.functional()
```

## Run order

| Step | Script | Output (under `[outputs].interm_dir`) |
|------|--------|----------------------------------------|
| 1 | `src/run_depth_model.jl` | `depth_model_stem` (JLD2 `df`: columns `dir`, `ti`, `te`, `pos`) |
| 2 | `src/run_depth_volume.jl` | `depth_volume_stem` (JLD2: `img`, `res`) |
| 3 | `src/run_flatmap.jl` | `flatmap_stem` (JLD2: `obj` 2D mesh, `objphy` clipped 3D mesh) |
| 4 | `src/run_uvw_export.jl` | `[uvw_export].output_csv` |

From the **repo root** (`D:\projectome_analysis`), example:

```powershell
cd D:\projectome_analysis
julia --project=tools/nmt_flatmap_pipeline tools/nmt_flatmap_pipeline/src/run_depth_model.jl tools/nmt_flatmap_pipeline/config.toml
julia --project=tools/nmt_flatmap_pipeline tools/nmt_flatmap_pipeline/src/run_depth_volume.jl tools/nmt_flatmap_pipeline/config.toml
julia --project=tools/nmt_flatmap_pipeline tools/nmt_flatmap_pipeline/src/run_flatmap.jl tools/nmt_flatmap_pipeline/config.toml
julia --project=tools/nmt_flatmap_pipeline tools/nmt_flatmap_pipeline/src/run_uvw_export.jl tools/nmt_flatmap_pipeline/config.toml
```

Or `cd` into `tools\nmt_flatmap_pipeline` and use `--project=.` and `src\run_....jl config.toml`.

`NNHelpers.sample_objs` from upstream (undefined in `1-unibrain.jl`’s standalone scope) is replaced by **`sample_objs_pointcloud`** in `run_depth_model.jl`, equivalent triangle sampling on `GeometryBasics` meshes.

The **lefths** / **leftfrontal** flatmap masks use `LocalNmtDat.load_images()`; **`test`** does not load volumes (fastest smoke path).

## Directory roles (vs Gou code)

| Role | Gou hardcodes | Your config |
|------|----------------|-------------|
| NMT NIfTIs | `NmtDat.essential_files` in `zz0atlas.jl` | `[nmt]` paths |
| Surfaces | `1-unibrain.jl` / `2-meshes.jl` | `[surfaces]` |
| Intermediates | `monkeyrec/interm/*.dat` | `[outputs].interm_dir` + `*_stem` names |
| `0-meshes.jl` | same repo | default: `references/analysis-code_gou_etal_2025/monkeyrec` or `[paths].monkeyrec_root` |

## Coupling to R (laterality / INS Rmd)

1. Export a **soma CSV** with physical XYZ columns (same space as the NMT-aligned `Soma_NII_*` coordinates used in your analysis).
2. Run steps 1–4 above.
3. Read `somauv.csv` in R (`read.csv` / `readr::read_csv`) and **join** on `id` (or row index if `id_col` is omitted) to add `(u, v, w)` for flatmap-aware plots or summaries.

## Minimal smoke check (wiring only)

1. Set `[depth_model].train_iters` to a **very small** value (e.g. 2–20) only to verify scripts and paths; results will **not** be scientifically usable.
2. Use `[flatmap].tag = "test"` so no full NMT volume load is required for the mask.
3. For `run_uvw_export.jl`, point `[uvw_export].soma_csv` at a tiny CSV with two rows and the configured `x_col`, `y_col`, `z_col`.

## Risks (same as upstream)

- **VRAM / batch size**: lower `[depth_model].batch_vertices` if training OOMs.
- **Version skew**: mismatched CHARM / SS / surfaces break rasterization or the `@assert` that equates mid-mesh vertices with `df.pos`.
- **Headless servers**: without a display, GLMakie steps may fail until you add an offscreen backend or X forwarding.

## Key file map

| File | Purpose |
|------|---------|
| `config.example.toml` | Documented template; map keys ↔ Gou paths |
| `src/LocalNmtDat.jl` | TOML-driven `load_images` + `pos2idx` / `idx2pos` / `idx2ann` |
| `src/include_monkeyrec_meshes.jl` | Resolves and `include`s `0-meshes.jl` once |
| `src/run_depth_model.jl` | Depth MLP / adhesion loss (port of `depth-mapping-mac-model`) |
| `src/run_depth_volume.jl` | CHARM raster through columns (port of `depth-mapping-mac`) |
| `src/run_flatmap.jl` | Refine → clip → flatten (port of `flatmap_data(Tag)`) |
| `src/run_uvw_export.jl` | `xyz2uvw` → CSV |
