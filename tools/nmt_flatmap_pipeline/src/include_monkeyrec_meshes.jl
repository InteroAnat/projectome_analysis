# Load Gou et al. mesh helpers (refine / clip / flatten / sample_mesh / dijkstra_vertex).
# Call `init_monkeyrec_meshes!(cfg)` after parsing config so `paths.monkeyrec_root` can override defaults.

using GeometryBasics
using MeshIO
using FileIO
using LinearAlgebra
using Statistics
using DataStructures
using NearestNeighbors
using ProgressMeter
using Flux
using GLMakie
using Makie
using Meshes
using Distributions
using StaticArrays

const _MESHESSRC = Ref{Union{Nothing,String}}(nothing)

function default_monkeyrec_root()
    # tools/nmt_flatmap_pipeline/src -> repo root is three levels up
    joinpath(@__DIR__, "..", "..", "..", "references", "analysis-code_gou_etal_2025", "monkeyrec")
end

function mesh_jl_path(cfg::AbstractDict)
    root = get(get(cfg, "paths", Dict()), "monkeyrec_root", default_monkeyrec_root())
    root = string(root)
    joinpath(root, "0-meshes.jl")
end

"""
    init_monkeyrec_meshes!(cfg) -> path_included

`include` Gou `0-meshes.jl` once (idempotent). Requires OPEN display for `flatten_mesh` / depth viz.
"""
function init_monkeyrec_meshes!(cfg::AbstractDict)
    path = mesh_jl_path(cfg)
    isfile(path) || error("""
        0-meshes.jl not found at:
          $path
        Set [paths].monkeyrec_root in config.toml to your monkeyrec checkout, or sync
        references/analysis-code_gou_etal_2025/monkeyrec under the repo.
        """)
    if _MESHESSRC[] == path
        return path
    end
    _MESHESSRC[] = path
    Base.include(Main, path)
    return path
end
