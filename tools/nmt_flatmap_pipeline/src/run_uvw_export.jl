#!/usr/bin/env julia
#
# Mirrors references/.../monkeyrec/1-unibrain.jl xyz2uvw for INS / soma CSV → (u,v,w).

import Pkg
const PROJ = joinpath(@__DIR__, "..")
Pkg.activate(PROJ)

using CSV
using DataFrames
using JLD2
using LinearAlgebra
using NearestNeighbors
using ProgressMeter
using StaticArrays
using TOML

include(joinpath(@__DIR__, "include_monkeyrec_meshes.jl"))
include(joinpath(@__DIR__, "LocalNmtDat.jl"))
using .LocalNmtDat

function xyz2dxyz(xyzs, depthimg, depthres; nneighbor = 1)
    allci = CartesianIndices(@view depthimg[:, :, :, 1])
    idxs = map(xyzs) do xyz
        LocalNmtDat.pos2idx(xyz, depthres)
    end
    idx2dxyz =
        @showprogress "mapxyz" map(unique(idxs)) do idx
            rng = intersect(CartesianIndices(((idx[d] .+ (-nneighbor:nneighbor) for d = 1:3)...,)), allci)
            dxyzm = SVector(Inf32, NaN32)
            for ii in rng
                dxyz = @view(depthimg[ii, :])
                dd = norm(idx .- Float32.(Tuple(ii)))
                dd = isnan(dxyz[1]) ? dd * 100 + 100 : dd
                dxyz = SVector(dd, dxyz...)
                dxyzm = min(dxyzm, dxyz)
            end
            idx => (dxyzm[2], SVector(dxyzm[3:end]...))
        end |> Dict
    dxyzs = map(xyzs) do xyz
        idx2dxyz[LocalNmtDat.pos2idx(xyz, depthres)]
    end
    @info "NaN $(count(isnan, first.(dxyzs)))"
    dxyzs
end

function dxyz2uvw(dxyzs, objflat, objphy; perface::Int = 64)
    samples = Pair[]
    sample_mesh(objflat, objflat.position, objphy.position; perface = perface) do vec
        @assert length(vec) == 5
        push!(samples, SVector(vec[1:2]...) => SVector(vec[3:end]...))
    end
    tree = NearestNeighbors.KDTree(last.(samples))
    nnn = first.(NearestNeighbors.knn(tree, last.(dxyzs), 1)[1])
    map(dxyzs, nnn) do dxyz, nn
        uv = isnan(dxyz[1]) ? (NaN32, NaN32) : first(samples[nn])
        SVector(uv..., dxyz[1])
    end
end

function xyz2uvw(xyzs, depthimg, depthres, objflat, objphy; nneighbor = 1, perface::Int = 64)
    dxyzs = xyz2dxyz(xyzs, depthimg, depthres; nneighbor = nneighbor)
    dxyz2uvw(dxyzs, objflat, objphy; perface = perface)
end

function run!(cfg::AbstractDict)
    init_monkeyrec_meshes!(cfg)
    uvw = cfg["uvw_export"]
    outdir = cfg["outputs"]["interm_dir"]
    dpath = joinpath(outdir, cfg["outputs"]["depth_volume_stem"])
    fpath = joinpath(outdir, cfg["outputs"]["flatmap_stem"])
    isfile(dpath) || error("Missing depth volume JLD2: $dpath")
    isfile(fpath) || error("Missing flatmap JLD2: $fpath")
    img = JLD2.load(dpath, "img")
    res = JLD2.load(dpath, "res")
    objflat = JLD2.load(fpath, "obj")
    objphy = JLD2.load(fpath, "objphy")
    soma_path = expanduser(uvw["soma_csv"])
    isfile(soma_path) || error("Missing soma CSV: $soma_path")
    tbl = CSV.read(soma_path, DataFrame)
    xcol, ycol, zcol = Symbol(uvw["x_col"]), Symbol(uvw["y_col"]), Symbol(uvw["z_col"])
    all(hasproperty(tbl, c) for c in (xcol, ycol, zcol)) || error("CSV must contain columns $xcol, $ycol, $zcol")
    idcol = let x = get(uvw, "id_col", nothing)
        x === nothing || x == "" ? nothing : Symbol(x)
    end
    idcol !== nothing && !hasproperty(tbl, idcol) && error("Missing id column $idcol")
    xyzs = map(eachrow(tbl)) do r
        SVector{3,Float32}(Float32(r[xcol]), Float32(r[ycol]), Float32(r[zcol]))
    end
    perface = Int(get(uvw, "perface_uv", 64))
    nneighbor = Int(get(uvw, "nneighbor", 1))
    uvws = xyz2uvw(xyzs, img, res, objflat, objphy; nneighbor = nneighbor, perface = perface)
    out_path = expanduser(uvw["output_csv"])
    mkpath(dirname(out_path))
    nbad = count(u -> any(isnan, u), uvws)
    @info "Finite UVW $(length(uvws) - nbad) / $(length(uvws))"
    ids = idcol === nothing ? collect(1:length(uvws)) : tbl[:, idcol]
    outdf = DataFrame(;
        id = ids,
        u = [u[1] for u in uvws],
        v = [u[2] for u in uvws],
        w = [u[3] for u in uvws],
    )
    CSV.write(out_path, outdf)
    @info "Wrote" out_path
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=$PROJ $(@__FILE__) config.toml")
    cfg = TOML.parsefile(ARGS[1])
    run!(cfg)
end

main()
