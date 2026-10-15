#!/usr/bin/env julia
#
# Parameterized port of references/.../monkeyrec/2-meshes.jl `flatmap_data(Tag)` target.

import Pkg
const PROJ = joinpath(@__DIR__, "..")
Pkg.activate(PROJ)

using FileIO
using MeshIO
using GeometryBasics
using JLD2
using LinearAlgebra
using NearestNeighbors
using StaticArrays
using TOML

include(joinpath(@__DIR__, "include_monkeyrec_meshes.jl"))
include(joinpath(@__DIR__, "LocalNmtDat.jl"))
using .LocalNmtDat

function build_clipcfg(tag::AbstractString, imgs)
    tag = lowercase(string(strip(tag)))
    if tag == "test"
        return (;
            clipfn = (obj) -> begin
                vmax = findmin(p -> -p[2], obj.position)[2]
                dist1 = dijkstra_vertex(obj, vmax)
                (v) -> dist1[v][1] < 40000
            end,
            rightfn = p -> p[1],
        )
    elseif tag == "lefths"
        return (;
            clipfn = (obj1) -> begin
                nmt = imgs
                charmpos = map(i -> LocalNmtDat.idx2pos(SVector(Tuple(i)), nmt.res), CartesianIndices(nmt.seg)[nmt.charm[:, :, :, 1].>0])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj1, findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 10000
            end,
            rightfn = p -> p[2],
        )
    elseif tag == "leftfrontal"
        return (;
            clipfn = (obj1) -> begin
                nmt = imgs
                charmpos =
                    map(i -> LocalNmtDat.idx2pos(SVector(Tuple(i)), nmt.res), CartesianIndices(nmt.seg)[nmt.charm[:, :, :, 1].==1])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj1, findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 60000
            end,
            rightfn = p -> p[1],
        )
    else
        error("Unknown [flatmap].tag \"$(tag)\" (use test, lefths, or leftfrontal)")
    end
end

function run!(cfg::AbstractDict)
    init_monkeyrec_meshes!(cfg)
    fm = cfg["flatmap"]
    tag = fm["tag"]
    surf = get(fm, "mid_surface", cfg["surfaces"]["lh_mid"])
    outdir = cfg["outputs"]["interm_dir"]
    mkpath(outdir)
    outpath = joinpath(outdir, cfg["outputs"]["flatmap_stem"])
    imgs = if tag == "test" || lowercase(string(strip(tag))) == "test"
        nothing
    else
        LocalNmtDat.load_images(cfg)
    end
    cfg_clip = build_clipcfg(tag, imgs)
    obj05 = FileIO.load(string(surf))
    obj = refine_mesh(obj05)
    obj2 = clip_mesh(cfg_clip.clipfn(obj), obj)
    vmax = findmin(cfg_clip.rightfn, obj2.position)[2]
    obj3 = flatten_mesh(obj2, vmax)
    JLD2.jldsave(outpath, true; obj = obj3, objphy = obj2)
    @info "Wrote flatmap" outpath
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=$PROJ $(@__FILE__) config.toml")
    cfg = TOML.parsefile(ARGS[1])
    run!(cfg)
end

main()
