#!/usr/bin/env julia
#
# Parameterized port of references/.../monkeyrec/1-unibrain.jl depth-mapping-mac target.

import Pkg
const PROJ = joinpath(@__DIR__, "..")
Pkg.activate(PROJ)

using FileIO
using MeshIO
using Images
using Interpolations
using JLD2
using LinearAlgebra
using ProgressMeter
using StaticArrays
using Statistics
using TOML

include(joinpath(@__DIR__, "include_monkeyrec_meshes.jl"))
include(joinpath(@__DIR__, "LocalNmtDat.jl"))
using .LocalNmtDat

function run!(cfg::AbstractDict)
    init_monkeyrec_meshes!(cfg)
    outdir = cfg["outputs"]["interm_dir"]
    mkpath(outdir)
    mpath = joinpath(outdir, cfg["outputs"]["depth_model_stem"])
    vpath = joinpath(outdir, cfg["outputs"]["depth_volume_stem"])
    isfile(mpath) || error("Missing depth model JLD2: $mpath (run run_depth_model.jl first)")
    df = JLD2.load(mpath, "df")
    surf = cfg["surfaces"]
    midp = cfg["surfaces"]["lh_mid"]
    obj = FileIO.load(midp)
    @assert reduce(hcat, obj.position) == reduce(hcat, df.pos) "Mid surface vertices must match depth model df.pos (same lh_mid file as training)."
    imgs = LocalNmtDat.load_images(cfg)
    charm = imgs.charm[1:end÷2, :, :, 1]
    img = charm
    res = imgs.res
    ratio = Int(cfg["depth_volume"]["charm_resize_ratio"])
    perface = Int(cfg["depth_volume"]["perface_sample"])
    img = Images.imresize(img; ratio = ratio, method = Interpolations.Constant())
    res = res ./ ratio
    buf = map(v -> (true, Inf32, NaN32, NaN32, NaN32, NaN32), img)
    hits = Any[]
    sample_mesh(obj, obj.position, df.dir, df.ti, df.te; perface = perface) do vec
        @assert length(vec) == 8
        cent = SVector(vec[1:3]...)
        dir = SVector(vec[4:6]...)
        ti, te = vec[7:8]
        empty!(hits)
        @assert te >= 0 "$te >=0"
        @assert ti <= 0 "$ti <=0"
        te2, ti2 = map((0:60:te*1.5f0, 0:-60:ti*1.5f0)) do rng
            outside = false
            tt = last(rng)
            for t in rng
                p = cent + dir * t
                i = LocalNmtDat.pos2idx(p, res)
                ii = LocalNmtDat.pos2idx(p, imgs.res)
                if !(checkbounds(Bool, charm, ii...) && charm[ii...] != 0) && !outside
                    tt = t
                    outside = true
                end
                pp = LocalNmtDat.idx2pos(i, res)
                dd = norm(pp - cent)
                push!(hits, (outside, i, dd, t))
            end
            tt
        end
        if te2 > ti2
            for (outside, i, dd, t) in hits
                if checkbounds(Bool, buf, i...)
                    buf[i...] = min(buf[i...], (outside, dd, clamp((t - te2) / (ti2 - te2), 0, 1), cent...))
                end
            end
        end
    end
    img = cat(map(k -> map(v -> v[k], buf), 3:6)...; dims = 4)
    img = cat(img, img[end:-1:1, :, :, :], dims = 1)
    JLD2.jldsave(vpath, true; img, res)
    @info "Wrote depth volume" vpath
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=$PROJ $(@__FILE__) config.toml")
    cfg = TOML.parsefile(ARGS[1])
    run!(cfg)
end

main()
