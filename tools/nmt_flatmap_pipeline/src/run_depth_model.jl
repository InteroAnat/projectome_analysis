#!/usr/bin/env julia
#
# Parameterized port of references/.../monkeyrec/1-unibrain.jl depth-mapping-mac-model target.

import Pkg
const PROJ = joinpath(@__DIR__, "..")
Pkg.activate(PROJ)

using CUDA
using DataFrames
using FileIO
using MeshIO
using Flux
using GLMakie
using GeometryBasics
using JLD2
using LinearAlgebra
using NearestNeighbors
using ProgressMeter
using StaticArrays
using TOML

macro check_size(ex, sz)
    esc(ex)
end

function sample_objs_pointcloud(fn::Function, objs, perface::Int)
    for obj in objs
        samps = map(eachcol(rand(Float32, 2, perface))) do (l1, l2)
            r1 = 1 - sqrt(l1)
            r2 = (1 - r1) * l2
            r3 = 1 - r1 - r2
            SVector(r1, r2, r3)
        end
        for f in GeometryBasics.faces(obj)
            a, b, c = map(i -> obj.position[i], Tuple(f))
            for r123 in samps
                p = r123[1] * a + r123[2] * b + r123[3] * c
                fn(p)
            end
        end
    end
end

function gen_tree(objs, perface::Int)
    pts = SVector{3,Float32}[]
    sample_objs_pointcloud(objs, perface) do p
        push!(pts, p)
    end
    pts = reduce(hcat, pts)
    (tree = NearestNeighbors.KDTree(pts), pts = pts)
end

function run!(cfg::AbstractDict)
    dev = CUDA.functional() ? Flux.gpu : identity
    surf = cfg["surfaces"]
    dm = cfg["depth_model"]
    outdir = cfg["outputs"]["interm_dir"]
    mkpath(outdir)
    outpath = joinpath(outdir, cfg["outputs"]["depth_model_stem"])
    objs = (; outer = [FileIO.load(surf["lh_gray"])], middle = [FileIO.load(surf["lh_mid"])], inner = [FileIO.load(surf["lh_white"])])
    perface = Int(dm["perface_sample"])
    innertree, outertree = gen_tree(objs.inner, perface), gen_tree(objs.outer, perface)

    mid = hcat([reduce(hcat, obj.position) for obj in objs.middle]...)
    adjs = SVector{2,Int}[]
    let offset = 0
        for obj in objs.middle
            for f in GeometryBasics.faces(obj)
                a, b, c = f .+ offset
                push!(adjs, sort(SVector(a, b)))
                push!(adjs, sort(SVector(b, c)))
                push!(adjs, sort(SVector(c, a)))
            end
            offset += length(obj.position)
        end
        adjs = reduce(hcat, unique(adjs))
    end
    adjs = adjs |> dev
    mid = mid |> dev
    nvert = size(mid, 2)
    get_ends(dir, len, sel) = mid[:, sel] .+ dir[:, sel] .* len[:, sel]
    calc_tite(tite, sgn) = (sgn * 100.0f0) .* Flux.softplus(tite)
    batch = Int(dm["batch_vertices"])
    kk = Int(dm["knn_inner"])
    function loss_dist(pts, tree, K)
        pts2 = Flux.ignore_derivatives() do
            inds, dists = NearestNeighbors.knn(tree.tree, pts |> Flux.cpu, kk)
            tree.pts[:, reduce(vcat, inds)] |> dev
        end
        @check_size pts2 (3, :)
        @check_size pts (3, :)
        pts = reshape(pts, 3, 1, :)
        pts2 = reshape(pts2, 3, kk, :)
        dpts = sum((pts .- pts2) .^ 2, dims = 1)
        Flux.mean(dpts) / 1.0f4
    end
    function loss(m, K, obs)
        sel = Flux.ignore_derivatives() do
            rand(axes(mid, 2), batch) |> dev
        end
        len2 = sum(m.dir .^ 2, dims = 1)
        mti = calc_tite(m.ti, -1)
        mte = calc_tite(m.te, 1)
        l0 = Flux.mean(m.ti .^ 2 .+ m.te .^ 2)
        inners = get_ends(m.dir, mti, :)
        outers = get_ends(m.dir, mte, :)
        l1 = loss_dist(inners[:, sel], innertree, K)
        l2 = loss_dist(outers[:, sel], outertree, K)
        l3 =
            100.0f0 * (
                1.0f0 - Flux.mean(
                    sum(m.dir[:, adjs[1, :]] .* m.dir[:, adjs[2, :]], dims = 1) ./
                    sqrt.(len2[:, adjs[1, :]] .* len2[:, adjs[2, :]]),
                )
            )
        l4 =
            Flux.mean(sum((inners[:, adjs[1, :]] .- inners[:, adjs[2, :]]) .^ 2, dims = 1)) /
            (K < 3000 ? 1.0f4 : 1.0f5)
        l5 =
            Flux.mean(sum((outers[:, adjs[1, :]] .- outers[:, adjs[2, :]]) .^ 2, dims = 1)) /
            (K < 3000 ? 1.0f4 : 1.0f5)
        l = l0 + l1 + l2 + l3 + l4 + l5
        Flux.ignore_derivatives() do
            if K % 10 == 0
                e1 = get_ends(m.dir, mti, :)
                e2 = get_ends(m.dir, mte, :)
                obs.vecs[] = reshape(vcat(e1, e2), 3, :) |> Flux.cpu
                obs.from[] = e1 |> Flux.cpu
                obs.to[] = e2 |> Flux.cpu
                @info "$K, loss $l = $l0 [$l1 $l2] $l3 [$l4 $l5]"
            end
        end
        l
    end

    obs = (; vecs = Ref{Any}(), from = Ref{Any}(), to = Ref{Any}())
    model = (; dir = rand(Float32, 3, nvert) .- 0.5f0, te = rand(Float32, 1, nvert), ti = -rand(Float32, 1, nvert)) |> dev
    loss(model, 0, obs)

    obs = let
        fig = GLMakie.Figure()
        display(fig)
        ax = GLMakie.LScene(fig[1, 1])
        foreach(o -> GLMakie.mesh!(ax, o, color = :orange), objs.inner)
        GLMakie.linesegments!(ax, reshape(mid[:, adjs], 3, :) |> Flux.cpu, color = :red)
        vecs = GLMakie.Observable(obs.vecs[])
        from = GLMakie.Observable(obs.from[])
        to = GLMakie.Observable(obs.to[])
        GLMakie.linesegments!(ax, vecs, color = :gray)
        GLMakie.scatter!(ax, from, color = :red)
        GLMakie.scatter!(ax, to, color = :green)
        (; vecs = vecs, from = from, to = to)
    end

    optst = Flux.setup(Flux.Adam(0.015), model)
    niter = Int(dm["train_iters"])
    @showprogress for K in 1:niter
        l, g = Flux.withgradient(m -> loss(m, K, obs), model)
        Flux.update!(optst, model, g[1])
    end
    model.ti .= calc_tite(model.ti, -1)
    model.te .= calc_tite(model.te, 1)
    model = model |> Flux.cpu
    mid_cpu = mid |> Flux.cpu
    df =
        map(eachcol(model.dir), eachcol(model.ti), eachcol(model.te), eachcol(mid_cpu)) do dir, ti, te, pos
            dir = SVector(dir...)
            ldir = norm(dir)
            dir = dir / ldir
            ti = ti[1] * ldir
            te = te[1] * ldir
            pos = SVector(pos...)
            (; dir, ti, te, pos)
        end |> DataFrame
    JLD2.jldsave(outpath, true; df)
    @info "Wrote depth model" outpath
end

function main()
    length(ARGS) >= 1 || error("Usage: julia --project=$PROJ $(@__FILE__) config.toml")
    cfg = TOML.parsefile(ARGS[1])
    run!(cfg)
end

main()
