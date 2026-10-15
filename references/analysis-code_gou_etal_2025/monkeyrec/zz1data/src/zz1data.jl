module zz1data

using DataFrames

import Gapr
using GeometryBasics

using LinearAlgebra
using ProgressMeter


using StaticArrays

using Random
import Optim
import StatsBase

import zz0atlas: NmtDat, Allen

export common_regions, istypit, is_projcol
export cached_result
export load_swcs_as_segs
export @check_size

import Distributions
import HypothesisTests
function prop_test(x, n, p0)
    x2 = mapreduce(+, (x, n - x), (p0, 1 - p0)) do c, q
        (abs(c - n * q) - 0.5)^2 / (n * q)
    end
    p = HypothesisTests.pvalue(Distributions.Chisq(1), x2, tail=:right)
    s = sign(x / n - p0)
    (; p, s)
end

arbor_size(v) = cbrt(6 / π * v)


function mirrorby(Mod)
    (p, ref=nothing) -> begin
        Mod === nothing && return p
        Mod.isleft(ref === nothing ? p : ref) && return p
        Mod.mirror(p)
    end
end
function mirrorswcby(Mod, ref=nothing)
    (swc) -> begin
        Mod === nothing && return swc
        Mod.isleft(ref !== nothing ? ref : swc[1].pos) && return swc
        for n in swc.nodes
            n.pos = Mod.mirror(n.pos)
        end
        swc
    end
end

export matchhier
function matchhier(hiers...)
    h -> begin
        hh = iterate(hiers)
        k = 0
        N = length(h)
        while hh !== nothing
            i = findfirst(hh[1], h[k+1:end])
            i === nothing && return false
            if i.start + k != 1 && h[i.start+k-1] != '|'
                k = i.start + k
                continue
            end
            if i.stop + k != N && h[i.stop+k+1] != '|'
                k = i.start + k
                continue
            end
            k = i.stop + k
            hh = iterate(hiers, hh[2])
        end
        true
    end
end

using StatsBase
function vote(df, keygrp, keydedup)
    vv = Dict()
    for r in eachrow(df)
        k = r[keygrp]
        ss = get!(() -> Set(), vv, k)
        push!(ss, r[keydedup])
    end
    sort(map(p -> p.first => length(p.second), collect(vv)), by=last)
end

function find_best_coor_axis(xvec::AbstractVector, yvec::AbstractVector; aligned_x=0, aligned_y=0)
    Random.seed!(0)
    xvec = reduce(hcat, xvec)
    yvec = reduce(hcat, yvec)
    M = size(xvec, 1)
    N = size(yvec, 1)
    obj_func(x) = begin
        @assert length(x) == M + N
        xx = reshape(reshape(x[1:M], 1, :) * xvec, :)
        yy = reshape(reshape(x[M+1:end], 1, :) * yvec, :)
        #-StatsBase.corspearman(xx, yy)
        -StatsBase.cor(xx, yy)
    end
    get_vec(L, aligned, C) = begin
        tmp = zeros(L, C)
        if aligned == 0
            tmp .= randn(L, C)
        elseif aligned > 0
            tmp[aligned, :] .= rand((-1, 1), C)
        else
            for (j, a) in enumerate(rand(1:2*L, size(tmp, 2)))
                i, e = divrem(a - 1, 2)
                tmp[i+1, j] = e > 0 ? 1 : -1
            end
        end
        tmp
    end
    x00 = vcat(get_vec(M, aligned_x, 30000), get_vec(N, aligned_y, 30000))
    minv, jj = findmin(j -> obj_func(view(x00, :, j)), axes(x00, 2))
    @warn "$minv"
    xx = if aligned_x == 0 && aligned_y == 0
        res = Optim.optimize(obj_func, x00[:, jj], Optim.BFGS())
        display(res)
        Optim.minimizer(res)
    else
        x00[:, jj]
    end
    get_norm(v) = begin
        round.(v ./ norm(v), digits=2)
    end
    get_norm.((SVector(xx[1:M]...), SVector(xx[M+1:end]...)))
end

# manual selections, suspectable to data change
function select_neurons(tag::Symbol)
    (
        all_itc=r -> istypit(r.typ) && in(r.cls, 1:5),
        all_its=r -> istypit(r.typ) && in(r.cls, 6:9),
        all_ct=r -> !istypit(r.typ) && in(r.cls, 10:10),
        all_pt=r -> !istypit(r.typ) && in(r.cls, 11:12),
        all_iti=r -> istypit(r.typ) && in(r.cls, 13:32),
        all_demo=r -> in(r.cls, (1, 8, 11, 15)),
        long_it_ipsi_lip=r -> istypit(r.typ) && in(r.cls, (15,)),
        long_it_ipsi_pcc=r -> istypit(r.typ) && in(r.cls, (32,)),
        long_it_ipsi_te=r -> istypit(r.typ) && in(r.cls, (13,)),
        long_it_ipsi_stsd=r -> istypit(r.typ) && in(r.cls, (14,)),
        long_it_ipsi=r -> istypit(r.typ) && in(r.cls, (13, 14, 15, 32)),
        long_it_ipsi_temp=r -> istypit(r.typ) && in(r.cls, (13, 14)),
        long_it_ipsi_pari=r -> istypit(r.typ) && in(r.cls, (15, 32)),
        long_it_ipsi_g97=r -> r.inj == "G97-1bL" && r.cls in (13, 15) && istypit(r.typ),
        pt_strict=r -> r.typ == "PT" && in(r.cls, (11, 12)),
        ct_strict=r -> r.typ == "CT" && in(r.cls, (10,)),
        itc_strict=r -> r.typ == "ITc" && in(r.cls, (1, 2, 3, 4, 5)),
        its_strict=r -> r.typ == "ITs" && in(r.cls, (6, 7, 8, 9)),
        its_ipsi_strict=r -> r.typ == "ITs" && in(r.cls, (7, 8, 9)),
        pt_mus=r -> r.typ == "PT",
        ct_mus=r -> r.typ == "CT",
    )[tag]
end
# end manual selections

function calc_twoside_projection!(df)
    curcols = filter(is_projcol, names(df))
    cols = sort(unique([replace(c, r"^(\w+)[IC]\|" => s"\1|") for c in curcols]))
    for col in cols
        val = zeros(Float64, size(df, 1))
        for ic in (:I, :C)
            colic = replace(col, r"^(\w+)" => SubstitutionString("\\1$ic"))
            if colic in curcols
                val .= val .+ df[:, colic]
            end
        end
        insertcols!(df, col => val)
    end
end

function get_flatmap_highlights()
    [
        "nmt|charm|Frontal" => 1.0
        "nmt|charm|Frontal|lat_PFC" => :gray80
        "nmt|charm|Frontal|lat_PFC|dlPFC" => 0.75
        "nmt|charm|Frontal|lat_PFC|vlPFC" => 0.75
        "nmt|charm|Frontal|lat_PFC|area_8A|area_8Ad"
        "nmt|charm|Frontal|lat_PFC|area_8A|area_8Av"
        "nmt|charm|Frontal|lat_PFC|dlPFC|area_8B|area_8Bd"
        "nmt|charm|Frontal|lat_PFC|dlPFC|area_8B|area_8Bm"
        "nmt|charm|Frontal|lat_PFC|dlPFC|area_8B|area_8Bs"
        "nmt|charm|Frontal|lat_PFC|dlPFC|rostral_dlPFC|area_46d"
        "nmt|charm|Frontal|lat_PFC|dlPFC|rostral_dlPFC|area_9|area_9d"
        "nmt|charm|Frontal|lat_PFC|dlPFC|rostral_dlPFC|area_9|area_9m"
        "nmt|charm|Frontal|lat_PFC|vlPFC|caudal_vlPFC|area_44"
        "nmt|charm|Frontal|lat_PFC|vlPFC|caudal_vlPFC|area_45|area_45a"
        "nmt|charm|Frontal|lat_PFC|vlPFC|caudal_vlPFC|area_45|area_45b"
        "nmt|charm|Frontal|lat_PFC|vlPFC|rostral_vlPFC|area_12r/l|area_12l"
        "nmt|charm|Frontal|lat_PFC|vlPFC|rostral_vlPFC|area_12r/l|area_12r"
        "nmt|charm|Frontal|lat_PFC|vlPFC|rostral_vlPFC|area_46v/f|area_46f"
        "nmt|charm|Frontal|lat_PFC|vlPFC|rostral_vlPFC|area_46v/f|area_46v"
        "nmt|charm|Frontal|motor" => 0.75
        "nmt|charm|Frontal|ACgG" => 0.75
        "nmt|charm|Frontal|OFC" => 0.75
        "nmt|charm|Occipital" => 1.0
        "nmt|charm|Parietal" => 1.0
        "nmt|charm|Temporal" => 1.0
    ]
end

is_projcol(col) = startswith(col, "nmt") || startswith(col, "allen")
istypit(r) = startswith("IT")(r)
istypit(r::DataFrameRow) = startswith("IT")(r.typ)

function common_regions(select=:)
    all = include("monkeytemp/wyz-upload/julia/mouse-monkey-brain-regions.jl")
    reduce(vcat, all[select])
end

function load_swcs_as_segs(swcs::AbstractVector{<:AbstractString}, somacenter=nothing; Atlas=NmtDat)
    isempty(swcs) && return []
    segs = @showprogress "load swcs" map(enumerate(swcs)) do (nrn, f)
        swc = Gapr.load_swc(f)
        lrroot = Atlas.isleft(somacenter === nothing ? swc[1].pos : somacenter[nrn])
        map(swc.nodes) do n
            row = n.ind
            len = if n.parind == 0
                0.0f0
            else
                pp = swc[n.parind].pos
                norm(pp - n.pos)
            end
            ic = Atlas.isleft(n.pos) == lrroot ? :I : :C
            (; nrn, row, n.id, n.pos, len, n.typ, ic)
        end
    end
    reduce(vcat, segs)
end

function calc_ipsicontra_len(dfx, arborx; Atlas=NmtDat)
    @info "load swcs"
    segs = load_swcs_as_segs(dfx.path; Atlas)
    cnts = Dict{String,Tuple}()
    for s in segs
        k = dfx[s.nrn, :path]
        s.typ != 2 && continue
        v = ((s.ic == :I ? s.len : 0.0), (s.ic == :C ? s.len : 0.0), 0.0, 0.0)
        cnts[k] = get(cnts, k, (0.0, 0.0, 0.0, 0.0)) .+ v
    end
    if !("soma_pos" in names(arborx))
        arborx = leftjoin(arborx, dfx[:, [:path, :soma_pos]], on=:path) |> dropmissing!
    end
    @info "load arbors"
    segs = load_swcs_as_segs(arborx.arborpath, arborx.soma_pos)
    for s in segs
        k = arborx[s.nrn, :path]
        v = (0.0, 0.0, (s.ic == :I ? s.len : 0.0), (s.ic == :C ? s.len : 0.0))
        cnts[k] = get(cnts, k, (0.0, 0.0, 0.0, 0.0)) .+ v
    end
    map(p -> (v = p.second; (; path=p.first, lenipsi=v[1], lencontra=v[2], arboripsi=v[3], arborcontra=v[4])), collect(cnts)) |> DataFrame
end


function set_pos_rawxyz!(swc, rawinfo)
    for n in swc.nodes
        n.pos = rawinfo[n.id].rawxyz
    end
end

export cached_result
function cached_result(f::Function, tag, args...)
    global xxxcached_result
    tag = (tag, hash.(args)...)
    try
        xxxcached_result[tag]
    catch
        v = f(args...)
        try
            xxxcached_result[tag] = v
        catch
            xxxcached_result = Dict{Tuple,Any}()
            xxxcached_result[tag] = v
        end
        v
    end |> deepcopy
end

function twopasshclust(distm; K=20, kwargs...)
    h1 = Clustering.hclust(distm; kwargs...)
    cls = Clustering.cutree(h1, k=K)
    init = Int[findfirst(==(k), cls) for k in 1:K]
    @assert cls[init] == 1:K
    cls2 = Clustering.kmedoids(distm, K, init=init).assignments
    foreach(cls, cls2) do c1, c2
        @info "$c1 $c2"
    end
    #@assert cls==cls2
    distm = deepcopy(distm)
    for idx in CartesianIndices(distm)
        i, j = Tuple(idx)
        if cls2[i] != cls2[j]
            distm[idx] += 2000
        end
    end
    Clustering.hclust(distm; kwargs...)
end

macro check_size(arr, siz)
    expr = "$__source__ size($arr) == $siz"
    quote
        tgt = $(esc(siz))
        s = size($(esc(arr)))
        msg = "failed size check $($expr): $s"
        length(tgt) != length(s) && error(msg)
        for i in 1:length(tgt)
            if tgt[i] != Colon() && tgt[i] != s[i]
                error(msg)
            end
        end
    end
end

end # module zz1data
