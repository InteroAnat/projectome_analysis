module zz0atlas


module Allen

import JSON
using ColorTypes
import Images
using StaticArrays


const _hier_url = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"
function get_hier(fn=nothing)
    @info "loading allen annot ..."
    if fn === nothing
        fn = get(essential_files, :json, "/data/dat/allen-1.json")
    end
    jsonstr = open(fn === nothing ? `curl -s $_hier_url` : fn) do io
        read(io, String)
    end
    json = JSON.parse(jsonstr)
    anndata = Dict{Int,NamedTuple}()
    function handle(arr, par)
        for obj in arr
            id = obj["id"]
            col = obj["color_hex_triplet"]
            obj["parent_structure_id"] == par || error("bad par")
            abbr = obj["acronym"]
            nm = obj["name"]
            ord = obj["graph_order"]
            anndata[id] = (par=(par === nothing ? 0 : par), col=parse(Color, "#" * col), abbr=abbr, name=nm, order=ord)
            handle(obj["children"], id)
        end
    end
    handle(json["msg"], nothing)
    anndata
end

const essential_files = NamedTuple([
    :average => "/data/dat/average_template_25_space_3.nrrd"
    :seg => "/data/dat/annotation_25.nrrd"
    :json => "/data/dat/allen-1.json"
    :depth => "/data/dat/laplacian_10.nrrd"
])

function isleft(pos)
    pos[3] < 456 * 25.0f0 / 2
end
function mirror(pos::AbstractArray)
    siz1, siz2p... = size(pos)
    @assert siz1 == 3
    pos = reshape(pos, siz1, :)
    y = similar(pos)
    for i in axes(y, 1)
        if i == 3
            y[i, :] = 456 * 25.0f0 .- pos[i, :]
        else
            y[i, :] = pos[i, :]
        end
    end
    reshape(y, siz1, siz2p...)
end

function load_images()
    ave = Images.load(essential_files.average).data
    seg = Images.load(essential_files.seg).data
    # XXX:
    hier = get_hier(get(essential_files, :json, "/data/dat/allen-1.json"))
    res = SVector(25.0f0, 25.0f0, 25.0f0)
    (average=ave, seg=seg,
        res=res,
        range=SVector(res .* size(ave)...),
        hier,
    )
end

pos2idx(x::AbstractVector, res) = begin
    SVector(ceil.(Int32, x ./ res)...)
end
idx2pos(x::AbstractVector, res) = begin
    SVector((x .- 0.5f0) .* res)
end

function idx2ann(imgs, idx::AbstractVector)
    imgs.seg[idx...]
end
function ann2hier(imgs, ann; stack=Int[])
    hier = imgs.hier
    empty!(stack)
    cur = ann
    while cur != 0
        push!(stack, cur)
        cur = hier[cur].par
    end
    if isempty(stack)
        ""
    else
        string("allen", ("|$(hier[v].abbr)" for v in Iterators.reverse(stack))...)
    end
end

function getter(imgs)
    getter(imgs.hier, imgs)
end
function getter(hier, imgs)
    getvox(img, i) = begin
        img[i...]
    end
    # getvoxm(img, i) = begin
    #     SVector(view(img, i..., :)...)
    # end
    areas = filter(collect(keys(hier))) do id
        res = []
        cur = hier[id]
        while true
            push!(res, cur.abbr)
            cur.par == 0 && break
            cur = hier[cur.par]
        end
        length(res) < 8
        #(id, length(res), join(reverse(res), "/"))
    end
    sort!(areas, lt=(a, b) -> hier[a].order < hier[b].order)
    area2idx = Dict{Int,Int}()
    for (i, a) in enumerate(areas)
        area2idx[a] = i
    end

    function (pos::AbstractArray)
        siz = size(pos)
        @assert length(siz) > 0 && siz[1] == 3 && eltype(pos) <: AbstractFloat
        pos = reshape(pos, siz[1], :)
        res = zeros(Float32, 128 * 6, size(pos, 2))
        shift = 10
        hh = []
        hhh = map(enumerate(eachcol(pos))) do (col, p)
            p = SVector(p...)
            i = pos2idx(p, imgs.res)
            i = checkbounds(Bool, imgs.average, i...) ? i : map((_) -> Int32(1), i)
            seg = idx2ann(imgs, i)

            res[1, col] = seg > 0 ? 1.0f0 : 0.0f0
            res[shift, col] = 0.123123f0
            for cur in hh
                idx = get(area2idx, cur, 0)
                if idx != 0
                    res[idx+shift, col] = 1.0f0
                end
            end
            ann2hier(imgs, seg; stack=hh)
        end
        res[2, :] = res[get(area2idx, 315, 0)+shift, :]
        res[3, :] = res[get(area2idx, 512, 0)+shift, :]
        res = reshape(res, size(res, 1), siz[2:end]...)
        res, hhh
    end
end

function center()
    SVector(6600.0, 4000.0, 5700.0)
end

using ProgressMeter
function volume(imgs)
    volume(imgs.hier, imgs)
end
function volume(hier, imgs)
    cnttag = Dict{Int,Int}()
    idxtag = Dict{Int,Any}()
    @showprogress for ind in CartesianIndices(imgs.seg)
        i = Tuple(ind)
        t = imgs.seg[i...]
        pc = get(cnttag, t, 0)
        if pc == 0
            idxtag[t] = SVector(i)
        end
        cnttag[t] = pc + 1
    end
    tag2pos = map(collect(idxtag)) do (t, i)
        t => idx2pos(i, imgs.res)
    end
    cnts = Dict{String,Int}()
    foreach(first.(tag2pos), getter(hier, imgs)(reduce(hcat, last.(tag2pos)))[2]) do tag, h
        h == "" && return
        c = cnttag[tag]
        cnts[h] = get(cnts, h, 0) + c
        while contains('|')(h)
            h = replace(h, r"\|[^|]+$" => "")
            cnts[h] = get(cnts, h, 0) + c
        end
    end
    cnts = map(collect(cnts)) do (a, b)
        a => b * prod(imgs.res) / 1e9
    end
    Dict(cnts)
end

end

module NmtDat

import NIfTI
import DelimitedFiles
using StaticArrays

const essential_files = NamedTuple([
    :cerebellum => "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/supplemental_masks/NMT_v2.0_sym_cerebellum_mask.nii.gz"
    :charm => "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/CHARM_in_NMT_v2.0_sym.nii.gz"
    :charm_key => "/data/dat/NMT_v2.0_sym/tables_CHARM/CHARM_key_all.txt"
    :nmr => "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz"
    :sarm => "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/SARM_in_NMT_v2.0_sym.nii.gz"
    :sarm_key => "/data/dat/NMT_v2.0_sym/tables_SARM/SARM_key_all.txt"
    :seg => "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_segmentation.nii.gz"
])

function isleft(pos)
    pos[1] < 256 * 250.0f0 / 2
end

function mirror(pos::AbstractArray)
    siz1, siz2p... = size(pos)
    @assert siz1 == 3
    y = similar(pos)
    pos2 = reshape(pos, siz1, :)
    y2 = reshape(y, siz1, :)
    for i in axes(y, 1)
        if i == 1
            y2[i, :] = 256 * 250.0f0 .- pos2[i, :]
        else
            y2[i, :] = pos2[i, :]
        end
    end
    y
end

function center()
    SVector(32000.0f0, 39000.0f0, 25000.0f0)
end

function load_images()
    nmr = NIfTI.niread(essential_files.nmr).raw
    # int16
    cb_img = NIfTI.niread(essential_files.cerebellum).raw .> 0
    # 0/1
    charm = reshape(NIfTI.niread(essential_files.charm).raw, size(nmr)..., :)
    # (256, 312, 200, 6)
    # 239
    sarm = reshape(NIfTI.niread(essential_files.sarm).raw, size(nmr)..., :)
    # max=318
    seg = NIfTI.niread(essential_files.seg).raw
    #  0x00 0x03 0x04 0x01 0x05 0x02
    sarm_key = DelimitedFiles.readdlm(essential_files.sarm_key, '\t', header=true)[1]
    charm_key = DelimitedFiles.readdlm(essential_files.charm_key, '\t', header=true)[1]
    res = SVector(250.0f0, 250.0f0, 250.0f0)
    tag2tbl = map((:sarm => sarm_key, :charm => charm_key)) do (tag, tbl)
        tbl = map(eachrow(tbl)) do r
            r[1] => r[2]
        end |> Dict
        tag => tbl
    end |> NamedTuple
    (nmr=nmr, cb_msk=cb_img, charm=charm, charm_key=charm_key, sarm=sarm, sarm_key=sarm_key, seg=seg,
        res=res,
        range=SVector(res .* size(nmr)...),
        tag2tbl,
    )
end

function normalize_tissue(nmt)
    map(nmt.seg) do vv
        vv = if vv < 0.5
            0.0f0
        elseif vv < 1.5
            0.03f0
        elseif vv < 2.5
            1.0f0
        elseif vv < 3.5
            0.5f0
        elseif vv < 4.5
            0.3f0
        elseif vv < 5.5
            0.05f0
        else
            @assert false
        end
        Float32(vv > 0.1 ? 1.0 : 0.0)
    end
end

pos2idx(x::AbstractVector, res) = begin
    SVector(ceil.(Int32, x ./ res)...)
end
idx2pos(x::AbstractVector, res) = begin
    SVector((x .- 0.5f0) .* res)
end

function idx2ann(imgs, idx::AbstractVector)
    getvoxm(img, i) = begin
        SVector(view(img, i..., :)...)
    end
    (; charm=getvoxm(imgs.charm, idx), sarm=getvoxm(imgs.sarm, idx))
end
function ann2hier(imgs, ann; stack=Int[])
    gethier(h, t) = begin
        tbl = imgs.tag2tbl[t]
        string("nmt|", t, ("|$(tbl[v])" for v in unique(h))...)
    end
    if ann.charm[1] != 0
        gethier(ann.charm, :charm)
    elseif ann.sarm[1] != 0
        gethier(ann.sarm, :sarm)
    else
        ""
    end
end

function getter(imgs)
    getvox(img, i) = begin
        img[i...]
    end
    tissues = SVector(sort(unique(imgs.seg[:]))...)
    charms = SVector(sort(unique(imgs.charm_key[:, 1]))...)
    sarms = SVector(sort(unique(imgs.sarm_key[:, 1]))...)
    function (pos::AbstractArray)
        siz = size(pos)
        @assert length(siz) > 0 && siz[1] == 3 && eltype(pos) <: AbstractFloat
        pos = reshape(pos, siz[1], :)
        res0 = map(eachcol(pos)) do p
            p = SVector(p...)
            i = pos2idx(p, imgs.res)
            i = checkbounds(Bool, imgs.nmr, i...) ? i : map((_) -> Int32(1), i)
            seg = getvox(imgs.seg, i)
            ann = idx2ann(imgs, i)
            charm, sarm = ann[(:charm, :sarm)]
            hh = ann2hier(imgs, ann)
            hh => SVector(
                seg > 0 ? 1.0f0 : 0.0f0,
                0.0f0,
                0.0f0,
                0.0f0,
                0.0f0,
                0.123123f0,
                map(t -> seg === t ? 1.0f0 : 0.0f0, tissues)...,
                0.123123f0,
                getvox(imgs.cb_msk, i) ? 1.0f0 : 0.0f0,
                0.123123f0,
                map(c -> in(c, charm) ? 1.0f0 : 0.0f0, charms)...,
                0.123123f0,
                map(s -> in(s, sarm) ? 1.0f0 : 0.0f0, sarms)...,
            )
        end
        res = reduce(hcat, map(last, res0))
        res = reshape(res, size(res, 1), siz[2:end]...)
        out = zeros(Float16, 128 * 6, size(res, 2))
        out[axes(res, 1), :] .= res
        out[2, :] = out[9, :]
        out[3, :] = out[14, :]
        out, map(first, res0)
    end
end

using ProgressMeter
function volume(imgs)
    cntall = 0
    cnttag = Dict{SVector,Int}()
    idxtag = Dict{SVector,Any}()
    @showprogress for ind in CartesianIndices(imgs.seg)
        i = Tuple(ind)
        if imgs.seg[ind] != 0
            cntall += 1
        end
        t = SVector(view(imgs.charm, i..., :)..., view(imgs.sarm, i..., :)...)
        pc = get(cnttag, t, 0)
        if pc == 0
            idxtag[t] = SVector(i)
        end
        cnttag[t] = pc + 1
    end
    tag2pos = map(collect(idxtag)) do (t, i)
        t => NmtDat.idx2pos(i, imgs.res)
    end
    cnts = Dict{String,Int}()
    foreach(first.(tag2pos), NmtDat.getter(imgs)(reduce(hcat, last.(tag2pos)))[2]) do tag, h
        h == "" && return
        c = cnttag[tag]
        cnts[h] = get(cnts, h, 0) + c
        while contains('|')(h)
            h = replace(h, r"\|[^|]+$" => "")
            cnts[h] = get(cnts, h, 0) + c
        end
    end
    cnts["nmt"] = cntall
    cnts = map(collect(cnts)) do (a, b)
        a => b * prod(imgs.res) / 1e9
    end
    Dict(cnts)
end

end

end # module zz0atlas
