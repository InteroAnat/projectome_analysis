"""
TOML-driven stand-in NMT volume bundle with `load_images()` return fields compatible with
`zz0atlas.NmtDat.load_images()` for downstream scripts (depth volume + flatmap clip masks).
"""
module LocalNmtDat

using NIfTI
using DelimitedFiles
using StaticArrays
using TOML

export load_images, pos2idx, idx2pos, idx2ann, ann2hier

function _parse_delim(s::AbstractString)
    s == "\\t" && return '\t'
    length(s) == 1 && return first(s)
    error("config nmt.key_delim must be a single character or \"\\\\t\"; got: $(repr(s))")
end

function _read_key_matrix(path::AbstractString, delim)
    readdlm(path, delim, header = true)[1]
end

function _rows_to_pairdict(data::AbstractMatrix)
    d = Dict{Any,Any}()
    for i in axes(data, 1)
        d[data[i, 1]] = data[i, 2]
    end
    d
end

"""
    load_images(dict) -> NamedTuple

`dict` is the parsed `config.toml` root (must contain `[nmt]` table). Paths may be plain strings.
"""
function load_images(cfg::AbstractDict)
    nmt = cfg["nmt"]
    delim = _parse_delim(get(nmt, "key_delim", "\t"))
    nmr = NIfTI.niread(nmt["nmr"]).raw
    cb_img = NIfTI.niread(nmt["cerebellum"]).raw .> 0
    charm = reshape(NIfTI.niread(nmt["charm"]).raw, size(nmr)..., :)
    sarm = reshape(NIfTI.niread(nmt["sarm"]).raw, size(nmr)..., :)
    seg = NIfTI.niread(nmt["seg"]).raw
    sarm_key = _read_key_matrix(nmt["sarm_key"], delim)
    charm_key = _read_key_matrix(nmt["charm_key"], delim)
    res = let r = collect(get(nmt, "res", [250.0, 250.0, 250.0]))
        length(r) == 3 || error("nmt.res must have length 3")
        SVector{3,Float32}(Float32.(r)...)
    end
    tag2tbl = (; charm = _rows_to_pairdict(charm_key), sarm = _rows_to_pairdict(sarm_key))
    (
        nmr = nmr,
        cb_msk = cb_img,
        charm = charm,
        charm_key = charm_key,
        sarm = sarm,
        sarm_key = sarm_key,
        seg = seg,
        res = res,
        range = SVector(res .* size(nmr)...),
        tag2tbl = tag2tbl,
    )
end

load_images(path::AbstractString) = load_images(TOML.parsefile(path))

# --- Same API as zz0atlas.NmtDat (subset needed for depth + flatmap) ---

pos2idx(x::AbstractVector, res) = SVector(ceil.(Int32, x ./ res)...)
idx2pos(x::AbstractVector, res) = SVector((x .- 0.5f0) .* res)

function idx2ann(imgs, idx::AbstractVector)
    getvoxm(img, i) = SVector(view(img, i..., :)...)
    (; charm = getvoxm(imgs.charm, idx), sarm = getvoxm(imgs.sarm, idx))
end

function ann2hier(imgs, ann; stack = Int[])
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

end # module
