using StaticArrays
using LinearAlgebra
using CodecZlib
using GeometryBasics

import SparseArrays
import DataStructures

export plotswc, plotswc!
function plotswc end
function plotswc! end

export SwcNodePos, SwcNode, SwcNodeAnnot, SwcTableData, SwcTable
export load_swc!, load_swc, split_swc, sort_swc!
export eachnode, eachannot, eachloop, eachnodeattr

#const SwcNodePos = Point3d
const SwcNodePos = SVector{3,Float64}

# XXX: use @enum for node types

"""
    SwcNode

Basic data fields that are compatible to the SWC file format.
Note that, parent node index (`parind`) is used instead of parent node ID.
"""
mutable struct SwcNode
    id::Int64
    parind::Int64
    pos::SwcNodePos
    typ::Int32
    rad::Float32
end
Base.propertynames(_::SwcNode, _::Bool) =
    (@inline; (:type, fieldnames(SwcNode)...))
function Base.getproperty(node::SwcNode, name::Symbol)
    @inline
    name in (:type,) && return getfield(node, :typ)
    name in (:radius,) && return getfield(node, :rad)
    name in (:paridx,) && return getfield(node, :parind)
    name in (:parindex,) && return getfield(node, :parind)
    name in (:typename,) && return begin
        t = getfield(node, :typ)
        return get([:undefined, :soma, :axon, :dendrite, :apical_dendrite], t + 1, :other)
    end
    getfield(node, name)
end
Base.setproperty!(node::SwcNode, name::Symbol, v) = begin
    @inline
    name in (:type,) && return _set_field_with_conv(node, :typ, v)
    name in (:radius,) && return _set_field_with_conv(node, :rad, v)
    _set_field_with_conv(node, name, v)
end
Base.show(io::IO, n::SwcNode) = begin
    print(io, "SwcNode(")
    _print_node_info(io, n, nothing)
    print(io, ")")
end
function SwcNode(n::Union{SwcNode,T}; parind::Union{Int64,Missing}=missing) where {T<:NamedTuple}
    SwcNode(n.id, ismissing(parind) ? n.parind : parind, n.pos, n.typ, n.rad)
end
function Base.NamedTuple(row::SwcNode)
    (; row.id, row.parind, row.pos, row.typ, row.rad)
end

struct SwcNodeAnnot
    id::Int64
    key::Symbol
    val::String
end
function SwcNodeAnnot(ann::T; id::Union{Int64,Missing}=missing) where {T<:NamedTuple}
    SwcNodeAnnot(ismissing(id) ? ann.id : id, Symbol(ann.key), ann.val)
end

"""
    struct SwcTableRow{Ext}

Access a row of `SwcTable`.
Allowed properties: `.index`, `.node`, `.extended`, and node properties.
`Ext` is the user supplied user data per node accessed by `extended` property.
"""
struct SwcTableRow{Ext}
    # XXX: XXX: abstract slow!!! (runtime polymorphism); concret fast (compiletime poly.)
    nodes::Vector{SwcNode}
    nodes_ext_::Ext #extended node attributes
    ind::Int64
end
Base.propertynames(swc::SwcTableRow{Ext}, private::Bool=false) where {Ext} =
    (:ind, :index, :node, :parent, fieldnames(Ext)..., propertynames(swc.nodes[swc.ind])..., (private ? fieldnames(typeof(swc)) : ())...)
function Base.getproperty(row::SwcTableRow{Ext}, name::Symbol) where {Ext}
    @inline
    @inbounds begin
        nodes = getfield(row, :nodes)
        ind = getfield(row, :ind)
        nodesext = getfield(row, :nodes_ext_)
        name in (:index,) && return ind
        name in (:node,) && return nodes[ind]
        # in(name, (:ext, :extended)) && return node.second
        if in(name, (:parent,))
            pind = nodes[ind].parind
            pind == 0 && return nothing
            return SwcTableRow(nodes, nodesext, pind)
        end
        if in(name, (:length, :len))
            pind = nodes[ind].parind
            pind == 0 && return 0
            return norm(nodes[pind].pos .- nodes[ind].pos)
        end
        hasfield(SwcTableRow{Ext}, name) && return getfield(row, name)
        hasfield(Ext, name) && return getfield(nodesext, name)[ind]
        getproperty(nodes[ind], name)
    end
end
function Base.setproperty!(row::SwcTableRow{Ext}, name::Symbol, v) where {Ext}
    @inline
    @inbounds begin
        nodes = getfield(row, :nodes)
        ind = getfield(row, :ind)
        nodesext = getfield(row, :nodes_ext_)
        name === :node && return nodes[ind] = v
        hasfield(Ext, name) && return getfield(nodesext, name)[ind] = v
        setproperty!(nodes[ind], name, v)
    end
end
function Base.show(io::IO, swc::SwcTableRow{Ext}) where {Ext}
    ind = swc.ind
    print(io, "SwcTableRow[", ind, "](")
    n = swc.nodes[ind]
    _print_node_info(io, n, n.parind == 0 ? nothing : swc.nodes[n.parind].id)
    print(io, "; ")
    join(io, map(n -> "$n=$(getfield(swc.nodes_ext_, n)[ind])", fieldnames(Ext)), ", ")
    print(io, ")")
end

struct SwcTableNodeIter{Ext,Sel}
    # abstract
    nodes::AbstractVector{SwcNode}
    nodes_ext_::Ext
    node_sel_::Sel
end
Base.length(iter::Gapr.SwcTableNodeIter{Ext,Sel}) where {Ext,Sel} =
    Sel === Missing ? length(iter.nodes) : length(iter.node_sel_)
function Base.show(io::IO, swc::SwcTableNodeIter{Ext}) where {Ext}
    print(io, "SwcTableNodeIter{")
    _print_ext_spec(io, Ext)
    print(io, "}(")
    _print_filtered_count(io, swc.node_sel_, swc.nodes, " nodes")
    print(io, ")")
end

function filtered_iterate_impl(::Type{Res}, ::Val{DIR}, iter::SwcTableNodeIter{Ext,Sel}, i::Union{Integer,Missing}) where {Res,DIR,Ext,Sel}
    # XXX: no inline: slow;  inline fast
    @inline
    inds = Sel === Missing ? iter.nodes : iter.node_sel_
    i = @coalesce i (DIR > 0 ? firstindex(inds) : lastindex(inds))
    !checkbounds(Bool, inds, i) && return nothing
    @inbounds Res(iter.nodes, iter.nodes_ext_, Sel === Missing ? i : iter.node_sel_[i]), i + DIR
end
Base.iterate(iter::SwcTableNodeIter, i::Union{Integer,Missing}=missing) =
# XXX: noinline: slow; inline fast
    (@inline; filtered_iterate_impl(SwcTableRow, Val(1), iter, i))
Base.iterate(iter::Base.Iterators.Reverse{<:SwcTableNodeIter}, i::Union{Integer,Missing}=missing) =
    (@inline; filtered_iterate_impl(SwcTableRow, Val(-1), iter.itr, i))

"""
    SwcTableData

Data container for SWC files.
To access the data, use `SwcTable` (`take!(::SwcTableData)`).
"""
mutable struct SwcTableData{Ext}
    # XXX: abstractVector slow!!!: concret fast
    nodes::Vector{SwcNode}
    nodes_ext_::Ext # Extended node attributes
    annots_::AbstractVector{SwcNodeAnnot} # Gapr extension
    loops_::AbstractVector{Pair{Int64,Int64}} # Gapr extension
    id2ind__::AbstractDict{Int64,Int64} # Aux. data
end
"""
    SwcTableData(; extinfos...)

Create an empty data container.
Use `extinfos` to specify extended node attributes.
Eg. `SwcTableData(; ext1=Bool, ext2=Float64)`.
"""
function SwcTableData(; extinfos...)
    ext = map(T -> T[], NamedTuple(extinfos))
    SwcTableData(SwcNode[], ext, SwcNodeAnnot[], Pair{Int64,Int64}[], Dict{Int,Int}())
end
function SwcTableData(nodes::AbstractVector{<:SwcNode})
    r = SwcTableData(nodes, (;), SwcNodeAnnot[], Pair{Int64,Int64}[], Dict{Int,Int}())
    d = r.id2ind__
    for (i, n) in enumerate(r.nodes)
        d[n.id] = i
    end
    r
end
function SwcTableData(swcx::SwcTableData{Ext}; extinfos...) where {Ext}
    for (k, v) in extinfos
        @assert length(v) == length(swcx.nodes)
        @assert !hasfield(Ext, k)
    end
    SwcTableData(swcx.nodes, (; swcx.nodes_ext_..., extinfos...), swcx.annots_, swcx.loops_, swcx.id2ind__)
end

"""
    struct SwcTable{Ext}

A table represention for a SWC file.
Access `.nodes` property to iterate over nodes;
`.annots` for annotations; and `.loops` for loops.
Index with `[index]` or `[!, id]` to access a row.
"""
struct SwcTable{Ext,Sel}
    data_::SwcTableData{Ext}
    node_sel_::Sel #Union{AbstractVector{Int64},Missing}
    annot_sel_::Sel #Union{AbstractVector{Int64},Missing}
    loop_sel_::Sel #Union{AbstractVector{Int64},Missing}
end
SwcTable(; kw...) = SwcTable(SwcTableData(; kw...), missing, missing, missing)
SwcTable(nodes::AbstractVector{<:SwcNode}) = SwcTable(SwcTableData(nodes), missing, missing, missing)
Base.propertynames(swc::SwcTable{Ext}, private::Bool=false) where {Ext} =
    (:nodes, :annots, :loops, fieldnames(Ext)..., (private ? fieldnames(typeof(swc)) : ())...)
Base.getindex(swc::SwcTable, ind::Integer) =
    SwcTableRow(swc.data_.nodes, swc.data_.nodes_ext_, ind)
Base.getindex(swc::SwcTable, _::typeof(!), id::Integer) =
    getindex(swc, swc.data_.id2ind__[id])
Base.getindex(swc::SwcTable, _::typeof(!), id::Integer, def) =
    (ind = get(swc.data_.id2ind__, id, 0); ind == 0 ? def : getindex(swc, ind))
Base.parent(swc::SwcTable) = SwcTable(swc.data_, missing, missing, missing)
function Base.show(io::IO, swc::SwcTable{Ext}) where {Ext}
    print(io, "SwcTable{")
    _print_ext_spec(io, Ext),
    print(io, "}("),
    _print_filtered_count(io, swc.node_sel_, swc.data_.nodes, " nodes, ")
    _print_filtered_count(io, swc.annot_sel_, swc.data_.annots_, " annots, ")
    _print_filtered_count(io, swc.loop_sel_, swc.data_.loops_, " loops")
    print(io, ")")
end
function ignore(swc::SwcTable{Ext}; annots=false, loops=false) where {Ext}
    SwcTable{Ext}(swc.data_, swc.node_sel_,
        annots ? Int64[] : swc.annot_sel_, loops ? Int64[] : swc.loop_sel_)
end

"""
    eachnode(swc::SwcTable)

Iterate over all nodes as `SwcTableRow`s.
Shortcut: `swc.nodes`.
"""
function eachnode(swc::SwcTable)::SwcTableNodeIter
    SwcTableNodeIter(swc.data_.nodes, swc.data_.nodes_ext_, swc.node_sel_)
end
"""
    eachannot(swc::SwcTable)

Iterate over all node annotations as `SwcNodeAnnot`s.
Shortcut: `swc.annots`.
"""
function eachannot(swc::SwcTable)
    _simpleiterate(swc.data_.annots_, swc.annot_sel_)
end
"""
    eachloop(swc::SwcTable)

Iterate over all loops as `Pair{Int64,Int64}`s.
Shortcut: `swc.loops`.
"""
function eachloop(swc::SwcTable)
    _simpleiterate(swc.data_.loops_, swc.loop_sel_)
end
"""
    eachnodeattr(swc::SwcTable, ::Val{attr})

Iterate over attribute `attr` for all nodes.
Shortcut: `swc.attr`.
"""
function eachnodeattr(swc::SwcTable{Ext}, ::Val{attr}) where {Ext,attr}
    data = getfield(swc, :data_)
    sel = getfield(swc, :node_sel_)
    Iterators.map(n -> getproperty(n, attr), SwcTableNodeIter(data.nodes, data.nodes_ext_, sel))
end

function Base.getproperty(swc::SwcTable{Ext}, name::Symbol) where {Ext}
    @inline
    name === :nodes && return eachnode(swc)
    name === :annots && return eachannot(swc)
    name === :loops && return eachloop(swc)
    hasfield(SwcTable{Ext}, name) && return getfield(swc, name)
    eachnodeattr(swc, Val(name))
end

"""
    Base.take!(swcx::SwcTableData{Ext})::SwcTable{Ext}

Take from `swcx` and return a `SwcTable` for accessing the data.
"""
function Base.take!(swcx::SwcTableData{Ext})::SwcTable{Ext} where {Ext}
    nodes, swcx.nodes = swcx.nodes, empty(swcx.nodes)
    nodes_ext, swcx.nodes_ext_ = swcx.nodes_ext_, map(empty, swcx.nodes_ext_)
    annots, swcx.annots_ = swcx.annots_, empty(swcx.annots_)
    loops, swcx.loops_ = swcx.loops_, empty(swcx.loops_)
    id2ind, swcx.id2ind__ = swcx.id2ind__, empty(swcx.id2ind__)
    for n in fieldnames(Ext)
        @assert length(getfield(nodes_ext, n)) == length(nodes)
    end
    @assert length(id2ind) == length(nodes)
    SwcTable(SwcTableData(nodes, nodes_ext, annots, loops, id2ind), missing, missing, missing)
end

const SwcTableXxx = SwcTable{Bool}

struct SwcRow
    tab::SwcTableXxx
    # row::Int64
end
function Base.NamedTuple(row::SwcRow)
    t = row.tab
    r = row.row
    tup = NamedTuple(t.nodes[r])
    par = tup.paridx == 0 ? -1 : t.nodes[tup.paridx].id
    (; tup..., idx=r, par)
end


# XXX:
function addannot!(swc::SwcTable, ann::NamedTuple)
    push!(swc.data_.annots_, SwcNodeAnnot(ann))
end

function addannot!(swc::SwcTableXxx, ann::NamedTuple)
    push!(swc.annots, SwcNodeAnnot(ann))
end

function addloop!(swc::SwcTableXxx, (a, b))
    push!(swc._loops, a => b)
end


function parse_swc_line(l; idoffset=0)
    f = eachsplit(l, r"\s+")
    @inline parseff(::Type{T}, f, valst) where {T} = begin
        valst === nothing && error("wrong swc line: $l")
        parse(T, valst[1]), iterate(f, valst[2])
    end
    id, valst = parseff(Int64, f, iterate(f))
    typ, valst = parseff(Int32, f, valst)
    x, valst = parseff(Float64, f, valst)
    y, valst = parseff(Float64, f, valst)
    z, valst = parseff(Float64, f, valst)
    rad, valst = parseff(Float32, f, valst)
    par, valst = parseff(Int64, f, valst)
    if idoffset != 0
        id += idoffset
        if par != -1
            par += idoffset
        end
    end
    (; id, typ, pos=SwcNodePos(x, y, z), rad, par)
end
function parse_gapr_annot(l; idoffset=0)
    !startswith("#Gapr!")(l) && return nothing
    m = match(r"^#Gapr!(-1|\d+)(@([\.\w]+)=(.*)|/(\d+)|=000(1|0)[0-9A-F]{4})$", l)
    m === nothing && error("unknown annot: $l")
    id = parse(Int64, m.captures[1])
    id != -1 && (id += idoffset)
    if nothing !== m.captures[3]
        return (type=:annot, id=id, key=m.captures[3], val=m.captures[4])
    end
    if nothing !== m.captures[5]
        return (type=:loop, id=id, loop=parse(Int64, m.captures[5]) + idoffset)
    end
    if nothing !== m.captures[6]
        return (type=:pr, id=id, pr=m.captures[6] == "1")
    end
    @assert false
end

function read_swc(f::Function, fname::AbstractString)
    io = Ref{IO}(open(fname))
    try
        if lowercase(splitext(fname)[2]) == ".gz"
            io[] = GzipDecompressorStream(io[])
        end
        f(io[])
    finally
        close(io[])
    end
end
function write_swc(f::Function, fname::AbstractString)
    io = Ref{IO}(open(fname, "w"))
    try
        if lowercase(splitext(fname)[2]) == ".gz"
            io[] = GzipCompressorStream(io[]; level=6)
        end
        f(io[])
    finally
        close(io[])
    end
end

#XXX: view or data???
#view data???
#view data???
#view data???
function renamenodes!(f::Function, swc::SwcTableXxx)
    empty!(swc._id2idx)
    for (i, n) in enumerate(swc.nodes)
        n.id = id = f(n.id)
        swc._id2idx[id] = i
    end
    for (i, a) in enumerate(swc.annots)
        swc.annots[i] = SwcNodeAnnot(f(a.id), a.key, a.val)
    end
    for (i, l) in enumerate(swc._loops)
        swc._loops[i] = f(l[1]) => f(l[2])
    end
end

"""
    load_swc!(swcx::SwcTableData, io; idoffset=0)
    load_swc(io; idoffset=0, sort=false)

Load SWC data from `io` into `swcx`.
Node IDs are offset by `idoffset`.
A filename can also be sepcificed as `io`.
"""
# TODO: ?idgroup to avoid collision
function load_swc!(swcx::SwcTableData{Ext}, io::IO; idoffset=0) where {Ext}
    previd = -1
    pr = hasfield(Ext, :pr) && eltype(fieldtype(Ext, :pr)) === Bool ? swcx.nodes_ext_.pr : nothing
    nodes = swcx.nodes
    annots = swcx.annots_
    loops = swcx.loops_
    id2ind = swcx.id2ind__
    for l in eachline(io)
        l = lstrip(l)
        isempty(l) && continue
        if l[1] == '#'
            ann = parse_gapr_annot(l; idoffset)
            ann === nothing && continue
            id = ann.id
            if id == -1
                @assert previd != -1
                id = previd
            end
            if ann.type === :annot
                push!(annots, SwcNodeAnnot(ann; id))
            elseif ann.type === :loop
                push!(loops, id => ann.loop)
            elseif ann.type === :pr
                if !isnothing(pr)
                    idx = id2ind[id]
                    @assert idx == length(nodes)
                    pr[idx] = ann.pr
                end
            end
        else
            n = parse_swc_line(l; idoffset)
            parind = n.par != -1 ? id2ind[n.par] : 0
            push!(nodes, SwcNode(n; parind))
            !isnothing(pr) && push!(pr, false)
            previd = n.id
            idx = length(nodes)
            prev = get!(id2ind, previd, idx)
            @assert prev == idx
        end
    end
end
function load_swc!(swcx::SwcTableData{Ext}, fn::AbstractString; kw...) where {Ext}
    read_swc(fn) do io
        load_swc!(swcx, io; kw...)
    end
end
function load_swc(io::IO; sort=false, kw...)
    swcx = SwcTableData(; pr=Bool)
    load_swc!(swcx, io; kw...)
    sort && (@info "sort swc..."; sort_swc!(swcx))
    take!(swcx)
end
function load_swc(fn::AbstractString; kwargs...)
    read_swc(fn) do io
        load_swc(io; kwargs...)
    end
end

function format_swc_line(n::T) where {T<:NamedTuple}
    id, typ, pos, rad, par = n[(:id, :typ, :pos, :rad, :par)]
    (x, y, z) = pos
    "$id $typ $x $y $z $rad $par\n"
end
format_swc_line(n::SwcNode, par::Int64) =
    format_swc_line((id=n.id, typ=n.typ, pos=n.pos, rad=n.rad, par=par))

function format_gapr_annot(ann::T) where {T<:NamedTuple}
    id, key, val = ann[(:id, :key, :val)]
    "#Gapr!$id@$key=$val\n"
end
format_gapr_annot(ann::SwcNodeAnnot) =
    format_gapr_annot((id=ann.id, key=ann.key, val=ann.val))

format_gapr_loop(a::Int64, b::Int64) = "#Gapr!$a/$b\n"

parindex_checker(::Missing) = begin
    (_, pind) -> pind !== 0
end
function parindex_checker(sel::Sel) where {Sel}
    a, b = first(sel), last(sel)
    @assert 0 < a
    wl = Set{Int}()
    (ind, pind) -> begin
        @assert pind <= b
        r = !(pind < a || pind > b) && pind in wl
        (push!(wl, ind); r)
    end
end
function save_swc(io::IO, swc::SwcTable{Ext,Sel}) where {Ext,Sel}
    # XXX: _pr not saved
    ck = parindex_checker(swc.node_sel_)
    for row in swc.nodes
        n = row.node
        parind = n.parind
        par = !ck(row.ind, parind) ? -1 : swc[parind].id
        write(io, format_swc_line(n, par))
    end
    for ann in swc.annots
        write(io, format_gapr_annot(ann))
    end
    for (a, b) in swc.loops
        write(io, format_gapr_loop(a, b))
    end
end
function save_swc(fn::AbstractString, swc::SwcTable{Ext}; kwargs...) where {Ext}
    write_swc(fn) do io
        save_swc(io, swc; kwargs...)
    end
end

function _root_prio(name)
    if startswith("seg")(name)
        1
    elseif startswith("root")(name)
        3
    elseif startswith("qroot")(name)
        3
    else
        0
    end
end

function sort_swc!(swcx::SwcTableData{Ext}) where {Ext}
    oldnodes = swcx.nodes
    oldloops = swcx.loops_
    oldid2ind = swcx.id2ind__
    adjs = (id2head=Dict{Int64,Int64}(), idx2adj=Pair{Int64,Int64}[])
    adj2seq = Dict{Pair{Int,Int},Int}()
    function add_adj!(adjs, (a, b)::Pair)
        seq = length(adj2seq)
        if get!(adj2seq, a => b, seq) != seq
            @warn "duplicated link $a => $b"
            return
        end
        h = get(adjs.id2head, a, 0)
        push!(adjs.idx2adj, b => h)
        adjs.id2head[a] = length(adjs.idx2adj)
    end
    function del_adj!(adjs, (a, b)::Pair)
        hit = 0
        h = get(adjs.id2head, a, 0)
        while h != 0
            bb, hh = adjs.idx2adj[h]
            if bb == b
                hit += 1
                adjs.idx2adj[h] = -1 => hh
            end
            h = hh
        end
        @assert hit == 1 "hit==1? ($a,$b) $hit???"
    end
    function pop_adjs!(adjs, a)
        h = get(adjs.id2head, a, 0)
        adjs.id2head[a] = 0
        h
    end
    function foreach_adj(f, adjs, h)
        while h != 0
            bb, hh = adjs.idx2adj[h]
            if bb != -1
                f(bb)
            end
            h = hh
        end
    end
    for n in oldnodes
        if n.parind != 0
            pid = oldnodes[n.parind].id
            add_adj!(adjs, n.id => pid)
            add_adj!(adjs, pid => n.id)
        end
    end
    for (a, b) in oldloops
        add_adj!(adjs, a => b)
        add_adj!(adjs, b => a)
    end
    todo = Pair{Int64,Int64}[]
    indinfos = Pair{Int64,Int64}[]
    loops = empty!(oldloops)
    id2idx = Dict{Int64,Int64}()
    function add_node(id, parind)
        oldidx = oldid2ind[id]
        push!(indinfos, oldidx => parind)
        idx = length(indinfos)
        previdx = get!(id2idx, id, idx)
        @assert previdx == idx
    end
    for n in Iterators.reverse(oldnodes)
        push!(todo, n.id => -1)
    end
    todoroots = []
    for ann in Iterators.reverse(swcx.annots_)
        if ann.key === :root
            push!(todoroots, ann.id => _root_prio(ann.val))
        end
    end
    for prio in unique(sort(map(last, todoroots)))
        for (id, p) in todoroots
            if p == prio
                if p >= 3
                    add_node(id, 0)
                end
                push!(todo, id => -1)
            end
        end
    end
    while !isempty(todo)
        id, par = pop!(todo)
        idx = get(id2idx, id, 0)
        if par == -1
            if idx == 0
                add_node(id, 0)
            end
        else
            if idx != 0
                push!(loops, id => par)
            else
                add_node(id, id2idx[par])
            end
        end
        hh = pop_adjs!(adjs, id)
        foreach_adj(adjs, hh) do b
            del_adj!(adjs, b => id)
            push!(todo, b => id)
        end
    end
    swcx.nodes = map(indinfos) do (oldidx, parind)
        SwcNode(oldnodes[oldidx]; parind)
    end
    swcx.nodes_ext_ = map(swcx.nodes_ext_) do ext
        map(indinfos) do (oldidx, _)
            ext[oldidx]
        end
    end
    swcx.loops_ = loops
    swcx.id2ind__ = id2idx
    swcx
end

# XXX: just a combination of node.iter???, node.length, node.typ, node.typename
# sum((n.len for n in s.nodes))
# sum((n.len for n in s.nodes if n.typename==:axon))
function neurite_length(swc::SwcTable)
    acc = Dict{Symbol,Float64}()
    for n in swc.nodes
        n.parind == 0 && continue
        d = norm(swc[n.parind].pos .- n.pos)
        k = Symbol(n.typ)
        acc[k] = get(acc, k, 0.0) + d
    end
    NamedTuple(acc)
end

# XXX: return SwcTable (wraps SwcTableXxx and filters)
function split_swc(fn::Function, swc::SwcTable{Ext,Sel}) where {Ext,Sel}
    alloc_sel() = (nodes=Int64[], annots=Int64[], loops=Int64[])
    TagT = first(Base.return_types(fn))
    SelT = first(Base.return_types(alloc_sel))
    filters = Dict{TagT,SelT}()
    id2tag = Dict{Int64,TagT}()
    sizehint!(id2tag, length(swc.nodes))
    for n in swc.nodes
        tag = fn(n)
        id2tag[n.id] = tag
        nn = get!(alloc_sel, filters, tag)
        push!(nn.nodes, n.ind)
    end
    annots = swc.data_.annots_
    for (i, a) in (Sel === Missing ? pairs(annots) : Base.Pairs(annots, swc.annot_sel_))
        tag = id2tag[a.id]
        push!(filters[tag].annots, i)
    end
    pair_uniq(p::Tuple{A,B}) where {A,B} = p[1] == p[2] ? (p[1],) : p
    loops = swc.data_.loops_
    for (i, p) in (Sel === Missing ? pairs(loops) : Base.Pairs(loops, swc.loop_sel_))
        for tag in pair_uniq(map(id -> id2tag[id], Tuple(p)))
            push!(filters[tag].loops, i)
        end
    end
    [(; tag, swc=SwcTable(swc.data_, nn...)) for (tag, nn) in filters]
end

"""
    split_swc(swc::SwcTable)

Split the SWC table by `:root` annotations and connectivity.
"""
function split_swc(swc::SwcTable{Ext}) where {Ext}
    # XXX: split: works on a view.
    idx2root = Vector{Pair{Int64,Int64}}(undef, length(swc.nodes))
    roots = Dict{Int64,Pair{Int64,String}}()
    for ann in swc.annots
        if ann.key === :root
            push!(roots, ann.id => (_root_prio(ann.val) => ann.val))
        end
    end
    for (idx, n) in enumerate(swc.nodes)
        root = -10 => n.id
        if (v = get(roots, n.id, nothing)) !== nothing
            root = max(root, v[1] => n.id)
        end
        if n.parind != 0
            root = max(root, idx2root[n.parind])
        end
        idx2root[idx] = root
    end
    swcs = split_swc(swc) do n
        last(idx2root[n.ind])::Int64
    end
    map(t -> (; rootid=t.tag, name=get(roots, t.tag, 0 => nothing).second, t.swc), swcs)
end

struct SwcSegmentIter{Ext,Ord}
    nodes::Vector{SwcNode}
    nodes_ext_::Ext
    taboo::Set{Int64}
    order::Ord
end
Base.IteratorSize(::Type{<:SwcSegmentIter}) = Base.SizeUnknown()
struct SwcSegment{Ext}
    nodes::Vector{SwcNode}
    nodes_ext_::Ext
    taboo::Set{Int64}
    tail::Int64
end
Base.IteratorSize(::Type{<:SwcSegment}) = Base.SizeUnknown()

function Base.iterate(iter::SwcSegmentIter, i::Integer=lastindex(iter.order))
    while true
        firstindex(iter.order) > i && return nothing
        @inbounds ind = iter.order[i]
        if !(ind in iter.taboo)
            return SwcSegment(iter.nodes, iter.nodes_ext_, iter.taboo, ind), i - 1
        end
        i -= 1
    end
end
function Base.iterate(s::SwcSegment{Ext}, i::Integer=s.tail) where {Ext}
    i == 0 && return nothing
    r = SwcTableRow(s.nodes, s.nodes_ext_, i)
    i in s.taboo && return r, 0
    push!(s.taboo, i)
    r, r.parind
end

function eachsegment(swc::SwcTable{Ext,Sel}, prop::Symbol; rev=false) where {Ext,Sel}
    depth = getfield(swc.data_.nodes_ext_, prop)
    @assert eltype(fieldtype(Ext, prop)) <: Number
    order = if Sel === Missing
        sortperm(depth; rev)
    else
        sort(swc.node_sel_, by=i -> depth[i]; rev)
    end
    SwcSegmentIter(swc.data_.nodes, swc.data_.nodes_ext_, Set{Int64}(), order)
end

function eachsegment(swc::SwcTable{Ext,Sel}) where {Ext,Sel}
    order = Sel === Missing ? axes(swc.data_.nodes, 1) : swc.node_sel_
    SwcSegmentIter(swc.data_.nodes, swc.data_.nodes_ext_, Set{Int64}(), order)
end

# XXX:
function join_swc!(swc1::SwcTableData{Ext}, swcs::SwcTableData{Ext}...) where {Ext}
    for swc in swcs
        idxoff = length(swc1.nodes)
        for (idx, n) in enumerate(swc.nodes)
            parind = n.paridx == 0 ? n.paridx : n.paridx + idxoff
            push!(swc1.nodes, SwcNode(n; parind))
            curidx = length(swc1.nodes)
            for f in fieldnames(Ext)
                push!(swc1.nodes_ext_[f], swc.nodes_ext_[f][idx])
            end
            previdx = get!(swc1.id2ind__, n.id, curidx)
            @assert previdx == curidx
        end
        for a in swc.annots_
            push!(swc1.annots_, a)
        end
        for l in swc.loops_
            push!(swc1.loops_, l)
        end
    end
    swc1
end

# XXX: direct access degree with property, cache in swctable ()
# forward sweep: dist to soma; tier or order;
# backward sweep: max/min dist to term; subtree length; subtree branches; reverse order;
# order independent: node degree;
function nodedegree(swc::SwcTable{Ext}) where {Ext}
    if hasfield(Ext, :degree)
        @assert eltype(fieldtype(Ext, :degree)) <: Integer
        return swc
    end
    data = swc.data_
    nodes = data.nodes
    degree = zeros(Int, length(nodes))
    add_degree(ind) = (degree[ind] += 1)
    for (ind, n) in enumerate(nodes)
        if n.parind != 0
            add_degree(ind)
            add_degree(n.parind)
        end
    end
    SwcTable(SwcTableData(data; degree), swc.node_sel_, swc.annot_sel_, swc.loop_sel_)
end

function addattr(swc::SwcTable{Ext}; extinfos...) where {Ext}
    exts = map(T -> similar(swc.data_.nodes, T), NamedTuple(extinfos))
    SwcTable(SwcTableData(swc.data_; exts...), swc.node_sel_, swc.annot_sel_, swc.loop_sel_)
end

# XXX: refactor:::
# 1. Data can be 'hcat' for more attr
# 2. Once loaded, Data cannot be 'vcat' (length fixed)
# 3. For 'vcat'/change length/add annot/loops, make a Data copy? or in place?
# 4. Data[Selector] to get SwcView, and ????
# Three types: view into a table, a table for incremental building, a whole table.
# Choose a correct for a specfic API.

# XXX: generic
# XXX: ext_attr that Gapr uses (private ones) should begin with '_'; but user can create an alias to easily access.
# So, it should be readonly, cannot be changed.
# E.G. _path_depth/path_depth, and _degree/degree...
# XXX: What condition can we create such a ext_attr? should inplace (xxx!(x)) or return (x=xxx(x))?
# Can we gen ext_attr when in a view???
function nodedepth(swc::SwcTable{Ext}) where {Ext}
    if hasfield(Ext, :depth)
        @assert eltype(fieldtype(Ext, :depth)) === Float64
        return swc
    end
    # XXX: iterate over selected or over all???? all data data
    data = swc.data_
    nodes = data.nodes
    depth = Vector{Float64}(undef, length(nodes))
    for (i, n) in enumerate(nodes)
        parind = n.parind
        parind == 0 && (depth[i] = 0; continue)
        depth[i] = depth[parind] + norm(swc[parind].pos .- n.pos)
    end
    SwcTable(SwcTableData(swc.data_; depth), swc.node_sel_, swc.annot_sel_, swc.loop_sel_)
end

function distmat(swc::SwcTable{Ext}; alpha=0.0) where {Ext}
    degrees = getfield(swc.data_.nodes_ext_, :degree)
    dist = Dict{SVector,Float64}()
    distkey(i, j) = sort(SVector(i, j))
    pushdist(i, j, d) = begin
        k = distkey(i, j)
        dist[k] = min(get(dist, k, Inf), d)
    end
    getdist(i, j) = get(dist, distkey(i, j), Inf)
    pts = Set{Int}()
    for n in swc.nodes
        ind = n.ind
        if degrees[ind] != 2 || n.parind == 0
            push!(pts, ind)
            l = 0.0
            idx2 = ind
            pushdist(ind, idx2, l)
            while n.parind != 0
                idx2 = n.parind
                n2 = swc[idx2]
                l += norm(n2.pos .- n.pos)
                n = n2
                degrees[idx2] != 2 && break
            end
            pushdist(ind, idx2, l)
        end
    end
    adjs = Dict()
    for ((i, j), _) in dist
        vec = get!(Set, adjs, i)
        push!(vec, j)
        vec = get!(Set, adjs, j)
        push!(vec, i)
    end
    pts = sort(collect(pts), rev=true)
    pq = DataStructures.PriorityQueue{Int,Float64}()
    taboo = Set{Int}()
    for i in pts
        empty!(pq)
        empty!(taboo)
        push!(taboo, i)
        for j in adjs[i]
            pq[j] = getdist(i, j)
        end
        while !isempty(pq)
            j, d = DataStructures.dequeue_pair!(pq)
            pushdist(i, j, d)
            push!(taboo, j)
            for k in adjs[j]
                if !(k in taboo)
                    pq[k] = min(d + getdist(j, k), get(pq, k, Inf))
                end
            end
        end
    end
    I = Int[]
    J = Int[]
    V = Float64[]
    for ((i, j), v) in dist
        push!(I, i)
        push!(J, j)
        d2 = norm(swc[i].pos .- swc[j].pos)
        push!(V, v * (1 - alpha) + d2 * alpha)
    end
    SparseArrays.sparse(Symmetric(SparseArrays.sparse(I, J, V), :U))
end

isterminal(n, d) = !isroot(n, d) && d == 1
ismiddle(n, d) = !isroot(n, d) && d == 2
isbranch(n, d) = !isroot(n, d) && d > 2
isroot(n, d) = n.paridx == 0

function trim_swc(swc::SwcTableXxx, thr::Number)
    degrees = nodedegree(swc)
    nodemsk = map(one, degrees)
    tmp = []
    for (i, n) in enumerate(swc)
        d = degrees[i]
        if isterminal(n, d)
            empty!(tmp)
            push!(tmp, i)
            l = 0.0
            while true
                ii = n.paridx
                nn = swc[ii]
                l += norm(nn.pos .- n.pos)
                !ismiddle(nn, degrees[ii]) && break
                push!(tmp, ii)
                n = nn
            end
            if l < thr
                for ii in tmp
                    nodemsk[ii] = 0
                end
            end
        end
    end
    nodes = findall(>(0), nodemsk)
    nodeids = Set(map(i -> swc[i].id, nodes))
    annots = findall(a -> in(a.id, nodeids), swc.annots)
    loops = findall(l -> in(l[1], nodeids) && in(l[2], nodeids), swc._loops)
    (; nodes, annots, loops)
end

function decompose_swc_depthfirst!(swc; K=30, L=300)
    depth = fill(Inf, length(swc.nodes))
    taboo = Set(1)
    depth[1] = 0.0
    lens = Float64[]
    tips = Int[]
    for k in 1:1000
        K !== nothing && k > K && break
        mm = (0.0, 0)
        for (i, n) in enumerate(swc)
            i in taboo && continue
            @assert n.paridx != 0
            d = depth[i] = depth[n.paridx] + (n.typ == 2 ? norm(n.pos - swc[n.paridx].pos) : 0.0)
            mm = max(mm, (d, i))
        end
        push!(lens, mm[1])
        i = mm[2]
        push!(tips, i)
        i == 0 && break
        while !in(i, taboo)
            n = swc[i]
            n.typ = -k
            push!(taboo, i)
            depth[i] = 0.0
            i = n.paridx
        end
        L !== nothing && mm[1] < L && break
    end
    len9 = 0.0
    for (i, n) in enumerate(swc)
        if n.typ == 2 && !in(i, taboo)
            len9 += norm(n.pos - swc[n.paridx].pos)
        end
    end
    lens, len9, tips
end

# XXX: generic change node attr
# YYY:
# id/type/pos/rad/par
# id/type/pos/rad/par
# with view?
function xform_swc!(f::Function, swc::SwcTable{Ext,Sel})::SwcTable{Ext,Sel} where {Ext,Sel}
    foreach(n -> n.pos = f(n), swc)
    swc
end
xform_swc(f::Function) = (swc::SwcTableXxx) -> xform_swc!(f, swc)

function filter_impl(fn::Function, swc::SwcTable, sel::Set, node_sel::Vector, _::Val{false}, _::Val{ToTerm})::Nothing where {ToTerm}
    for row in swc.nodes
        cond = if fn(row)
            true
        elseif ToTerm && (pind = row.parind; pind != 0 && swc[pind].id in sel)
            true
        else
            false
        end
        if cond
            push!(sel, row.id)
            push!(node_sel, row.ind)
        end
    end
end
filter_impl(fn::Function, swc::SwcTable, sel::Set, node_sel::Vector, _::Val{true}, _::Val{false})::Nothing = begin
    for row in Iterators.reverse(swc.nodes)
        parind = if fn(row)
            push!(sel, row.id)
            push!(node_sel, row.ind)
            row.parind
        elseif row.id in sel
            push!(node_sel, row.ind)
            row.parind
        else
            0
        end
        parind != 0 && push!(sel, swc[parind].id)
    end
    reverse!(node_sel)
    # XXX: no need to loop again?
    nodes2 = Int64[]
    for row in swc.nodes
        row.id in sel && push!(nodes2, row.ind)
    end
    @assert node_sel == nodes2
end
filter_impl(fn::Function, swc::SwcTable, sel::Set, node_sel::Vector, _::Val{true}, _::Val{true})::Nothing = begin
    for row in Iterators.reverse(swc.nodes)
        parind = if fn(row)
            (push!(sel, row.id); row.parind)
        elseif row.id in sel
            row.parind
        else
            0
        end
        parind != 0 && push!(sel, swc[parind].id)
    end
    for row in swc.nodes
        cond = if row.id in sel
            true
        elseif (pind = row.parind; pind != 0 && swc[pind].id in sel)
            push!(sel, row.id)
            true
        else
            false
        end
        cond && push!(node_sel, row.ind)
    end
end

"""
    Base.filter(fn::Function, swc::SwcTable{Ext}; toroot=true, toterm=false)::SwcTable{Ext}

Produce a sub-tabe with filtered nodes.
`fn` accepts a row (`SwcTableRow{Ext}`) and returns `true` to keep it.
All ancesstors of filtered nodes are also kept.
"""
function Base.filter(fn::Function, swc::SwcTable{Ext,Sel}; toroot=true, toterm=false)::SwcTable{Ext} where {Ext,Sel}
    sel = Set{Int64}()
    node_sel, annot_sel, loop_sel = Int64[], Int64[], Int64[]
    filter_impl(fn, swc, sel, node_sel, Val(toroot), Val(toterm))
    annots = swc.data_.annots_
    for (ind, ann) in (Sel === Missing ? pairs(annots) : Base.Pairs(annots, swc.annot_sel_))
        ann.id in sel && push!(annot_sel, ind)
    end
    loops = swc.data_.loops_
    for (ind, loop) in (Sel === Missing ? pairs(loops) : Base.Pairs(loops, swc.loop_sel_))
        if any(map(id -> id in sel, loop))
            push!(loop_sel, ind)
        end
    end
    SwcTable(swc.data_, node_sel, annot_sel, loop_sel)
end

_print_filtered_count(io, a, b, sfx) = begin
    if !ismissing(a)
        Base.with_output_color(:italic, io) do io
            print(io, length(a))
        end
        print(io, "/")
    end
    print(io, length(b), sfx)
end
_print_ext_spec(io, ::Type{Ext}) where {Ext} =
    join(io, map((n, T) -> "$n::$(eltype(T))", fieldnames(Ext), fieldtypes(Ext)), ", ")
_print_node_info(io::IO, n, parid) = begin
    Base.with_output_color(:bold, io) do io
        print(io, "@", n.id)
    end
    print(io, ", ", n.typ, ", (")
    Base.with_output_color(:italic, io) do io
        join(io, n.pos, ",")
    end
    print(io, "), ", n.rad, ", ")
    print(io, n.parind)
    if !isnothing(parid)
        Base.with_output_color(:bold, io) do io
            print(io, "@", parid)
        end
    end
end
# XXX: XXX: abstractVector fast***; concret fast
_simpleiterate(dat::AbstractVector{T}, sel::Sel) where {T,Sel} =
    Sel === Missing ? dat : Iterators.map(i -> dat[i], sel)

_set_field_with_conv(x::T, f::Symbol, v) where {T} = begin
    @inline
    Tf = fieldtype(T, f)
    setfield!(x, f, convert(Tf, v))
end

function GeometryBasics.Rect{3,T}(swc::SwcTable) where {T}
    vmin = Point{3,T}(typemax(T))
    vmax = Point{3,T}(typemin(T))
    for n in swc.nodes
        vmin, vmax = GeometryBasics.minmax(Point{3,T}(n.pos), vmin, vmax)
    end
    GeometryBasics.Rect{3,T}(SVector(vmin, vmax))
end
function GeometryBasics.Rect{T}(swc::SwcTable) where {T}
    Rect{3,T}(swc)
end
GeometryBasics.Rect(swc::SwcTable) = Rect{3,Float64}(swc)
GeometryBasics.coordinates(swc::SwcTable) = swc.pos
