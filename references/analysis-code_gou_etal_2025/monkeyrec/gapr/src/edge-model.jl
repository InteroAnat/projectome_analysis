import Gapr_jll
import Graphs
import GeometryBasics

const libgapr = Gapr_jll.libgaprcore

function print_thread_id(tag::AbstractString)
    @info "xxx"
    @ccall libgapr.print_thread_id(tag::Cstring)::Cvoid
    @info "yyy"
end

mutable struct SimpleFut{T}
    @atomic s::Bool
    v::Union{Some{T},Nothing}
    l::Base.ThreadSynchronizer
end
SimpleFut(::Type{T}) where {T} = SimpleFut{T}(false, nothing, Base.ThreadSynchronizer())

function Base.wait(f::SimpleFut{T}) where {T}
    @info "bf wait"
    if !@atomic f.s
        donenotify = f.l::Base.ThreadSynchronizer
        lock(donenotify)
        try
            while !@atomic f.s
                wait(donenotify)
            end
        finally
            unlock(donenotify)
        end
    end
    @info "af wait"
    something(f.v)
end
function Base.notify(f::SimpleFut{T}, v::T) where {T}
    @assert @atomic(f.s) == false
    f.v = Some(v)
    @atomic f.s = true

    @info "bf notify"
    donenotify = f.l
    lock(donenotify)
    try
        if !isempty(donenotify.waitq)
            notify(donenotify)
        end
    finally
        unlock(donenotify)
    end
    @info "af notify"
end

# macro c_async_cb(func, Tret, Targs)
#     @assert Tret.head===:tuple
#     Terrret = Expr(:tuple, Cint, Tret.args...,)
#     quote
#         wrapper(promise::Ptr{Promise}, args...) = begin
#             try
#                 ret = $func(args...)
#                 @info "$Terrret"
#                 ccall((:gapr_async_return, libgapr), Cvoid, $Terrret, 0, ret...)
#             catch
#                 ccall((:gapr_async_return, libgapr), Cvoid, (Cint), 1)
#             end
#         end
#         @cfunction(wrapper, Cvoid, $Targs)
#     end
# end
struct Promise end
# function async_return()
#     ccall
# end
#     void (gapr_promise *promise, int error, ...);

struct Looper end
function looper_create()
    @ccall libgapr.gapr_looper_create()::Ptr{Looper}
end
function looper_destroy(looper::Ptr{Looper})
    @ccall libgapr.gapr_looper_destroy(looper::Ptr{Looper})::Cvoid
end
function looper_start(looper::Ptr{Looper})
    @ccall libgapr.gapr_looper_start(looper::Ptr{Looper})::Cvoid
end
function looper_stop(looper::Ptr{Looper})
    @ccall libgapr.gapr_looper_stop(looper::Ptr{Looper})::Cvoid
end


const trypass = ["testpass", "ggg"]
function ask_passwd(user::AbstractString, host::AbstractString, port::Cushort)
    @info "before thread id"
    Gapr.print_thread_id("do ask passwd in julia")
    @info "after thread id"
    p = pop!(trypass)
    return "", p
    p = Base.getpass("$user@$host:$port's password")
    println(stdout)
    return "", read(p, String)
end
function ask_passwd_wrapper(promise::Ptr{Promise}, user::Cstring, host::Cstring, port::Cushort)::Cvoid
    @info "julia passwd 1"
    user = unsafe_string(user)
    host = unsafe_string(host)
    ###@info "asdf $user@$host:$port:w"
    Threads.@spawn :default begin
        @info "julia passwd 2"
        usr, pass = ask_passwd(user, host, port)
        ###@info "user: pass $usr, $pass"
        @info "julia passwd $usr $pass"
        if isempty(pass)
            @ccall libgapr.gapr_async_return(promise::Ptr{Promise}, 1::Cint;)::Cvoid
        else
            @ccall libgapr.gapr_async_return(promise::Ptr{Promise}, 0::Cint; usr::Cstring, pass::Cstring)::Cvoid
            @ccall printf("setttt %d %s %s %p %p\n"::Cstring; 0::Cint, usr::Cstring, pass::Cstring, usr::Cstring, pass::Cstring)::Cint
        end
        @info "julia passwd end"
    end
    nothing
end

struct Engine end
function engine_create(looper::Ptr{Looper}, uri::AbstractString, testfunc::Function)
    cf = @cfunction($testfunc, Cvoid, (Cstring,))
    @ccall libgapr.gapr_engine_create(looper::Ptr{Looper}, uri::Cstring, cf::Ptr{Cvoid})::Ptr{Engine}
end
function engine_destroy(engine::Ptr{Engine})
    @info "engine destroy julia"
    @ccall libgapr.gapr_engine_destroy(engine::Ptr{Engine})::Cvoid
end

function _engine_start_cb(pw::Ptr{SimpleFut{Cint}}, error::Cint)::Cvoid
    @info "start end"
    w = unsafe_pointer_to_objref(pw)
    notify(w, error)
end
function engine_start(engine::Ptr{Engine})
    func_c = @cfunction(ask_passwd_wrapper, Cvoid, (Ptr{Promise}, Cstring, Cstring, Cushort))
    #@ccall $func_c(Ptr{Promise}()::Ptr{Promise}, "asdf"::Cstring, "asdf"::Cstring, 3::Cushort)::Cvoid
    #func_c = @cfunction(ask_passwd, Cvoid, (Cstring, Cstring, Cushort))

    w = SimpleFut(Cint)
    pw = pointer_from_objref(w)
    notify_c = @cfunction(_engine_start_cb, Cvoid, (Ptr{SimpleFut{Cint}}, Cint,))
    @info "wait start ready 000"
    @ccall libgapr.gapr_engine_start(engine::Ptr{Engine}, func_c::Ptr{Cvoid}, notify_c::Ptr{Cvoid}, pw::Ptr{Cvoid})::Cvoid
    @info "wait start ready 111"
    wait(w)
end

function _engine_load_commits_cb(pw::Ptr{SimpleFut{Cint}}, error::Cint)::Cvoid
    @info "load_commits end"
    w = unsafe_pointer_to_objref(pw)
    notify(w, error)
end
function engine_load_commits(engine::Ptr{Engine}, upto::UInt32)
    w = SimpleFut(Cint)
    # block fut task
    # 
    #
    # fut
    # Ccb
    func_c = @cfunction(_engine_load_commits_cb, Cvoid, (Ptr{SimpleFut{Cint}}, Cint,))
    #@ccall $func_c(Ptr{Promise}()::Ptr{Promise}, "asdf"::Cstring, "asdf"::Cstring, 3::Cushort)::Cvoid
    #func_c = @cfunction(ask_passwd, Cvoid, (Cstring, Cstring, Cushort))
    pw = pointer_from_objref(w)
    @ccall libgapr.gapr_engine_load_commits(engine::Ptr{Engine}, upto::UInt32, func_c::Ptr{Cvoid}, pw::Ptr{Cvoid})::Cvoid
    wait(w)
end

struct EdgeModelReader end
@assert isbitstype(EdgeModelReader)
struct EdgeModelNode
    data::Pair{NTuple{3,Int32},UInt32}
end
@assert isbitstype(EdgeModelNode)
function EdgeModelNode(::Nothing)
    EdgeModelNode(Pair{NTuple{3,Int32},UInt32}((0, 0, 0), 0))
end
Base.convert(::Type{EdgeModelNode}, ::Nothing) = EdgeModelNode(nothing)

const NodePos = GeometryBasics.Point3{Float64}
function Base.getproperty(x::EdgeModelNode, f::Symbol)
    if f === :pos
        return NodePos(_from_fixed_point.(x.data.first, Val(0), Val(sizeof(Int32) * 8 - 1), Val(10), Float64))
    end
    if f === :coverage
        return (x.data.second & 0x10000) >> 16
    end
    Base.getfield(x, f)
end

function _from_fixed_point_helpers(::Type{T}, ::Val{LSB}, ::Val{MSB}, ::Val{RSHIFT}, ::Type{Float}) where {T<:Signed,LSB,MSB,RSHIFT,Float}
    _sgn_bit = T(1) << MSB
    _sb_001 = T(1) << LSB
    _sb_mask = _sgn_bit - _sb_001
    # constexpr static double _rt_max_f=to_float(_sb_mask);
    # constexpr static double _inf=INFINITY;
    # constexpr static double _nan=NAN;
    _sgn_bit, _sb_001, _sb_mask
end
function _from_fixed_point_to_float(v::T, ::Val{LSB}, ::Val{MSB}, ::Val{RSHIFT}, ::Type{Float})::Float where {T<:Signed,LSB,MSB,RSHIFT,Float}
    return Float(v >> LSB) / Float(1 << RSHIFT)
end
function _from_fixed_point(host::T, ::Val{LSB}, ::Val{MSB}, ::Val{RSHIFT}, ::Type{Float})::Float where {T<:Signed,LSB,MSB,RSHIFT,Float}
    _sgn_bit, _sb_001, _sb_mask = _from_fixed_point_helpers(T, Val(LSB), Val(MSB), Val(RSHIFT), Float)
    sgn = host & _sgn_bit
    if sgn != 0
        sb = (_sb_001 - host) & _sb_mask
        if sb > _sb_001
            return -_from_fixed_point_to_float(sb - _sb_001, Val(LSB), Val(MSB), Val(RSHIFT), Float)
        end
        if sb == 0
            return -Float(Inf)
        end
        return Float(NaN)
    else
        sb = host & _sb_mask
        if sb < _sb_mask
            return _from_fixed_point_to_float(sb, Val(LSB), Val(MSB), Val(RSHIFT), Float)
        end
        return Float(Inf)
    end
end
# fixed_point<host_type, 0, sizeof(host_type)*8-1, 10>
# fixed_point<host_type, 0, sizeof(host_type)*8-1, 10>
# fixed_point<host_type, 0, sizeof(host_type)*8-1, 10>
# 			constexpr static double to_float(UT v) {
# 			}
# 			constexpr static UT from_float(double v) {
# 				auto l=std::lround(std::ldexp(v, RSHIFT));
# 				return static_cast<UT>(l)<<LSB;
# 			}
# /
#
# 		}
# 	template<typename T, unsigned int LSB, unsigned int MSB, int RSHIFT>
# 		inline constexpr T fixed_point<T, LSB, MSB, RSHIFT>::enc(double value) {
# 			switch(std::fpclassify(value)) {
# 				case FP_INFINITE:
# 					if(std::signbit(value))
# 						return _sgn_bit|_sb_001;
# 					else
# 						return _sb_mask;
# 				case FP_NAN:
# 					return _sgn_bit;
# 				case FP_ZERO:
# 					return 0;
# 				default:
# 					break;
# 			}
# 			if(std::signbit(value)) {
# 				if(value<=-_rt_max_f)
# 					return _sgn_bit|_sb_001;
# 				return from_float(value)&(_sgn_bit|_sb_mask);
# 			} else {
# 				if(value>=_rt_max_f)
# 					return _sb_mask;
# 				return from_float(value);
# 			}
# 		}

struct EdgeModelEdge <: Graphs.AbstractEdge{UInt32}
    edge_id::UInt32
    left_vid::UInt32
    right_vid::UInt32

    n_nodes::UInt32
    nodes::Ptr{UInt32}
    points::Ptr{EdgeModelNode}
    # gapr::edgemodel::edge
    #xxx::Int64
end
struct EdgeModelEdgeExt <: Graphs.AbstractEdge{UInt32}
    edge::EdgeModelEdge
    src::UInt32
    tgt::UInt32
end
function Base.getproperty(x::EdgeModelEdgeExt, f::Symbol)
    e = Base.getfield(x, :edge)
    if f === :points
        return unsafe_wrap(Array, e.points, e.n_nodes)
    end
    if f === :nodes
        return unsafe_wrap(Array, e.nodes, e.n_nodes)
    end
    if f in (:src, :tgt,)
        return Base.getfield(x, f)
    end
    return Base.getfield(e, f)
end
struct EdgeModelEdgeIter <: Graphs.AbstractEdgeIter
    edges::Vector{EdgeModelEdge}
    _vid2idx::Dict{UInt32,UInt32}
end
function Base.iterate(itr::EdgeModelEdgeIter, state=UInt32(1))
    edges = itr.edges
    vid2idx = itr._vid2idx
    if state <= length(edges)
        e = edges[state]
        s = vid2idx[e.left_vid]
        t = vid2idx[e.right_vid]
        return EdgeModelEdgeExt(e, s, t), state + 1
    end
    nothing
end
@assert isbitstype(EdgeModelEdge)
struct EdgeModelVertex
    vertex_id::UInt32
    attr::EdgeModelNode
    # gapr::vertex
end
@assert isbitstype(EdgeModelVertex)
struct EdgeModelProp
    id::UInt32
    key::Cstring
    val::Cstring
end
@assert isbitstype(EdgeModelProp)
function NamedTuple(prop::EdgeModelProp)
    key = prop.key != Ptr{UInt8}() ? unsafe_string(prop.key) : nothing
    val = prop.val != Ptr{UInt8}() ? unsafe_string(prop.val) : nothing
    (; prop.id, key, val)
end
function EdgeModelProp(tup::NamedTuple)
    key = tup.key === nothing ? Ptr{UInt8}() : Base.unsafe_convert(Cstring, tup.key)
    val = tup.val === nothing ? Ptr{UInt8}() : Base.unsafe_convert(Cstring, tup.val)
    EdgeModelProp(tup.id, key, val)
end

function edge_model_acquire(engine::Ptr{Engine})
    @ccall libgapr.gapr_edge_model_acquire(engine::Ptr{Engine})::Ptr{EdgeModelReader}
end
function edge_model_release(reader::Ptr{EdgeModelReader})
    @ccall libgapr.gapr_edge_model_release(reader::Ptr{EdgeModelReader})::Cvoid
end
function edge_model_edges(reader::Ptr{EdgeModelReader})::Vector{EdgeModelEdge}
    n_edges = Ref{UInt32}()
    edges = @ccall libgapr.gapr_edge_model_edges(reader::Ptr{EdgeModelReader}, n_edges::Ptr{UInt32})::Ptr{EdgeModelEdge}
    unsafe_wrap(Array, edges, n_edges[])
end
function edge_model_vertices(reader::Ptr{EdgeModelReader})::Vector{EdgeModelVertex}
    n_vertices = Ref{UInt32}()
    vertices = @ccall libgapr.gapr_edge_model_vertices(reader::Ptr{EdgeModelReader}, n_vertices::Ptr{UInt32})::Ptr{EdgeModelVertex}
    unsafe_wrap(Array, vertices, n_vertices[])
end
function edge_model_props(reader::Ptr{EdgeModelReader})::Vector{EdgeModelProp}
    n_props = Ref{UInt32}()
    props = @ccall libgapr.gapr_edge_model_props(reader::Ptr{EdgeModelReader}, n_props::Ptr{UInt32})::Ptr{EdgeModelProp}
    unsafe_wrap(Array, props, n_props[])
end

struct DelPatch
    n_props::UInt32
    props::Ptr{EdgeModelProp}
    n_nodes::UInt32
    nodes::Ptr{UInt32}
end
@assert isbitstype(DelPatch)
function engine_del_patch(engine::Ptr{Engine}, props::AbstractVector{EdgeModelProp}, nodes::AbstractVector{UInt32}, fake_nid0::UInt32)
    w = SimpleFut(Cint)
    func_c = @cfunction(_engine_load_commits_cb, Cvoid, (Ptr{SimpleFut{Cint}}, Cint,))
    pw = pointer_from_objref(w)
    op = Ref(DelPatch(length(props), Base.unsafe_convert(Ptr{EdgeModelProp}, props), length(nodes), Base.unsafe_convert(Ptr{UInt32}, nodes)))
    @ccall libgapr.gapr_engine_del_patch(engine::Ptr{Engine}, op::Ptr{DelPatch}, fake_nid0::UInt32, func_c::Ptr{Cvoid}, pw::Ptr{Cvoid})::Cvoid
    wait(w)
end
function engine_del_patch(engine::Ptr{Engine}, props::AbstractVector{<:Any}, nodes::AbstractVector{<:Integer}, fake_nid0::Integer)
    if eltype(props) !== EdgeModelProp
        props = map(EdgeModelProp, props)
    end
    if eltype(nodes) !== UInt32
        nodes = map(UInt32, nodes)
    end
    engine_del_patch(engine, props, nodes, UInt32(fake_nid0))
end
struct AddPatch
    n_links::UInt32
    links::Ptr{Pair{UInt32,UInt32}}
    n_props::UInt32
    props::Ptr{EdgeModelProp}
    n_nodes::UInt32
    nodes::Ptr{Pair{EdgeModelNode,UInt32}}
end
@assert isbitstype(AddPatch)
function engine_add_patch(engine::Ptr{Engine}, links::AbstractVector{Pair{UInt32,UInt32}}, props::AbstractVector{EdgeModelProp}, nodes::AbstractVector{Pair{EdgeModelNode,UInt32}}, fake_nid0::UInt32)
    w = SimpleFut(Cint)
    func_c = @cfunction(_engine_load_commits_cb, Cvoid, (Ptr{SimpleFut{Cint}}, Cint,))
    pw = pointer_from_objref(w)
    op = Ref(AddPatch(length(links), Base.unsafe_convert(Ptr{Pair{UInt32,UInt32}}, links), length(props), Base.unsafe_convert(Ptr{EdgeModelProp}, props), length(nodes), Base.unsafe_convert(Ptr{Pair{EdgeModelNode,UInt32}}, nodes)))
    @ccall libgapr.gapr_engine_add_patch(engine::Ptr{Engine}, op::Ptr{AddPatch}, fake_nid0::UInt32, func_c::Ptr{Cvoid}, pw::Ptr{Cvoid})::Cvoid
    wait(w)
end
function engine_add_patch(engine::Ptr{Engine}, links::AbstractVector{<:Any}, props::AbstractVector{<:Any}, nodes::AbstractVector{<:Any}, fake_nid0::Integer)
    if eltype(links) !== Pair{UInt32,UInt32}
        if eltype(links) <: Pair
            links = map(t -> Pair{UInt32,UInt32}(t.first, t.second), links)
        else
            links = map(t -> Pair{UInt32,UInt32}(t.id, t.link), links)
        end
    end
    if eltype(props) !== EdgeModelProp
        props = map(EdgeModelProp, props)
    end
    if eltype(nodes) !== Pair{EdgeModelNode,UInt32}
        if eltype(nodes) <: Pair
            nodes = map(t -> Pair{EdgeModelNode,UInt32}(t.first, t.second), nodes)
        else
            nodes = map(t -> Pair{EdgeModelNode,UInt32}(t.node, t.par), nodes)
        end
    end
    engine_add_patch(engine, links, props, nodes, UInt32(fake_nid0))
end

struct EdgeModel <: Graphs.AbstractGraph{UInt32}
    reader::Ptr{EdgeModelReader}
    edges::Vector{EdgeModelEdge}
    vertices::Vector{EdgeModelVertex}
    props::Vector{EdgeModelProp}
    _vid2idx::Dict{UInt32,UInt32}
    _adjs::Vector{Vector{UInt32}}
end
####@assert isbitstype(EdgeModel)

Graphs.nv(g::EdgeModel)::UInt32 = length(g.vertices)
# XXX: Graphs assumes contiguous vertex IDs.
Graphs.vertices(g::EdgeModel) = Base.OneTo(Graphs.nv(g))
Graphs.is_directed(::Type{EdgeModel}) = false
Graphs.outneighbors(g::EdgeModel, v::UInt32) = g._adjs[v]
Graphs.inneighbors(g::EdgeModel, v::UInt32) = g._adjs[v]
node_attr(g::EdgeModel, v::UInt32) = g.vertices[v].attr
node_id(g::EdgeModel, v::UInt32) = g.vertices[v].vertex_id
Graphs.ne(g::EdgeModel)::UInt32 = length(g.edges)
Graphs.edges(g::EdgeModel) = EdgeModelEdgeIter(g.edges, g._vid2idx)
Graphs.src(e::EdgeModelEdgeExt) = e.src
Graphs.dst(e::EdgeModelEdgeExt) = e.tgt
#inneighbors(g::AbstractSimpleGraph, v::Integer) = fadj(g, v)
#fadj(g::AbstractSimpleGraph, v::Integer) = g.fadjlist[v]

#
# #     
# # AbstractEdgeIter
# # 
# # AbstractGraphFormat
# # AbstractPathState
# #
#
#
function with_edge_model(f::Function, engine::Ptr{Engine})
    reader = edge_model_acquire(engine)
    try
        edges = edge_model_edges(reader)
        @warn "got edges $(length(edges))"
        for (i, e) in enumerate(edges)
            #println("edge $i $(e.left_vid) $(e.right_vid)")
        end
        vertices = edge_model_vertices(reader)
        for (i, v) in enumerate(vertices)
            #println("edge $i $(v.vertex_id)")
        end
        @warn "got vertices $(length(vertices))"
        props = edge_model_props(reader)
        @warn "got props $(length(props))"
        id2idx = Dict(map(p -> p[2].vertex_id => p[1], enumerate(vertices)))
        adjs = [Vector{UInt32}() for _ in axes(vertices, 1)]
        for e in edges
            left = id2idx[e.left_vid]
            right = id2idx[e.right_vid]
            push!(adjs[left], right)
            push!(adjs[right], left)
        end
        f(EdgeModel(reader, edges, vertices, props, id2idx, adjs))
    finally
        edge_model_release(reader)
    end
end

