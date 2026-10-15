# isbits
# Symbol
# type
# Symbol("day", 4)
#
#
# function add(a, b)
# a + b
# end
#
# add(a, b) = begin
# a + b
# end
#
# x -> begin
# y
# end
#
# output_fig(arg)
# output_fig{arg}

module Make


import Dates
import Downloads
import Base: _sizeof_uv_fs, uv_error

function setmtime(path::AbstractString, mtime::Real, atime::Real=mtime; follow_symlinks::Bool=true)
    req = Libc.malloc(_sizeof_uv_fs)
    try
        if follow_symlinks
            ret = ccall(:uv_fs_utime, Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring, Cdouble, Cdouble, Ptr{Cvoid}),
                C_NULL, req, path, atime, mtime, C_NULL)
        else
            ret = ccall(:uv_fs_lutime, Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Cstring, Cdouble, Cdouble, Ptr{Cvoid}),
                C_NULL, req, path, atime, mtime, C_NULL)
        end
        ccall(:uv_fs_req_cleanup, Cvoid, (Ptr{Cvoid},), req)
        ret < 0 && uv_error("utime($(repr(path)))", ret)
    finally
        Libc.free(req)
    end
end

setmtime(path::AbstractString, mtime::Dates.AbstractDateTime, atime::Dates.AbstractDateTime=mtime; follow_symlinks::Bool=true) =
    setmtime(path, Dates.datetime2unix(mtime), Dates.datetime2unix(atime); follow_symlinks=follow_symlinks)

const _mtime_cache = Dict{String,Base.Filesystem.StatStruct}()
const _mtime_dir_cache = Dict{String,Tuple{Float64,Float64,Float64,Float64}}()
function clear_file_meta_cache(p)
    # time
    if p in _mtime_cache
        pop!(_mtime_cache, p)
    end
    pp = dirname(p)
    while pp != "/" && pp != ""
        if pp in _mtime_dir_cache
            pop!(_mtime_dir_cache, pp)
        end
        pp = dirname(pp)
    end
end

function cached_stat(p::AbstractString)
    get!(_mtime_cache, p) do
        stat(p)
    end
end
function cached_mtime(p::AbstractString)
    s = cached_stat(p)
    mtime(s)
end
function cached_mctime(p::AbstractString)
    s = cached_stat(p)
    mtime(s), ctime(s)
end

struct input_file{path}
end

abstract type output_file{path} end
function output_file(path)
end
#getproperty(x, f::Symbol) = (@inline; getfield(x, f))

function dump_expr(expr, lvl=0)
    return
    if expr isa Expr
        hd = expr.head
        println("*", repeat("  ", lvl), typeof(hd), " ", hd)
        for a in expr.args
            dump_expr(a, lvl + 1)
        end
    else
        println("*", repeat("  ", lvl), typeof(expr), ": ", expr)
    end
end
macro hello(e)
    dump_expr(e)
    quote end
end

struct FileTarget{Tup}
end
struct ByProduct{T}
end

mutable struct WeakDeps{Tag,Tup}
end
mutable struct WeakDepsVar{Tag,Tup}
end

function LuatexDeps(fn::AbstractString)
    WeakDeps{:LuatexDeps,Symbol(normpath(fn))}
end

function _tup_from_expr(s)
    dump_expr(s)
    args = ()
    if s isa Expr && s.head === :string
        for x in s.args
            if x isa Symbol
                #args=(args..., call, :esc, x))
                args = (args..., esc(x))
            elseif x isa String
                args = (args..., QuoteNode(Symbol(x)))
            else
                @assert false "Wrong type: $x"
            end
        end
    elseif s isa AbstractString
        args = (args..., QuoteNode(Symbol(s)))
    else
        @assert false "Wrong type: $s"
    end
    Expr(:curly, :Tuple, args...)
end
macro LuatexDeps(s)
    tup = _tup_from_expr(s)
    quote
        WeakDepsVar{:LuatexDeps,FileTarget{$(tup)}}
    end
end

macro x(s)
    tup = _tup_from_expr(s)
    quote
        #x=Int
        FileTarget{$(tup)}
    end
    #@x "xxx"
    #*Symbol string
    #*  Symbol: 
    #*  String: +1
end

# macro x_str(s)
#     typeassert(s, AbstractString)
#     dump_expr(s)
#     tup=Expr(:curly, Core.Tuple)
#     i=1
#     cur=[]
#     while i<= ncodeunits(s)
#         c=s[i]
#         if c!='$'
#             push!(cur, c)
#             i=nextind(s, i)
#             continue
#         end
#         i=nextind(s, i)
#         @assert i<=ncodeunits(s)
#         d=s[i]
#         if d==='$'
#             push!(cur, d)
#             i=nextind(s, i)
#             continue
#         end
#         e, i=Meta.parseatom(s, i)
#         push!(tup.args, QuoteNode(Symbol(join(cur))))
#         push!(tup.args, esc(e))
#         empty!(cur)
#     end
#     push!(tup.args, QuoteNode(Symbol(join(cur))))
#     Expr(:curly, FileTarget, tup)
# end

function path(::Type{FileTarget{T}}) where {T<:Tuple}
    join(T.parameters)
end
function path(::Type{ByProduct{T}}) where {T}
    path(T)
end
function dir(::Type{FileTarget{T}}) where {T<:Tuple}
    "/asdf"
end
function abspath(t::Type{FileTarget{T}}) where {T<:Tuple}
    joinpath(dir(t), path(t))
end
function Base.getproperty(x::FileTarget{T}, f::Symbol) where {T<:Tuple}
    if f === :path
        return path(FileTarget{T})
    else
        Base.getfield(x, f)
    end
end
function Base.getproperty(x::ByProduct{T}, f::Symbol) where {T}
    if f === :path
        return path(T)
    else
        Base.getfield(x, f)
    end
end

function _get_deps!(deps::AbstractDict, t::Type{FileTarget{T}}, ds) where {T}
    if t in ds
        error("dependency loop")
    end
    _make_recurse!(ds, t)
    p = path(t) |> normpath
    deps[p] = max(2, get(deps, p, 0))
end
struct List2{fff,Tup}
end
function _get_deps!(deps::AbstractDict, t::Type{List2{fff,T}}, ds) where {fff,T<:Tuple}
    for t in T.parameters
        _get_deps!(deps, fff(t...), ds)
    end
end

function _get_deps!(::AbstractDict, t::Type{ByProduct{T}}, ds) where {T}
end

function _parse_luatex_fls(fn)
    # TODO: cache
    inputs = String[]
    outputs = String[]
    getarg(l) = l[findfirst(" ", l)[1]+1:end]
    !isfile(fn) && return inputs, outputs
    open(fn) do io
        pwd = nothing
        for l in eachline(io)
            if startswith("PWD ")(l)
                pwd = getarg(l)
            elseif startswith("INPUT ")(l)
                d = getarg(l)
                d = joinpath(pwd, d) |> normpath
                push!(inputs, d)
            elseif startswith("OUTPUT ")(l)
                d = getarg(l)
                d = joinpath(pwd, d) |> normpath
                push!(outputs, d)
            else
                @assert false "Unknown line $l"
            end
        end
    end
    inputs, outputs
end
function _get_deps!(deps::AbstractDict, t::Type{WeakDepsVar{tag,FileTarget{U}}}, ds) where {tag,U}
    T = path(FileTarget{U})
    inputs, _ = _parse_luatex_fls(String(T))
    @info "lua deps $T"
    for d in inputs
        deps[d] = max(1, get(deps, d, 0))
    end

end
function _get_deps!(deps::AbstractDict, t::Type{WeakDeps{:LuatexDeps,T}}, ds) where {T}
    @assert T isa Symbol
    inputs, _ = _parse_luatex_fls(String(T))
    @info "lua deps $T"
    for d in inputs
        deps[d] = max(1, get(deps, d, 0))
    end
end

import CodecZlib
import Serialization
import Base64
# relpath->ts
const _target2meta = Dict{String,Dict{Symbol,Any}}()
get_meta_path(path::AbstractString) = get!(() -> Dict{Symbol,Any}(), _target2meta, path)
function get_meta(path::AbstractString, field::Symbol, def)
    s = get_meta_path(path)
    get(s, field, def)
end

function _save_meta(path, key, val)
    buf = IOBuffer()
    str = CodecZlib.GzipCompressorStream(buf)
    Serialization.serialize(str, (; path, key, val))
    write(str, CodecZlib.TranscodingStreams.TOKEN_END)
    flush(str)
    l = take!(buf)
    Base64.base64encode(l)
end
function _load_meta(l)
    l = Base64.base64decode(l)
    buf = IOBuffer(l)
    str = CodecZlib.GzipDecompressorStream(buf)
    Serialization.deserialize(str)
end
function put_metas(f::Function; save=true)
    ff(io) = (path::AbstractString, field::Symbol, val) -> begin
        if io !== nothing
            l = _save_meta(path, field, val)
            println(io, l)
        end
        s = get_meta_path(path)
        s[field] = val
    end
    if save
        open(f ∘ ff, "makelog", "a")
    else
        f ∘ ff(nothing)
    end
end
function init_metas()
    isfile("makelog") || return
    open("makelog") do io
        for l in eachline(io)
            t = _load_meta(l)
            s = get_meta_path(t.path)
            s[t.key] = t.val
        end
    end
end


import SHA

function _shafile!(ctx, p)
    open(p) do io
        while true
            nread = 4096
            dat = read(io, nread)
            SHA.update!(ctx, dat)
            length(dat) < nread && break
        end
    end
end
function mctime_dir_minmax(p::AbstractString)
    m1, m2 = Inf, -Inf
    c1, c2 = Inf, -Inf
    for (r, ds, fs) in walkdir(p)
        for f in fs
            t, t2 = cached_mctime(joinpath(r, f))
            m2 = max(m2, t)
            m1 = min(m1, t)
            c2 = max(c2, t2)
            c1 = min(c1, t2)
        end
    end
    m1, m2, c1, c2
end
function mtime_dir(p)
    _, m2, _, _ = get!(_mtime_dir_cache, p) do
        mctime_dir_minmax(p)
    end
    m2
end
function mtime_dir_min(p)
    m1, _, _, _ = get!(_mtime_dir_cache, p) do
        mctime_dir_minmax(p)
    end
    m1
end
function ctime_dir_min(p)
    _, _, c1, _ = get!(_mtime_dir_cache, p) do
        mctime_dir_minmax(p)
    end
    c1
end
function get_file_sha(p)
    # TODO: cache this
    ctx = SHA.SHA256_CTX()
    files = []
    if isdir(p)
        for (r, ds, fs) in walkdir(p)
            for f in fs
                push!(files, Base.abspath(joinpath(r, f)))
            end
        end
    else
        push!(files, p)
    end
    for d in sort(unique(files))
        _shafile!(ctx, d)
    end
    SHA.digest!(ctx)
end
# relpath->hash/ts
const cachefilets = Dict{String,Any}()
function _fix_ts_impl(p; prep=false)
    if haskey(_mtime_cache, p)
        pop!(_mtime_cache, p)
    end
    if prep
        pts, ph = get(cachefilets, p, (-Inf, nothing))
        if ph === nothing && ispath(p)
            h = get_file_sha(p)
            ts = cached_mtime(p)
            @info "saved mtime $p $h $ts"
            cachefilets[p] = (ts, h)
        end
        return
    end
    #@info "--- fix mtime $p"
    h = get_file_sha(p)
    pts, ph = get(cachefilets, p, (-Inf, nothing))
    ts = cached_mtime(p)
    @info "fix??? $h $ph"
    if h == ph
        if ts > pts
            @info "fix mtime $p"
            setmtime(p, pts)
            if haskey(_mtime_cache, p)
                pop!(_mtime_cache, p)
            end
        end
    else
        cachefilets[p] = (ts, h)
    end
end

function _fix_ts(t::Type{FileTarget{T}}, ::Nothing; kw...) where {T}
    _fix_ts_impl(path(t); kw...)
end
function _fix_ts(::Nothing, t::Type{FileTarget{T}}; kw...) where {T}
end
function _fix_ts(::Nothing, t::Type{List2{f,T}}; kw...) where {f,T}
end
function _fix_ts(::Nothing, t::Type{ByProduct{T}}; kw...) where {T}
    _fix_ts(T, nothing; kw...)
end
function _fix_ts(::Nothing, t::Type{WeakDepsVar{tag,U}}; kw...) where {tag,U<:FileTarget}
    T = path(U)
    inputs, outputs = _parse_luatex_fls(String(T))
    for d in outputs
        if d in inputs
            _fix_ts_impl(d; kw...)
        end
    end
end
function _fix_ts(::Nothing, t::Type{WeakDeps{:LuatexDeps,T}}; kw...) where {T}
    inputs, outputs = _parse_luatex_fls(String(T))
    for d in outputs
        if d in inputs
            _fix_ts_impl(d; kw...)
        end
    end
end

function _get_cts(t::Type{FileTarget{T}}) where {T}
    p = path(t)
    s = cached_stat(p)
    if !ispath(s)
        -Inf
    elseif isfile(s)
        ctime(s)
    elseif isdir(s)
        ctime_dir_min(p)
    else
        @assert false "unknown cond"
    end
end
function _get_cts(t::Type{ByProduct{T}}) where {T}
    return _get_cts(T)
end
function _guess_runts(Out, Args)
    t = _get_cts(Out)
    for d in Args
        if d <: ByProduct
            t = min(t, _get_cts(d))
        end
    end
    #@warn "runts $t"
    t
end

function _make_recurse!(ds::AbstractSet, t::Type{FileTarget{T}}; force=()) where {T}
    ret = nothing
    while true
        p = path(t)
        #@info "TEMPT $p"
        imps = code_typed(t())
        open("/tmp/make-touch.log", "a") do io
            println(io, p)
        end
        if isempty(imps) && ispath(p)
            #@info "no imps $imps"
            return
        end
        isempty(imps) && error("no imps $p")
        f = imps[1]
        totyp(v) = begin
            if v isa Core.Const
                return typeof(v.val)
            else
                return v
            end
        end
        Args = map(totyp, f.first.slottypes[2:end])
        deps = Dict{String,Int}()
        push!(ds, t)
        for a in Args
            _get_deps!(deps, a, ds)
        end
        pop!(ds, t)

        deps = sort(collect(deps))
        runts0 = _guess_runts(t, Args)
        runts = get_meta(p, :runts, runts0)
        #@warn "$runts0 $runts"
        if isinf(runts0)
            runts = runts0
        end
        #@warn "force $p $force?"
        if p in force
            runts = -Inf
            @warn "force $p"
            pop!(force, p)
        end
        maxts, maxtspath = -Inf, nothing
        for (d, dt) in deps
            #@info "deps: $d $dt"
            dt == 0 && continue
            s = cached_stat(d)
            if isfile(s)
                ts = mtime(s)
                maxts, maxtspath = max((maxts, maxtspath), (ts, d))
            elseif isdir(s)
                ts = mtime_dir(d)
                maxts, maxtspath = max((maxts, maxtspath), (ts, d))
            elseif !ispath(s)
                if dt > 1
                    error("dependency not found: $d")
                end
            else
                @assert false "type not supported $s"
            end
            #@info "deps $dt $d"
        end
        if !isinf(runts) && maxts <= runts
            ##@info "SKIP $p: $maxtspath $maxts $runts"
            return ret
        end
        @warn " $runts $maxts ($maxtspath)"

        _fix_ts(t, nothing; prep=true)
        for d in Args
            _fix_ts(nothing, d; prep=true)
        end


        args = map(v -> v(), Args)
        @info "MAKE $p: $maxtspath"
        runts = max(maxts, time())
        ret = t()(args...)
        _fix_ts(t, nothing)
        for d in Args
            _fix_ts(nothing, d)
        end
        put_metas() do ff
            ff(p, :runts, runts)
        end
    end
end

import JLD2

# TODO: reconsider hashes of each run
function make(t::Type{FileTarget{T}}; force=false, open=false) where {T}
    if isempty(_target2meta)
        init_metas()
    end
    empty!(_mtime_cache)
    empty!(_mtime_dir_cache)
    ds = Set()
    force = if force
        Set((path(t),))
    else
        ()
    end
    t0 = time()
    _make_recurse!(ds, t; force)
    t1 = time() - t0
    if t1 > 60
        input = IOBuffer("task done ($(round(Int, t1))s): $(path(t))")
        headers = ["Priority" => "high", "Tags" => "tada"]
        Downloads.request("https://ntfy.sh/ggg-comp-task-done"; input, output=IOBuffer(), method="POST", headers)
    end
    if open
        if endswith(".dat")(path(t))
            JLD2.@load path(t) df
            return df
        else
            if filesize(path(t)) > 0
                run(`xdg-open $(path(t))`, wait=false)
            end
        end
    end
    #   outts runts
    #
    #   contmtime in file in file
    #   
    # .

end
function make(t::Type{List2{fff,Tup}}; force=false, open=false) where {fff,Tup}
    deps = map(a -> fff(a...), Tup.parameters)
    for d in deps
        make(d; force, open)
    end
end


function _gen_args!(args, prev)
    push!(args, prev)
end
function _gen_args!(args, prev, val, vals...)
    for v in val
        v = v isa AbstractString ? Symbol(v) : v
        v = v isa Tuple ? v : (v,)
        _gen_args!(args, (prev..., v...), vals...)
    end
end
function list(f, vals...)
    args = []
    _gen_args!(args, (), vals...)
    List2{f,Tuple{args...}}
end

function iter(T::Type{List2{fff,Tup}}) where {fff,Tup<:Tuple}
    map(a -> (a, path(fff(a...))), Tup.parameters)
end

function Base.getproperty(x::List2{F,T}, f::Symbol) where {F,T<:Tuple}
    if f === :iter
        return iter(List2{F,T})
    else
        Base.getfield(x, f)
    end
end

macro ttt(e)
    Base.dump(e)
    quote end
end


end

###
# XXX: 
# Taking advantage of Revise in other packages
# record run ts only when mtime is intentionally changed

##To make it easier for other packages to benefit from Revise without needing to add it as a dependency or understand Revise's internals, Revise interfaces with CodeTracking, which is a small package acting as Revise's "query" interface.
#
#mtime for versions, ctime for runts.
