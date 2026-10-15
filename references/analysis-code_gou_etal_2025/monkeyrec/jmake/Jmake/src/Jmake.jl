module Jmake

export @rule, @norule, @make
export @deps, @weakdeps, @byproducts


"""
    Jmake.@rule [tags] target let [deps]
        [alt_deps]...
        body
    end

Define a rule to build target.
Allowed tags are:
`:phony` for virtual targets, make unconditionally;
`:data` for saved julia data structures;
`:datamem` for non-saved julia data structures, make unconditionally if cold boot;
`:checksum` for checksum (content) based (instead of timestamp) update.

`target` can be `name`, `[name=]"filename.ext"`, or `[name=]"filename.\$(pattern)"`.
`name` and `pattern` will be available when body is executed.

Omitting the `let` block define pure input files with applicable tags.

Specifying `deps` and `alt_deps` is similar to `target` without tags.
`alt_deps` are recognized by `Jmake.@deps` and `Jmake.@weakdeps` calls.
Use `Jmake.@make` for dynamic dependencies (known only after running the rule).

This make system has three stages:
1. parse all rules;
2. build dependency graph for specified targets;
3. run active rules.

## Example makefile.jl

```julia
#!/bin/env julia
using Revise
using Jmake

@rule tgt1 let; end
@rule tgt2 let tgt1; end

@make \$ARGS
```

To use as a script:
```shell
\$ ./makefile.jl tgt2
```

To use in REPL:
```julia
> include("makefile.jl")
> @make tgt2
```

!!! note
    It is assumed that content change implies timestamp update;
    otherwise, update might work incorrectly.
"""
macro rule(args...)
    mkfile = String(__source__.file)
    priv.parse_rule(__module__, mkfile, args...)
    quote end
end
macro norule(args...)
    quote end
end

"""
    Jmake.@deps dep1 dep2 ...
    Jmake.@weakdeps dep1 dep2 ...

Specify multiple dependencies.
Call this macro only in rule body.

`depN` can be `name`, `[name=]"filename.ext"`, or `[name=]"filename.\$(pattern)"`.
`name`s are available and `pattern`s are substituted when rule body is executed.

With `Jmake.@weakdeps`, rule body can run without weak dependencies satisfied.
After weak dependencies became available, rule body will run again.
Use this to solve circular dependencies, e.g. a latex document with bibtex.
"""
macro deps(args...)
    priv.pass_deps(args, false)
end
macro weakdeps(args...)
    priv.pass_deps(args, true)
end

"""
    Jmake.@make tgt1 tgt2 ...

Make specified targets.
`tgtN` can be a `name`; a `"filename.ext"`; or a `\$expr`, which will be evaluated.

Returns absolute paths for file targets, or data values for :data targets.
"""
macro make(args...)
    dir = dirname(String(__source__.file))
    targets = priv.parse_make_targets(__module__, dir, args)
    targets = Expr(:vect, targets...)
    quote
        priv.make_targets($targets)
    end
end

"""
    Jmake.@byproducts [tags] tgt1 tgt2 ...

Define byproducts with.
The syntax is similar to that for the primary target in `@rule`.
Use `put!(name, val)` to return values for data structure byproducts.
"""
macro byproducts(args...)
    quote end
end


module priv

import SHA
import Serialization
import Base64
import Cassette
import CodeTracking
import CodecZlib

const idofrevise_ = Base.PkgId(Base.UUID("295af30f-e4ad-537b-8983-00126c2a3abe"), "Revise")
const Revise = Ref(Base)
function __init__()
    if Base.root_module_exists(idofrevise_)
        Revise[] = Base.root_module(idofrevise_)
    end
end
Cassette.@context OdCtx

struct Target
    file::Union{Nothing,String}
    pattern::Vector{Union{String,Symbol}}
    sym::Union{Nothing,Symbol}
    dir::AbstractString
    checksum::Bool
    data::Bool
    nosave::Bool
    rgx::Union{Nothing,Regex}
    prio::Int
end

struct DepEntry
    files::Union{AbstractString,AbstractVector{<:AbstractString},Expr}
    sym::Union{Nothing,Symbol}
    inlet::Bool
    optional::Bool
    dynamic::Bool
end

struct Rule
    seq::Int
    mkfile::AbstractString
    target::Target
    phony::Bool
    stems::Dict{Symbol,Union{Nothing,String}}
    deps::Vector{DepEntry}
    byproducts::Vector{Target}
    code::Pair{Function,Vector{UInt8}}
end
function Base.print(io::IO, r::Rule)
    Base.print(io, @something(r.target.file, r.target.pattern))
end
########################################################
#1.*2.3
#########################################################

const rules = Dict{String,Rule}()
const inputs = Dict{String,Target}()

function parse_tags(args)
    tags = Dict{Symbol,Int}()
    while !isempty(args) && (tag = first(args)) isa QuoteNode
        tags[tag.value] = get(tags, tag.value, 0) + 1
        args = (args[2:end]...,)
    end
    tags, args
end

function check_tag!(tags, tag::Symbol)
    hastag = get(tags, tag, 0) > 0
    if hastag
        pop!(tags, tag)
    end
    hastag
end
function count_tag!(tags, tag::Symbol)
    hastag = get(tags, tag, 0)
    if hastag != 0
        pop!(tags, tag)
    end
    hastag
end
const debug = true
function parse_rule(mod::Module, mkfile::AbstractString, args...)
    _during_make[] && return
    if debug && false
        for a in args
            dump_expr(a)
        end
    end
    tags, args = parse_tags(args)
    phony = check_tag!(tags, :phony)
    if length(args) == 1
        @assert !phony
        tgt = Target(mkfile, tags, args[1])
        register_input(tgt)
    elseif length(args) == 2
        tgt = Target(mkfile, tags, args[1])
        rule = parse_rule_body(mod, mkfile, tgt, args[2], phony)
        register_rule(rule)
    else
        error("wrong number of arguments, need: Target [Recipe]: $args")
    end

    #@info "totrack $mod $mkfile"
    push!(totrack, (get_mod_for_include(mod), mkfile))
end

function Target(mkfile::AbstractString, tags, tgt; assigned=nothing)
    checksum = check_tag!(tags, :checksum)
    data = check_tag!(tags, :data)
    prio = count_tag!(tags, :+)
    nosave = false
    if !data && check_tag!(tags, :memdata)
        data = true
        nosave = true
    end
    for tag in tags
        error("Unknown tag: $tag")
    end
    file = nothing
    pat = []
    sym = nothing
    rgx = nothing
    if tgt isa Expr && tgt.head === :(=) && length(tgt.args) == 2 && tgt.args[1] isa Symbol
        assigned = tgt.args[1]
        tgt = tgt.args[2]
    end
    if tgt isa AbstractString
        file = tgt
    elseif assigned === nothing && tgt isa Symbol
        file = replace(String(tgt), "_" => "-")
        sym = tgt
    elseif tgt isa Expr && tgt.head === :string
        pattt = resolve_pat(tgt.args)
        if pattt isa AbstractArray
            pat = pattt
            rgx = build_regex(pattt)
        else
            file = pattt
        end
    else
        error("wrong target specifier $tgt")
    end
    if assigned !== nothing
        @assert sym === nothing
        sym = assigned
    end
    Target(file, pat, sym, dirname(mkfile), checksum, data, nosave, rgx, prio)
end
function build_regex(args; match_any=true)
    @info "pattern: $args"
    args = map(args) do s
        if s isa String
            "\\Q$s\\E"
        elseif s isa Symbol
            if match_any
                "([^/]+)"
            else
                "(\\w+)"
            end
        else
            @assert false
        end
    end
    pushfirst!(args, "^")
    push!(args, "\$")
    Regex(join(args))
end

const rules_seq = Ref{Int}(0)
function parse_rule_body(mod::Module, mkfile::AbstractString, tgt::Target, body, phony)
    (body isa Expr && body.head === :let && length(body.args) == 2) || error("wrong recipe: $body")
    stems = Dict{Symbol,String}()
    for x in tgt.pattern
        if x isa Symbol
            stems[x] = ""
        end
    end
    decls, body = body.args
    deps1 = DepEntry[]
    if decls isa Symbol
        decls = [decls]
    elseif decls isa Expr && decls.head === :(=)
        decls = [decls]
    elseif decls isa Expr && decls.head === :block
        decls = decls.args
    else
        error("wrong declaration: $decls")
    end
    depenv = ()#mod=mod, stems=stems)
    for decl in decls
        push!(deps1, resolve_dep!(depenv, decl; inlet=true))
    end

    deps2 = DepEntry[]
    byproducts = []
    if body isa Expr && body.head === :block
    else
        error("wrong body: $body")
    end
    for (linei, line) in enumerate(body.args)
        function handle_body_deps(optional)
            syms = []
            for item in line.args[3:end]
                dep = resolve_dep!(depenv, item; optional=optional)
                push!(deps2, dep)
                if dep.sym !== nothing
                    push!(syms, dep.sym)
                end
            end
            if !isempty(syms)
                body.args[linei] = Expr(:(=), Expr(:tuple, syms...), Expr(:tuple, map(s -> Symbol("@@@", s, "@@@"), syms)...))
            end
        end
        if line isa Expr && line.head === :macrocall
            name = resolve_macro(mod, line.args[1])
            @assert line.args[2] isa LineNumberNode
            if name === getproperty(parentmodule(@__MODULE__), Symbol("@deps"))
                handle_body_deps(false)
            elseif name === getproperty(parentmodule(@__MODULE__), Symbol("@weakdeps"))
                handle_body_deps(true)
            elseif name === getproperty(parentmodule(@__MODULE__), Symbol("@byproducts"))
                tags, args = parse_tags(line.args[3:end])
                for item in args
                    push!(byproducts, Target(mkfile, copy(tags), item))
                end
            elseif name === getproperty(Base.CoreLogging, Symbol("@info"))
                push!(line.args, Expr(:(=), :_id, nameof(mod)))
                #elseif name===getproperty(Base.CoreLogging, Symbol("@warning"))
                #push!(line.args, Expr(:(=), :_id, nameof(mod)))
            elseif name === getproperty(Base.CoreLogging, Symbol("@error"))
                push!(line.args, Expr(:(=), :_id, nameof(mod)))
            else
            end
        end
    end

    code = build_function(mod, body, stems, tgt, deps1, deps2)
    rule = Rule(rules_seq[] += 1, mkfile, tgt, phony, stems, push!(deps1, deps2...), byproducts, code)

    filter!(dep -> !dep.dynamic, rule.deps)
    rule
end

function print(io::IO, files, sym)
    Base.print(io, files)
    if sym !== nothing
        Base.print(io, "[$sym]")
    end
end
function print(io, file, pattern, sym)
    print(io, [@something(file, pattern)], sym)
end
function Base.print(io::IO, dep::DepEntry)
    if dep.optional
        Base.print(io, "?")
    end
    if dep.dynamic
        Base.print(io, "*")
    end
    print(io, dep.files, dep.sym)
end
function Base.print(io::IO, tgt::Target)
    print(io, tgt.file, tgt.pattern, tgt.sym)
    if tgt.checksum
        Base.print(io, "?s")
    end
    if tgt.data
        Base.print(io, "?d")
    end
end
function print(io::IO, rule::Rule)
    Base.print(io, rule.target)
    for t in rule.byproducts
        Base.print(io, " ", t)
    end
    Base.print(io, ":")
    for d in rule.deps
        Base.print(io, " ", d)
    end
    println(io)
    println(io, "\t@make -f $(rule.mkfile) && echo dummy\n")
end

function makefile(outfn)
    @info "save $outfn"
    open(outfn, "w") do io
        phonys = []
        for rule in sort(collect(values(rules)), lt=(a, b) -> a.seq < b.seq)
            print(io, rule)
            if rule.phony
                push!(phonys, rule.target)
            end
        end
        if !isempty(phonys)
            Base.print(io, ".PHONY:")
            for t in phonys
                Base.print(io, " ", t)
            end
        end
    end
end

function resolve_dep!(rule, expr; assigned=nothing, optional=false, dynamic=false, inlet=false, invec=nothing)
    files = nothing
    sym = nothing
    if expr isa Expr
        if expr.head == :(=) && length(expr.args) == 2 && expr.args[1] isa Symbol
            @assert assigned === nothing
            return resolve_dep!(rule, expr.args[2]; assigned=expr.args[1], optional, dynamic, inlet, invec)
        else
            files = expr
        end
    elseif expr isa String
        files = expr
    elseif expr isa AbstractArray{<:AbstractString}
        files = expr[:]
    elseif assigned === nothing && expr isa Symbol
        files = replace(String(expr), "_" => "-")
        sym = expr
    else
        error("wrong dep expr: $(typeof(expr)) $expr")
    end
    if assigned !== nothing
        @assert sym === nothing
        sym = assigned
    end
    DepEntry(files, sym, inlet, optional, dynamic)
end
const PathOrPat = Int
function xxxresolve_dep!(rule, expr; assigned=nothing, optional=false, dynamic=false, inlet=false, invec=nothing)
    function handle_plural(val)
        if val isa AbstractVector
            dep = DepEntry(PathOrPat[], assigned, inlet, optional, dynamic, true)
            for v in val
                if v isa AbstractString || (v isa Expr && v.head === :string)
                    resolve_dep!(rule, v; optional, dynamic, inlet, invec=dep.files)
                else
                    error("Wrong item $(typeof(v)) $v")
                end
            end
            return dep
        else
            error("Wrong replacement $val")
        end
    end
    file = nothing
    sym = nothing
    if expr isa AbstractString
        file = joinpath(rule.dir, expr)
    elseif assigned === nothing && invec === nothing && expr isa Symbol
        file = joinpath(rule.dir, replace(String(expr), "_" => "-"))
        sym = expr
    elseif expr isa Expr && expr.head === :string
        pattt = resolve_pat(expr.args)
        if pattt isa AbstractString
            file = joinpath(rule.dir, pattt)
        else
            if pattt[1] isa AbstractString
                pattt[1] = joinpath(rule.dir, pattt[1])
            else
                pushfirst!(pattt, joinpath(rule.dir, ""))
            end
            file = pattt
        end
    elseif assigned === nothing && invec === nothing && expr isa Expr && expr.head === :(=) && length(expr.args) == 2 && expr.args[1] isa Symbol
        return resolve_dep!(rule, expr.args[2]; assigned=expr.args[1], optional, dynamic, inlet, invec)
        #elseif !inlet && expr isa Expr && expr.head === :$
        ## XXX
        #file="XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        #dynamic=true
    elseif expr isa Expr && expr.head === :$ && length(expr.args) == 1
        val = rule.mod.eval(expr.args[1])
        return handle_plural(val)
    elseif expr isa Expr && expr.head in (:vect, :hcat, :vcat)
        return handle_plural(expr.args)
    else
        error("Wrong dep $(typeof(expr)) $expr")
    end
    if file isa AbstractVector
        for s in file
            if s isa Symbol
                @assert haskey(rule.stems, s)
            end
        end
    end
    if invec !== nothing
        @assert assigned === nothing
        @assert sym === nothing
        push!(invec, file)
        return
    end
    if assigned !== nothing
        @assert sym === nothing
        sym = assigned
    end
    DepEntry([file], sym, inlet, optional, dynamic, false)
end

function register_key(tgt::Target)
    if tgt.file !== nothing
        "file:" * tgt.file
    else
        parts = map(tgt.pattern) do s
            if s isa AbstractString
                s
            else
                "@@@@@@@@$s@@@@@@@@@@@"
            end
        end
        "pattern:" * join(parts)
    end
end
function register_rule(rule::Rule)
    id = register_key(rule.target)
    rules[id] = rule
    @info "register rule $id"
end
function register_input(tgt::Target)
    id = register_key(tgt)
    inputs[id] = tgt
end

function resolve_pat(args)
    pat = Union{String,Symbol}[]
    nstems = 0
    for s in args
        if s isa String
            push!(pat, s)
        elseif s isa Symbol
            push!(pat, s)
            nstems += 1
        else
            error("wrong pattern: $tgt")
        end
    end
    nstems == 0 ? join(pat) : pat
end

function dump_expr(expr, lvl=0)
    debug || return
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

function resolve_mod(mod, node)
    if node isa Expr && node.head === :.
        @assert length(node.args) == 2 && node.args[2] isa QuoteNode
        mod = resolve_mod(mod, node.args[1])
        getproperty(mod, node.args[2].value)
    elseif node isa Symbol
        getproperty(mod, node)
    else
        error("wrong module $node")
    end
end

function resolve_macro(mod, node)
    if node isa Expr && node.head === :.
        @assert length(node.args) == 2 && node.args[2] isa QuoteNode
        mod = resolve_mod(mod, node.args[1])
        getproperty(mod, node.args[2].value)
    elseif node isa Symbol
        getproperty(mod, node)
    elseif node isa GlobalRef
        mod.eval(node)
    else
        error("wrong macro call $(typeof(node)) $node")
    end
end

function parse_target!(mod::Module, targets, tgt; inexpr=false)
    #priv.dump_expr(tgt)
    if tgt isa Symbol
        push!(targets, replace(String(tgt), "_" => "-"))
    elseif tgt isa AbstractString
        push!(targets, tgt)
    elseif tgt isa AbstractVector
        for t in tgt
            parse_target!(mod, targets, t; inexpr)
        end
    elseif !inexpr && tgt isa Expr && tgt.head === :$ && length(tgt.args) == 1
        parse_target!(mod, targets, mod.eval(tgt.args[1]), inexpr=true)
    elseif tgt isa Expr && tgt.head === :string
        push!(targets, esc(tgt))
    else
        error("wrong target: $tgt $(typeof(tgt))")
    end
end

function check_match_once(tgt::Target, path::AbstractString)
    #@info "find $path $(tgt.rgx)"
    @assert tgt.rgx !== nothing
    m = match(tgt.rgx, path)
    m === nothing && return
    values = Dict{Symbol,String}()
    i = 1
    for p in tgt.pattern
        if p isa Symbol
            @assert i <= length(m.captures)
            v = m.captures[i]
            pv = get!(values, p, v)
            pv != v && return
            i += 1
        end
    end
    @assert length(m.captures) + 1 == i
    values
end
function check_match(tgt::Target, absp::AbstractString)
    @assert isabspath(absp)
    @assert isabspath(tgt.dir)
    relp = relpath(absp, tgt.dir)
    for path in (absp, relp)
        m = check_match_once(tgt, path)
        m !== nothing && return m
    end
    nothing
end

function exact_hit(rule::Rule, tgt::AbstractString)
    @assert isabspath(tgt)
    tgt == joinpath(dirname(rule.mkfile), rule.target.file)
end
function find_rule(tgt::AbstractString)
    hit_rules = []
    for (_, rule) in rules
        #@info "find $tgt $(rule.target.file)"
        if rule.target.file !== nothing && exact_hit(rule, tgt)
            push!(hit_rules, subst_rule(rule))
        end
        for bp in rule.byproducts
            if bp.file !== nothing && tgt == bp.file
                push!(hit_rules, subst_rule(rule))
            end
        end
    end
    for (_, inp) in inputs
        if inp.file !== nothing && tgt == inp.file
            push!(hit_rules, subst_rule(inp))
        end
    end
    length(hit_rules) > 1 && return nothing
    length(hit_rules) == 1 && return hit_rules[1]
    for (_, rule) in rules
        #@info "find $tgt $(rule.target.rgx)"
        if rule.target.file === nothing && (subst = check_match(rule.target, tgt)) !== nothing
            @assert Set(keys(subst)) == Set(keys(rule.stems))
            push!(hit_rules, subst_rule(rule, subst))
        end
        for bp in rule.byproducts
            if bp.file === nothing && (subst = check_match(bp, tgt)) !== nothing
                if Set(keys(subst)) == rule.stems
                    push!(hit_rules, subst_rule(rule, subst))
                end
            end
        end
    end
    for (_, inp) in inputs
        if inp.file === nothing && (subst = check_match(inp, tgt)) !== nothing
            push!(hit_rules, subst_tgt(inp, subst))
        end
    end
    sort!(hit_rules, lt=(a, b) -> a.target.prio > b.target.prio)
    if !isempty(hit_rules)
        bestprio = hit_rules[1].target.prio
        filter!(r -> r.target.prio == bestprio, hit_rules)
    end
    if length(hit_rules) > 1
        @info "ambiguous ... $(hit_rules)"
        return nothing
    end
    length(hit_rules) == 1 && return hit_rules[1]
    isfile(tgt) && return tgt
    nothing
end

function prepare_target!(list, tgt::AbstractVector, downstream, optional)
    for f in tgt
        res = prepare_target!(list, f, downstream, optional)
        res isa AbstractString && return res
    end
end
function prepare_target!(list, deps::AbstractVector{DepEntry}, downstream)
    for dep in deps
        dep.optional && continue
        res = prepare_target!(list, dep.files, downstream, dep.optional)
        res isa AbstractString && return res
    end
end
function prepare_target!(list, tgt::AbstractString, downstream::Dict{String,Array}, optional)::Union{Nothing,AbstractString}
    @info "prepare $tgt"
    for l in list
        l.target.file == tgt && return
        for bp in l.byproducts
            bp.file == tgt && return
        end
    end
    for inp in input_types_x
        inp == tgt && return
    end
    rule = find_rule(tgt)
    if rule === nothing
        error("no rule (ambiguous rules) for $tgt")
    elseif rule isa Union{AbstractString,Target}
        @info "add inp: $tgt $rule"
        add_input_types(rule)
        return
    end

    @assert rule isa Rule
    haskey(downstream, rule.target.file) && return rule.target.file

    postpone = downstream[rule.target.file] = Rule[]
    res = prepare_target!(list, rule.deps, downstream)
    delete!(downstream, rule.target.file)

    if res isa AbstractString
        !optional && return res
        addpostpone = get(downstream, res, nothing)
        @assert addpostpone !== nothing
        push!(addpostpone, rule, postpone...)
        @error "postpone $rule to $res"
        return
    end

    push!(list, rule)

    for r in postpone
        @assert !haskey(downstream, r.target.file)
        pp = downstream[r.target.file] = Rule[]
        res = prepare_target!(list, r.deps, downstream)
        delete!(downstream, r.target.file)
        if res isa AbstractString
            addpostpone = get(downstream, res, nothing)
            @assert addpostpone !== nothing
            push!(addpostpone, r, pp...)
            continue
        end
        push!(list, r)
    end
end

@kwdef mutable struct RunInfo
    runts::Float64 = -Inf
    codecond::Vector{UInt8} = UInt8[]
    datacond::Vector{UInt8} = UInt8[]
    codesigs::Vector{Type{<:Tuple}} = Type{<:Tuple}[]
end

@kwdef struct InputInfo
    path::AbstractString
    phony::Bool
    checksum::Bool
    data::Bool
    nosave::Bool
    mtime::Base.RefValue{Float64}
    datats::Vector{Float64}
    datasum::Vector{String}
    filesum::Vector{String}
    cachedvals::Vector
    dataval::Base.RefValue{Any}
    runinfo::RunInfo
end
function Base.print(io::IO, ii::InputInfo)
    Base.print(io, ii.path, " ", ii.mtime[])
end
#function InputInfo()
#InputInfo(path, phony, checksum, data, Ref(nothing))
#end
const input_types = Dict{String,InputInfo}()
const input_types_x = Set{String}()
function update_info(info; datats=nothing, datasum=nothing)
    if datats !== nothing
        push!(info.datats, datats)
    end
    if datasum !== nothing
        push!(info.datasum, datasum)
    end
    ts = stat(info.path).mtime
    if ts > info.mtime[]
        info.mtime[] = ts
        push!(info.cachedvals, precache_vals(info))
    elseif info.data
        push!(info.cachedvals, precache_vals(info))
    end
end
function precache_vals(info)
    info.phony && return
    if !info.checksum
        if info.data
            #if info.nosave
            (isempty(info.datats) ? -Inf : last(info.datats)), nothing
            #else
            #last(info.datats), nothing
            #end
        else
            info.mtime[], nothing
        end
    else
        chk = if info.data
            isempty(info.datasum) ? "" : last(info.datasum)
        else
            if isfile(info.path)
                @info "SHA $(info.path)"
                sum = open(info.path) do io
                    SHA.sha256(io) |> bytes2hex
                end
                last(push!(info.filesum, sum))
            else
                "NOFILE"
            end
        end
        -Inf, chk
    end
end
function add_input_types(fn::AbstractString, path, phony, checksum, data, nosave)
    @assert isabspath(fn)
    if in(fn, input_types_x)
        pinfo = input_types[fn]
        @assert (pinfo.path, pinfo.phony, pinfo.checksum, pinfo.data) == (path, phony, checksum, data)
        return
    end
    push!(input_types_x, fn)
    pinfo = get(input_types, fn, nothing)
    ts = stat(path).mtime
    info = InputInfo(path, phony, checksum, data, nosave, Ref(ts), [], [], [], [], Ref{Any}(nothing), RunInfo())
    if pinfo !== nothing
        for i in 1:fieldcount(RunInfo)
            setfield!(info.runinfo, i, getfield(pinfo.runinfo, i))
        end
    end
    if pinfo === nothing || ts > pinfo.mtime[]
        push!(info.cachedvals, precache_vals(info))
    else
        push!(info.datats, pinfo.datats...)
        push!(info.datasum, pinfo.datasum...)
        push!(info.filesum, pinfo.filesum...)
        push!(info.cachedvals, pinfo.cachedvals...)
        info.dataval[] = pinfo.dataval[]
    end
    #@info "add info: $fn $info"
    input_types[fn] = info
end
function add_input_types(fn::AbstractString)
    add_input_types(fn, fn, false, false, false, false)
end
function add_input_types(tgt::Target, phony=false)
    add_input_types(joinpath(tgt.dir, tgt.file), tgt.file, phony, tgt.checksum, tgt.data, tgt.nosave)
end
const coldboot_dbs = Set()
const coldboot_data = Set()
function build_graph(targets::AbstractVector{<:AbstractString})
    list = Rule[]
    empty!(input_types_x)
    empty!(cached_expr_hash)
    if prepare_target!(list, targets, Dict{String,Array}(), false) isa AbstractString
        error("circular dependencies")
    end
    dirs = []
    for rule in list
        push!(dirs, dirname(rule.mkfile))
        add_input_types(rule.target, rule.phony)
        for bp in rule.byproducts
            add_input_types(bp)
        end
    end
    for dir in dirs
        dir in coldboot_dbs && continue
        push!(coldboot_dbs, dir)
        #@info "coldboot $dir ..."
        dbfn = joinpath(dir, dbfile())
        isfile(dbfn) || continue
        open(dbfn) do io
            for l in eachline(io)
                tup = decode_dbline(l)
                tup === nothing && continue
                @assert tup isa NamedTuple
                #@info "coldboot $tup"
                if tup.type === :runinfo
                    path = abspath(joinpath(dir, tup.path))
                    pinfo = get(input_types, path, nothing)
                    if pinfo === nothing
                        pinfo = input_types[path] = InputInfo(path, false, false, false, false, Ref(-Inf), [], [], [], [], Ref{Any}(nothing), RunInfo())
                    end
                    pinfo.runinfo.runts = tup.runts
                    pinfo.runinfo.codecond = tup.codecond
                    pinfo.runinfo.datacond = tup.datacond
                    pinfo.runinfo.codesigs = tup.codesigs !== nothing ? tup.codesigs : Tuple[]
                else
                    error("unknown line: $tup")
                end
            end
        end
    end

    for rule in list
        if rule.target.data && !rule.target.nosave
            rule.target.file in coldboot_data && continue
            push!(coldboot_data, rule.target.file)
            tup = load_data(rule.target.file)
            tup === nothing && continue
            @assert tup isa NamedTuple
            @assert tup.type === :data
            #@info "coldboot $(rule.target.file) ..."
            path = abspath(joinpath(rule.target.dir, tup.path))
            pinfo = get(input_types, path, nothing)
            if pinfo === nothing
                pinfo = input_types[path] = InputInfo(path, false, false, false, false, Ref(-Inf), [], [], [], [], Ref{Any}(nothing), RunInfo())
            end
            pinfo.dataval[] = tup.data
            push!(pinfo.datats, tup.runts)
            push!(pinfo.datasum, tup.datasum)
            pinfo.runinfo.runts = tup.runts
            pinfo.runinfo.codecond = tup.codecond
            pinfo.runinfo.datacond = tup.datacond
            pinfo.runinfo.codesigs = tup.codesigs !== nothing ? tup.codesigs : Tuple[]
        end
    end
    list
end

function subst_impl(file, pattern, subst)
    if file !== nothing
        @assert isempty(pattern)
        return file, pattern
    end
    @assert !isempty(pattern)
    parts = map(pattern) do i
        if i isa AbstractString
            i
        else
            subst[i]
        end
    end
    join(parts), []
end
function subst_impl(mod::Module, files, subst, dir::AbstractString)
    vals = Expr(:block, map(p -> Expr(:(=), p.first, p.second), collect(subst))...)
    expr = Expr(:let, vals, Expr(:block, files))
    files = mod.eval(expr)
    files isa AbstractString && return joinpath(dir, files)
    function flatten!(out, vec::AbstractArray)
        for v in vec
            if v isa AbstractString
                push!(out, joinpath(dir, v))
            elseif v isa AbstractArray
                flatten!(out, v)
            else
                error("wrong deps subst: $v")
            end
        end
        out
    end
    if files isa AbstractArray
        res = AbstractString[]
        return flatten!(res, files)
    end
    error("wrong type of deps $(typeof(files)) $files")
end
function subst_tgt(tgt::Target, subst)
    file, pat = subst_impl(tgt.file, tgt.pattern, subst)
    Target(file, pat, tgt.sym, tgt.dir, tgt.checksum, tgt.data, tgt.nosave, nothing, tgt.prio)
end
function subst_rule(rule::Rule, subst=Dict())
    @assert Set(keys(subst)) == Set(keys(rule.stems))
    target = subst_tgt(rule.target, subst)
    deps = map(rule.deps) do dep
        files = subst_impl(parentmodule(rule.code.first), dep.files, subst, dirname(rule.mkfile))
        DepEntry(files, dep.sym, dep.inlet, dep.optional, dep.dynamic)
    end
    byproducts = map(rule.byproducts) do bp
        subst_tgt(bp, subst)
    end
    Rule(rule.seq, rule.mkfile, target, rule.phony, subst, deps, byproducts, rule.code)
end
function duptgt(tgt::Target)
    Target(tgt.file, tgt.pattern, tgt.sym, tgt.dir, tgt.checksum, tgt.data, tgt.nosave, nothing, tgt.prio)
end
function duprule(rule::Rule)
    @assert isempty(rule.stems)
    target = duptgt(rule.target)
    deps = map(rule.deps) do dep
        DepEntry(dep.files, dep.sym, dep.inlet, dep.optional, dep.dynamic)
    end
    byproducts = map(rule.byproducts) do bp
        duptgt(bp)
    end
    Rule(rule.seq, rule.dir, target, rule.phony, rule.stems, deps, byproducts, rule.code)
end

function record_func(ctx::OdCtx, sig::Type{<:Tuple})
    push!(ctx.metadata, sig)
end
function get_world_info()
    latest = Base.get_world_counter()
    curr = ccall(:jl_get_tls_world_age, UInt, ())
    curr, latest
end

function override_func_where_arg(var::TypeVar, repvar=true)
    if !repvar
        return var
    end
    nsym = repvar ? Symbol(var.name, 666) : var.name
    return Expr(:comparison, var.lb, :(<:), nsym, :(<:), var.ub)
    #return TypeVar(nsym, var.lb, var.ub)
    if var.lb === Union{}
        if var.ub === Any
            nsym
        else
            Expr(:(<:), nsym, var.ub)
        end
    else
        if var.ub === Any
            Expr(:(<:), var.lb, nsym)
        else
            Expr(:comparison, var.lb, :(<:), nsym, :(<:), var.ub)
        end
    end
end

function override_func_funcarg(var::Type{Core.Type{T}}, wheres) where {T}
    @info "Type::: $var $(typeof(var))"
    Expr(:curly, Core.Type, map(p -> override_func_funcarg(p, wheres), var.parameters)...)
end
function override_func_funcarg(var::Type{Tuple{Vararg{Any}}}, wheres)
    #function override_func_funcarg(var::Type{Core.Tuple{Vararg{Any}}}, wheres)
    @info "Tuple::: $var $(typeof(var))"
    Expr(:curly, Core.Tuple, map(p -> override_func_funcarg(p, wheres), var.parameters)...)
end



function override_func_funcarg(var::Number, wheres)
    var
end

function override_func_funcarg(var::UnionAll, wheres)
    args = []
    lastbound = 0
    while true
        @info "$(var.var.name) $wheres"
        if Symbol(var.var.name, 666) in wheres
            @assert lastbound == length(args)
            lastbound += 1
        end
        push!(args, var.var)
        var = var.body
        var isa UnionAll || break
    end
    @assert lastbound == 0
    w = var.name.wrapper
    wparameters = Base.unwrap_unionall(w).parameters
    @assert length(wparameters) == length(var.parameters)
    pars = map(i -> (var.parameters[i] isa TypeVar && !(Symbol(var.parameters[i], 666) in wheres)) ? var.parameters[i] : wparameters[i], 1:length(var.parameters))
    nvar = w{pars...}
    for i in length(args):-1:1
        nvar = UnionAll(var.parameters[i], nvar)
    end
    for i in length(var.parameters):-1:1
        if !(var.parameters isa TypeVar)
            nvar = UnionAll(pars[i], nvar)
        end
    end
    pars = []
    for p in var.parameters
        if !(p isa TypeVar && !(Symbol(p, 666) in wheres))
            push!(pars, override_func_funcarg(p, wheres))
        end
    end
    for i in 1:lastbound
        push!(pars, override_func_funcarg(args[i], wheres))
    end
    @info "Union::: $var $(typeof(var)) $(pars) $lastbound $args"
    Expr(:curly, nvar, pars...)
end
#################
##################
#################
###############
###########
function override_func_funcarg(var, wheres)
    if var isa TypeVar
        if (nsym = Symbol(var.name, 666)) in wheres
            nsym
        else
            var
        end
    elseif var isa Core.TypeofVararg
        Expr(:curly, Core.Vararg, override_func_funcarg(var.T, wheres), override_func_funcarg(var.N, wheres))
    elseif var isa UnionAll
        @info "UnionAll $var $(typeof(var))"
        nstrip = 0
        while true
            nstrip += 1
            var = var.body
            var isa UnionAll || break
        end
        x = var
        for p in Iterators.reverse(var.parameters)
            if p isa TypeVar
                x = UnionAll(p, x)
            end
        end
        Expr(:curly, x, map(p -> (p.name in wheres ? p.name : override_func_funcarg(p, wheres)), var.parameters[1:(end-nstrip)])...)
    else
        @info "DataType $var $(typeof(var))"
        @assert var isa DataType
        if isempty(var.parameters)
            var
        else
            x = var
            hit = 0
            reps = map(p -> override_func_funcarg(p, wheres), var.parameters[1:end])
            isempty(reps) && return var
            #hit==0 && return var
            Expr(:curly, x.name.wrapper, reps...)
        end
    end
end
function override_func_signature(sig::UnionAll)
    args = []
    wheres = Set()
    while true
        push!(args, sig.var)
        push!(wheres, Symbol(sig.var.name, 666))
        sig = sig.body
        sig isa UnionAll || break
    end
    Expr(:where,
        Expr(:call,
            Expr(:., :Cassette, QuoteNode(:prehook)), Expr(:(::), :ctx, OdCtx),
            map(t -> Expr(:(::), override_func_funcarg(t, wheres)), sig.types)...),
        map(override_func_where_arg, args)...,
    )
end
############
function override_func_signature(sig::DataType)
    Expr(:call, Expr(:., :Cassette, QuoteNode(:prehook)), Expr(:(::), :ctx, OdCtx),
        map(t -> Expr(:(::), t), sig.types)...)
end
#*        Symbol ::
#*          Symbol: y
#*          Symbol: T
#*        Symbol ::
#*          Symbol: z
#*          Symbol: Vector
#*        Symbol ::
#*          Symbol: w
#*          Symbol: S
#*    Symbol block
#*      LineNumberNode: #= REPL[123]:1 =#
#*      LineNumberNode: #= REPL[123]:1 =#
#*      Symbol call
#*        Symbol: xx
#*        Symbol curly
#*          Symbol: Tuple
#*          Symbol curly
#*            Symbol: Vector
#*            Symbol: T
#*          Symbol: T
#*          Symbol: S
#
#
#*
#############
function override_od(sig::Type{<:Tuple})
    @info "Overload prehook: $sig $(objectid(sig))"

    ##return
    ##return
    lhs = override_func_signature(sig)
    expr = Expr(:function, lhs, Expr(:call, record_func, :ctx, sig))
    @info "Overload prehook: $expr"
    eval(expr)
end

function remap_type(t::DataType)
    @info "map eval $t $(objectid(t))"
    t = eval(reduce((a, b) -> Expr(:., a, QuoteNode(b)), (fullname(parentmodule(t))..., nameof(t))))
    @info "map  out $t $(objectid(t))"
    t
end
# XXX
function remap_type(t::Type{Type{T}}) where {T}
    Type{remap_type(T)}
end
function remap_type(t::Core.TypeofVararg)
    @info "map id $t"
    t
end
function prepare_overdub()
    whitelistx = Set()
    if Revise[] !== Base
        #@info "expressions withrevise"
        for (mm, mkfile) in totrack
            #@info "start track $mm $mkfile"
            #XXX :sigs #:includet #:eval #:sigs, :eval, :evalmeth, :evalassign
            Revise[].track(mm, mkfile)
            id = Base.moduleroot(mm) == Main ? Base.PkgId(mm, string(mm)) : Base.PkgId(mm)
            for (m, sigss) in Revise[].get_expressions(id, mkfile)
                for (ex, sigs) in sigss
                    for sig in sigs
                        push!(whitelistx, sig)
                        #@info "$sig"
                    end
                end
            end
        end
        #empty!(totrack)
    end
    whitelist = Vector()
    for sig in whitelistx
        #@info "white list xxx $sig $(objectid(sig.types[1]))"
        #types=map(remap_type, sig.types)
        #sig=Tuple{types...}
        push!(whitelist, sig => true)
        #@info "white list yyy $sig $(objectid(sig.types[1]))"
        #@info "white list $sig $(objectid(sig.types[1]))"
    end
    #@info "world named $(@eval objectid(typeof(Main.Xxx.named)))"
    #@info "world target $(@eval objectid(typeof(Main.target)))"
    allmethods = methods(Cassette.prehook)
    toremove = []
    for m in allmethods
        sig = m.sig
        vars = []
        while sig isa UnionAll
            push!(vars, sig.var)
            sig = sig.body
        end
        @assert sig.name.wrapper === Tuple
        types = sig.types
        @assert length(types) >= 2
        tph, tctx, tfargs... = types
        @assert Cassette.prehook isa tph
        sig = Tuple{tfargs...}
        for v in Iterators.reverse(vars)
            sig = UnionAll(v, sig)
        end
        if (hit = findfirst(isequal(sig => true), whitelist)) !== nothing
            whitelist[hit] = sig => false
        elseif tctx === OdCtx
            @info "delete method: $m $sig $(typeof(whitelistx))"
            for s2 in whitelistx
                eq1 = sig == s2
                eq2 = sig === s2
                @info "$eq1 $eq2 $sig $s2"
            end
            Base.delete_method(m)
        end
    end
    for (sig, add) in whitelist
        if add
            override_od(sig)
        end
    end
end

run_seq = Ref{Int}(0)
_during_make = Ref{Bool}(false)
function make_targets(tgts::AbstractVector{<:AbstractString})
    isempty(tgts) && return
    GC.gc()
    _during_make[] = true
    try
        #prepare_overdub()
        list = build_graph(tgts)
        Base.invokelatest(run_graph, list)
        res = []
        for t in tgts
            @assert isabspath(t)
            info = input_types[t]
            v = if info.phony
                nothing
            elseif info.data
                info.dataval[]
            else
                info.path
            end
            push!(res, v)
        end
        if length(res) == 1
            return res[1]
        else
            (res...,)
        end
    finally
        _during_make[] = false
    end
end
function run_graph(list::AbstractVector{Rule})
    run_inst = (run_seq[] += 1)
    for l in list
        @info "MAKE..... $run_inst $l"
    end
    hasupd = true
    while hasupd
        hasupd = false
        for (i, l) in enumerate(list)
            if check_and_run(l, run_inst)
                hasupd = true
                break
            end
        end
    end
end

function depshash(io, d::DepEntry, dir)::Float64
    if d.sym !== nothing
        println(io, d.sym)
    end
    maxts = -Inf
    files = d.files
    if files isa AbstractString
        files = [files]
    end
    for file in files
        @assert file isa AbstractString
        @assert isabspath(file)
        println(io, relpath(file, dir))
        #@info file
        info = @something(get(input_types, file, nothing))
        #@info "check: $file $info"
        if info.phony
            @assert !info.checksum && !info.data
        else
            ts, chk = getchkinfo(info)
            #@info "check: $file $ts $chk"
            if chk !== nothing
                println(io, chk)
            end
            maxts = max(maxts, ts)
        end
    end
    maxts
end

function decode_dbline(str::AbstractString)
    try
        Serialization.deserialize(IOBuffer(Base64.base64decode(str)))
    catch e
        @error "failed to read dbline $e"
        return nothing
    end
end

function dbfile()
    return "julia-makefile-state.log"
end

function format_dbline(path::AbstractString; kwargs...)
    res = (; type=:runinfo, path=path)
    for (k, v) in kwargs
        if k === :codesigs
            # XXX
            res = (; res..., k => v)
        else
            res = (; res..., k => v)
        end
    end
    format_dbline(res)
end
function format_dbline(t::NamedTuple)::AbstractString
    res = Base64.base64encode() do io
        Serialization.serialize(io, t)
    end
    @assert decode_dbline(res) == t
    return res
end

const cached_expr_hash = IdDict{Type{<:Tuple},Array{UInt8}}()
function get_sig_expr_hash(sig::Type{<:Tuple})::Array{UInt8}
    #sig=sig|>sig2str|>str2sig
    if (c = get(cached_expr_hash, sig, nothing)) !== nothing
        return c
    end
    meth = try
        Base.which(sig)
    catch
        return UInt8[]
    end
    expr = CodeTracking.definition(Expr, meth)
    io = IOBuffer()
    calc_hash_expr(io, expr)
    cached_expr_hash[sig] = take!(io)
end

function sig2str(sig::Type{<:Tuple})
    io = IOBuffer()
    Serialization.serialize(io, sig)
    take!(io)
end
function str2sig(str)::Type{<:Tuple}
    Serialization.deserialize(IOBuffer(str))
end

function load_data(file)
    file = "$file.dat"
    @info "load $file"
    !isfile(file) && return nothing
    open(file) do io
        str = CodecZlib.GzipDecompressorStream(io)
        try
            Serialization.deserialize(str)
        catch e
            e isa UndefVarError && return nothing
            e isa TypeError && return nothing
            rethrow()
        end
    end
end

function save_data(file, data)
    file = "$file.dat"
    @info "save $file"
    open(file, "w") do io
        str = CodecZlib.GzipCompressorStream(io)
        Serialization.serialize(str, data)
        close(str)
    end
end

function get_total_code_hash(rulecode, sigs::AbstractVector{Type{<:Tuple}})
    io = IOBuffer()
    write(io, rulecode.second)
    for h in sort(unique(map(get_sig_expr_hash, sigs)))
        #@info "code cond $(String(copy(h))))"
        write(io, h)
    end
    take!(io) |> SHA.sha256
end

function check_and_run(rule::Rule, run_inst::Int)
    info = @something(get(input_types, joinpath(rule.target.dir, rule.target.file), nothing))
    #@info "hhhh"
    runinfo = info.runinfo
    if !info.data && !info.phony && !isfile(info.path)
        # XXX check only active outputs
        @info "reset ts... $(rule.target.file)"
        runinfo.runts = -Inf
    end
    if info.data && !info.nosave && !info.phony && !isfile(info.path * ".dat")
        # XXX check only active outputs
        @info "reset ts... $(rule.target.file) $(info.path)"
        runinfo.runts = -Inf
    end
    codecond = get_total_code_hash(rule.code, runinfo.codesigs)
    maxts, datacond = let
        maxts = -1e9
        io = IOBuffer()
        if rule.phony
            println(io, run_inst)
        end
        for (s, v) in sort(collect(pairs(rule.stems)))
            println(io, "$s=$v")
        end
        for d in rule.deps
            ts = depshash(io, d, dirname(rule.mkfile))
            maxts = max(maxts, ts)
        end
        maxts, take!(io) |> SHA.sha256
    end

    ##@info "cond:  $(rule.target.file) $maxts>$(runinfo.runts) $codecond $datacond"
    if maxts > runinfo.runts || (codecond, datacond) != (runinfo.codecond, runinfo.datacond)
        #=
        dbgfn="/tmp/testjmake/$(basename(rule.target.file)).txt"
        open(dbgfn, "w") do io2
        write(io2, bytes)
        end
        =#
        if !rule.target.nosave
            maxts = max(maxts, time())
        end

        codesigs = nothing
        args = prepare_args(rule)
        curr_age, latest_age = get_world_info()
        ret = if codecond == runinfo.codecond || true
            @info "RUN------------------ $curr_age/$latest_age $(rule.target.file)"
            @time rule.code.first(args...)
        else
            @info "RUN++++++++++++++++++ $curr_age/$latest_age $(rule.target.file)"
            ctx = OdCtx(metadata=codesigs = Set{Type{<:Tuple}}())
            @time Cassette.overdub(ctx, rule.code.first, args...)
        end
        if codesigs !== nothing
            empty!(runinfo.codesigs)
            codesigs = append!(runinfo.codesigs, codesigs)
            codecond = get_total_code_hash(rule.code, codesigs)
        end

        if info.data
            info.dataval[] = ret
            push!(info.datats, maxts)
            datasum = string(hash(ret))
            push!(info.datasum, datasum)
            if !rule.target.nosave
                tosave = (type=:data, path=rule.target.file,
                    data=ret, datasum=datasum,
                    runts=maxts,
                    codecond=codecond,
                    datacond=datacond,
                    codesigs=codesigs,
                )
                save_data(rule.target.file, tosave)
            end
        end
        update_info(info)
        for bp in rule.byproducts
            update_info(@something(get(input_types, joinpath(bp.dir, bp.file), nothing)))
        end

        runinfo.runts = maxts
        runinfo.codecond = codecond
        runinfo.datacond = datacond

        let
            info = @something(get(input_types, joinpath(rule.target.dir, rule.target.file), nothing))
            runinfo = info.runinfo
        end

        if !rule.phony && !rule.target.nosave && !rule.target.data
            runinfo = format_dbline(rule.target.file;
                runts=maxts,
                codecond=codecond,
                datacond=datacond,
                codesigs=codesigs
            )
            open(joinpath(dirname(rule.mkfile), dbfile()), "a") do io
                println(io, runinfo)
            end
        end
        true
    else
        false
    end
end
####################################################################################
##dyn need: restore: dynamic_deps
#################################################################################
################################################################
###############################################################################
#############################################################################
#######################################################
#=
=#
function getchkinfo(info::InputInfo)
    return last(info.cachedvals)
end

function get_value_for(tgt::AbstractString)
    @assert isabspath(tgt)
    info = @something(get(input_types, tgt, nothing))
    if info.phony
        nothing
    elseif info.data
        info.dataval[]
    else
        info.path
    end
end
function get_value_for(tgts::AbstractVector{<:AbstractString})
    map(get_value_for, tgts)
end

function prepare_args(rule::Rule)
    args = []
    if rule.target.sym !== nothing
        push!(args, get_value_for(joinpath(rule.target.dir, rule.target.file)))
    end
    push!(args, Symbol("@@SEP1@@"))
    push!(args, Symbol("@@SEP2@@"))
    for s in sort(collect(keys(rule.stems)))
        push!(args, rule.stems[s])
    end
    push!(args, rule)
    for dep in rule.deps
        if dep.inlet && dep.sym !== nothing
            push!(args, get_value_for(dep.files))
        end
    end
    for dep in rule.deps
        if !dep.inlet && dep.sym !== nothing
            push!(args, get_value_for(dep.files))
        end
    end
    return args
end

#=
function Cassette.prehook(ctx::OdCtx, f, a1::Float64, a2::Function)
    println("FFF: $(objectid(typeof(f))) $f")
end
=#

const totrack = Set()

function hash_or_content(str)
    if debug
        str
    else
        SHA.sha256(str)
    end
end

function build_function(mod::Module, body::Expr, stems, target::Target, deps, deps2)
    hash = let
        io = IOBuffer()
        calc_hash_expr(io, body)
        hash_or_content(take!(io))
    end
    args = []
    if target.sym !== nothing
        push!(args, target.sym)
    end
    push!(args, Symbol("@@SEP1@@"))
    #=
    for bp in byproducts
        if bp.sym!==nothing
            push!(args, bp.sym)
        end
    end
    =#
    push!(args, Symbol("@@SEP2@@"))
    for s in sort(collect(keys(stems)))
        push!(args, s)
    end
    push!(args, Symbol("ENDDEEEEEEEEEEEE"))
    for dep in deps
        @assert !dep.dynamic
        if dep.sym !== nothing
            push!(args, dep.sym)
        end
    end
    for dep in deps2
        @assert !dep.dynamic
        if dep.sym !== nothing
            push!(args, Symbol("@@@", dep.sym, "@@@"))
        end
    end
    func = mod.eval(Expr(:function, Expr(:tuple, args...), body))
    fn = @something(target.file, target.pattern)
    fn = basename("$fn")
    mkpath("/tmp/testjmake")
    dbgfn = "/tmp/testjmake/$fn.code"
    if isfile(dbgfn)
        #mv(dbgfn, "$dbgfn.bak", force=true)
    end
    #=
    open(dbgfn, "w") do io
        println(io, code_for_func(func))
    end
    =#
    @assert func isa Function
    func => hash
end



##################################################################
##################################################################
##############################################################
##############################################################
#############################################################




function pass_deps(args, weak)
    deps = []
    for a in args
        if a isa Expr
            if a.head === :(=) && length(a.args) == 2 && a.args[1] isa Symbol
                push!(deps, Symbol("@@@", a.args[1], "@@@"))
            else
            end
        elseif a isa AbstractArray
        elseif a isa AbstractString
        else
            error("wrong deps pass: $a")
        end
        #=
        if a isa AbstractString
        else
        =#
    end
    Expr(:tuple, deps...)
end
function parse_make_targets(mod, dir, args)
    targets = []
    for a in args
        priv.parse_target!(mod, targets, a)
    end
    #@info "dir: $(typeof(dir)) $(typeof(__source__))"
    if dir === nothing || isempty(dir)
        dir = pwd()
    end
    map(targets) do t
        Expr(:call, joinpath, dir, t)
    end
end

function resolve_sym(mod, expr)
    #if expr isa Expr && expr.head===:escape && length(expr.args)==1
    #return resolve_sym(mod, expr.args[1])
    if expr isa Symbol
        return GlobalRef(mod, expr)
    elseif expr isa QuoteNode
        return resolve_sym(mod, expr.value)
    elseif expr isa Expr && expr.head === :. && length(expr.args) == 2
        return resolve_sym(resolve_mod(mod, expr.args[1]), expr.args[2])
    else
        error("wrong sym: $(typeof(expr)) $expr")
    end
end

function parse_sig(mod, a, typ, inkw=false)::Union{GlobalRef,Missing}
    if typ && a isa LineNumberNode
    elseif a isa Expr && a.head === :(::) && length(a.args) == 2
        return resolve_sym(mod, a.args[2])
    elseif a isa Symbol
        return GlobalRef(Core, :Any)
    elseif !typ && a isa Expr && a.head === :parameters
    elseif !typ && !inkw && a isa Expr && a.head === :kw && length(a.args) == 2
        return parse_sig(mod, a.args[1], typ, true)
    else
        error("wrong sig: $a")
    end
    missing
end
function getlhs(mod, expr)
    if expr isa Expr && expr.head === :call
        name = resolve_sym(mod, expr.args[1])
        sigs = (skipmissing(map(a -> parse_sig(mod, a, false), expr.args[2:end]))...,)
        return name, sigs
    elseif expr isa Symbol
        # XXX
        return resolve_sym(mod, expr), tuple()
    else
        error("wrong lhs: $(typeof(expr)) $expr")
    end
end

function remove_escape(expr, lvl)
    @assert lvl >= 0
    lvl == 0 && return expr
    if expr isa Expr && expr.head === :escape && length(expr.args) == 1
        remove_escape(expr.args[1], lvl - 1)
    elseif expr isa Expr
        Expr(expr.head, map(a -> remove_escape(a, lvl), expr.args)...)
    elseif expr isa Union{Symbol,Number,AbstractString}
        expr
    elseif expr isa LineNumberNode
        expr
    else
        error("wrong xxx $(typeof(expr))")
    end
end

const exprhash_db = Dict{String,Dict{Tuple,String}}()
const exprhash_todo = Vector{Tuple{GlobalRef,Tuple,String}}()
function ref_to_name(ref::GlobalRef)
    mod = which(ref.mod, ref.name)
    get_abs_mod(mod) * ".$(ref.name)"
end
function update_exprhash()
    for (lhs, sig, str) in exprhash_todo
        lhs = ref_to_name(lhs)
        sig = map(ref_to_name, sig)
        mm = get!(Dict{Tuple,String}, exprhash_db, lhs)
        mm[sig] = SHA.sha256(str) |> bytes2hex
        if debug
            open("/tmp/testjmake/$lhs.$sig.expr", "w") do io
                Base.print(io, str)
            end
        end
    end
    empty!(exprhash_todo)
end
function dump_exprhash()
    update_exprhash()
    for (n, mm) in sort(exprhash_db)
        println(n)
        for (s, h) in sort(mm)
            println("-- ", s, " = ", h)
        end
    end
end

function get_abs_mod(mod::Module)::String
    pmod = parentmodule(mod)
    if pmod === mod
        String(nameof(mod))
    else
        get_abs_mod(pmod) * "." * String(nameof(mod))
    end
end
function calc_hash_expr(io::IO, expr, lvl=0)
    if expr isa Expr
        println(io, repeat("  ", lvl), "Expr ", expr.head)
        for a in expr.args
            calc_hash_expr(io, a, lvl + 1)
        end
    elseif expr isa LineNumberNode
    else
        println(io, repeat("  ", lvl), typeof(expr), " ", expr)
    end
end
function add_hash_expr(name::GlobalRef, sig::Tuple, expr)
    io = IOBuffer()
    calc_hash_expr(io, expr)
    str = String(take!(io))
    push!(exprhash_todo, (name, sig, str))
end

const cached_mod_infos = IdDict()
function get_mod_for_include(mod::Module)
    if (m = get(cached_mod_infos, mod, nothing)) !== nothing
        return m
    end
    ppp = mod
    for f in stacktrace(true)
        if f.from_c
            if f.func === :jl_eval_module_expr
                p = parentmodule(mod)
                @assert p !== mod
                mod = p
            elseif f.func === :ijl_toplevel_eval_in
                break
            end
        end
    end
    if fullname(mod) == (:Main, :Xxx, :Yyy)
        for f in stacktrace(true)
            @error "stack $f"
        end
    end
    #@info "hahahahahah $ppp $mod"
    cached_mod_infos[ppp] = mod
    mod
end

function clearmem()
    for (k, v) in input_types
        if v.data
            if v.nosave
                v.runinfo.runts = -Inf
                empty!(v.runinfo.datacond)
            elseif v.path in coldboot_data
                pop!(coldboot_data, v.path)
            end
            v.dataval[] = nothing
        end
    end
end

end



macro makefile(arg)
    #@info "mod $(__module__), eval $arg"
    arg = __module__.eval(arg)
    quote
        priv.makefile($arg)
    end
end

macro clearmem()
    quote
        priv.clearmem()
    end
end






end # module Jmake

#= Author's notes

# XXX @make for dynamic deps
#Features:
#latex aux/bibtex loops (weekdeps)
#
#?filebased: metadata in central store
#?data(jld2)based: data in eachjld, metadata in eachjld
#
#
#########################################
############################
#######################
#
#preserve rel/abs path
#
#
# 
#######################
############################
#
#Later: 
#*** if include again ? Main.Xxx is new; definitions is old; degrade Main.Xxx or upgrade definitions, or hack definitions;
#will sig outdated???
#3. efficiency comparable
#   od 0.000242 seconds (561 allocations: 39.078 KiB)
#   od 0.000304 seconds (560 allocations: 38.781 KiB)
#   od 0.000313 seconds (561 allocations: 39.078 KiB)
#   dd 0.000235 seconds (447 allocations: 31.766 KiB)
#   dd 0.000280 seconds (447 allocations: 31.766 KiB)
#   dd 0.000238 seconds (447 allocations: 31.766 KiB)
#
#
#
#* byproduct must be a file!!! ( for data, no need for this!, put all in return value)
=#
