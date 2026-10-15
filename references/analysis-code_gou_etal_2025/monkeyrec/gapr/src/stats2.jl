using Distributed
if (length(workers()) < 3)
    addprocs(5)
end
@everywhere using DelimitedFiles
@everywhere using DataFrames
@everywhere using LinearAlgebra
@everywhere using Random
using Dates
import XLSX
using StaticArrays
using Statistics
import NearestNeighbors
using Plots
using StatsPlots
using Images
using JSON
import LaTeXStrings
import IniFile
import HypothesisTests

pgfplotsx()
function tex(s)
    LaTeXStrings.LaTeXString(s)
end

@everywhere function gapr_build_dir()
    homedir = ENV["HOME"]
    "$homedir/Desktop/gapr/build"
end
@everywhere function gapr_cmd(cmd::Cmd)
    build_dir = gapr_build_dir()
    c = cmd.exec[1]
    cmd.exec[1] = "$build_dir/source/$c/gapr-$c"
    return cmd
end
@everywhere function check_exit(cc)
    if typeof(cc) == Base.Process
        if cc.exitcode != 0 || cc.termsignal != 0
            error("error run: $cc")
        end
    elseif typeof(cc) == Base.ProcessChain
        for p in cc.processes
            check_exit(p)
        end
    end
end

const _async_run_all_cmds = []
function _async_run_popone(cmds)
    cc, bb = popfirst!(cmds)
    wait(cc)
    check_exit(cc)
    bb()
end
function async_run(npar, cmd, cb)
    if length(_async_run_all_cmds) >= npar
        _async_run_popone(_async_run_all_cmds)
    end
    cmd = run(cmd, wait=false)
    #@info "enque cmd: $cmd"
    push!(_async_run_all_cmds, (cmd, cb))
end
function async_run_finish()
    while !isempty(_async_run_all_cmds)
        _async_run_popone(_async_run_all_cmds)
    end
end

import Revise
Revise.includet("0-make.jl")
Revise.track(@__FILE__)

cache_file_stats_gz(ds::String) = "/data/tmp/raw-stats-$ds.stats.gz"
function cache_file_stats(ds::String, stg::Int)
    f, t = range_from_stage(ds, stg)
    "/data/tmp/stats-$ds-$(stg+1000)-$f-$t.stats"
end
cache_file_progress(ds::String, from::Int, to::Int) = "/data/tmp/progress-$ds-$from-$to.stats"
function cache_file_errors(ds::String)
    f, t = range_from_stage(ds, 9)
    "/data/tmp/errors-$ds-$f-$t.txt"
end
function cache_file_swcdir(ds::AbstractString, stg::Int)
    f, t = range_from_stage(ds, stg)
    "/data/tmp/results-$ds-$(stg+1000)-$f-$t.swc"
end
cache_dir_swcs_xform() = "/data/tmp/swcs-xform"
cache_dir_swcs_xform_mirror() = "/data/tmp/swcs-xform-mirror"

datasets() = (Make.@make datasets)
function dataset_from_id(ds::AbstractString)
    dss = datasets()
    idx = findfirst(v -> (v.id == ds), dss)
    dss[idx]
end
function range_from_stage(ds::AbstractString, stg::Int)
    d = dataset_from_id(ds)
    f = map(p -> (p.first < stg ? p.second[1] : 0), collect(d.stgs)) |> maximum
    t = d.stgs[stg][1]
    f, t
end
function mirror_files(ds::AbstractString, typ::Symbol)
    homedir = ENV["HOME"]
    mirror_dir = "$homedir/Desktop/gapr-mirror"
    if typ == :repo
        return "$mirror_dir/confdir/REPO/fmost$ds.repo"
    elseif typ == :cat
        return "$mirror_dir/confdir/DATA/fmost$ds.catalog"
    else
        error("wrong typ")
    end
end

function parse_stats(buf)
    pop_punct(s) = SubString(s, 1, lastindex(s) - 1)
    filesize(buf) != 0 || return DataFrame()
    v = readdlm(buf, ' ')
    r = DataFrame([
        :id => [parse(Int, pop_punct(x)) for x in v[:, 3]],
        :who => String[pop_punct(x) for x in v[:, 5]],
        :time => [parse(DateTime, x[1:(length(x)-6)]) for x in v[:, 7]],
        :type => String[pop_punct(x) for x in v[:, 9]],
        :score => [x for x in v[:, 11]],
        :nums => [
            (
                v[i, 12:15]...,
                parse(Int, pop_punct(v[i, 16])),
                v[i, 17:19]...,
                parse(Int, pop_punct(v[i, 20])),
                parse(Int, pop_punct(v[i, 21])),
                parse(Int, pop_punct(v[i, 22])),
                v[i, 23:25]...,
                parse(Int, pop_punct(v[i, 26])),
                v[i, 27:28]...
            )
            for i in 1:size(v)[1]]
    ])
    r
end

function __check_neuron_qc(ds::AbstractString, stg::Int, ord::Int, nrn::AbstractString, fn::AbstractString)
    stg % 10 != 9 && return 0
    ord >= 999000 && return 0

    if match(r"^(-\d+-TREES|-JUNK)\b", nrn) !== nothing
        return 0
    end

    name = nothing
    if (m = match(r"^@\s*(\d{3}|q?root\d{3}|a\d{3}|b\d{3}|c\d{3}|d\d{3}|s\d{3})(\b|[a-z])", nrn)) !== nothing
        name = m.captures[1]
    elseif match(r"^@\s*(seg\d{3}\b.*|seg\b.*|attach\b.*|)$", nrn) !== nothing
        return 0
    else
        @error "$fn"
        @assert false
    end

    function get_table(ds, fn)
        global __qc_table__
        if !@isdefined(__qc_table__) || __qc_table__ === nothing
            __qc_table__ = Dict()
        end
        qc = get(__qc_table__, ds, nothing)
        qc === nothing || return qc
        qc = Dict{String,String}()
        v, hdr = readdlm("precious/neuron_annotation/$fn", ',', String, header=true, quotes=true)
        @assert hdr[1] == "Name"
        for i = 1:size(v, 1)
            n = v[i, 1]
            q = if size(v, 2) >= 2
                v[i, 2]
            else
                "Good"
            end
            @assert q in ("Good", "Bad")
            qq = get!(qc, n, q)
            qq == q || error("duplicate name $ds/$n")
        end
        __qc_table__[ds] = qc
    end

    if ds in ("17391", "192095", "192096")
        qc = get_table(ds, "$ds.csv")
        q = get(qc, name, nothing)
        if q === nothing
            println("!!! empty qc for $ds/$nrn")
            return 1
        end
        if q == "Good"
            return 2
        elseif q == "Bad"
            return 1
        else
            error("unknown qc $q")
        end
    end

    if ds in ("18708", "18709", "18133", "192100", "192102", "202813", "202814", "202815", "202816")
        m = match(r"(\b|\d)(good|bad)(\b|\d)", nrn)
        if m !== nothing
            return m.captures[2] == "good" ? 2 : 1
        end
        return 1
    end

    if ds in ("18925", "18926", "192101")
        if name[1] == 'c'
            return 2
        end
        return 1
    end
    return -1
end

function _check_neuron_qc(ds::AbstractString, stg::Int, ord::Int, nrn::AbstractString, fn::AbstractString)
    ret = __check_neuron_qc(ds, stg, ord, nrn, fn)
    open("/tmp/gapr-chk-nrn.txt", "a") do io
        println(io, "$ret ds:$ds stg:$stg fn:$fn")
    end
    ret
end
function check_neuron_qc(fn::AbstractString)
    if (m = match(r"^(\w+)-10(\d+)-(\d+(@.*|-\d+-TREES|-JUNK)\.swc(\.gz)?)$", fn)) !== nothing
        ds = m.captures[1]
        stg = m.captures[2]
        fn = m.captures[3]
        check_neuron_qc(ds, stg, fn)
    else
        error("cannot guess ds nrn: $fn")
    end
end
function check_neuron_qc(ds, stg::AbstractString, fn)
    check_neuron_qc(ds, parse(Int, stg), fn)
end
function check_neuron_qc(ds, stg::Int, fn)
    m = match(r"((\d+)(@\s*(\w*).*|-\d+-TREES|-JUNK)\.swc(\.gz)?)$", fn)
    @assert m !== nothing
    bn = m.captures[1]
    ord = parse(Int, m.captures[2])
    _check_neuron_qc(ds, stg, ord, m.captures[3], fn)
end

plotstyle(extra...; scaleonlyaxis=false) = (
    colorbar_tickfontsize=5,
    colorbar_titlefontsize=6,
    grid=:none,
    guidefontsize=6,
    legend_font_pointsize=5,
    linewidth=0.5,
    markerstrokewidth=0.5,
    tickfontsize=5,
    extra_kwargs=Dict(:subplot => Dict(
        "scale only axis" => scaleonlyaxis,
        "every axis legend/.append style" => "line width={.5}",
        "every outer x axis line/.append style" => "line width={0.5}",
        "every outer y axis line/.append style" => "line width={0.5}",
        extra...,
    )),
)

module My
struct DatasetInfo
    id::String
    stgs
    area
    autopr
    prober
    halfz
    fix_diff_term
    mode
    strain
    label
    imaging
    host
    function DatasetInfo(id, stgs;
        area="unknown", autopr=false,
        prober=false, fix_diff_term=false, halfz=false,
        mode=nothing, strain=nothing, label=nothing,
        imaging=nothing, host=nothing)
        new(id, stgs, area, autopr, prober, halfz,
            fix_diff_term, mode, strain, label, imaging, host)
    end
end
end

Make.@target datasets () -> begin
    dss = [
        My.DatasetInfo("18133", Dict(
                0 => (2027),
                1 => (377080, "open"),
                2 => (551691, "open", "partial1"),
                3 => (697760, "open", "partial3"),
                9 => (-1)
            ),
            area="VMH",
            fix_diff_term=true,
            mode="diff",
            strain="Esr1-2A-Cre",
            label="virus stereotaxic injection",
            imaging="TDI",
        ),
        My.DatasetInfo("17391", Dict(
                0 => (3814),
                1 => (1135773, "open"),
                2 => (1612097, "restricted", "all"),
                9 => (-1),
            ),
            area="VMH",
            strain="Esr1-2A-Cre",
            label="virus stereotaxic injection",
            imaging="TDI",
        ),
        My.DatasetInfo("18925", Dict(
                0 => (1868),
                1 => (106109, "open"),
                2 => (163467, "restricted", "all"),
                9 => (176886),
                11 => (218778, "restricted", "all"),
                19 => (220402),
                21 => (265039, "restricted", "all"),
                29 => (267220),
                31 => (312235, "restricted", "all"),
                39 => (314609),
                41 => (363762, "restricted", "all"),
                49 => (-1),
            ),
            area="HY", #RCH AHN MPO
            autopr=true,
            prober=true,
            strain="Prokr2-Cre",
            label="virus stereotaxic injection",
            imaging="TDI",
        ),
        My.DatasetInfo("18926", Dict(
                0 => (330),
                9 => (-1),
            ),
            area="HY",
            autopr=true,
            prober=true,
            mode="fast",
            strain="Prokr2-Cre",
            label="virus stereotaxic injection",
            imaging="TDI",
        ),
        My.DatasetInfo("18926.3", Dict(
                0 => (0),
                9 => (-1),
            ),
            area="SCN",
            mode="manual",
        ),
        My.DatasetInfo("18926.4", Dict(
                0 => (0),
                9 => (-1),
            ),
            area="SCN",
            mode="manual",
        ),
        My.DatasetInfo("192095", Dict(
                0 => (876),
                1 => (78396, "open"),
                2 => (115587, "restricted", "all"),
                9 => (118476),
                11 => (160264, "restricted", "all"),
                19 => (161768),
                21 => (197617, "restricted", "all"),
                29 => (199134),
                31 => (233204, "restricted", "all"),
                39 => (234570),
                41 => (276009, "restricted", "all"),
                49 => (-1),
            ),
            area="VMH",
            strain="Nr5a1-Cre",
            label="virus stereotaxic injection",
            autopr=2,
            prober=true,
            halfz=true,
        ),
        My.DatasetInfo("192096", Dict(
                0 => (2411),
                1 => (83529, "open"),
                2 => (125948, "restricted", "all"),
                9 => (129398),
                11 => (174653, "restricted", "all"),
                19 => (176447),
                21 => (216585, "restricted", "all"),
                29 => (218703),
                31 => (252940, "restricted", "all"),
                39 => (254559),
                41 => (298500, "restricted", "all"),
                49 => (-1),
            ),
            area="VMH",
            strain="Nr5a1-Cre",
            label="virus stereotaxic injection",
            autopr=2,
            prober=true,
            halfz=true,
        ),
        My.DatasetInfo("192101", Dict(
                0 => (4190),
                1 => (52325, "open"),
                2 => (71558, "restricted", "all"),
                9 => (-1),
            ),
            area="Whole brain", #MEA PVH
            autopr=2,#XXX
            prober=true,
            halfz=true,
            mode="selected",
            strain="GRPR-CreERT2:Ai139",
            label="tamoxifen induction",
            imaging="LIMO",
        ),
        My.DatasetInfo("202814", Dict(
                0 => (6695),
                1 => (210683, "open"),
                2 => (326395, "restricted", "all"),
                9 => (-1),
            ),
            area="ARH", #VMH LHA
            strain="AGRP-Cre",
            label="virus stereotaxic injection",
            autopr=true,
            prober=true,
        ),
        My.DatasetInfo("202813", Dict(
                0 => (5524),
                1 => (128478, "open"),
                2 => (190848, "restricted", "all"),
                9 => (-1),
            ),
            area="ARH",
            strain="AGRP-Cre",
            label="virus stereotaxic injection",
            autopr=true,
            prober=true,
        ),
        My.DatasetInfo("202816", Dict(
                0 => (5293),
                1 => (173938, "open"),
                2 => (266596, "restricted", "all"),
                9 => (-1),
            ),
            area="ARH",
            strain="AGRP-Cre",
            label="virus stereotaxic injection",
            autopr=true,
            prober=true,
        ),
        My.DatasetInfo("202815", Dict(
                0 => (5464),
                1 => (144035, "open"),
                2 => (217983, "restricted", "all"),
                9 => (-1),
            ),
            area="ARH",
            strain="AGRP-Cre",
            label="virus stereotaxic injection",
            autopr=true,
            prober=true,
        ),
        My.DatasetInfo("18708", Dict(
                0 => (3253),
                1 => (494707, "open"),
                2 => (736141, "open", "partial1"),
                9 => (-1),
            ),
            area="MPO",
            fix_diff_term=true,
            strain="Esr1-2A-Cre",
            label="virus stereotaxic injection",
            imaging="TDI",
            mode="diff",
        ),
        My.DatasetInfo("18709", Dict(
                0 => (9345),
                1 => (367689, "open"),
                2 => (663369, "open", "partial1"),
                9 => (-1),
            ),
            area="MPO",
            autopr=true,
            fix_diff_term=true,
            mode="diff",
            strain="Esr1-2A-Cre",
            label="virus stereotaxic injection",
            imaging="TDI",
        ),
        My.DatasetInfo("192100", Dict(
                0 => (500),
                1 => (246283, "open"),
                2 => (330776, "open", "partial??"),
                9 => (-1),
            ),
            area="Whole brain",
            fix_diff_term=true,
            mode="diff",
            strain="GRPR-CreERT2:Ai139",
            label="tamoxifen induction",
            imaging="TDI",
        ),
        My.DatasetInfo("192102", Dict(
                0 => (992),
                1 => (134444, "open"),
                2 => (188694, "open", "partial??"),
                9 => (-1),
            ),
            area="Whole brain",
            fix_diff_term=true, #all reset
            mode="diff",
            strain="GRPR-CreERT2:Ai139",
            label="tamoxifen induction",
            imaging="TDI",
        ),
        My.DatasetInfo("18925.resnet", Dict(0 => (365052), 10 => (-1),),
            host="goulf-ts:8003",
        ),
        My.DatasetInfo("18926.resnet", Dict(0 => (3487), 10 => (3703), 19 => (-1),),
            host="goulf-ts:8003",
        ),
        My.DatasetInfo("192096.resnet", Dict(0 => (300194), 10 => (-1),),
            host="goulf-ts:8003",
        ),
        My.DatasetInfo("18925.unet", Dict(0 => (-1),),
            host="goulf-ts:8003",
        ),
        My.DatasetInfo("18926.unet", Dict(0 => (-1),),
            host="goulf-ts:8003",
        ),
        My.DatasetInfo("192096.unet", Dict(0 => (-1),),
            host="goulf-ts:8003",
        ),
    ]
    function cmp_id(a, b)
        aa = split(a.id, '.')
        bb = split(b.id, '.')
        aaa = parse(Float64, aa[1])
        bbb = parse(Float64, bb[1])
        (aaa, a.id) < (bbb, b.id)
    end
    sort!(dss, lt=cmp_id)
    for ds in dss
        n = ds.id
        host = @something(ds.host, "localhost:30009")
        fn = cache_file_stats_gz(n)
        if !isfile(fn)
            cc = run(`curl -s -k -o $fn.temp https://$host/api/stats/fmost$n`)
            check_exit(cc)
            mv("$fn.temp", fn)
        end
        try
            open(`gzip -cd $fn`) do f
                m = match(r"Stats: \[(\d+), (\d+)\)", readline(f))
                if m.captures[1] != "0"
                    error("error 0: $n")
                end
                throw(parse(Int64, m.captures[2]))
            end
        catch tot
            isa(tot, Int64) || rethrow()
            for si in keys(ds.stgs)
                x = ds.stgs[si]
                if x[1] == -1
                    ds.stgs[si] = (tot)
                elseif tot < x[1]
                    error("err cnt: $n: $tot")
                elseif tot > x[1]
                end
            end
        end
    end
    dss
end

Make.@target df_stats (datasets) -> begin
    stats_cache = map(datasets) do ds
        parts = [(id=j, max=s[1], lines=[]) for (j, s) in pairs(ds.stgs)]
        parts = sort(parts)
        fn = cache_file_stats_gz(ds.id)
        open(`gzip -cd $fn`) do f
            r = r"^Commit: id: (\d+);"
            for l in eachline(f, keep=true)
                m = match(r, l)
                m === nothing && continue
                id = parse(Int64, m.captures[1])
                for part in parts
                    if id < part.max
                        push!(part.lines, l)
                        break
                    end
                end
            end
            for part in parts
                open(cache_file_stats(ds.id, part.id), "w") do of
                    for l in part.lines
                        write(of, l)
                    end
                end
            end
        end
        statdat = map(parts) do part
            v = parse_stats(cache_file_stats(ds.id, part.id))
            insertcols!(v, 1, :ds => ds.id, :stg => part.id)
            part.id => v
        end
        ds.id => statdat
    end

    to_join = []
    for (ds, statdat) in stats_cache, (stg, dat) in statdat
        if size(dat)[1] > 1
            push!(to_join, dat)
        end
    end
    df = vcat(to_join...)
    sort!(df, [:ds, :id])
    df.nums = map(v -> map(i -> Int32(i), v), df.nums)

    insertcols!(df, :time, :time_diff => NaN, after=true)
    allowmissing!(df, :time_diff)
    for pp in eachrow(unique(df[!, [:who, :ds]]))
        prev_time = DateTime(0)
        prev_stg = 0
        dsinfo = dataset_from_id(pp.ds)
        for i in findall(df.ds .== pp.ds .&& df.who .== pp.who)
            stg = df[i, :stg]
            @assert stg >= prev_stg
            time = df[i, :time]
            if time < prev_time
                error("error time")
            end
            df[i, :time_diff] = prev_time == DateTime(0) ? missing : Millisecond(time - prev_time).value / 1000.0
            prev_stg = stg
            prev_time = time
        end
    end

    tv = df[:, :time_diff]
    insertcols!(df, :time_dd => tv)
    max_dt = 3000
    to_clamp = [
        ("add_edge_", [(0, 60), (1, 30), (2, 30), (3, 30), (9, 60)]),
        ("add_patch_", [(0, 600), (9, 60)]),
        ("add_prop_", [(1, 60), (2, 30), (3, 30), (9, 60)]),
        ("chg_prop_", [(1, 30), (2, 30), (9, 60)]),
        ("del_patch_", [(0, 60), (1, 30), (2, 30), (3, 30), (9, 60)]),
        ("proofread_", [(0, 600), (1, 30), (2, 30), (3, 30), (9, 60)]),
        ("reset_proofread_", [(0, 60), (1, 30), (2, 30), (3, 30), (9, 60)]),
        ("reset_proofread_0_", [(0, 60), (1, 30), (2, 30), (3, 30), (9, 60)]),
    ]
    for ccc in to_clamp, ddd in ccc[2]
        sel = df[:, :type] .== ccc[1] .&& (df[:, :stg] .% 10) .== ddd[1] .&& (ismissing.(df[:, :time_dd]) .|| df[:, :time_dd] .> ddd[2])
        df[sel, :time_dd] .= ddd[2]
    end
    df[ismissing.(df[:, :time_dd]), :time_dd] .= max_dt
    return df
end

Make.@target snapshots (df_stats) -> begin
    df = df_stats
    snapshots = unique(df[!, [:ds, :stg]])
    insertcols!(snapshots, :time => 0.0)
    for iii = 1:size(snapshots)[1]
        ds = snapshots[iii, :ds]
        stg = snapshots[iii, :stg]
        l = df.ds .== ds .&& df.stg .== stg
        tv = df[l, :time_dd]
        tt = sum(tv) / 3600
        snapshots[iii, :time] = tt
    end
    snapshots
end

Make.@target snapshots2 (df_stats, snapshots) -> begin
    function get_swc_tot_sizes(files)
        totlen, totcnt = 0, 0
        if !isempty(files)
            open(gapr_cmd(`brief $files`), "r") do f
                v = readdlm(f, ' ')
                totlen = sum(map(x -> Float64(x), v[:, 1]))
            end
            open(pipeline(`gzip -cd $files`, `grep '^[0-9]\+\s[0-9]\+\s'`, `wc -l`), "r") do f
                totcnt = parse(Int64, readline(f))
            end
        end
        return totlen, totcnt
    end

    snapshots = deepcopy(snapshots)
    botid = 1
    function get_swc(ds, stg)
        dataset = dataset_from_id(ds)
        cnt = dataset.stgs[stg][1]
        fn = cache_file_swcdir(ds, stg)
        if !isdir(fn)
            rm("$fn.temp", force=true, recursive=true)
            mkdir("$fn.temp")
            host = @something(dataset.host, "localhost:30009")
            cmd = gapr_cmd(`export -z -c $(gapr_build_dir())/../confdir/client.conf bot$botid@$host/fmost$ds -s $fn.temp`)
            if cnt != 0
                cmd = gapr_cmd(`export -z -n $cnt -c $(gapr_build_dir())/../confdir/client.conf bot$botid@$host/fmost$ds -s $fn.temp`)
            end
            async_run(1, cmd, function ()
                mv("$fn.temp", fn)
            end)
            #botid=botid%64+1
        end
    end
    for i in findall((snapshots[!, :stg] .% 10) .>= 0)
        ds = snapshots[i, :ds]
        stg = snapshots[i, :stg]
        get_swc(ds, stg)
    end
    async_run_finish()

    insertcols!(snapshots, :n_node => 0, :n_node_sel => 0, :n_nrn => 0, :n_nrn2 => 0, :len => 0.0, :len_sel => 0.0, :err_tot => 0, :err_fix => 0, :loops => 0, :terms => 0, :bad_terms => 0)
    tmp = map(eachrow(snapshots)) do row
        fn = cache_file_swcdir(row.ds, row.stg)
        swcs = filter(x -> match(r".*\.swc(\.gz)?$", x) !== nothing, readdir(fn))
        swcs_unsel = filter(x -> match(r"^999.*\.swc(\.gz)?$", x) !== nothing, swcs)
        swcs_sel = filter(x -> !(x in swcs_unsel), swcs)
        neurons = filter(x -> check_neuron_qc(row.ds, row.stg, x) != 0, swcs_sel)
        row.n_nrn = length(filter(x -> check_neuron_qc(row.ds, row.stg, x) > 1, neurons))
        row.n_nrn2 = length(neurons)
        (i=rownumber(row),
            fn_sel=[joinpath(fn, x) for x in swcs_sel],
            fn_unsel=[joinpath(fn, x) for x in swcs_unsel],)
    end
    tmp2 = pmap(tmp) do x
        sel_len, sel_cnt = get_swc_tot_sizes(x.fn_sel)
        unsel_len, unsel_cnt = get_swc_tot_sizes(x.fn_unsel)
        (x.i, sel_len + unsel_len, sel_cnt + unsel_cnt, sel_len, sel_cnt)
    end
    for (i, a, b, c, d) in tmp2
        snapshots[i, :len] = a
        snapshots[i, :n_node] = b
        snapshots[i, :len_sel] = c
        snapshots[i, :n_node_sel] = d
    end
    snapshots
end

Make.@target df_errs (df_stats) -> begin
    df_errs = unique(df_stats[!, [:ds]])
    sel = df_stats[df_stats[:, :stg].==9, :ds]
    filter!(r -> in(r.ds, sel), df_errs)
    insertcols!(df_errs, :loops => 0, :terms => 0, :bad_terms => 0)
    for i in 1:size(df_errs)[1]
        ds = df_errs[i, :ds]
        tmp_pth = cache_file_errors(ds)
        cmd = gapr_cmd(`show --playback "$(mirror_files(ds, :repo))" --script misc/gen-err-stats.lua --image "file://$(mirror_files(ds, :cat))"`)
        dat = dataset_from_id(ds)
        if dat.mode == "selected"
            continue
        end
        after_9 = dat.stgs[9][1]
        pre_9 = maximum(map(v -> v[1], skipmissing(map(v -> get(dat.stgs, v, missing), 0:8))))
        println("$tmp_pth $ds $pre_9 $after_9")
        if !isfile(tmp_pth)
            cmd = addenv(cmd, ["AUTOCLOSE=1", "GAPR_DEBUG=", "DISPLAY=:0",
                "LD_PRELOAD=/usr/lib64/libjemalloc.so.2",
                "GEN_ERROR_STATS=$tmp_pth",
                "PRE_9=$pre_9",
                "AFTER_9=$after_9",
            ])
            run(cmd)
        end
        open(pipeline(`cat $tmp_pth`, `perl ./misc/summ-errs.pl`), "r") do f
            after_0 = dat.stgs[0][1]
            if after_9 == after_0
                return
            end
            v = readdlm(f, ' ')
            for j in 1:size(v)[1]
                t = v[j, 2]
                if in(t, ["?fixed", "?deleted", "?invalid"])
                    continue
                end
                if in(t, ["deleted"])
                    t = "fixed"
                end
                if in(t, ["unresolved_end"])
                    t = "unresolved"
                end
                if in(t, ["deferred"])
                    t = "unresolved"
                end
                s = Symbol(t)
                if !hasproperty(df_errs, s)
                    insertcols!(df_errs, s => 0)
                end
                df_errs[i, s] += 1
            end
        end
        open(pipeline(`cat $tmp_pth`, `grep '^s @'`), "r") do f
            v = readdlm(f, ' ')
            if size(v)[1] != 2
                error("need two lines")
            end
            dl = v[2, 3] - v[1, 3]
            df_errs[i, :loops] = -dl
            df_errs[i, :terms] = v[2, 6]
            df_errs[i, :bad_terms] = v[2, 6] - v[2, 8] - v[2, 7]
        end
    end
    df_errs
end

Make.@target snapshots3 (snapshots2, df_errs) -> begin
    snapshots2 = deepcopy(snapshots2)
    err_tot = df_errs[:, :fixed] .+ df_errs[:, :invalid] .+ df_errs[:, :redundant] .+ df_errs[:, :wontfix] .+ df_errs[:, :unresolved]
    err_fix = df_errs[:, :fixed] .+ df_errs[:, :invalid] .+ df_errs[:, :redundant]
    for i = 1:length(err_tot)
        ds = df_errs[i, :ds]
        j = findfirst(snapshots2[:, :ds] .== ds .&& snapshots2[:, :stg] .== 9)
        if j === nothing
            continue
        end
        snapshots2[j, :err_tot] = err_tot[i]
        snapshots2[j, :err_fix] = err_fix[i]
        snapshots2[j, :loops] = df_errs[i, :loops]
        snapshots2[j, :terms] = df_errs[i, :terms]
        snapshots2[j, :bad_terms] = df_errs[i, :bad_terms]
    end
    snapshots2
end
function linesegment_at_node(nodes, n)
    a = n.pos
    b = if n.paridx == 0
        a
    else
        nodes[n.paridx].pos
    end
    a, b
end
function calc_length(nodes::AbstractVector{<:NamedTuple}, sel=:)
    len = 0.0
    for n in nodes[sel]
        a, b = linesegment_at_node(nodes, n)
        len += norm(a .- b)
    end
    len
end
function match_swcs(nodes0, nodes9; thr=1.0)
    function gen_knn(nodes)
        pts = SVector[]
        for n in nodes
            a, b = linesegment_at_node(nodes, n)
            for t in 0:0.1:1
                pt = t .* a .+ (1 - t) .* b
                push!(pts, SVector((pt...,)))
            end
        end
        pts = @time "reduce knn" reduce(hcat, pts)
        @time "gen tree" NearestNeighbors.KDTree(pts)
    end
    function mdistto(tree, pts)
        _, dist = @time "run knn" NearestNeighbors.knn(tree, pts, 1)
        map(first, dist)
    end
    function sample_at(nodes, t)
        pts = map(nodes) do n
            a, b = linesegment_at_node(nodes, n)
            SVector(a .* t .+ b .* (1 - t))
        end
        reduce(hcat, pts)
    end
    tree0 = gen_knn(nodes0)
    tree9 = gen_knn(nodes9)
    sel0 = findall(reduce(max, [mdistto(tree9, sample_at(nodes0, t)) for t in 0:0.1:1]) .< thr)
    sel9 = findall(reduce(max, [mdistto(tree0, sample_at(nodes9, t)) for t in 0:0.1:1]) .< thr)
    sel0, sel9
end
function write_swc(func, io::IO, nodes; idadd=0)
    for (idx, n) in enumerate(nodes)
        par = if n.par != -1
            n.par + idadd
        else
            -1
        end
        print(io, format_swc_line(n.id + idadd, func(idx, n), n.pos, n.r, par))
    end
end
Make.@target autorec_cor_datx (snapshots3) -> begin
    dat = deepcopy(snapshots3)[:, [:ds, :stg, :time, :n_node, :len]]
    filter!(dat) do r
        r.stg == 0 && !in(dataset_from_id(r.ds).mode, ["manual", "selected"]) && !endswith(r.ds, ".resnet")
    end
    insertcols!(dat, :fnr => NaN, :fdr => NaN, :fp => 0.0, :tp0 => 0.0, :tp9 => 0.0, :fn => 0.0, :pp => 0.0, :p => 0.0)
    for row in eachrow(dat)
        refds = row.ds
        if endswith(row.ds, ".unet")
            refds = split(row.ds, ".unet")[1]
        end
        refstg = maximum(filter(r -> r.ds == refds, snapshots3).stg)
        PP = row.len
        P = first(filter(r -> (r.ds == refds && r.stg == refstg), snapshots3)).len
        swcs0, swcs9 = map([(row.ds, row.stg), (refds, refstg)]) do (ds, stg)
            dir = cache_file_swcdir(ds, stg)
            files = filter(x -> match(r".*\.swc(\.gz)?$", x) !== nothing, readdir(dir, join=true))
            read_nodes_from_swcs(files)
        end
        sel0, sel9 = match_swcs(swcs0, swcs9, thr=1.0)
        row.p = calc_length(swcs9)
        row.pp = calc_length(swcs0)
        row.tp9 = calc_length(swcs9, sel9)
        row.fn = row.p - row.tp9
        row.tp0 = calc_length(swcs0, sel0)
        row.fp = row.pp - row.tp0
        row.fdr = row.fp / row.pp
        row.fnr = row.fn / row.p
        @info "$(row.pp-PP)"
        @info "$(row.p-P)"

        if in(refds, ("18925", "18926", "192096"))
            open("/tmp/gapr-compx-$(row.ds)-$(row.stg)-$refds-$refstg.swc", "w") do io
                sel0, sel9 = map(Set, (sel0, sel9))
                write_swc(io, swcs0, idadd=0) do idx, n
                    in(idx, sel0) ? 0 : 2
                end
                write_swc(io, swcs9, idadd=1000_000_000) do idx, n
                    in(idx, sel9) ? 0 : 3
                end
            end
        end
    end
    dat
end

Make.@target fig_auto_rec (autorec_cor_datx) -> begin
    ds_sel = [
        "192096", "192095", "18925", "17391", "18133",
        "202813", "202814", "202815", "202816",
    ]
    dat = deepcopy(autorec_cor_datx)
    dat = filter!(r -> in(r.ds, ds_sel), dat)
    sort!(dat, [:pp])

    usedprober(i) = dataset_from_id(dat[i, :ds]).prober
    idx0 = filter(usedprober, 1:size(dat, 1))
    idx1 = filter(!usedprober, 1:size(dat, 1))

    for (plt, lim) in [(:fdr, 6), (:fnr, 40)]
        p = bar(dat[idx0, :ds], dat[idx0, plt] * 100,
            ; plotstyle()...,
            xticks=:none,
            yguide=uppercase(string(plt)) * " (%)",
            ymirror=true,
            legend=false,
            size=(140, 160),
            permute=(:x, :y),
            xflip=true,
            ylim=(0, lim),
            xlim=(-0.1, size(dat, 1) + 0.5),
        )
        bar!(p, dat[idx1, :ds], dat[idx1, plt] * 100,
            ; plotstyle()...,
            permute=(:x, :y),
        )
        savefig(p, "extra/autorec-$plt.tex")
    end

    function ann(i)
        v = dat[i, :len]
        res = "\\SI{$(round(Int,v/1000))}{mm}"
        tex(res)
    end
    dsname(x) = tex("\\#$x")

    q = bar(dsname.(dat[idx0, :ds]), (dat[idx0, :len] ./ dat[idx0, :time]) / 1000,
        text=([ann(v) for v in idx0], font(5, :left)),
        ; plotstyle()...,
        xguide="Datasets",
        yguide=tex("Automatic reconstruction speed~(\\si{mm/h})"),
        ymirror=true,
        legend=:outerbottom,
        permute=(:x, :y),
        size=(180, 160),
        xflip=true,
        ylim=(0, 200),
        xlim=(-0.1, size(dat, 1) + 0.5),
        label="with U-Net",
    )
    bar!(q, dsname.(dat[idx1, :ds]), (dat[idx1, :len] ./ dat[idx1, :time]) / 1000,
        text=([ann(v) for v in idx1], font(5, :left)),
        ; plotstyle()...,
        permute=(:x, :y),
        label="without U-Net",
    )
    push!(q.o.the_plot.preamble, "\\usepackage{siunitx}")
    savefig(q, "extra/autorec-speed.tex")
    dat
end

Make.@target tab_datasets (df_stats, snapshots3) -> begin
    ss3 = deepcopy(snapshots3)
    begin
        nodes = Dict()
        n2r = Dict()
        segs = Dict()
        seg_n = Dict()
        seg_pr = Dict()
        files = readdir(cache_file_swcdir("18926", 9), join=true)
        map(files) do fn
            match(r"\.swc(\.gz)?$", fn) !== nothing || return
            prev_n = -1
            open(`gzip -cd $fn`, "r") do io
                for l in eachline(io)
                    length(l) > 0 || continue
                    if match(r"^#Gapr!-1=0001.*", l) !== nothing
                        r = n2r[prev_n]
                        seg_pr[r] = get(seg_pr, r, 0) + 1
                        continue
                    end
                    l[1] != '#' || continue
                    (id, typ, pos, rad, par) = parse_swc_line(l)
                    nodes[id] = pos
                    prev_n = id
                    if par == -1
                        n2r[id] = id
                        seg_n[id] = get(seg_n, id, 0) + 1
                        continue
                    end
                    r = n2r[id] = n2r[par]
                    seg_n[r] = get(seg_n, r, 0) + 1
                    dd = pos .- nodes[par]
                    d = norm(dd)
                    segs[r] = get(segs, r, 0.0) + d
                end
            end
        end
        len = 0.0
        for k in keys(seg_pr)
            if seg_pr[k] > 500
                println(seg_pr[k], '/', seg_n[k], ':', segs[k])
                if seg_pr[k] == seg_n[k] && seg_pr[k] > 500
                    len += segs[k]
                end
            end
        end
        i = findlast(ss3.ds .== "18926" .&& (ss3.stg .% 10) .== 9)
        ss3[i, :len_sel] = len
    end

    XLSX.openxlsx("extra/ds-info.xlsx", mode="w") do exc
        sheet = exc[1]
        ptri, ptrj = 1, 1
        to_time = function (v)
            return round(v, digits=1)
        end
        to_mm = (v) -> round(v / 1000, digits=1)
        to_nn = (v) -> round(v / 1e6, digits=1)
        to_mm_nn = (a, b) -> "$(to_mm(a)) ($(to_nn(b)))"
        to_pars = (a, b) -> "$a/$b"
        function to_stg(s)
            Dict(
                :open => "Regular proofreading",
                :restricted => "Restricted proofreading",
                :fix => "Fixing",
            )[Symbol(s)]
        end
        function add_cols(cols...)
            for x in cols
                if x !== nothing
                    sheet[ptri, ptrj] = x
                end
                ptrj += 1
            end
        end
        yes_no(v) = ("No", "Yes", "Failed")[v+1]
        function add_nl()
            ptri += 1
            ptrj = 1
        end
        function get_size(ds)
            lines = read(mirror_files(ds, :cat), String)
            chunks = split(lines, r"^\[\w+\]$"m)
            chunks = filter(v -> match(r"^pattern="m, v) !== nothing, chunks)
            sizes = map(chunks) do chk
                m = match(r"^size=(.*)$"m, chk)
                m !== nothing || error("missing size $chk")
                m.captures[1]
            end
            sizes = unique(sizes)
            length(sizes) == 1 || error("multi sizes $ds $sizes $lines")
            tb = round(Int, prod(map(x -> parse(Int64, x), split(sizes[1], r" +"))) * 2 / 1000 / 1000 / 1000 / 1000)
            replace(sizes[1], r" " => "x") * " ($(tb) TB)"
        end
        add_cols(["Dataset ID"
            "Brain area"
            "Mouse line"
            "Labeling"
            "Image size"
            "High quality neurons"
            "Total proofread neurons"
            "Proofread length (mm)"
            "Automatic reconstruction (mm/h)"
            "U-Net"
            "Automatic proofreading"
            "Stage"
            "Stage time usage (h)"])
        add_nl()
        sums = (0, 0, 0.0, 0.0)
        for ds in datasets()
            ss = filter(r -> r.ds == ds.id, ss3)
            s0 = findfirst(ss[!, :stg] .== 0)
            s9 = findfirst(ss.stg .== 9)
            s99 = findlast((ss[!, :stg] .% 10) .== 9)
            if s9 === nothing || s0 === nothing
                continue
            end
            s0val = (0.0, 0.0)
            for row in eachrow(ss)
                if row.stg == 0
                    s0val = s0val .+ (row.len, row.time)
                elseif row.stg % 10 == 0
                    prevrow = argmax(r -> r.stg, filter(r -> r.stg < row.stg, eachrow(ss)))
                    s0val = s0val .+ (row.len - prevrow.len, row.time)
                end
            end
            stgs = map(1:size(ss)[1]) do i
                ss[i, :ds] == ds.id || return missing
                stg = ss[i, :stg]
                (stg % 10) != 0 || return missing
                if stg % 10 == 9
                    return (ss[i, :time], "fix")
                end
                (ss[i, :time], ds.stgs[stg][2])
            end
            stgs = collect(skipmissing(stgs))
            length(stgs) != 0 || "not proofread"
            add_cols(ds.id,
                ds.area,
                ds.strain,
                ds.label,
                get_size(ds.id),
                ss[s99, :n_nrn],
                ss[s99, :n_nrn2],
                to_mm(ss[s99, :len_sel]),
                "$(to_mm(s0val[1]))/$(to_time(s0val[2]))",
                yes_no(ds.prober),
                yes_no(ds.autopr),
                to_stg(stgs[1][2]),
                to_time(stgs[1][1]),
            )
            sums = sums .+ (ss[s99, :n_nrn], ss[s99, :n_nrn2], ss[s99, :len_sel], stgs[1][1])
            add_nl()
            for i in 2:length(stgs)
                add_cols(
                    repeat([nothing], 8 + 2 + 2 - 1)...,
                    to_stg(stgs[i][2]),
                    to_time(stgs[i][1]),
                )
                sums = sums .+ (0, 0, 0.0, stgs[i][1])
                add_nl()
            end
        end
        add_cols(repeat([nothing], 10 + 2 + 2 - 1)...,)
        add_nl()
        add_cols("Sum:", repeat([nothing], 3 + 2 - 1)..., sums[1], sums[2], to_mm(sums[3]), repeat([nothing], 2 + 2)..., to_time(sums[4]))
        add_nl()
        XLSX.rename!(sheet, "Dataset Info")
    end
    0
end

function find_branches_and_alike!(nodes::AbstractVector{<:NamedTuple}; thr=2.5)
    degrees = map(n -> 0, nodes)
    link2edg = map(n -> 0, nodes)
    edgalloc = 10
    for (idx, n) in enumerate(nodes)
        if n.par != -1
            degrees[idx] += 1
            degrees[n.paridx] += 1
        else
            #degrees[idx]+=10
        end
    end
    points = []
    for (idx, n) in enumerate(nodes)
        if n.par != -1
            pidx = n.paridx
            edg = if degrees[pidx] == 2
                if nodes[pidx].par == -1
                    if link2edg[pidx] == 0
                        link2edg[pidx] = edgalloc += 1
                    else
                        link2edg[pidx]
                    end
                else
                    link2edg[pidx]
                end
            elseif degrees[pidx] == 1
                link2edg[pidx] = edgalloc += 1
            else
                edgalloc += 1
            end
            link2edg[idx] = edg
            ppos = nodes[pidx].pos
            for t in 0:0.1:1
                push!(points, (pos=n.pos .* t .+ (1 - t) .* ppos, edg=edg))
            end
        end
    end
    posvec = [SVector(p.pos) for p in points]
    @info "type $(typeof(posvec))"
    posvec = @time "reduce" reduce(hcat, posvec)
    tree = @time "build tree" NearestNeighbors.KDTree(posvec)
    nodepos = reduce(hcat, [SVector(n.pos) for n in nodes])
    indsvec = @time "inrange" NearestNeighbors.inrange(tree, nodepos, thr)
    for (idx, n) in enumerate(nodes)
        deg = degrees[idx]
        br = if deg == 0
            false
        elseif deg == 1 || deg == 2
            inds = indsvec[idx]
            ##@assert n.par!=-1
            edg = link2edg[idx]
            @assert edg != 0
            alien = count(i -> edg != points[i].edg, inds)
            alien > 0
        else
            true
        end
        nodes[idx] = (n..., br=br)
    end
    dist2att = map(n -> (n.pr ? Inf : 0.0), nodes)
    for _ in 1:6
        for (idx, n) in enumerate(nodes)
            if n.par != -1
                pidx = n.paridx
                dist = norm(nodes[pidx].pos .- n.pos)
                a = dist2att[idx]
                b = dist2att[pidx]
                dist2att[idx] = min(a, b + dist)
                dist2att[pidx] = min(b, a + dist)
            end
        end
    end
    for (i, n) in enumerate(nodes)
        if n.br && degrees[i] == 1
            @info "hit: $(n.id)"
        end
    end
    dist2att
end
Make.@target autopr_stats (df_stats, snapshots2) -> begin
    df = deepcopy(df_stats)
    stats = deepcopy(snapshots2)
    filter!(stats) do r
        special = endswith(r.ds, ".resnet")
        stg = special ? maximum(filter(rr -> rr.ds == r.ds, snapshots2).stg) : 0
        r.stg == stg && dataset_from_id(r.ds).mode === nothing
    end
    stats = stats[:, [:ds, :stg, :time, :n_node, :len]]
    insertcols!(stats, :np => 0, :nn => 0, :nbr => 0, :nmiss => 0, :natt => 0)
    insertcols!(stats, :pr_cov => 0.0, :time_1st_pr => 0.0, :n_1st_pr => 0, :time_1st_other => 0.0, :time_2nd_pr => 0.0, :n_2nd_pr => 0)

    for (iii, row) in enumerate(eachrow(stats))
        ds = stats[iii, :ds]
        l = df.ds .== ds .&& df.stg .== 1 .&& df.type .== "proofread_"
        tv = df[l, :time_dd]
        tt = sum(tv) / 3600
        stats[iii, :n_1st_pr] = length(findall(l))
        stats[iii, :time_1st_pr] = tt
        l = df.ds .== ds .&& df.stg .== 1 .&& df.type .!= "proofread_"
        tv = df[l, :time_dd]
        tt = sum(tv) / 3600
        stats[iii, :time_1st_other] = tt
        l = df.ds .== ds .&& df.stg .== 2 .&& df.type .== "proofread_"
        tv = df[l, :time_dd]
        tt = sum(tv) / 3600
        stats[iii, :n_2nd_pr] = length(findall(l))
        stats[iii, :time_2nd_pr] = tt
        swcs = readdir(cache_file_swcdir(row.ds, row.stg), join=true)
        nodes = read_nodes_from_swcs(swcs)

        if row.stg != 0
            dist2att = find_branches_and_alike!(nodes)
            row.natt = count(dist2att .< 6.0)
            row.nbr = count(n -> n.br, nodes)
            row.nmiss = count(enumerate(nodes)) do (i, n)
                res = n.br && dist2att[i] > 6.0
                if res
                    @info "$ds : $(n.id)"
                end
                res
            end
            if true
                open("/tmp/gapr-resnet-$ds-pr.swc", "w") do io
                    write_swc(io, nodes) do idx, n
                        n.pr ? 0 : 3
                    end
                end
                open("/tmp/gapr-resnet-$ds-br.swc", "w") do io
                    write_swc(io, nodes) do idx, n
                        n.br ? 3 : 0
                    end
                end
            end
        end
        row.nn = length(nodes)
        row.np = count(n -> n.pr, nodes)
        row.pr_cov = round(row.np / row.nn * 100) / 100
    end
    stats
end
Make.@target fig_autopr_resnet (autopr_stats) -> begin
    x = deepcopy(autopr_stats)
    filter!(r -> endswith(r.ds, ".resnet"), x)
    dslbl(x) = begin
        d = split(x, ".resnet")[1]
        tex("{\\#$d}")
    end
    p1 = bar(dslbl.(x[:, :ds]), x[:, :pr_cov] .* 100,
        ; plotstyle(scaleonlyaxis=true)...,
        yguide=tex("Coverage~(%)"),
        legend=false,
        xticks=:none,
        permute=(:x, :y),
        xflip=true,
        ylim=(0, 100),
        xlim=(-0.1, size(x, 1) + 0.1),
    )
    p2 = bar(dslbl.(x[:, :ds]), x.nmiss ./ x.nbr .* 100,
        ; plotstyle(scaleonlyaxis=true)...,
        yguide=tex("Error rate~(%)"),
        legend=false,
        permute=(:x, :y),
        xguide="Datasets",
        xflip=true,
        ylim=(0, 5),
        xlim=(-0.1, size(x, 1) + 0.1),
    )
    p = plot(p2, p1, layout=grid(1, 2, widths=[0.4, 0.6]),
        size=(120, 50),
        link=:y,
    )
    savefig(p, "extra/autopr-resnet.tex")
    x
end

Make.@target fig_collab_real (df_stats) -> begin
    function draw_progress_fig(df, ds::String, from::Int, to::Int, of::String)
        tmp_pth = cache_file_progress(ds, from, to)
        id2dfi = Dict()
        for i = 1:size(df)[1]
            id = df[i, :id]
            if df[i, :ds] == ds && id >= from - 1 && id < to
                id2dfi[id] = i
            end
        end
        cmd = gapr_cmd(`show --playback $(mirror_files(ds, :repo)) --script misc/gen-stats.lua --image file://$(mirror_files(ds, :cat))`)
        vals = nothing
        if !isfile(tmp_pth)
            cmd = addenv(cmd, ["AUTOCLOSE=1", "GAPR_DEBUG=", "DISPLAY=:0",
                "LD_PRELOAD=/usr/lib64/libjemalloc.so.2",
                "GEN_PROGRESS=$tmp_pth",
                "GEN_PROGRESS_FROM=$from",
                "GEN_PROGRESS_TO=$to"
            ])
            run(cmd)
        end
        vals = readdlm(tmp_pth, ' ')
        idxes = [id2dfi[i] for i in from:(to-1)]
        #xyz
        #xyz
        users = unique(df[collect(idxes), :who])
        begin
            df = deepcopy(df)
            night = minimum(df[collect(idxes), :time]) + Second(16 * 3600)
            prevs = df[:, :time] .< night
            nexts = df[:, :time] .> night
            gap = minimum(df[nexts, :time]) - maximum(df[prevs, :time]) - Second(10 * 60)
            df[nexts, :time] .-= gap
        end
        ts0, ts1 = extrema(df[collect(idxes), :time])
        println(ts0)
        per = 10 * 60 * 1000.0
        nbins = round(Int64, (ts1 - ts0).value / per) + 1
        freq = Matrix{Float64}(undef, nbins, length(users))
        freq .= 0.0
        for i = 1:length(users)
            for j in collect(values(id2dfi))
                id = df[j, :id]
                if df[j, :who] == users[i] && id >= from && id < to
                    bin = round(Int64, (df[j, :time] - ts0).value / per) + 1
                    freq[bin, i] += 1
                end
            end
        end
        ncmts = sum(freq, dims=1)[1, :]
        sel = ncmts .> 100
        sel = shuffle(findall(sel))
        freq = freq ./ per .* (60 * 1000)
        x2 = [max((df[id2dfi[x], :time] - ts0).value / 60.0 / 60000.0, 0) for x in Array{Int64}(vals[:, 1])]
        y2 = [(vals[x, 3] - vals[x, 4]) / vals[x, 3] for x in 1:size(vals)[1]]
        y3 = [(vals[x, 5] - vals[x, 6]) / vals[x, 5] for x in 1:size(vals)[1]]
        ylims = maximum(x2)
        ylims = max(ylims, maximum([x * (per / 60.0 / 60000.0) for x in 1:nbins]))
        xticks = 0:(4):floor(Int64, ylims / 8)*8
        p1 = heatmap([x * (per / 60.0 / 60000.0) for x in 1:nbins], users[sel], freq[:, sel]',
            ; plotstyle(scaleonlyaxis=true,
                "every colorbar/.append style" => "height=2mm,width=20mm,at={(0.5,1.1)},anchor=south,xlabel style={font={{\\fontsize{7 pt}{0pt}\\selectfont\\bfseries}},anchor=south,at={(0.5,1.03)}},xticklabel pos={lower}",
                "xlabel style" => "anchor=west,at={{(ticklabel cs:0.0)}}",
            )...,
            fillcolor=cgrad(:OrRd_9),
            colorbar=:top,
            colorbar_title=tex("Proofreading activity (\\si{Operations/min})"),
            xshowaxis=:false,
            xticks=xticks,
            grid=:x,
            framestyle=:grid,
            yticks=((1:length(sel)) .- 0.5, map(v -> tex("\\#$v"), 1:length(sel))),
            yguide="Annotators",
            xlims=(0, ylims),
            legend=:right,
        )
        display(x2)
        p2 = plot(x2, y2 * 100,
            ; plotstyle("clip" => false, scaleonlyaxis=true)...,
            leg=:none,
            ylims=(0, 100),
            xlims=(0, ylims),
            #labels=["" "proofread" "terminal"],
            showaxis=:xy,
            xticks=xticks,
            yticks=([0, 50, 100]),
            grid=:x,
            yguide="Progress (%)",
            xguide="Time (h)",
        )
        p = plot(p1, p2, layout=grid(2, 1, heights=[0.6, 0.4]),
            link=:x,
            size=(220, 180),
        )
        push!(p.o.the_plot.preamble, "\\usepackage{siunitx}")
        savefig(p, of)
        return (df[sort(collect(values(id2dfi))), [:id, :who, :time, :type]], vals)
    end
    id = "202814"
    ds = dataset_from_id(id)
    draw_progress_fig(df_stats, id, ds.stgs[1][1] + 1, ds.stgs[2][1], "./extra/collab-real.tex")
end

function to_kb(v::AbstractString)
    i = findfirst(c -> isletter(c), v)
    u = v[i:end]
    r = parse(Float64, v[1:i-1])
    if u == "KB"
        r = r * 1024
    elseif u == "MB"
        r = r * 1024 * 1024
    else
        error("unit $u")
    end
    return r / 1024.0
end

Make.@target fig_collab_simu () -> begin
    data_coll = (;)
    let
        x = readdlm("precious/server-bw.csv", ' ')
        x = DataFrame([:npar => map(v -> v, x[:, 1]),
            :bw => map(v -> to_kb(v), x[:, 2])])
        data_coll = (; data_coll..., server_bw=x)
        xticks = (3:7, map(v -> tex("\$$(2^v)\$"), 3:7))
        p = violin(log2.(x[:, :npar]), x[:, :bw] ./ x[:, :npar],
            ; plotstyle()...,
            xticks=xticks,
            yguide=tex("BW per client~(\$\\si{KiB/s}\$)"),
            legend=:none,
            size=(200, 160),
            xlim=(2.5, 7.5),
            ylims=(0, 16),
            color=palette(:auto)[1],
            ytickfontcolor=palette(:auto)[1],
        )
        medx = sort(unique(x[:, :npar]))
        medy = map(medx) do a
            median(x[x[:, :npar].==a, :bw]) / (a * a)
        end
        plot!(twinx(p), log2.(medx), medy,
            ; plotstyle()...,
            label=tex("\${\\operatorname{median} \\left(\\text{total BW}\\right)\\over\\left[\\text{# of clients}\\right]^2}\$"),
            legend=(0.05, 1.00),
            legend_foreground_color=:transparent,
            legend_background_color=:transparent,
            xlim=(2.5, 7.5),
            ylims=(0, 0.05),
            yticks=0:0.02:0.04,
            color=palette(:auto)[2],
            ytickfontcolor=palette(:auto)[2],
        )
        push!(p.o.the_plot.preamble, "\\usepackage{siunitx}")
        push!(p.o.the_plot.preamble, "\\usepackage{amsmath}")
        savefig(p, "extra/server-bw.tex")
    end
    let
        x = readdlm("precious/server-cpu.csv", ' ')
        x = DataFrame([:npar => map(v -> Integer(v), x[:, 1]),
            :cpu => map(v -> v, x[:, 2])])
        data_coll = (; data_coll..., server_cpu=x)
        xticks = (3:7, map(v -> tex("\$$(2^v)\$"), 3:7))
        p = violin(log2.(x[:, :npar]), x[:, :cpu],
            ; plotstyle()...,
            xticks=xticks,
            #xguide="Number of clients",
            yguide=tex("CPU usage~(\$%\$)"),
            legend=:none,
            size=(200, 160),
            xlim=(2.5, 7.5),
            ylims=(0, 77),
        )
        savefig(p, "extra/server-cpu.tex")
    end
    let
        x = readdlm("precious/server-freqs.csv", ' ')
        x = DataFrame([:npar => map(v -> Integer(v), x[:, 1]),
            :tp => map(v -> parse(DateTime, v), x[:, 2])])
        data_coll = (; data_coll..., server_freqs=x)
        medx = sort(unique(x[:, :npar]))
        medy = map(medx) do a
            tps = x[x[:, :npar].==a, :tp]
            dt = Millisecond(maximum(tps) - minimum(tps)).value / 1000.0
            n = length(tps)
            n / dt
        end
        xticks = (3:7, map(v -> tex("\$$(2^v)\$"), 3:7))
        p = plot(log2.(medx), [map(v -> (v * 0.5), medx) medy],
            ; plotstyle()...,
            xticks=xticks,
            label=["Theoretical upper limit" "Measured"],
            #xguide="Number of clients",
            yguide="Operations per second",
            size=(200, 160),
            xlim=(2.5, 7.5),
            legend=(0.05, 1.00),
            legend_foreground_color=:transparent,
            legend_background_color=:transparent,
            legend_font_halign=:left,
            ylim=(0, 65),
        )
        savefig(p, "extra/server-freqs.tex")
    end
    let
        x = readdlm("precious/client-lag.csv", ' ')
        x = DataFrame([:npar => map(v -> Integer(v), x[:, 1]),
            :lag => map(v -> v / 1000, x[:, 2])])
        data_coll = (; data_coll..., client_lag=x)
        xticks = (3:7, map(v -> tex("\$$(2^v)\$"), 3:7))
        p = violin(log2.(x[:, :npar]), x[:, :lag],
            ; plotstyle()...,
            xticks=xticks,
            xguide="Number of clients",
            yguide=tex("Round-trip time~(\$\\si{ms}\$)"),
            legend=:none,
            size=(200, 160),
            xlim=(2.5, 7.5),
        )
        push!(p.o.the_plot.preamble, "\\usepackage{siunitx}")
        savefig(p, "extra/client-lag.tex")
    end
    data_coll
end

function force_dir(func, outdir)
    outdir = deepcopy(outdir)
    rm("$outdir", force=true, recursive=true)
    rm("$outdir.temp", force=true, recursive=true)
    mkdir("$outdir.temp")
    func("$outdir.temp")
    mv("$outdir.temp", outdir)
    outdir
end

Make.@target xform_swcs (snapshots2) -> begin
    dss = deepcopy(snapshots2)[:, [:ds, :stg]]
    filter!(r -> begin
            v = dataset_from_id(r.ds)
            v.mode != "manual"
        end, dss)
    filter!(r -> !endswith(r.ds, r".resnet\d*"), dss)
    filter!(r -> !endswith(r.ds, r".unet\d*"), dss)
    filter!(r -> r.stg % 10 == 9, dss)
    filter!(r -> r.stg == maximum(filter(rr -> rr.ds == r.ds, dss).stg), dss)
    insertcols!(dss, :fn => map(r -> cache_file_swcdir(r.ds, r.stg), eachrow(dss)))

    indir = force_dir(cache_dir_swcs_xform()) do tmpdir
        function xform(ds, stg, fn)
            swcs = filter(x -> match(r".*@.*\.swc(\.gz)?$", x) !== nothing, readdir(fn))
            swcs = map(x -> joinpath(fn, x), swcs)
            !isempty(swcs) || return
            stg += 1000
            cmd = gapr_cmd(`map $swcs -o "$tmpdir/$ds-$stg-%s" -x "bash misc/antsxform.sh $ds"`)
            cc = run(cmd)
            check_exit(cc)
        end
        cond = startswith("20281")
        futs = []
        fut1 = remotecall(WorkerPool(workers())) do
            for r in filter(r -> cond(r.ds), eachrow(dss))
                xform(r.ds, r.stg, r.fn)
            end
        end
        push!(futs, fut1)
        for r in filter(r -> !cond(r.ds), eachrow(dss))
            fut2 = remotecall(xform, WorkerPool(workers()), r.ds, r.stg, r.fn)
            push!(futs, fut2)
        end
        foreach(wait, futs)
    end
    df = map(readdir(indir)) do fn
        @assert match(r"^\d+.*\.swc$", fn) !== nothing
        (fn=fn, dir=indir, path=joinpath(indir, fn))
    end
    df = sort(DataFrame(df), :fn)
    filter!(r -> check_neuron_qc(r.fn) != 0, df)
end

@everywhere function get_soma_pos(fn; id=false)
    open(fn, "r") do io
        while !eof(io)
            l = readline(io)
            (!isempty(l) && l[1] != '#') || continue
            f = split(l)
            id && return parse(Int64, f[1])
            return (map(v -> parse(Float64, v), f[3:5])...,)
        end
    end
end
function _get_allen_data(pos)
    global __allen_img_data__
    if !@isdefined(__allen_img_data__) || __allen_img_data__ === nothing
        @info "loading allen img"
        __allen_img_data__ = load("/data/dat/annotation_10.nrrd")
    end
    pos = (pos[3], pos[2], pos[1])
    ii = map(v -> round(Int64, v / 10.0 + 0.5), pos)
    ss = size(__allen_img_data__)
    for d in 1:3
        (ii[d] >= 1 && ii[d] <= ss[d]) || return 0
    end
    return Int64(__allen_img_data__[ii...])
end
function _get_allen_annot()
    global __allen_ann_data__
    if !(@isdefined __allen_ann_data__) || __allen_ann_data__ === nothing
        @info "loading allen annot"
        anndata = Dict{Int64,Any}()
        fn = "extra/allen-graph.json"
        if !isfile(fn)
            cc = run(`curl -s 'http://api.brain-map.org/api/v2/structure_graph_download/1.json' -o "$fn.temp"`)
            check_exit(cc)
            mv("$fn.temp", fn)
        end
        res = join(readlines(fn), "\n")
        res = JSON.parse(res)
        function handle(arr, par)
            for obj in arr
                id = obj["id"]
                col = obj["color_hex_triplet"]
                obj["parent_structure_id"] == par || error("bad par")
                abbr = replace(obj["acronym"], "/" => "|")
                nm = obj["name"]
                ord = obj["graph_order"]
                anndata[id] = (par=(par === nothing ? 0 : par), col=parse(Color, "#" * col), abbr=abbr, name=nm, order=ord)
                handle(obj["children"], id)
            end
        end
        handle(res["msg"], nothing)
        __allen_ann_data__ = anndata
    end
    __allen_ann_data__
end
function get_allen_hier_int(id::Int64)
    id != 0 || return "/"
    ann = _get_allen_annot()
    stk = []
    while id != 0
        a = ann[id]
        push!(stk, a.abbr)
        id = a.par
    end
    push!(stk, "")
    join(Iterators.reverse(stk), "/")
end
function get_allen_hier(sym::String)
    global __allen_sym2id
    if !(@isdefined __allen_sym2id) || __allen_sym2id === nothing
        vals = map(p -> (p.second.abbr, p.first), (_get_allen_annot() |> pairs |> collect))
        __allen_sym2id = Dict(vals)
    end
    get_allen_hier_int(get(__allen_sym2id, sym, 0))
end
function get_allen_hier(pos)
    id = _get_allen_data(pos)
    get_allen_hier_int(id)
end

Make.@target neurons_summary (xform_swcs) -> begin
    swcs = deepcopy(xform_swcs)
    roi = Set(["VMH", "MPO", "MPN", "ARH"])
    dstype = Dict(
        "17391" => "VMH", "18133" => "VMH", "192095" => "VMH", "192096" => "VMH",
        "18708" => "MPO", "18709" => "MPO",
        "202812" => "ARH", "202813" => "ARH", "202814" => "ARH", "202815" => "ARH", "202816" => "ARH",
    )
    df1 = map(swcs.fn, swcs.path) do fn, fnf
        m = match(r"^(\d+)-(\d+)-(\d+)(@.*\.swc)$", fn)
        m !== nothing || error("should match")
        ds = m.captures[1]
        stg = parse(Int64, m.captures[2]) - 1000
        ord = parse(Int64, m.captures[3])

        pos = get_soma_pos(fnf)
        ba = get_allen_hier(pos)
        hier = split(ba, "/")
        for i = 1:length(hier)
            if in(hier[i], roi)
                hier = hier[1:i]
                break
            end
        end
        ba = join(hier, "/")

        qc = check_neuron_qc(fn) > 1
        dst = get(dstype, ds, "MISC")

        rawdir = cache_file_swcdir(ds, stg)
        rawfnf = joinpath(rawdir, m.captures[3] * m.captures[4] * ".gz")
        (fn=fn,
            ds=ds,
            dst=dst,
            stg=stg,
            ord=ord,
            soma_ba=ba,
            soma_pos=pos,
            qc=qc,
            fnf=fnf,
            rawfnf=rawfnf,
        )
    end
    df = sort(DataFrame(skipmissing(df1)), :fn)
    df2 = pmap(eachrow(df)) do row
        len_tot, lens_typ = get_neurite_len(row.fnf)
        len_axon = get(lens_typ, 2, 0.0)
        len_tot_raw, lens_typ_raw = get_neurite_len(row.rawfnf)
        len_axon_raw = get(lens_typ_raw, 2, 0.0)
        (fn=row.fn,
            mm_tot=len_tot / 1000,
            mm_axon=len_axon / 1000,
            mm_tot_raw=len_tot_raw / 1000,
            mm_axon_raw=len_axon_raw / 1000,
        )
    end
    leftjoin!(df, DataFrame(df2), on=:fn)
    df[:, Not([:fnf, :rawfnf])]
end

Make.@target fig_len_dist (neurons_summary) -> begin
    length_dist = deepcopy(neurons_summary)[:, [:fn, :ds, :stg, :ord, :qc, :mm_tot_raw, :mm_axon_raw]]
    rename!(length_dist, :mm_tot_raw => :len)
    filter!(length_dist) do row
        @assert row.ord < 999000
        @assert match(r"^\d+-\d+-\d+@\.swc$", row.fn) === nothing
        @assert match(r"^\d+-\d+-\d+@seg\d*", row.fn) === nothing
        return true
    end
    length_dist[:, :len] *= 1000
    x = deepcopy(length_dist)
    xticks = [2, 3, 4, 5]
    xticks = (xticks, map(v -> round((10^v) / 1000, digits=1), xticks))
    p1 = histogram(log10.(x[:, :len]),
        ; plotstyle(scaleonlyaxis=true)...,
        bins=range(1, 6, length=30),
        legend=:topright,
        ylim=(0, 650),
        xlim=(1.0, 6.7),
        xticks=xticks,
        xtick_direction=:out,
        size=(200, 120),
        yguide="Count",
        xguide=tex("Length (\\si{mm})"),
        label=tex("All proofread neurons")
    )
    x = x[x.qc, :]
    histogram!(log10.(x[:, :len]),
        ; plotstyle(scaleonlyaxis=true)...,
        bins=range(1, 6, length=30),
        #label=tex("High quality neurons (\$N=$(size(x)[1])\$)"),
        label=tex("High quality subset")
    )
    p = p1
    push!(p.o.the_plot.preamble, "\\usepackage{siunitx}")
    savefig(p, "extra/length-dist.tex")
    length_dist
end

Make.@target fig_pr_speed (snapshots3) -> begin
    x = deepcopy(snapshots3)
    insertcols!(x, :stg, :type => "")
    transl = Dict("open" => "Regular proofreading", "restricted" => "Restricted proofreading")
    for r in eachrow(x)
        s = dataset_from_id(r.ds).stgs[r.stg]
        if length(s) >= 2
            r.type = get(transl, s[2], "")
        end
    end
    filter!(r -> r.type !== "", x)
    # only count datasets with standard procedure
    ds_sel = ["17391" "18925" "192095" "192096" "202813" "202814" "202815" "202816"]

    idx = filter(i -> x[i, :stg] .% 9 != 0 && x[i, :stg] < 10 && in(x[i, :ds], ds_sel), 1:size(x, 1))
    dsname(x) = tex("\\#$x")
    p1 = groupedbar(dsname.(x[idx, :ds]), (x[idx, :len_sel] ./ x[idx, :time]) / 1000,
        ; plotstyle(scaleonlyaxis=true)...,
        group=map(i -> x[i, :type], idx),
        xguide=tex("Datasets"),
        yguide=tex("Proofreading speed per annotator~(\\si{mm/h})"),
        legend_position=:bottomright,
        permute=(:x, :y),
        grid=:y
    )
    plot!(p1, ylim=(-0.1, length(idx) / 2 + 0.1), xlim=(0, 97),)
    let shown = x[idx, :]
        a = filter(r -> startswith("Regular")(r.type), shown)
        b = filter(r -> startswith("Restricted")(r.type), shown)
        a = a.len_sel ./ a.time
        b = b.len_sel ./ b.time
        t = HypothesisTests.MannWhitneyUTest(a, b)
        p = HypothesisTests.pvalue(t, tail=:left)
        show(t)
        @info "pvalue $p"
    end
    p2 = scatter([0, 1], [25.2, 4.5], yerror=[11.9, 2.8],
        ; plotstyle(scaleonlyaxis=true)...,
        color=:black,
        permute=(:x, :y),
        yaxis=:hide,
        legend=:none,
        grid=:y,
        background_color_inside=:lightgray,
        xticks=:none
    )
    annotate!(p2,
        [
            (96, 1, (tex("(from literature)"), 6, :right)),
        ]
    )
    annotate!(p2,
        [
            (25.2 + 11.9, 0, ("Janelia Workstation proofreading", 5, :left)),
            (4.5 + 2.8, 1, ("Janelia Workstation manual reconstruction", 5, :left)),
        ],
        ylim=(-0.7, 1.7),
        xlim=(0, 97),
    )
    p3 = scatter([0, 1, 2], [27.8, 9.0, 2.2], yerror=[0, 0, 0],
        ; plotstyle(scaleonlyaxis=true)...,
        color=:black,
        permute=(:x, :y),
        yaxis=:hide,
        legend=:none,
        grid=:y,
        background_color_inside=:lightgray,
        xticks=:none
    )
    annotate!(p3,
        [
            (96, 2, (tex("(test with \\#18926)"), 6, :right)),
        ]
    )
    annotate!(p3,
        [
            (28 + 1.5, 0, ("Gapr proofreading", 5, :left)),
            (9.0 + 1.5, 1, ("Gapr manual reconstruction", 5, :left)),
            (2.2 + 1.5, 2, ("Vaa3D manual reconstruction", 5, :left)),
        ],
        ylim=(-0.7, 2.7),
        xlim=(0, 97),
    )

    p = plot(p3, p2, p1, layout=grid(3, 1, heights=[0.6 * 0.3, 0.4 * 0.3, 0.7]),
        size=(260, 230),
        link=:x,
    )
    push!(p.o.the_plot.preamble, "\\usepackage{siunitx}")
    savefig(p, "extra/pr-speed.tex")
    filter(r -> in(r.ds, ds_sel), x)
end

Make.@target fig_errors_stat (df_errs, snapshots3) -> begin
    x = deepcopy(df_errs)
    insertcols!(x, :len => 0.0)
    # only count datasets with standard procedure
    ds_sel = ["17391" "18925" "192095" "192096" "202813" "202814" "202815" "202816"]
    for r in eachrow(x)
        idx = findfirst(snapshots3[:, :ds] .== r.ds .&& snapshots3[:, :stg] .== 9)
        if idx === nothing || !in(snapshots3[idx, :ds], ds_sel)
            continue
        end
        r.len = snapshots3[idx, :len]
    end
    filter!(r -> r.len > 0, x)

    yy = ([x[:, :fixed] ./ x[:, :len] x[:, :wontfix] ./ x[:, :len] x[:, :unresolved] ./ x[:, :len]]) * 1000 * 100
    p = groupedbar(repeat(x[:, :ds], outer=3), yy,
        ; plotstyle()...,
        bar_position=:stack,
        #bar_width=0.7,
        permute=(:x, :y),
        size=(210, 140),
        yguide=tex("Errors per \$\\SI{100}{mm}\$"),
        group=repeat(["Fixed", "Unresolvable", "Unresolved"], inner=length(ds_sel)),
        xticks=:none,
        #xguide="Dataset",
        legend=:topright,
    )
    plot!(p, xlim=(0, 240), ylim=(-0.1, size(x, 1) + 0.1),)
    push!(p.o.the_plot.preamble, "\\usepackage{siunitx}")
    savefig(p, "extra/error-types.tex")

    dsname(x) = tex("\\#$x")
    p = bar(dsname.(x[:, :ds]), (x[:, :loops] ./ x[:, :len]) * 1000 * 100,
        ; plotstyle()...,
        yguide=tex("Loops per \$\\SI{100}{mm}\$"),
        xguide="Dataset",
        permute=(:x, :y),
        size=(160, 140),
        leg=false,
        ylim=(0, 40),
        xlim=(-0.1, size(x, 1) + 0.1),
    )
    push!(p.o.the_plot.preamble, "\\usepackage{siunitx}")
    savefig(p, "extra/loop-cnts.tex")
    x
end

Make.@target f1scores (snapshots2) -> begin
    x = deepcopy(snapshots2)
    sel9 = (x[:, :stg] .% 10) .== 9
    last = maximum(x[sel9, :stg])
    dss = unique(x[x[:, :stg].==last, :ds])
    stgs = sort(unique(x[map(v -> in(v, dss), x[:, :ds]).&&sel9, :stg]))
    reg = r"\d+@(c\d+).*\.swc(\.gz)?$"

    seqs = []
    rng = Random.MersenneTwister(0)
    for ds in dss
        fn = cache_file_swcdir(ds, last)
        swcs = filter(v -> match(reg, v) !== nothing, readdir(fn))
        files = map(swcs) do f
            ff = joinpath(fn, f)
            s = filesize(ff)
            check_neuron_qc(ds, last, f) > 1 || (s = 0)
            (f, s, ds)
        end
        files = sort(files, lt=(x, y) -> x[2] < y[2], rev=true)
        files = files[1:10]
        sels = map(files) do (f, s, ds)
            n = match(reg, f).captures[1]
            seq = []
            for stg in stgs
                fn = cache_file_swcdir(ds, stg)
                ff = filter(v -> begin
                        m = match(reg, v)
                        m !== nothing && m.captures[1] == n
                    end, readdir(fn))
                if length(ff) == 1
                    push!(seq, (joinpath(fn, ff[1]), stg))
                end
            end
            (seq, n, ds)
        end
        sels = filter(v -> length(v[1]) == length(stgs), sels)
        sels = shuffle(rng, sels)
        sels = sels[1:5]
        for x in sels
            push!(seqs, x)
        end
    end

    res = pmap(seqs) do (seq, name, ds)
        ref = ""
        others = []
        others_s = []
        for (f, s) in seq
            if s == last
                ref = f
            end
            push!(others, f)
            push!(others_s, s)
        end
        if ref == ""
            error("no ref")
        end
        v = nothing
        open(gapr_cmd(`brief --f1score $ref $others`), "r") do io
            v = readdlm(io, ' ')
        end
        (v, others_s)
    end

    mat = Matrix{Float64}(undef, length(seqs), length(stgs))
    for ni in 1:length(seqs)
        (v, others_s) = res[ni]
        for i = 1:size(v)[1]
            ii = findfirst(v -> v == others_s[i], stgs)
            mat[ni, ii] = v[i, 1]
        end
    end
    mat, map(v -> (v[2], v[3]), seqs)
end

Make.@target fig_more_stgs (f1scores) -> begin
    scores = f1scores[1]
    p = plot(scores',
        ; plotstyle()...,
        xticks=(1:5, ["2", "3", "4", "5", "6"]),
        xguide="Number of proofreading rounds",
        yguide=tex("\$F_1\$ score"),
        legend=:none,
        size=(240, 200),
        color=:black,
        label=hcat(tex("\$15\$ neurons"), map(v -> "", permutedims(scores[2:end, 1]))),
    )
    plot!(p, 1:size(scores)[2], median(scores, dims=1)[1, :],
        ; plotstyle()...,
        color=:red,
        legend=(0.55, 0.25),
        label="Median",
        linewidth=1,
    )
    savefig(p, "extra/pr-more-stages.tex")
    f1scores
end

function affine_mapper(origin, direction)
    fw(off) = direction * off .+ origin
    invdir = inv(direction)
    bw(pos) = invdir * (pos .- origin)
    (fw=fw, bw=bw)
end

function clamp_swc_file(nodes, offx; affine, cubesize)
    corner = affine.fw(offx)
    center = affine.fw(offx .+ cubesize)
    radius = 1.3 * norm(corner .- center)
    isinside = map(nodes) do n
        norm(n.pos .- center) > radius && return false
        res = (affine.bw(n.pos) .- offx) ./ cubesize
        all(res .> 0) && all(res .< 2)
    end
    id2idx = Dict{Int,Int}()
    clamped = NamedTuple[]
    for (idx, n) in enumerate(nodes)
        if isinside[idx]
            pid = -1
            pidx = 0
            if n.paridx != 0 && isinside[n.paridx]
                pid = n.par
                pidx = id2idx[pid]
            end
            push!(clamped, (n..., par=pid, paridx=pidx))
            id2idx[n.id] = length(clamped)
        end
    end
    clamped
end

Make.@target autorec_stats (snapshots2) -> begin
    df = deepcopy(snapshots2)
    filter!(r -> endswith(".unet")(r.ds) && r.stg == 0, df)
    df = df[:, [:ds, :stg, :time, :n_node, :len]]
    parse_numbers(Typ, s) = map(v -> parse(Typ, v), split(s, r"\s+"))
    check_same(prev, val) = begin
        if prev !== nothing && prev != val
            error("diffvals")
        end
        val
    end
    function sort_swc(fnalt)
        fnaltsort = "$fnalt.sort.swc"
        if !isfile(fnaltsort)
            prog = expanduser("~/Desktop/gapr/nodist/nrn-model/build/sort-swc")
            run(`$prog $fnalt $fnaltsort`)
        end
        fnaltsort
    end
    extra_swcs = []
    for (root, _, files) in walkdir("wyz-figs/Vaa3D_test_20240419"; follow_symlinks=true)
        for f in files
            if lowercase(splitext(f)[2]) == ".swc"
                f = joinpath(root, f)
                m = match(r"^wyz-figs/Vaa3D_test_20240419/(\d_[\.\w-]+)/(\d+_(our_data)/(\d+)_(\d+)\.tif([\.\w]+)|\d_(vaa3d_test_data)/.*)\.swc$", f)
                @assert m !== nothing
                if m.captures[3] == "our_data"
                    pkg, ds, id, sfx = m.captures[[1, 4, 5, 6]]
                    sfx = replace(sfx, r"^_x\d+_y\d+_z\d+(_app\d+)$" => s"\1")
                    fn = f
                    if sfx in (".r2", "_ENT", "_app1", "_meanshift", "_detection.swc_connection")
                        fn = sort_swc(fn)
                    end
                    cube2 = read_nodes_from_swcs([fn])
                    push!(extra_swcs, (fn=f, meth="_$sfx-$pkg", cubeid="$ds/$id", nodes=cube2))
                elseif m.captures[7] == "vaa3d_test_data"
                else
                    @assert false
                end
            end
        end
    end
    csv = readdlm("wyz-figs/vaa3d-test/test_regions.csv", ',', header=true)[1]
    per_cube = []
    for row in eachrow(df)
        refds = split(row.ds, ".unet")[1]
        cat = read(IniFile.Inifile(), mirror_files(refds, :cat))
        cubesize = nothing
        origin = nothing
        direction = nothing
        for (_, sect) in IniFile.sections(cat)
            if haskey(sect, "pattern")
                direction = check_same(direction, reshape(parse_numbers(Float64, sect["direction"]), 3, 3))
                cubesize = check_same(cubesize, parse_numbers(Float64, sect["cubesize"]))
                origin = check_same(origin, parse_numbers(Float64, sect["origin"]))
            end
        end
        @info "$refds $direction"
        invdir = inv(direction)

        refstg = maximum(filter(r -> r.ds == refds, snapshots2).stg)
        nodes0, nodes9 = map([(row.ds, row.stg), (refds, refstg)]) do (ds, stg)
            dir = cache_file_swcdir(ds, stg)
            @info "read dir $dir"
            swcs = readdir(dir, join=true)
            swcs = filter(x -> match(r".*\.swc(\.gz)?$", x) !== nothing, readdir(dir, join=true))
            read_nodes_from_swcs(swcs)
        end

        affine = affine_mapper(origin, direction)

        map(eachrow(csv)) do (ds, id, loc)
            ds = "$ds"
            ds == refds || return
            loc = map(v -> parse(Float64, v), match(r"\((.+),(.+),(.+)\)", loc).captures)
            off = affine.bw(loc)
            off0 = (round.(Int, off ./ cubesize) .- 3) .* cubesize
            offx = nothing
            offd = Inf
            for (dx, dy, dz) in map(Tuple, CartesianIndices((7, 7, 7)))
                off = off0 .+ (dx, dy, dz) .* cubesize
                d = norm(affine.fw(off .+ cubesize) .- loc)
                if d < offd
                    offd, offx = d, off
                end
            end
            function fixcoords(nodes)
                for (idx, n) in enumerate(nodes)
                    x, y, z = n.pos .+ 0.5
                    x, y, z = affine.fw((x, cubesize[2] * 2 - y, z) .+ offx)
                    nodes[idx] = (n..., pos=(x, y, z))
                end
                nodes
            end
            cube9 = clamp_swc_file(nodes9, offx; affine, cubesize)
            cube0 = clamp_swc_file(nodes0, offx; affine, cubesize)
            cube1 = read_nodes_from_swcs(["wyz-figs/vaa3d-test/$(refds)_$(id)_neutube.swc"]) |> fixcoords
            todo = [(:unet, cube0), (:neut, cube1)]
            fnalt = "wyz-figs/vaa3d-test/$(refds)_$(id)_app2.swc"
            if isfile(fnalt)
                fnaltsort = sort_swc(fnalt)
                cube2 = read_nodes_from_swcs([fnaltsort]) |> fixcoords
                push!(todo, (:app2, cube2))
            end
            for extra in extra_swcs
                if extra.cubeid == "$refds/$id"
                    push!(todo, (Symbol(extra.meth), extra.nodes |> fixcoords))
                end
            end

            idadd = 0
            for (tag, cube) in todo
                @assert cube !== nothing
                sel0, sel9 = match_swcs(cube, cube9, thr=1.0)
                p = calc_length(cube9)
                pp = calc_length(cube)
                tp9 = calc_length(cube9, sel9)
                fn = p - tp9
                tp0 = calc_length(cube, sel0)
                fp = pp - tp0
                fdr = fp / pp
                fnr = fn / p
                push!(per_cube, (ds=ds, samp=id, meth=tag, fdr=fdr, fnr=fnr, fp=fp, tp0=tp0, tp9=tp9, fn=fn, pp=pp, p=p))
                open("/tmp/gapr-comp-$refds-$id-$tag-diff.swc", "w") do io
                    sel0, sel9 = map(Set, (sel0, sel9))
                    write_swc(io, cube; idadd=0) do idx, n
                        in(idx, sel0) ? 0 : 3
                    end
                    write_swc(io, cube9; idadd=1_000_000_0000) do idx, n
                        in(idx, sel9) ? 0 : 2
                    end
                end
            end
        end
    end
    DataFrame(per_cube)
end

Make.@target fig_comp_unet (autorec_stats) -> begin
    df = deepcopy(autorec_stats)
    filter!(r -> r.meth != :app2, df)
    filter!(r -> !startswith("_")(string(r.meth)), df)
    sort!(df, [:ds, order(:meth, rev=true)])

    fixlbl(x) = begin
        ord, x = Dict(:unet => (0, "U-Net+neuTube"), :neut => (1, "neuTube in Vaa3D"))[x]
        tex("\\iffalse $ord\\fi $x")
    end
    fixds(x) = tex("\\#$x")

    p1 = groupedboxplot(fixds.(df.ds), df.fdr * 100, group=fixlbl.(df.meth), ylims=(0, 100),
        ; plotstyle(scaleonlyaxis=true)...,
        yguide="FDR (%)",
        xticks=:none,
        linecolor=:black,
        markerstrokecolor=:transparent,
        markersize=3,
        legend=:outertop
    )
    groupeddotplot!(p1, fixds.(df.ds), df.fdr * 100, group=fixlbl.(df.meth),
        xticks=:none,
        label=:none,
        markercolor=:black,
        markersize=1,
    )
    p2 = groupedboxplot(fixds.(df.ds), df.fnr * 100, group=fixlbl.(df.meth), ylims=(0, 100),
        ; plotstyle(scaleonlyaxis=true)...,
        yguide="FNR (%)",
        xguide="Datasets",
        linecolor=:black,
        markerstrokecolor=:transparent,
        markersize=3,
        legend=:none)
    groupeddotplot!(p2, fixds.(df.ds), df.fnr * 100, group=fixlbl.(df.meth),
        markercolor=:black,
        markersize=1,
    )
    p = plot(p1, p2, layout=grid(2, 1, heights=[0.45, 0.45]), link=:x,
        size=(160, 240),
        ; plotstyle()...
    )
    savefig(p, "extra/comp-neutube.tex")
    let x = df
        a = filter(r -> r.meth == :unet, x)
        b = filter(r -> r.meth != :unet, x)
        for v in [:fdr, :fnr]
            t = HypothesisTests.MannWhitneyUTest(a[:, v], b[:, v])
            p = HypothesisTests.pvalue(t, tail=:left)
            show(t)
            @info "pvalue $p"
        end
    end
    df
end

Make.@target fig_autopr_ratio (autopr_stats) -> begin
    x = deepcopy(autopr_stats)
    sort!(x, :ds)
    insertcols!(x, :usedpr => map(d -> dataset_from_id(d).autopr, x[!, :ds]))
    filter!(r -> r.usedpr !== false, x)
    dsname(x) = tex("\\#$x")
    q = bar(dsname.(x[:, :ds]), x[:, :pr_cov] .* 100,
        ; plotstyle()...,
        xguide="Datasets",
        yguide=tex("Automatic proofreading coverage~(%)"),
        legend=false,
        permute=(:x, :y),
        size=(180, 180),
        xflip=true,
        ylim=(0, 100),
        xlim=(-0.1, size(x, 1) + 0.1),
    )
    push!(q.o.the_plot.preamble, "\\usepackage{siunitx}")
    savefig(q, "extra/autopr-coverage.tex")
    x
end

Make.@target tab_comp_algs (autorec_stats) -> begin
    df = deepcopy(autorec_stats)
    filter!(r -> startswith("_")(string(r.meth)), df)
    samp_id(r) = "$(r.ds)-$(r.samp)"
    allsamps = unique([samp_id(r) for r in eachrow(df)])
    allalgs = unique([r.meth for r in eachrow(df)])
    map(allalgs) do alg
        res = map(allsamps) do s
            rows = []
            for r in eachrow(df)
                if samp_id(r) == s && r.meth == alg
                    push!(rows, r)
                end
            end
            @assert length(rows) <= 1
            isempty(rows) && return missing
            first(rows)
        end
        res = collect(skipmissing(res))
        fdrs = map(v -> v.fdr * 100, res)
        fnrs = map(v -> v.fnr * 100, res)
        meansd(x) = (mean=round(mean(x), digits=3), sd=round(std(x), digits=2))
        (alg=alg, N=length(res), fdr=meansd(fdrs), fnr=meansd(fnrs))
    end
    df
end

Make.@target all (
    tab_datasets,
    tab_comp_algs,
    fig_auto_rec,
    fig_autopr_resnet,
    fig_collab_real,
    fig_collab_simu,
    fig_len_dist,
    fig_pr_speed,
    fig_errors_stat,
    fig_more_stgs,
    fig_comp_unet,
    fig_autopr_ratio,
) -> begin end

Make.@target fig2_src_dat (fig_auto_rec, fig_autopr_resnet) -> begin
    XLSX.openxlsx("extra/source-dat-fig2.xlsx", mode="w") do exc
        let sheet = exc[1]
            dat = deepcopy(fig_auto_rec)[:, DataFrames.Not([:stg])]
            XLSX.rename!(sheet, "panel-c")
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Dataset" "Time used (h)" "Node count" "Length (um)" "FNR" "FDR" "False positive (um)" "True positive (um)" "True positive alt (um)" "False negative (um)" "Predicted positive (um)" "Positive (um)"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-g")
            dat = deepcopy(fig_autopr_resnet)[:, [:ds, :time, :n_node, :len, :np, :nbr, :nmiss, :pr_cov]]
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Dataset" "Time usage (h)" "Node count" "Length (um)" "Proofread node" "No of Branch-link" "Missed Branch-like" "Proofread coverage"], :)
        end
    end
    0
end

Make.@target fig3_src_dat (fig_collab_real, fig_collab_simu) -> begin
    XLSX.openxlsx("extra/source-dat-fig3.xlsx", mode="w") do exc
        let sheet = exc[1]
            XLSX.rename!(sheet, "panel-b")
            dat = deepcopy(fig_collab_real[1])
            users = unique(dat[:, :who])
            anom = Dict("root" => "root")
            for u in users
                if u != "root"
                    n = length(anom)
                    anom[u] = "user $n"
                end
            end
            dat[:, :who] = map(u -> anom[u], dat[:, :who])
            id2row = Dict()
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
                id2row[r.id] = i + 1
            end
            for r in eachrow(fig_collab_real[2])
                i = id2row[r[1]]
                sheet[i, 5:6] = r[3:4]
            end
            sheet[1, :] = reshape(["Commit ID" "Annotator" "Time" "Commit type" "Node count" "Node not proofread"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-c")
            dat = deepcopy(fig_collab_simu).server_bw
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Parallelism", "Server band width (KB)"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-d")
            dat = deepcopy(fig_collab_simu).server_cpu
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Parallelism", "Server CPU usage (% of 1 core)"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-e")
            dat = deepcopy(fig_collab_simu).server_freqs
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Parallelism", "Commit time"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-f")
            dat = deepcopy(fig_collab_simu).client_lag
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Parallelism", "Client round-trip time (ms)"], :)
        end
    end
    0
end

Make.@target fig4_src_dat (fig_len_dist, fig_pr_speed, fig_errors_stat, fig_more_stgs) -> begin
    XLSX.openxlsx("extra/source-dat-fig4.xlsx", mode="w") do exc
        let sheet = exc[1]
            dat = deepcopy(fig_len_dist)[:, Not([:stg, :ord])]
            dat.:mm_axon_raw = map(v -> v == 0 ? missing : v, dat.mm_axon_raw)
            XLSX.rename!(sheet, "panel-b")
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Filename" "Dataset" "High quality" "Total length (um)" "Axon length (mm)"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-c")
            dat = deepcopy(fig_pr_speed)[:, [:ds, :type, :time, :n_node_sel, :len_sel]]
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Dataset" "Proofreading scheme" "Time usage (h)" "Node count" "Length (um)"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-de")
            dat = deepcopy(fig_errors_stat)
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = collect(r)
            end
            sheet[1, :] = reshape(["Dataset" "loops" "terms" "bad_terms" "fixed" "wontfix" "redundant" "invalid" "unresolved" "?wontfix" "len"], :)
        end
        let sheet = XLSX.addsheet!(exc, "panel-f")
            dat, name = deepcopy(fig_more_stgs)
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, 3:7] = collect(r)
                sheet[i+1, 1:2] = [name[i]...]
            end
            sheet[1, :] = reshape(["Name" "Dataset"], :)
            for j in 2:6
                sheet[1, j+1] = "stage $j"
            end
        end
    end
    0
end

Make.@target fige1_src_dat (fig_comp_unet) -> begin
    XLSX.openxlsx("extra/source-dat-fige1.xlsx", mode="w") do exc
        let sheet = exc[1]
            dat = deepcopy(fig_comp_unet)
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = map(v -> v isa Symbol ? string(v) : v, collect(r))
            end
            sheet[1, :] = reshape(["Dataset" "Sample cube ID" "Method" "FDR" "FNR" "FP (um)" "TP (um)" "TP alt (um)" "FN (um)" "Predicted Positive (um)" "Positive (um)"], :)
        end
    end
    0
end

Make.@target tabs1_src_dat (tab_comp_algs) -> begin
    XLSX.openxlsx("extra/source-dat-tabs1.xlsx", mode="w") do exc
        let sheet = exc[1]
            dat = deepcopy(tab_comp_algs)
            for (i, r) in enumerate(eachrow(dat))
                sheet[i+1, :] = map(v -> v isa Symbol ? string(v) : v, collect(r))
            end
            sheet[1, :] = reshape(["Dataset" "Sample cube ID" "Method (output suffix)" "FDR" "FNR" "FP (um)" "TP (um)" "TP alt (um)" "FN (um)" "Predicted Positive (um)" "Positive (um)"], :)
        end
    end
    0
end

function remap_groups_alt(swcs, outdir)
    function handle_mapping(row, ii, oo)
        idadd = (10 * rownumber(row) + row.cls) * 100000000
        chg(id) = (Int64(id) % 100000000 + idadd)
        while !eof(ii)
            l = readline(ii)
            length(l) > 0 || continue
            if l[1] == '#'
                a = parse_gapr_annot(l)
                if a !== nothing
                    write(oo, format_gapr_annot(chg(a.id), a.key, a.val))
                end
                continue
            end
            (id, typ, pos, r, par) = parse_swc_line(l)
            id = chg(id)
            if par != -1
                par = chg(par)
            end
            write(oo, format_swc_line(id, typ, pos, r, par))
        end
    end
    force_dir(outdir) do tmpdir
        swcs = pmap(eachrow(swcs)) do row
            newfn = "$(row.cls)-$(row.fn)"
            path = joinpath(tmpdir, newfn)
            open(path, "w") do oo
                open(row.path, "r") do ii
                    handle_mapping(row, ii, oo)
                end
            end
            row.path = joinpath(outdir, newfn)
            row
        end
        touch(joinpath(tmpdir, "dir"))
    end
    DataFrame(swcs)
end
Make.@target pick_mus_neurons (xform_swcs, neurons_summary) -> begin
    swcs = deepcopy(xform_swcs)
    filter!(r -> check_neuron_qc(r.fn) != 0, swcs)
    nrns = deepcopy(neurons_summary)
    # actually all the neurons
    outdir = "/data/tmp/swcs-random-pick"

    swcs = leftjoin(DataFrame(swcs), nrns, on=:fn)
    insertcols!(swcs, :cls => -1)

    for row in eachrow(swcs)
        row.cls = (row.qc ? 1 : 0)
        @assert row.stg % 10 == 9
        @assert row.ord < 999000
        @assert match(r"@seg\d*", row.fn) === nothing
        @assert match(r"@\.swc$", row.fn) === nothing
    end
    swcs = filter(v -> v.cls >= 0, swcs)

    remap_groups_alt(swcs, outdir)
end

Make.@target extra (
    all,
    fig2_src_dat,
    fig3_src_dat,
    fig4_src_dat,
    fige1_src_dat,
    tabs1_src_dat,
    pick_mus_neurons,
) -> begin end

