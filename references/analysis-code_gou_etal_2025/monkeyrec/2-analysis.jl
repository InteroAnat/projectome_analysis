const all_sample_channels = [
    (221059, 1)
    (221747, 1)
    (231623, 1)
    (231623, 2)
    (232717, 1)
    (232717, 2)
    (233001, 1)
    (233001, 2)
    (234162, 1)
    (234162, 2)
    (234197, 1)
    (240025, 1)
]

function sample_abbrev(samp::AbstractString)
    Dict(
        "221059" => "A59",
        "221747" => "B47",
        "231623" => "C23",
        "232717" => "D17",
        "233001" => "E01",
        "234162" => "F62",
        "234197" => "G97",
        "240025" => "H25",
    )[samp]
end
function sample_abbrev(samp::AbstractString, chan::AbstractString)
    string(sample_abbrev(samp), "-", Dict("1" => "1", "2" => "2")[chan])
end

split_swcs(ID, CH) = Make.@x "monkeytemp/split-sample$ID-ch$CH"

(out::split_swcs(ID, CH))(in::Make.@x("/home/goulf/Desktop/segreg/data/temp/xformed-fmost$ID.ch$CH.swc.gz")) where {ID,CH} = begin
    fix_types(swc::SwcTable, name) = begin
        axontyp = 2
        if startswith("qroot")(name)
            for row in swc.nodes
                row.typ = axontyp
            end
        else
            nhit = 0
            rootdeg = 0
            rootind = first(swc.nodes).ind
            for row in swc.nodes
                pidx = row.parind
                pidx == 0 && continue
                if swc[pidx].typ == axontyp
                    nhit += 1
                    row.typ = axontyp
                end
                if pidx == rootind
                    rootdeg += 1
                end
            end
            rootdeg < 1 && @error "$name has no root hits"
            if rootdeg == 1 && nhit == 0
                for row in swc.nodes
                    row.typ = axontyp
                end
            end
        end
    end

    outdir = out.path
    outdirtmp = "$outdir.temp"
    rm(outdir; force=true, recursive=true)
    rm(outdirtmp; force=true, recursive=true)
    mkdir(outdirtmp)
    @info "load $(in.path)..."
    swc = Gapr.load_swc(in.path, sort=true)
    filter!(swc.annots) do ann
        startswith(".")(String(ann.key)) && return false
        ann.key === :error && ann.val == "HIGHLIGHTTERMS" && return false
        ann.key === :previd && return false
        return true
    end
    @info "split ..."
    id2root = Dict{Int64,String}()
    for ann in swc.annots
        if ann.key === :root
            prevval = get!(id2root, ann.id, ann.val)
            @assert prevval == ann.val
        end
    end
    subswcs = Gapr.split_swc(swc)
    Gapr.write_swc(joinpath(outdirtmp, "noroot-others.swc.gz")) do io
        for x in subswcs
            tag, name = x.rootid, x.name
            name = get(id2root, tag, nothing)
            fn = if name !== nothing
                fix_types(x.swc, name)
                "root$tag-$name.swc.gz"
            elseif length(x.swc.nodes) >= 1000
                "noroot-$tag.swc.gz"
            else
                Gapr.save_swc(io, x.swc)
                continue
            end
            Gapr.save_swc(joinpath(outdirtmp, fn), x.swc)
        end
    end
    mv(outdirtmp, outdir)
end

all_datasets() = Make.list(split_swcs, all_sample_channels)
list_neurons() = Make.@x "interm/list-neurons.dat"
list_neurons_proj() = Make.@x("interm/projection.dat")
list_neurons_dist() = Make.@x "interm/list-neurons-dist.dat"
list_neurons_itptct() = Make.@x "interm/list-neurons-itptct.dat"
list_neurons_detail() = Make.@x "interm/list-neurons-detail.dat"
list_neurons_hclust() = Make.@x "interm/list-neurons-hclust.dat"
list_neurons_uvw() = Make.@x "interm/list-neurons-uvw.dat"
list_neurons_decomp() = Make.@x "interm/list-neurons-decomp.dat"

list_nodes_rawinfo() = Make.@x "interm/list-nodes-rawinfo.dat"
list_nodes_uvw() = Make.@x "interm/list-nodes-uvw.dat"

list_mus_neurons() = Make.@x "interm/list-mus-neurons.dat"
list_mus_neurons_proj() = Make.@x "interm/list-mus-neurons-proj.dat"
list_mus_neurons_dist() = Make.@x "interm/list-mus-neurons-dist.dat"
list_mus_neurons_hclust() = Make.@x "interm/list-mus-neurons-hclust.dat"
list_mus_neurons_detail() = Make.@x "interm/list-mus-neurons-detail.dat"
list_mus_neurons_decomp() = Make.@x "interm/list-mus-neurons-decomp.dat"

list_mus_nodes_depth() = Make.@x "interm/list-mus-nodes-depth.dat"

arbor_dist_for(samp) = Make.@x("interm/arbordist-$samp.dat")

output_fig(N) = Make.@x("output/fig$N.pdf")
output_tab(N) = Make.@x("output/tab$N.pdf")
output_figs() = Make.@x("output/figures.pdf")
output_figs_opt() = Make.@x("monkeytemp/00-upload/figures-opt.pdf")
output_vid(N) = Make.@x("output/vid$N.mp4")

topo_proj_data(tag) = Make.@x("interm/topo-proj-$tag.dat")
topo_proj_data_mus(tag) = Make.@x("interm/topo-proj-mus-$tag.dat")

mus_neuron_decomp(f) = begin
    m = match(r"^(\d+)_(\d+)\.swc(\.gz)?$", f)
    m === nothing && return m
    (; sample=String(m.captures[1]), name=String(m.captures[2]))
end
(out::Make.@x("monkeytemp/mouse-pfc"))() = begin
    rawdir = "/fugax/mouse-pfc/swc_allen_neurite"
    ensure_directory(out.path) do _, outdirtmp
        files = map(readdir(rawdir)) do f
            m = mus_neuron_decomp(f)
            @assert m !== nothing
            joinpath(rawdir, f)
        end
        asyncmap(files, ntasks=3, batch_size=128) do batch
            infiles = map(first, batch)
            run(cmd_gapr(`map --decimate=12:1.3 -o "$outdirtmp/%n.gz" $infiles`))
            infiles
        end
    end
end

(out::list_mus_neurons())(in::Make.@x("monkeytemp/mouse-pfc")) = begin
    df = map(readdir(in.path)) do f
        path = joinpath(in.path, f)
        m = mus_neuron_decomp(f)
        @assert m !== nothing "$path error"
        sample = m.sample
        (; sample, path)
    end |> DataFrame

    procs = addprocs(7)
    df2 = try
        @everywhere procs @eval using ProgressMeter, Gapr
        @showprogress pmap(WorkerPool(procs), df.path) do path
            swc = Gapr.load_swc(path)
            root = swc[1].id
            soma_pos = swc.nodes[1].pos
            len_by_t = Gapr.neurite_length(swc)
            len = sum(len_by_t, init=0.0)
            axonlen = get(len_by_t, Symbol(2), 0.0)
            loops = length(swc._loops)
            (; root, path, soma_pos, len, axonlen, loops)
        end
    finally
        rmprocs(procs)
    end
    leftjoin!(df, DataFrame(df2), on=:path) |> dropmissing!

    filter!(r -> r.loops == 0, df)
    filter!(r -> r.len > 0, df)
    filter!(r -> r.axonlen > 0, df)
    df = df[:, Not(:loops)]
    JLD2.@save out.path df
end
(out::list_mus_neurons_detail())(in::list_mus_neurons()) = begin
    df = JLD2.load(in.path, "df")
    csv = DelimitedFiles.readdlm("monkeytemp/wyz-upload/mouse_PFC_neuron_class.csv", ',', header=true)[1]
    insertcols!(df, :typ => "", :subtyp => 0)
    for r in eachrow(df)
        fn = basename(r.path)
        hits = findall(i -> startswith(csv[i, 1])(fn), axes(csv, 1))
        @assert length(hits) == 1
        nrn, subtyp, typ = @view csv[hits[1], :]
        r.subtyp = subtyp
        r.typ = typ
    end

    allen = Allen.load_images()
    getter = Allen.getter(allen)
    insertcols!(df, :somahier => getter(reduce(hcat, df.soma_pos))[2])

    JLD2.jldsave(out.path; df)
end

(out::list_mus_nodes_depth())(
    list::list_mus_neurons(),
) = begin
    df = JLD2.load(list.path, "df")
    depth_func = knn_based_depth_allen()
    sampname(r) = string(r.sample, "/", r.path)

    sampnames = unique(sampname.(eachrow(df)))
    df = @showprogress map(sampnames) do sn
        dats = map(eachrow(df)) do r
            sampname(r) != sn && return missing
            swc = Gapr.load_swc(r.path)
            map(swc.nodes, depth_func([n.pos for n in swc.nodes])) do n, depth
                (; n.id, depth)
            end
        end
        sn => reduce(vcat, collect(skipmissing(dats)))
    end
    JLD2.jldsave(out.path; df)
end

function decompose_swcs_depthfirst(df)
    procs = addprocs(15)
    dfx = try
        @everywhere procs @eval using ProgressMeter, Gapr
        @showprogress pmap(WorkerPool(procs), df.path) do path
            swc = Gapr.load_swc(path)
            majorlens, nonmajorlen, majortips = Gapr.decompose_swc_depthfirst!(swc, K=nothing, L=100)
            (; path, majorlens, majortips, nonmajorlen)
        end
    finally
        rmprocs(procs)
    end
    DataFrame(dfx)
end

(out::list_mus_neurons_decomp())(
    list::list_mus_neurons(),
) = begin
    df = JLD2.load(list.path, "df")
    df = decompose_swcs_depthfirst(df)
    JLD2.jldsave(out.path; df)
end
(out::list_neurons_decomp())(
    list::list_neurons(),
) = begin
    df = JLD2.load(list.path, "df")
    df = decompose_swcs_depthfirst(df)
    JLD2.jldsave(out.path; df)
end


# sample/channel/path/root/qroot
# + soma_pos/len/axonlen
(out::list_neurons())(in::all_datasets()) = begin
    df = map(in.iter) do ((samp, ch), p)
        map(readdir(p)) do f
            m = match(r"^(root(\d+)|noroot-others|noroot-\d+)(-(q?root)(\d+-good)?.*|-seg.*|-alien.*|-tmp.*|-||-sroot\d+|-up.*|-trace.*).swc.gz", f)
            @assert m !== nothing "unknown fn $p//$f"
            startswith("noroot")(m.captures[1]) && return missing
            root = parse(Int64, m.captures[2])
            m.captures[4] === nothing && return missing
            m.captures[5] === nothing && return missing
            qroot = m.captures[4] == "qroot"
            (sample="$samp", channel="$ch", path=joinpath(p, f), root=root, qroot=qroot)
        end
    end
    df = reduce(vcat, df) |> skipmissing |> DataFrame

    procs = addprocs(5)
    df2 = try
        @everywhere procs @eval using ProgressMeter, Gapr
        @showprogress pmap(WorkerPool(procs), df.path) do path
            swc = Gapr.load_swc(path)
            soma_pos = first(swc.nodes).pos
            len_by_t = Gapr.neurite_length(swc)
            len = sum(len_by_t, init=0.0)
            axonlen = get(len_by_t, Symbol(2), 0.0)
            loops = length(swc.loops)
            (; path, soma_pos, len, axonlen, loops)
        end
    finally
        rmprocs(procs)
    end

    df2 = DataFrame(df2)
    dropmissing!(leftjoin!(df, df2, on=:path))
    let
        for r in eachrow(df)
            if r.loops > 0
                raise_alarm("loops") do io
                    println(io, "CNT=$(r.loops) $(r.path)")
                end
            end
            if r.len == 0.0 || r.axonlen == 0.0
                raise_alarm("zero-len") do io
                    println(io, "TOT=$(r.len) AXON=$(r.axonlen) $(r.path)")
                end
            end
        end
        tree = NearestNeighbors.KDTree(df.soma_pos)
        inds, dists = NearestNeighbors.knn(tree, df.soma_pos, 5)
        pairs = []
        for r in eachrow(df)
            for (i, k) in enumerate(inds[rownumber(r)])
                d = dists[rownumber(r)][i]
                if d < 35 && rownumber(r) != k
                    a, b = sort([r.path, df[k, :path]])
                    push!(pairs, a => b)
                end
            end
        end
        for (a, b) in unique(pairs)
            val = try
                read(cmd_gapr("brief", "--f1score", a, b), String) |> rstrip
            catch
                "err"
            end
            raise_alarm("maybe-the-same") do io
                println(io, "F1=$val $a $b")
            end
        end
    end

    sampchan(r) = (r.sample, r.channel)
    for sc in unique(sampchan.(eachrow(df)))
        dupnames = Dict{String,Set{String}}()
        for r in eachrow(df)
            if sampchan(r) == sc
                m = match(r"/root\d+-(q?root\d+)-good.*", r.path)
                @assert m !== nothing
                push!(get!(() -> Set(), dupnames, m.captures[1]), r.path)
            end
        end
        for (n, pp) in dupnames
            if length(pp) != 1
                raise_alarm("dupname") do io
                    println(io, "SAMPCHAN=$(sc) $(n)")
                end
            end
        end
    end

    filter!(r -> r.loops == 0, df)
    filter!(r -> r.len > 0, df)
    filter!(r -> r.axonlen > 0, df)
    df = df[:, Not(:loops)]
    leftjoin!(df, group_into_inj_sites(df), on=:path) |> dropmissing!
    JLD2.jldsave(out.path; df)
end


function raise_alarm(f::Function, tag::AbstractString)
    open(f, joinpath("monkeytemp//alarms", "$tag.txt"), "a")
end

function cmd_gapr(cmd::Cmd)
    exe = cmd.exec[1]
    exe = expanduser("~/Desktop/gapr/build/source/$exe/gapr-$exe")
    cmd.exec[1] = exe
    cmd
end

(out::Make.@x("monkeytemp/decimate"))(
    byp::Make.ByProduct{Make.@x("interm/neuron-list-decimate.dat")},
    list::list_neurons(),
) = begin
    df = JLD2.load(list.path, "df")
    partof(r) = (r.sample, r.channel, ceil(Int, log10(r.len + 10)))
    pfx(part) = string("samp", part[1], "ch", part[2], "s", part[3], "-")
    parts = unique(map(partof, eachrow(df)))
    outputfn(r) = begin
        pfx(partof(r)) * basename(r.path)
    end
    mktempdir() do tempdir
        ensure_directory(out.path) do outdir, outdirtmp
            insertcols!(df, :pathx => "")
            @showprogress asyncmap(parts; ntasks=5) do part
                dfx = filter(r -> partof(r) == part, df)
                infiles = dfx.path
                par = [
                    "1:0",
                    "1:0",
                    "1:0",
                    "27:3",
                    "81:9",
                    "192:18",
                    "192:18",
                ][part[3]]
                run(cmd_gapr(`map --decimate=$par -o "$tempdir/$(pfx(part))%n" $infiles`))
            end
            for r in eachrow(df)
                outfn = outputfn(r)
                r.pathx = joinpath(outdir, outfn)
                swc = Gapr.load_swc(joinpath(tempdir, outfn))
                if !NmtDat.isleft(swc[1].pos)
                    for n in swc.nodes
                        n.pos = NmtDat.mirror(n.pos)
                    end
                end
                Gapr.save_swc(joinpath(outdirtmp, outfn), swc)
            end
        end
    end
    df = df[:, [:path, :pathx]]
    JLD2.jldsave(byp.path; df)
end

(out::Make.@x("monkeytemp/decimate-mouse"))(
    byp::Make.ByProduct{Make.@x("interm/neuron-list-decimate-mus.dat")},
    list::list_mus_neurons(),
) = begin
    df = JLD2.load(list.path, "df")
    groupof(r) = match(r"^([a-zA-Z0-9]+)", basename(r.path)).captures[1]
    groups = unique(map(groupof, eachrow(df)))

    intermfn(r) = bytes2hex(SHA.sha256(r.path)) * ".swc"
    outputfn(r) = basename(r.path)
    insertcols!(df, :pathx => "")

    mktempdir() do tempdir
        ensure_directory(out.path) do outdir, outdirtmp
            @showprogress asyncmap(groups; ntasks=5) do grp
                mktempdir() do grptempdir
                    dfx = filter(r -> groupof(r) == grp, df)
                    infiles = map(eachrow(dfx)) do r
                        swc = Gapr.load_swc(r.path)
                        filter = Gapr.trim_swc(swc, 6)
                        trimedpath = joinpath(grptempdir, intermfn(r))
                        Gapr.save_swc(trimedpath, swc; filter)
                        trimedpath
                    end
                    run(cmd_gapr(`map --decimate=192:18 -o "$tempdir/%n" $infiles`))
                end
            end
            for r in eachrow(df)
                outfn = outputfn(r)
                r.pathx = joinpath(outdir, outfn)
                swc = Gapr.load_swc(joinpath(tempdir, intermfn(r)))
                if !Allen.isleft(swc[1].pos)
                    for n in swc
                        n.pos = Allen.mirror(n.pos)
                    end
                end
                Gapr.save_swc(joinpath(outdirtmp, outfn), swc)
            end
        end
    end
    df = df[:, [:path, :pathx]]
    JLD2.jldsave(byp.path; df)
end

function load_model(modelpath)
    model = JLD2.load(modelpath, "df")
    adddim3(x) = begin
        @check_size x (3, :)
        reshape(x, 1, size(x)...)
    end
    deldim6(x) = begin
        @check_size x (1, 6, :)
        reshape(x, size(x)[2:end]...)
    end
    deldim3(x) = begin
        @check_size x (1, 3, :)
        reshape(x, size(x)[2:end]...)
    end
    Flux.fmap_with_path(model) do kp, x
        kp = kp[2:end]
        if kp == Flux.KeyPath(:from_uni, :layers, 1)
            x = adddim3
        elseif kp == Flux.KeyPath(:from_uni, :layers, 3)
            x = deldim3
        elseif kp == Flux.KeyPath(:to_uni, :layers, 2, :layers, 1, :layers, 1)
            x = adddim3
        elseif kp == Flux.KeyPath(:to_uni, :layers, 2, :layers, 1, :layers, 3)
            x = deldim6
        end
        @info "$kp $(typeof(x))"
        x
    end
end

function calc_dist_mat(df::DataFrame, dfdist::DataFrame, matdist::AbstractMatrix; on=:path)
    calc_dist_mat(df => df, dfdist, matdist; on)
end
function calc_dist_mat((df1, df2)::Pair{DataFrame,DataFrame}, dfdist::DataFrame, matdist::AbstractMatrix; on=:path)
    @info "calc dist mat"
    path2idx = Dict(reverse.(enumerate(dfdist[:, on])))
    ii, jj = ([get(path2idx, dfx[i, on], 0) for i in axes(dfx, 1)] for dfx in (df1, df2))
    [(i == 0 || j == 0) ? Inf : matdist[i, j] for i in ii, j in jj]
end
function load_dist_mat(dfx, path::AbstractString; on=:path)
    swcmap, distmat = JLD2.load(path, "df", "distmat")
    calc_dist_mat(dfx, swcmap, distmat; on)
end

(out::list_neurons_dist())(
    list::list_neurons(),
    ::Make.@x("monkeytemp/decimate"),
    listx::Make.@x("interm/neuron-list-decimate.dat"),
) = begin
    df = JLD2.load(list.path, "df")
    dfx = JLD2.load(listx.path, "df")
    leftjoin!(df, dfx, on=:path) |> dropmissing!
    filter!(r -> !endswith("X")(r.inj), df)
    sort!(df, :pathx)
    mat, swcs = call_gapr_dist_helper(df.pathx; npar=32, ngib=128)
    @assert swcs == df.pathx
    distmat = mat
    df = df[:, [:path, :pathx]]
    JLD2.jldsave(out.path; df, distmat)
end

(out::list_mus_neurons_dist())(
    list::list_mus_neurons(),
    ::Make.@x("monkeytemp/decimate-mouse"),
    listx::Make.@x("interm/neuron-list-decimate-mus.dat"),
) = begin
    df = JLD2.load(listx.path, "df")
    sort!(df, :pathx)
    mat, swcs = call_gapr_dist_helper(df.pathx; npar=15, ngib=30)
    @assert swcs == df.pathx
    distmat = mat
    df = df[:, [:path, :pathx]]
    JLD2.jldsave(out.path; df, distmat)
end

function calc_common_root_path(paths)
    p = paths[1]
    for p2 in paths[2:end]
        while startswith("..")(relpath(p2, p))
            p = dirname(p)
        end
    end
    p
end

login_info() = (; user="wangyz", host="172.20.10.250", port=20007)

function call_gapr_dist_helper(swcs; npar=12, ngib=40, rowrng=nothing, colrng=nothing, usepartial::Bool=false)
    swcs = map(p -> replace(p, r"^monkeytemp/" => "/fugax/monkeytemp/"), swcs)
    @info "calcdist $(swcs[1])"
    if rowrng === nothing
        rowrng = 1:length(swcs)
    end
    if colrng === nothing
        colrng = 1:length(swcs)
    end
    rowrng = UnitRange(rowrng)
    colrng = UnitRange(colrng)
    root = normpath(calc_common_root_path(swcs))
    swcs_rel = map(p -> relpath(p, root), swcs)

    swcs_hash = bytes2hex(SHA.sha256(root))
    task_hash = open(setenv(`zcat $swcs_rel`; dir=root)) do io
        SHA.sha256(
            reduce(vcat,
                [
                    SHA.sha256(string(rowrng, "\n", colrng, "\n")),
                    SHA.sha256(root),
                    SHA.sha256(join(swcs_rel, "\n")),
                    empty!(SHA.sha256(io)),
                ]
            )
        ) |> bytes2hex
    end

    login = login_info()
    cache_dir = "gapr-dist-cache"
    cache_swcs_dir = "$cache_dir/swcs-$swcs_hash.dir"
    @info "rsync ..."
    run(`rsync -acz --delete $root/ -e "ssh -p $(login.port)" $(login.user)@$(login.host):"~/$cache_swcs_dir"`)
    cache_task_scr = "$cache_dir/run-$task_hash.sh"
    cache_task_out = "$cache_dir/run-$task_hash.out"
    cache_task_err = "$cache_dir/run-$task_hash.err"
    @info "upload list ..."
    open(`ssh $(login.user)@$(login.host) -p $(login.port) tee "~/$cache_task_scr"`, "w") do io
        print(
            io,
            """#!/bin/sh
#SBATCH -J gapr-dist
#SBATCH --partition=pro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$npar
#SBATCH -o $cache_task_err.txt
#SBATCH -e $cache_task_err
#SBATCH --time=72:00:00
 #--mem=36720
 
export OMP_NUM_THREADS=$npar;
export OMP_SCHEDULE=dynamic;
export GAPR_MEM_AVAIL=$ngib
cd ~/$cache_dir;
flock -n ~/$cache_task_scr ~/bin/gapr-brief --dist --dist-row=$(rowrng.-1) --dist-col=$(colrng.-1) >~/$cache_task_out.tmp """
        )
        for s in swcs_rel
            print(io, " \\\n  ~/'", joinpath(cache_swcs_dir, s), "'")
        end
        println(
            io,
            """
            \nif [ \$? = 0 ]; then
mv -i ~/$cache_task_out.tmp ~/$cache_task_out
fi """
        )
    end

    cmd = if !usepartial
        @info "run gapr-brief dist"
        run(`ssh $(login.user)@$(login.host) -p $(login.port) "if [ ! -e ~/$cache_task_out ]; then sbatch ~/$cache_task_scr; fi"`)
        `ssh -C $(login.user)@$(login.host) -p $(login.port) "cat ~/$cache_task_out"`
    else
        `ssh -C $(login.user)@$(login.host) -p $(login.port) "head -n-2 ~/$(cache_task_out).tmp"`
    end

    @info "retrieve result ~/$cache_task_out"
    mat = fill(NaN, size(swcs, 1), size(swcs, 1))
    open(cmd) do io
        v = readdlm(io, '\t', header=true)[1]
        for (i, j, s, m, nm) in eachrow(v)
            i, j = Int64.((i, j) .+ 1)
            dd = s / (m + nm)
            mat[i, j] = dd
            mat[j, i] = dd
        end
    end
    if usepartial
        vmat = view(mat, rowrng, colrng)
        vmat[isnan.(vmat)] .= Inf
    end
    if count(isnan, view(mat, rowrng, colrng)) > 0
        error("incomplete res")
    end
    swcs = map(p -> replace(p, r"^/fugax/monkeytemp/" => "monkeytemp/"), swcs)
    mat, swcs
end

function classify_itptct(df, x_cols; N1=8, act=:relu, depth=2)
    classes = sort(unique((df.typ0)))
    get_input(df1) = begin
        x_mat = zeros(Float32, 1024, size(df1, 1))
        x_mat[axes(x_cols, 1), :] .= Matrix(df1[:, x_cols])'
        log.(x_mat .+ 1)
    end
    model = let dfx = filter(r -> r.typ0 != "", df) |> DataFrame
        dfx = shuffle(dfx)[1:2000, :] |> DataFrame
        y_label = Flux.onehotbatch(dfx.typ0, classes)
        y_label = Flux.label_smoothing(y_label, 0.3f0)
        x_mat = get_input(dfx)
        N2 = N1 * 2
        act = (relu=Flux.relu, tanh=Flux.tanh_fast)[act]
        model = Flux.Chain(
            Flux.Dense(1024 => N1),
            act,
            Flux.Dense(N1 => N2),
            [
                Flux.SkipConnection(Flux.Chain(Flux.Dense(N2 => N1, tanh), Flux.BatchNorm(N1), Flux.Dense(N1 => N2)), +) for _ in 1:depth
            ]...,
            act,
            Flux.Dense(N2 => length(classes)),
        ) |> Flux.gpu

        loss(m, x, y2) = begin
            y = m(x)
            Flux.logitcrossentropy(y, y2)
        end

        x_mat, y_label = (x_mat, y_label) |> Flux.gpu
        st = Flux.setup(Flux.Adam(), model)
        @showprogress for K in 1:20000
            nbat = 32
            rows = rand(axes(dfx, 1), nbat) |> Flux.gpu
            x = x_mat[:, rows]
            y = y_label[:, rows]
            l, g = Flux.withgradient((m) -> loss(m, x, y), model)
            Flux.update!(st, model, g[1])
        end

        x_mat = get_input(df) |> Flux.gpu
        y = Flux.softmax(model(x_mat)) |> Flux.cpu
        assign = classes[map(v -> v[1], findmax(y, dims=1)[2])]
        for r in eachrow(df)
            r.typ = assign[rownumber(r)]
        end
    end
    nerr = 0
    open("/tmp/xxx-itptct-errs.txt", "w") do io
        for r in eachrow(df)
            if r.typ0 != "" && r.typ0 != r.typ
                println(io, r.path)
                nerr += 1
            end
            if r.typ0 != ""
                r.typ = r.typ0
            end
        end
    end
    nerr
end

(out::list_neurons_itptct())(
    list::list_neurons(),
    proj::list_neurons_proj(),
) = begin
    df = JLD2.load(list.path, "df")
    proj = JLD2.load(proj.path, "df")
    leftjoin!(df, proj, on=intersect(names(df), names(proj)))

    insertcols!(df, :typ => "", :typ0 => "")
    ann = string.(readdlm("monkeytemp/wyz-upload/IT_PT_CT_confirmed.csv", ','; header=true)[1])
    for r in eachrow(df)
        fn = match(r"/root\d+-(q?root\d+-good.*)", r.path).captures[1]
        hits = filter(i -> (ann[i, 1:2]...,) == (r.sample, r.channel) && startswith(ann[i, 3])(fn), axes(ann, 1))
        ## XXX: @assert length(hits) < 2 "mult $(r.path)"
        if length(hits) == 1
            a = ann[hits[1], 4]
            if a == "ITs_ITc"
                a = "ITs"
            end
            if a in ("IT", "PT", "CT", "ITc", "ITi", "ITs")
                r.typ0 = a
            else
                @error "unknown itptct $a"
            end
        end
    end

    x_cols = push!(filter!(Data.is_projcol, names(df)), "len", "axonlen")
    for N1 in (12, 16, 20)
        for act in (:relu, :tanh)
            for depth in (3, 4, 5)
                nerrs = []
                for _ in 1:3
                    #push!(nerrs, classify_itptct(df, x_cols; N1, act, depth))
                end
                @info "N1/$N1 act/$act depth/$depth = $(sort(nerrs))"
            end
        end
    end

    nerr = classify_itptct(df, x_cols; N1=20, act=:relu, depth=3)
    @warn "Num err $nerr"
    df = df[:, Cols(:path, :typ0, :typ,)]
    JLD2.jldsave(out.path; df)
end

# + charm/sarm
(out::list_neurons_proj())(list::list_neurons()) = begin
    df = JLD2.load(list.path, "df")
    nmt = NmtDat.load_images()
    procs = addprocs(7)
    accumtmps = try
        @everywhere procs @eval using LinearAlgebra, Gapr, ProgressMeter, StaticArrays
        @everywhere procs @eval using zz0atlas: Allen, NmtDat
        @everywhere procs begin
            nmtshare = (sarm=$(nmt.sarm), charm=$(nmt.charm), res=$(nmt.res))
        end
        @showprogress pmap(WorkerPool(procs), enumerate(df.path)) do (rownum, path)
            accumtmp = Dict{Tuple,Float64}()
            swc = Gapr.load_swc(path)
            lrroot = NmtDat.isleft(swc[1].pos)
            for l in swc.nodes
                pl = l.parent
                pl === nothing && continue
                pl.typ != 2 && continue
                d = norm(pl.pos .- l.pos)
                ic = NmtDat.isleft(l.pos) == lrroot ? :I : :C
                idx = NmtDat.pos2idx(l.pos, nmtshare.res)
                @assert checkbounds(Bool, nmtshare.sarm, idx..., 1)
                for tag in (:sarm, :charm)
                    img = nmtshare[tag]
                    ba = unique((view(img, idx..., :)...,))
                    ba[1] == 0 && continue
                    key = (rownum, tag, ic, ba...)
                    accumtmp[key] = get(accumtmp, key, 0.0) + d
                end
            end
            accumtmp
        end
    finally
        rmprocs(procs)
    end
    accum = Dict{Tuple,Float64}()
    for accumtmp in accumtmps
        for (key, val) in pairs(accumtmp)
            @assert length(key) >= 4
            while length(key) >= 3
                accum[key] = get(accum, key, 0.0) + val
                key = key[1:end-1]
            end
        end
    end
    tag2tbl = map((:sarm, :charm)) do tag
        tbl = nmt[Symbol(tag, :_key)]
        tbl = map(eachrow(tbl)) do r
            r[1] => r[2]
        end |> Dict
        tag => tbl
    end |> NamedTuple
    t2name(t) = begin
        tbl = tag2tbl[t[1]]
        Symbol(:nmt, t[2], "|", t[1], join(("|$(tbl[v])" for v in t[3:end])))
    end
    targets = sort(unique(map(p -> (p[2:end]...,), collect(keys(accum)))))
    for t in targets
        insertcols!(df, t2name(t) => 0.0)
    end
    for r in eachrow(df), t in targets
        r[t2name(t)] = get(accum, (rownumber(r), t...), 0.0)
    end
    df = df[:, pushfirst!(filter(Data.is_projcol, names(df)), "path")]
    JLD2.jldsave(out.path; df)
end

(out::list_mus_neurons_proj())(list::list_mus_neurons()) = begin
    df = JLD2.load(list.path, "df")
    allen = Allen.load_images()
    hiers = Allen.get_hier()
    procs = addprocs(7)
    accumtmps = try
        @everywhere procs @eval using LinearAlgebra, Gapr, ProgressMeter, StaticArrays, ColorTypes
        @everywhere procs @eval using zz0atlas: Allen, NmtDat
        @everywhere procs begin
            allenshare = $allen
        end
        @showprogress pmap(WorkerPool(procs), enumerate(df.path)) do (rownum, path)
            accumtmp = Dict{Tuple,Float64}()
            swc = Gapr.load_swc(path)
            lrroot = Allen.isleft(swc[1].pos)
            for l in swc
                pl = l.parrow
                pl === nothing && continue
                pl.typ != 2 && continue
                d = norm(pl.pos .- l.pos)
                ic = Allen.isleft(l.pos) == lrroot ? :I : :C
                idx = Allen.pos2idx(l.pos, allenshare.res)
                @assert checkbounds(Bool, allenshare.seg, idx...)
                ba = allenshare.seg[idx...]
                ba == 0 && continue
                key = (rownum, ic, ba)
                accumtmp[key] = get(accumtmp, key, 0.0) + d
            end
            accumtmp
        end
    finally
        rmprocs(procs)
    end
    accum = Dict{Tuple,Float64}()
    for accumtmp in accumtmps
        for (key, val) in pairs(accumtmp)
            @assert length(key) == 3
            k = key[end]
            while k != 0
                key = (key[1:end-1]..., k)
                accum[key] = get(accum, key, 0.0) + val
                k = hiers[k].par
            end
        end
    end
    t2name(t) = begin
        hh = ()
        k = t[end]
        while k != 0
            hhh = hiers[k]
            hh = (hhh.abbr, hh...)
            k = hhh.par
        end
        Symbol(:allen, t[1], join(("|$h" for h in hh)))
    end
    targets = sort(unique(map(p -> (p[2:end]...,), collect(keys(accum)))))
    for t in targets
        insertcols!(df, t2name(t) => 0.0)
    end
    for r in eachrow(df), t in targets
        r[t2name(t)] = get(accum, (rownumber(r), t...), 0.0)
    end
    df = df[:, Cols(:path, Data.is_projcol)]
    JLD2.jldsave(out.path; df)
end

function group_into_inj_sites(df)
    df = deepcopy(df)
    sampchan_of(r) = string(r.sample, "/", r.channel)
    injinfos = [
        (221059, 1) => (
            "L-MT-2/3-5/6 L-V4-2/3-5/6" => (18868, 17645, 34086),
        )
        (221747, 1) => (
            "L-46d-5/6" => (23143, 66171, 37293),
            "L-9m-5/6" => (28403, 63631, 39919),
            "L-V4-2/3" => (7255, 24089, 29346),
            "R-MT-5/6" => (52102, 20802, 28681),
            "R-46d-5/6" => (38439, 65211, 37901),
        )
        (231623, 1) => (
            "R-9d-2/3-5/6" => (38900, 65936, 39285),
            "R-V1-2/3" => (45114, 11449, 35260),
        )
        (231623, 2) => (
            "L-V1-2/3" => (14110, 12469, 31466),
            "L-10-2/3-5/6" => (29293, 71085, 35673),
        )
        (232717, 1) => (
            "R-46d-2/3" => (42789, 66473, 36318),
            "R-8B-2/3" => (37127, 60197, 41134),
            "R-V4-5/6" => (45714, 17797, 38119),
        )
        (232717, 2) => (
            "L-V2-5/6" => (27920, 15207, 41770),
            "L-8B-2/3" => (25625, 57539, 42788),
            "L-46d-2/3" => (23399, 65932, 38798),
        )
        (233001, 1) => (
            "R-TEa-2/3" => (57129, 31648, 20645),
            "R-8B-2/3" => (37173, 56347, 44021),
        )
        (233001, 2) => (
            "L-13m-2/3" => (23965, 56731, 25513),
            "L-V6-2/3" => (27152, 12273, 29732),
        )
        (234162, 1) => (
            "L-44-2/3-5/6" => (14350, 52358, 32091),
            "L-8Bm-2/3-5/6" => (31754, 58381, 41787),
            "L-TEm-2/3-5/6" => (9695, 41695, 13941),
        )
        (234162, 2) => (
            "R-44-2/3-5/6" => (48649, 51797, 30015),
            "R-V1-5/6" => (45292, 8314, 34249),
        )
        (234197, 1) => (
            "L-45-2/3-5/6" => (14104, 55062, 32410),
            "L-10-2/3-5/6" => (29019, 69853, 36652),
        )
        (240025, 1) => (
            "L-46v-5/6" => (18510, 67156, 34708),
            "L-8B-5/6" => (28494, 62446, 42038),
            "L-9m-2/3" => (27492, 64518, 40828),
        )
    ]
    procs = addprocs(5)
    dfs = try
        @everywhere procs @eval using Clustering, DataFrames, LinearAlgebra, Statistics, Gapr
        pmap(WorkerPool(procs), unique([sampchan_of(r) for r in eachrow(df)])) do sampchan
            hits = findall((x) -> join(first(x), '/') == sampchan, injinfos)
            @assert length(hits) == 1 "$sampchan no hits"
            injinfo = injinfos[hits[1]].second
            K = length(injinfo)
            df1 = filter(r -> sampchan_of(r) == sampchan, df)
            df2 = filter(r -> !r.qroot, df1)
            distmat = Matrix{Float64}(undef, size(df2, 1), size(df2, 1))
            for ij in CartesianIndices(distmat)
                distmat[ij] = norm(df2.soma_pos[ij[1]] .- df2.soma_pos[ij[2]])
            end
            nearest_idx(pos) = begin
                pos isa Pair || return rand(axes(df2, 1))
                pos = last(pos)
                findmin((x) -> norm(x .- pos), df2.soma_pos)[2]
            end
            assign = Clustering.kmedoids(distmat, K, init=collect(nearest_idx.(injinfo))).assignments
            injanterior = [median(reduce(hcat, df2[assign.==k, :soma_pos]), dims=2) for k in 1:K]
            injorder = sortperm(injanterior, rev=true, by=v -> v[2]) |> invperm
            injid(i) = string('a' + injorder[i] - 1, (injanterior[i][2] > 50000 ? "" : "X"))
            injname(i) = first(injinfo[i])
            insertcols!(df2, :inj => injid.(assign), :injname => injname.(assign))
            selcols = [:path, :inj, :injname]
            leftjoin!(df1, df2[:, selcols], on=:path)
            if size(df1, 1) != size(df2, 1)
                node2row = Pair[]
                for r in eachrow(df2)
                    swc = Gapr.nodedepth(Gapr.load_swc(r.path))
                    seg = reverse([n.pos for n in first(Gapr.eachsegment(swc, :depth))])
                    @assert seg[1] == r.soma_pos "$(r.path) $(seg[1])==$(r.soma_pos)"
                    for (idx, n) in enumerate(seg)
                        push!(node2row, n => rownumber(r))
                    end
                end
                tree = NearestNeighbors.KDTree(map(first, node2row))
                row2nodes = map(vcat, NearestNeighbors.inrange(tree, df1.soma_pos, 3000),
                    NearestNeighbors.knn(tree, df1.soma_pos, 100)[1])
                for r in eachrow(df1)
                    !r.qroot && continue
                    votes = Dict{Any,Int}()
                    for idx in row2nodes[rownumber(r)]
                        r2 = node2row[idx][2]
                        key = (df2[r2, :inj], df2[r2, :injname])
                        votes[key] = get(votes, key, 0) + 1
                    end
                    r2 = findmax(last, votes)[2]
                    r.inj = r2[1]
                    r.injname = r2[2]
                end
            end
            df1[:, selcols]
        end
    finally
        rmprocs(procs)
    end
    leftjoin!(df, reduce(vcat, dfs), on=:path) |> dropmissing!
    for r in eachrow(df)
        r.inj = string(sample_abbrev(r.sample, r.channel), r.inj)
        if !endswith("X")(r.inj)
            r.inj = r.inj * (NmtDat.isleft(r.soma_pos) ? 'L' : 'R')
        end
    end
    let injs = unique(df.inj)
        for inj in injs
            df2 = filter(r -> r.inj == inj, df)
            @assert size(df2, 1) != 0
            somas = filter(r -> !r.qroot, df2)
            cent = median(reduce(hcat, somas[:, :soma_pos]), dims=2)
            dists = map(v -> norm(v .- cent), df2.soma_pos)
            insertcols!(df2, :dist => dists)
            for r in eachrow(df2)
                if r.dist > 3500
                    raise_alarm("far-from-inj-center") do io
                        println(io, "DIST=$(r.dist) $(r.path)")
                    end
                end
            end
        end
    end
    df[:, Cols(:path, :inj, :injname)]
end

(out::list_neurons_detail())(
    list::list_neurons(),
    typ::list_neurons_itptct(),
    uvw::list_neurons_uvw(),
    hclust::list_neurons_hclust(),
) = begin
    typ = JLD2.load(typ.path, "df")
    uvw = JLD2.load(uvw.path, "df")
    cls = JLD2.load(hclust.path, "df")
    df = JLD2.load(list.path, "df")
    leftjoin!(df, typ, on=intersect(names(df), names(typ)))
    leftjoin!(df, uvw, on=intersect(names(df), names(uvw)))
    leftjoin!(df, cls, on=intersect(names(df), names(cls)))
    dropmissing!(df)

    insertcols!(df, :root_pos => df.soma_pos)
    sample_soma_pos_for_qroot!(df)
    nmt = NmtDat.load_images()
    getter = NmtDat.getter(nmt)
    insertcols!(df, :somahier => getter(reduce(hcat, df.soma_pos))[2])

    sort!(df, [:sample, :channel, :inj, :qroot, :path])
    let excel = "monkeytemp/neurons.xlsx"
        XLSX.openxlsx(excel, mode="w") do xf
            sheet = xf[1]
            XLSX.rename!(sheet, "neurons")
            for (j, col) in enumerate(names(df))
                dat = df[:, col]
                if col == "path"
                    d = dirname(excel)
                    dat = map(v -> relpath(v, d), dat)
                end
                if eltype(dat) <: AbstractVector{<:AbstractFloat}
                    dat = map(v -> round.(Float64.(v), digits=2), dat)
                end
                if eltype(dat) <: AbstractFloat
                    dat = map(v -> round(Float64(v), digits=2), dat)
                end
                if eltype(dat) <: Union{AbstractVector{<:AbstractFloat},Symbol}
                    dat = map(v -> string(v), dat)
                end
                sheet[:, j] = [col, dat...]
            end
        end
    end
    filter!(r -> !endswith("X")(String(r.inj)), df)
    JLD2.jldsave(out.path; df)
end



function show_brainarea_sarm(nmt, y, bas...)
    obj1 = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.mid_surface.rsl.gii.obj")
    display(GLMakie.wireframe(obj1, color=(:orange, 0.3), transparency=true, inspectable=false))
    id2name = Dict()
    map(bas) do ba
        idx = findall(r -> r[2] == ba, eachrow(nmt.sarm_key))
        row = nmt.sarm_key[idx, :]
        @info "row: $row"
        id2name[row[1]] = row[2]
    end
    points = Pair{SVector{3,Float32},String}[]
    for idx in CartesianIndices(nmt.sarm)
        name = get(id2name, nmt.sarm[idx], nothing)
        if name !== nothing
            push!(points, NmtDat.idx2pos(SVector(Tuple(idx)[1:3]...), nmt.res) => name)
        end
    end
    #filter!(NmtDat.isleft ∘ first, points)
    labels = CategoricalArrays.categorical(map(last, points))
    GLMakie.scatter!(map(first, points), color=labels.refs, colormap=:tab10, markersize=1, transparency=true)
    if y !== nothing
        @error "$y"
        injs = CategoricalArrays.categorical(string.(y.inj)).refs
        GLMakie.scatter!(y.soma_pos, color=injs, markersize=30, marker=[v ? :cross : :circle for v in y.qroot], label=y.path, colormap=:tab10, transparency=true)
    end
    GLMakie.DataInspector()
end

function show_brainarea(nmt, y, bas...)
    obj1 = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.mid_surface.rsl.gii.obj")
    display(GLMakie.wireframe(obj1, color=(:orange, 0.3), transparency=true, inspectable=false))
    id2name = Dict()
    map(bas) do ba
        idx = findall(r -> r[2] == ba, eachrow(nmt.charm_key))
        row = nmt.charm_key[idx, :]
        @info "row: $row"
        id2name[row[1]] = row[2]
    end
    points = Pair{SVector{3,Float32},String}[]
    for idx in CartesianIndices(nmt.charm)
        name = get(id2name, nmt.charm[idx], nothing)
        if name !== nothing
            push!(points, NmtDat.idx2pos(SVector(Tuple(idx)[1:3]...), nmt.res) => name)
        end
    end
    #filter!(NmtDat.isleft ∘ first, points)
    labels = CategoricalArrays.categorical(map(last, points))
    GLMakie.scatter!(map(first, points), color=labels.refs, colormap=:tab10, markersize=1, transparency=true)
    if y !== nothing
        @error "$y"
        injs = CategoricalArrays.categorical(string.(y.inj)).refs
        GLMakie.scatter!(y.soma_pos, color=injs, markersize=30, marker=[v ? :cross : :circle for v in y.qroot], label=y.path, colormap=:tab10, transparency=true)
    end
    GLMakie.DataInspector()
end

function show_swcs(path, pat)
    obj1 = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.gray_surface.rsl.gii.obj")
    display(GLMakie.wireframe(obj1, color=(:gray, 0.1), transparency=true, inspectable=false))
    files = readdir(path, join=false)
    for f in files
        if match(pat, f) !== nothing
            swc = Gapr.load_swc(joinpath(path, f))
            seg = reduce(vcat, [push!([n.pos for n in seg], Vec3f(NaN)) for seg in Gapr.eachsegment(swc)])
            Makie.lines!(seg)
        end
    end

    GLMakie.DataInspector()
end

(out::Make.@x("interm/monkey-cortex-graph.dat"))(
    flatmapdat::flatmap_data(:lefths),
) = begin
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    nmt = NmtDat.load_images()
    samps = []
    sample_mesh(objflat, objflat.position, objphy.position; perface=16) do vec
        p1 = SVector(vec[1:2]...)
        p2 = SVector(vec[3:end]...)
        ba = SVector(view(nmt.charm, NmtDat.pos2idx(p2, nmt.res)..., :)...)
        if ba[1] != 0
            push!(samps, (p1, p2, ba))
        end
    end
    bas = sort(unique(last.(samps)))
    ba2img = Dict()
    @showprogress "sort" for ba in bas
        sampsx = filter(==(ba) ∘ last, samps)
        ba = unique(ba)
        img = get!(() -> [], ba2img, ba)
        for (a, b, _) in sampsx
            push!(img, (a, b))
        end
    end
    ba2img = collect(ba2img)
    medoids = @showprogress "medoids" map(ba2img) do (ba, img)
        if length(img) > 600
            img = shuffle(img)[1:600]
        end
        dm = [norm(img[i][1] - img[j][1]) for i in axes(img, 1), j in axes(img, 1)]
        k = Clustering.kmedoids(dm, 1).medoids[1]
        img[k]
    end
    hiers = NmtDat.getter(nmt)(reduce(hcat, last.(medoids)))[2]

    fig = Makie.Figure()
    display(fig)
    abbr(h) = replace(h, r".*\|" => "")
    ax, _ = Makie.text(fig[1, 1], first.(medoids), text=abbr.(hiers))
    Makie.scatter!(ax, first.(medoids),
        inspector_label=(plt, idx, pos) -> hiers[idx])
    Makie.DataInspector()
    for (ba, img) in ba2img
        img = reduce(hcat, first.(img))
        a, b = minimum(img, dims=2)
        c, d = maximum(img, dims=2)
        canvas = zeros(Float32, ceil(Int, (c - a) * 1000) + 20, ceil(Int, (d - b) * 1000) + 20)
        for p in eachcol(img)
            idx = ceil.(Int, (p - SVector(a, b)) .* 1000) .+ 10
            canvas[idx...] = 1
        end
        Makie.contour!(ax, range(a - 10 / 1000, c + 10 / 1000, length=size(canvas, 1)), range(b - 10 / 1000, d + 10 / 1000, length=size(canvas, 2)), canvas, levels=2)
        Main.xxxcanvas = canvas
    end
end

function generate_hiers()
    allen = Allen.load_images()
    nmt = NmtDat.load_images()
    brains = (;
        mus=(g=Allen.getter(Allen.get_hier(), allen), aa=() -> begin
            a2p = Dict()
            for idx in CartesianIndices(allen.seg)
                a = allen.seg[idx]
                if a != 0
                    get!(() -> Allen.idx2pos(SVector(Tuple(idx)), allen.res), a2p, a)
                end
            end
            a2p
        end),
        mac=(g=NmtDat.getter(nmt), aa=() -> begin
            a2p = Dict()
            for t in (:sarm, :charm)
                img = nmt[t]
                @showprogress for idx in CartesianIndices(view(img, :, :, :, 1))
                    if img[idx, 1] != 0
                        a = (t, view(img, Tuple(idx)..., :)...)
                        get!(() -> NmtDat.idx2pos(SVector(Tuple(idx)), nmt.res), a2p, a)
                    end
                end
            end
            a2p
        end
        ),
    )
    for (i, t) in enumerate((:mus, :mac))
        @info "do $t"
        h = brains[t]
        a2p = collect(h.aa())
        x = h.g(reduce(hcat, last.(a2p)))[2]
        open("/tmp/hier$i.txt", "w") do io
            for l in sort(unique(x))
                println(io, l)
            end
        end
    end
end


function sample_soma_pos_for_qroot!(df)
    nmt = NmtDat.load_images()
    perinjs = map(unique(df.inj)) do inj
        dfx = filter(r -> r.inj == inj && !r.qroot, df)
        cent = median(reduce(hcat, dfx.soma_pos), dims=2)
        dists = map(eachrow(dfx)) do r
            norm(r.soma_pos .- cent)
        end
        mdist = quantile(dists, 0.5) * 0.5
        (inj => (; cent, mdist))
    end |> Dict
    Random.seed!(1)
    for r in eachrow(df)
        if r.qroot
            perinj = perinjs[r.inj]
            lr = NmtDat.isleft(r.soma_pos)
            while true
                r.soma_pos = perinj.cent .+ randn(3) .* perinj.mdist
                idx = NmtDat.pos2idx(r.soma_pos, nmt.res)
                checkbounds(Bool, nmt.charm, idx..., 1) && nmt.charm[idx..., 1] > 0 && lr == NmtDat.isleft(r.soma_pos) && break
            end
            @assert lr == NmtDat.isleft(r.soma_pos)
        end
    end
end

(out::Make.@x("output/cover-graphics.pdf"))(
    list::list_neurons_detail(),
    rawinfo::list_nodes_rawinfo(),
) = begin
    df = JLD2.load(list.path, "df")
    rawinfo = nodes_info_factory(rawinfo.path)
    obj = load("interm/mesh-sample-fmost234197.ch1-lr.obj")
    objfh = load("interm/mesh-sample-fmost234197.ch1.fh-lr.obj")
    objss = load("interm/mesh-sample-fmost234197.ch1.ss-lr.obj")
    objhs = load("interm/mesh-sample-fmost234197.ch1.hs-lr.obj")

    cent = (26956.588, 40457.43, 34111.242)
    cams = [
        (; fov=45.0, cent, deye=(-0.3, -0.39, -0.81) .* 90000.0, up=(.-((0.888, 0.011, 0.459))));
        (; fov=45.0, cent, deye=(0.0, -1.0, 0.0) .* 90000.0, up=(-0.92, -0.00, -0.39));
        (; fov=45.0, cent, deye=(-1.0, 0.0, -0.0) .* 90000.0, up=(-0.00, 1.0, 0.00))
        #(fov=45.0, cent=(26956.59, 40457.43, 34111.24), deye=(-0.3542806658463125, -0.6241881696132779, -0.6963291884750482) .* 122196.08399059522, up=(-0.91, 0.41, 0.09))
    ]
    typ2c = zz2fig.typ2color_func(df)
    df = filter(r -> r.sample == "234197", df)
    info = rawinfo("234197", 1)
    xformby(n) = info[n.id].rawxyz
    fixswc!(s) = (foreach(n -> n.pos = xformby(n) ./ 1000, s.nodes); s)
    fixobj(obj) = normal_mesh(obj.position ./ 1000, faces(obj))

    glass(kw...) = (; type=:Uber, reflection_color=Vec4f(0.9), reflection_weight=Vec4f(1.0),
        reflection_roughness=Vec4f(0.3), reflection_anisotropy=Vec4f(0.3),
        reflection_anisotropy_rotation=Vec4f(0.3),
        reflection_ior=Vec4f(1.5), refraction_color=Vec4f(0.9), refraction_weight=Vec4f(0.0),
        refraction_roughness=Vec4f(0.0), refraction_ior=Vec4f(1.5),
        refraction_thin_surface=false, refraction_absorption_color=Vec4f(0.6, 0.8, 0.6, 0.0),
        refraction_absorption_distance=Vec4f(150), refraction_caustics=true,
        coating_weight=Vec4f(0.1), sheen_weight=Vec4f(0), emission_weight=Vec4f(0),
        reflection_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
        transparency=Vec4f(0), kw...)
    ensure_bbox(ax) = begin
        obj2 = fixobj(obj)
        objfh2 = fixobj(objfh)
        objss2 = fixobj(objss)
        objhs2 = fixobj(objhs)
        r = reduce(union, [Rect3d(o) for o in (obj2, objfh2, objss2, objhs2)])
        Makie.meshscatter!(ax, unique(GeometryBasics.coordinates(r)), markersize=0, color=:black,)
    end
    testing = false
    maxrows = testing ? 30 : 3000
    iterations = testing ? 30 : 300
    scenesize = testing ? 4096 : 4096 * 2
    conts = [
        (ax) -> begin
            Random.seed!(1)
            for r in eachrow(shuffle(df))
                s = Gapr.load_swc(r.path) |> fixswc!
                Gapr.swcplot!(ax, s, color=typ2c(r.typ) .* 0.9, linewidth=9, rpr=true,) #= material=(; type=:Uber) =#
                if !r.qroot
                    Makie.meshscatter!(s[1].pos, markersize=0.09, color=typ2c(r.typ) .* 3)
                end
                rownumber(r) >= maxrows && break
            end
            obj2 = fixobj(obj)
            objfh2 = fixobj(objfh)
            objss2 = fixobj(objss)
            objhs2 = fixobj(objhs)

            Makie.mesh!(ax, objhs2, color=:ivory)
            Makie.mesh!(ax, obj2, color=:gray24, material=glass())
            ensure_bbox(ax)
            return
            Makie.mesh!(ax, obj2, color=:ivory, material=(; color=:ivory,
                reflection_color=Vec4f(0.9), reflection_weight=Vec4f(1.0),
                reflection_roughness=Vec4f(0.0), reflection_anisotropy=Vec4f(0.0),
                reflection_anisotropy_rotation=Vec4f(0.0),
                refraction_weight=Vec4f(1.0), refraction_thin_surface=false,
                sheen_weight=Vec4f(0), emission_weight=Vec4f(0),
                transparency=Vec4f(0.0),
                refraction_color=Vec4f(0.9f0),
                refraction_roughness=Vec4f(0.00),
                refraction_caustics=true,
                refraction_ior=1.09f0,
                reflection_ior=1.09f0,
                reflection_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
                refraction_absorption_color=Vec4f(0.9, 0.9, 0.9, 0),
                refraction_absorption_distance=Vec4f(20.0f0),
                coating_color=Vec4f(1.0f0),
                coating_weight=0.1f0,
                coating_roughness=0.01f0,
                coating_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_METALNESS,
                coating_metalness=1.0f0,
                coating_transmission_color=Vec4f(1.0f0),
                coating_thickness=0.1f0,
            ))
            # Makie.mesh!(ax, obj2, color=(:ivory, 0.1), transparency=true)
            # zz2fig.draw_neurons(df, ax; rowcolor=r -> typ2c(r.typ), xformby, linewidth=3)
        end,
        (ax) -> begin
            Random.seed!(1)
            for r in eachrow(shuffle(df))
                s = Gapr.load_swc(r.path) |> fixswc!
                Gapr.swcplot!(ax, s, color=typ2c(r.typ) .* 0.9, linewidth=9, rpr=true,) #= material=(; type=:Uber) =#
                if !r.qroot
                    Makie.meshscatter!(s[1].pos, markersize=0.09, color=typ2c(r.typ) .* 3)
                end
                rownumber(r) >= maxrows && break
            end
            obj2 = fixobj(obj)
            objfh2 = fixobj(objfh)
            objss2 = fixobj(objss)
            objhs2 = fixobj(objhs)

            Makie.mesh!(ax, objhs2, color=:ivory)
            ensure_bbox(ax)
            return
            Makie.mesh!(ax, obj2, color=:ivory, material=(; color=:ivory,
                reflection_color=Vec4f(0.9), reflection_weight=Vec4f(1.0),
                reflection_roughness=Vec4f(0.0), reflection_anisotropy=Vec4f(0.0),
                reflection_anisotropy_rotation=Vec4f(0.0),
                refraction_weight=Vec4f(1.0), refraction_thin_surface=false,
                sheen_weight=Vec4f(0), emission_weight=Vec4f(0),
                transparency=Vec4f(0.0),
                refraction_color=Vec4f(0.9f0),
                refraction_roughness=Vec4f(0.00),
                refraction_caustics=true,
                refraction_ior=1.09f0,
                reflection_ior=1.09f0,
                reflection_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
                refraction_absorption_color=Vec4f(0.9, 0.9, 0.9, 0),
                refraction_absorption_distance=Vec4f(20.0f0),
                coating_color=Vec4f(1.0f0),
                coating_weight=0.1f0,
                coating_roughness=0.01f0,
                coating_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_METALNESS,
                coating_metalness=1.0f0,
                coating_transmission_color=Vec4f(1.0f0),
                coating_thickness=0.1f0,
            ))
            # Makie.mesh!(ax, obj, color=(:ivory, 0.1), transparency=true)
            # zz2fig.draw_neurons(df, ax; rowcolor=r -> typ2c(r.typ), xformby, linewidth=3)
        end,
        (ax) -> begin
            Random.seed!(1)
            for r in eachrow(shuffle(df))
                s = Gapr.load_swc(r.path) |> fixswc!
                Gapr.swcplot!(ax, s, color=:white, linewidth=9, rpr=true,) #= material=(; type=:Uber) =#
                if !r.qroot
                    Makie.meshscatter!(s[1].pos, markersize=0.09, color=:white)
                end
                rownumber(r) >= maxrows && break
            end
            obj2 = fixobj(obj)
            objfh2 = fixobj(objfh)
            objss2 = fixobj(objss)
            objhs2 = fixobj(objhs)

            ensure_bbox(ax)
        end,
        (ax) -> begin
            Random.seed!(1)
            for r in eachrow(shuffle(df))
                s = Gapr.load_swc(r.path) |> fixswc!
                Gapr.swcplot!(ax, s, color=typ2c(r.typ) .* 0.9, linewidth=9, rpr=true,)
                if !r.qroot
                    Makie.meshscatter!(s[1].pos, markersize=0.09, color=typ2c(r.typ) .* 3)
                end
                rownumber(r) >= maxrows && break
            end
            obj2 = fixobj(obj)
            objfh2 = fixobj(objfh)
            objss2 = fixobj(objss)
            objhs2 = fixobj(objhs)

            Makie.mesh!(ax, objfh2, color=:ivory)
            ensure_bbox(ax)
        end,
        (ax) -> begin
            Makie.mesh!(ax, obj, color=:ivory)
        end,
        (ax) -> begin
            zz2fig.draw_neurons(df, ax; rowcolor=r -> typ2c(r.typ), xformby, linewidth=3)
        end,
        (ax) -> begin
            Makie.mesh!(ax, obj, color=(:ivory, 0.1), transparency=true)
            zz2fig.draw_neurons(df, ax; rowcolor=r -> typ2c(r.typ), xformby, linewidth=3)
        end,
    ]
    mktempdir() do path
        for (k, cam) in enumerate(cams)
            for (j, cont) in enumerate(conts)
                Makie.with_theme(Makie.theme_dark()) do
                    f = Makie.Figure(size=(scenesize, scenesize), backgroundcolor=(:black, 0.0))
                    ax = Makie.LScene(f[1, 1], show_axis=false,
                        scenekw=(; backgroundcolor=Makie.to_color(:skyblue4), light_color=Makie.to_color(:white), light_direction=normalize(Vec3f(-0.5, -0.2, 0.7)), camera_relative_light=false),
                    )
                    cont(ax)
                    Makie.cameracontrols(ax.scene).settings.fixed_axis = false
                    zz2fig.inspect_cameras(f)
                    zz2fig.setup_camera(ax => cam)
                    Makie.update_cam!(ax.scene)
                    Makie.update_state_before_display!(f)
                    ##p = zz2fig.alpha_colorbuffer(ax.scene)
                    ## p = Makie.colorbuffer(ax.scene)
                    save(joinpath(path, "cam$k-$j.png"), ax.scene; backend=RPRMakie, update=true, iterations, plugin=RPRMakie.Northstar)
                end
                j >= 4 && break
            end
            a = load(joinpath(path, "cam$k-1.png"))
            b = load(joinpath(path, "cam$k-2.png"))
            c = load(joinpath(path, "cam$k-3.png"))
            blend(a, b, c) = begin
                c0 = Makie.to_color(:skyblue4)
                c1 = Makie.to_color(:black)
                map(a, b, c) do a, b, c
                    am = mapreducec(identity, max, 0, a)
                    bm = mapreducec(identity, max, 0, b)
                    w = clamp(min((Flux.sigmoid((norm((c .- cx)) - 0.09) * 90) for cx in (c0, c1))...) .* 0.8 + 0.16, 0, 1)
                    w .* b .+ (1 - w) .* a
                    #mapc((a, b) -> clamp(a * 0.6 + b * 0.9, 0, 1), a, b)
                end
            end
            save(joinpath(path, "cam$k-9.png"), blend(a, b, c))
            break
        end
        in = map(readdir(path, join=true)) do f1
            f2 = "$f1.pdf"
            run(`magick $f1 $f2`)
            f2
        end
        run(`pdfunite $in $(out.path)`)
    end
end

fig1_collect_data(tag) = Make.@x("interm/fig1-$tag.txt")
(out::fig1_collect_data(Tag))() where {Tag} = begin
    dataset_to_show = "234197.ch1"
    datasetid = replace(dataset_to_show, r"\..*" => "")
    cmd = (;
        tiffinfo=`ssh -C dell@10.10.48.111 "find '/mnt/samba$datasetid/' -type f -printf '%s %T@ %p\n' && echo 'END'"`,
        webminfo=`ssh -C admin@10.10.41.224 "find '/run/media/admin/GaprDataStore/goulf-conv-setup/conf/DATA/fmost$dataset_to_show/' -type f -printf '%s %T@ %p\n' && echo 'END'"`,
        userstats=`ssh -C gapr_srv@yanlab.org.cn "curl https://172.16.164.86:30003/api/stats/fmost$dataset_to_show -k | zgrep '^Commit' && echo END"`,
    )[Tag]
    @info "run $cmd"
    Main.lines = lines = open(cmd) do io
        readlines(io)
    end
    @assert lines[end] == "END"
    open(out.path, "w") do io
        for l in lines
            println(io, l)
        end
    end
end

function show_case_info()
    resolution = (0.65, 0.65, 3)
    size = (83988, 112005, 23000)
    (; resolution, size,
        dsres=(65, 65, 60),
        cubesize=(400, 400, 80),
        center=(size .* resolution) ./ 2,
        mix="/data/dat/dsimgs/fmost234197.ch1@ch00-mix.nrrd",
        avg="/data/dat/dsimgs/fmost234197.ch1@ch00-avg.nrrd",
        mip="/data/dat/dsimgs/fmost234197.ch1@ch00-mip.nrrd",
    )
end




function add_subplot_lbl(grid, r, c, lbl)
    Makie.Label(grid[r, c, Makie.TopLeft()], lbl, font=:bold, fontsize=7, halign=:right)
end

(out::Make.@x("interm/sample-slices"))() = begin
    dsimg = "/data/dat/dsimgs/fmost234197.ch1@ch00-mip.nrrd"
    dsimg = load(dsimg)
    zrange = axes(dsimg, 3)
    nsel = 13
    rng = round.(Int, range(length(zrange) / 50, length(zrange) * 46 / 50, length=nsel))
    sel = zrange[rng]
    ensure_directory(out.path) do outdir, outtmp
        for z in sel
            save(joinpath(outtmp, string(z * 60, pad=5) * ".tif"), view(dsimg, :, :, z))
        end
    end
end

(out::Make.@x("monkeytemp/soma-G97-1bL.png"))(
    list::list_neurons_detail(),
    rawinfo::list_nodes_rawinfo(),
) = begin
    df = JLD2.load(list.path, "df")
    rawinfo = nodes_info_factory(rawinfo.path)
    rowcolor(r) = :red
    filter!(r -> r.inj == "G97-1bL" && !r.qroot, df)
    img = draw_hires(df, rawinfo; rowcolor)
    save(out.path, img)
end

function draw_hires_sc(cam, sample, channel)
    r = max(cam.zrange, cam.xyrange)
    samppts = vec(reduce((x, y) -> [(a..., b) for a in x, b in y], [(-r+x):50:(r+x) for x in cam.center]))
    df = map(samppts) do p
        (; root=Point3d(p), sample, channel)
    end |> DataFrame
    eyedir = cam.eye .- cam.center
    eyedir = eyedir ./ norm(eyedir)
    @info "len $(size(df, 1))"
    filter!(df) do r
        dp = r.root .- cam.center
        dz = dot(dp, eyedir)
        abs(dz) > cam.zrange && return false
        dxy = norm(dp .- dz .* eyedir)
        dxy > cam.xyrange && return false
        true
    end
    @info "len $(size(df, 1))"
    d = map(eachrow(df)) do r
        r.root => (; rawxyz=Point3d(r.root))
    end
    d = Dict(d)
    rawinfo(_, _) = begin
        d
    end
    rowcolor(r) = begin
        Makie.RGBf(0, 0, 0)
    end

    img = zz1vis.draw_hires(df, rawinfo, cam; rowcolor)
    global xxximg = deepcopy(img)
    img
end
function draw_hires_sc_fiber(cam, sample, channel, dfx, rawinfo, rowcolor)
    dfx = shuffle(dfx[1:1, :])[1:1, :]
    zz1vis.draw_hires_fibers(dfx, rawinfo, cam; rowcolor)
end

(out::Make.@x("monkeytemp/sc-layers-D17.tiff"))(
# list::list_neurons_detail(),
# rawinfo::list_nodes_rawinfo(),
) = begin
    cam = (;
        center=(28160.2, 28967.2, 35243.1),
        eye=(28128, 28962.2, 34744.2),
        up=(24393.6, 29017.2, 35486),
        zrange=600 / 2,
        xyrange=4800,
    )
    img = draw_hires_sc(cam, "232717-1", 2)
    save(out.path, rotl90(img))
end
(out::Make.@x("monkeytemp/sc-layers-G97.tiff"))(
# list::list_neurons_detail(),
# rawinfo::list_nodes_rawinfo(),
) = begin
    cam = (;
        center=(28383.7, 38111, 41963.8),
        eye=(28355.1, 38109.9, 41615),
        up=(24297.9, 38017.2, 42299.5),
        zrange=600 / 2,
        xyrange=4800,
    )
    img = draw_hires_sc(cam, "234197", 1)
    save(out.path, rotl90(img))
end

(out::output_fig(1))(
    list::list_neurons_detail(),
    proj::list_neurons_proj(),
    rawinfo::list_nodes_rawinfo(),
    sampslices::Make.@x("interm/sample-slices"),
    nmr::Make.@x("interm/inj-popu-proj.dat"),
    flatmapdat::flatmap_data(:leftfrontal),
) = begin
    df = JLD2.load(list.path, "df")
    proj = JLD2.load(proj.path, "df")
    rawinfo = @time "load nodes" nodes_info_factory(rawinfo.path)
    obj = load("interm/mesh-nmt-brain-lr.obj")
    objg97 = load("interm/mesh-sample-fmost234197.ch1-lr.obj")
    sampinfo = show_case_info()
    nmr = JLD2.load(nmr.path, "df")
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))

    caminj = (fov=9, cent=sampinfo.center, deye=normalize(Vec3f(-0.5f0, -0.7f0, -0.5f0)) .* 3.6f5, up=normalize(Vec3f(-1.0f0, 0.0f0, -0.2f0)), dcent=(0, 0, -1_500))

    inj_vert_colors() = begin
        dfx = filter(r -> r.sample == "234197" && r.channel == "1", df)
        injpos = map(sort(unique(dfx.inj))) do inj
            somapos = map(eachrow(filter(r -> r.inj == inj && !r.qroot, dfx))) do r
                rawinfo(r.sample, r.channel)[r.root].rawxyz
            end
            inj => SVector(median(reduce(hcat, somapos), dims=2)...)
        end
        @time "color vert" color_mesh_verts(objg97, Makie.to_color((:lightgray, 1.0)), (last.(injpos) .=> Makie.to_color(:green))...)
    end
    function get_objinfl(nmrdat)
        nmrdat.surf
    end
    nmrdat = load_nmr_data()
    objinfl = get_objinfl(nmrdat)

    highlights = zz1data.get_flatmap_highlights()

    with_figure(1, out.path, (; df, objflat, objphy, highlights, obj, objg97, objinfl, nmr, nmrdat, caminj, inj_vert_colors, sampslices, sampinfo, rawinfo)) do fig, env
        plot_fig_recon(fig, env; env...)
    end
end

function load_popu_proj(fn, df, dfproj)
    dfproj = leftjoin(df, dfproj, on=:path)
    filter!(r -> r.inj == "G97-1bL", dfproj)
    nmtvol = zz1data.cached_result(:vol) do
        nmt = NmtDat.load_images()
        NmtDat.volume(nmt)
    end

    fn = "monkeytemp/wyz-upload/elife-72534-fig3-data1-v1.xlsx"
    df = DataFrame(XLSX.readtable(fn, 1))
    rename!(df, 3 => :v1, 4 => :v2, 5 => :a) |> dropmissing!

    insertcols!(df, :proji => NaN, :projc => NaN, :proj => NaN, :vol => NaN)
    for r in eachrow(df)
        ppi = 0.0
        ppc = 0.0
        hiti = 0
        hitc = 0
        hh = nothing
        for n in filter(zz1data.is_projcol, names(dfproj))
            if replace(n, r"^(\w+I)\|.*\|" => "") == r.a
                hiti += 1
                ppi = sum(dfproj[:, n])
                hh = replace(n, r"^(\w+)[IC]\|" => s"\1|")
            end
            if replace(n, r"^(\w+C)\|.*\|" => "") == r.a
                hitc += 1
                ppc = sum(dfproj[:, n])
                hh = replace(n, r"^(\w+)[IC]\|" => s"\1|")
            end
        end
        if hiti <= 1 && hitc <= 1
            r.proji = ppi
            r.projc = ppc
            r.proj = ppi + ppc
            r.vol = nmtvol[hh]
        end
    end
    dropmissing!(df)
end

(out::output_fig(:S01))(
    tiffinfo::fig1_collect_data(:tiffinfo),
    webminfo::fig1_collect_data(:webminfo),
    userstats::fig1_collect_data(:userstats),
    list::list_neurons_detail(),
    proj::list_neurons_proj(),
    nmr::Make.@x("interm/inj-popu-proj.dat"),
) = begin
    stats = NamedTuple[]
    for l in eachline(tiffinfo.path)
        l == "END" && break
        m = match(r"^(\d+) ([\d.]+) /mnt/samba234197/234197-CH1[^/]*/234197_0*(\d+)_CH1(-tiled)?.tif$", l)
        m === nothing && continue
        @assert m.captures[4] == "-tiled"
        push!(stats, (; type=:tiff,
            size=parse(Int, m.captures[1]),
            time=unix2datetime(parse(Float64, m.captures[2])),
            slice=parse(Int, m.captures[3]),
        ))
    end
    for l in eachline(webminfo.path)
        l == "END" && break
        m = match(r"^(\d+) ([\d.]+) /run/media/admin/GaprDataStore/goulf-conv-setup/conf/DATA/fmost234197.ch1/ch00webm/z0*(\d+)/y0*(\d+).x0*(\d+).webm$", l)
        m === nothing && continue
        push!(stats, (; type=:webm,
            size=parse(Int, m.captures[1]),
            time=unix2datetime(parse(Float64, m.captures[2])),
            zoff=parse(Int, m.captures[3]),
            yoff=parse(Int, m.captures[4]),
            xoff=parse(Int, m.captures[5]),
        ))
    end
    for l in eachline(userstats.path)
        l == "END" && break
        m = match(r"^Commit: id: (\d+); author: ([^;]+?)(\.fix|\.qc)?; time: ([^;]+)\+(\d+); type: (\w+): stats:.*$", l)
        m === nothing && continue
        @assert m.captures[5] == "0800"
        push!(stats, (; type=:edit,
            id=parse(Int, m.captures[1]),
            user=String(m.captures[2]),
            usertag=String(@something(m.captures[3], "")),
            time=parse(DateTime, m.captures[4]) - Hour(8),
            edit=String(m.captures[6]),
        ))
    end

    df = JLD2.load(list.path, "df")
    proj = JLD2.load(proj.path, "df")
    nmr = JLD2.load(nmr.path, "df")
    nmrdat = load_nmr_data()
    cmpdat = load_popu_proj(nothing, df, proj)
    leftjoin!(df, proj, on=:path)
    conndat = (; flne="flne_value", areamap="region_name")
    conndat = map(s -> DataFrame(XLSX.readtable("monkeytemp/wyz-upload/markov_flne_value_region_name.xlsx", s)), conndat)
    ensure_str(T, t, cols) = begin
        mapcols!(vv -> map(v -> (@assert !isnothing(v); @assert !ismissing(v); T(v)), vv), t; cols=cols)
    end
    ensure_str(string, conndat.flne, [:MONKEY, :TARGET, :SOURCE])
    ensure_str(Float64, conndat.flne, [:FLNe])
    ensure_str(string, conndat.areamap, [:region_name_markov, :region_name_nmt])

    with_figure(:S01, out.path, (; stats, df, cmpdat, nmrdat, nmr, conndat)) do fig, env
        plot_fig_compare2(fig, env; env...)
    end
end

(out::output_fig(:S02))(
    neurons::list_neurons_detail(),
    proj::list_neurons_proj(),
    hclust::list_neurons_hclust(),
    dist::list_neurons_dist(),
) = begin
    df = JLD2.load(neurons.path, "df")
    proj = JLD2.load(proj.path, "df")
    leftjoin!(df, proj, on=:path) |> dropmissing!
    hclust = loadhclust!(df, hclust.path)
    sort!(df, :hclustord)
    distm = load_dist_mat(df, dist.path)
    umap = @time "umap" UMAP.umap(distm, 2, metric=:precomputed)
    obj = load("interm/mesh-nmt-brain-lr.obj")

    with_figure(:S02, out.path, (; df, umap, obj)) do fig, env
        zz2fig.plot_fig_subtyp_supp(fig, env; env...)
    end
end

(out::output_fig(:SD01))(
    detail::list_neurons_detail(),
    proj::list_neurons_proj(),
) = begin
    nmt = NmtDat.load_images()
    df = JLD2.load(detail.path, "df")
    proj = JLD2.load(proj.path, "df")
    leftjoin!(df, proj, on=:path) |> dropmissing!
    obj = load("./interm/mesh-nmt-brain-lr.obj")

    with_figure(:SD01, out.path, (; df, obj, nmt)) do fig, env
        plot_all_injs(fig, env; env...)
    end
end

(out::output_fig(:SD02))(
    list::list_neurons_detail(),
    hclust::list_neurons_hclust(),
    project::list_neurons_proj(),
    dist::list_neurons_dist(),
) = begin
    df = JLD2.load(list.path, "df")
    projs = JLD2.load(project.path, "df")
    hclust = loadhclust!(df, hclust.path)
    sortas!(projs, df, :path, filter=true)
    distm = load_dist_mat(df, dist.path)
    umap = UMAP.umap(distm, 2, metric=:precomputed)
    obj = load("./interm/mesh-nmt-brain-lr.obj")
    with_figure(:SD02, out.path, (; df, umap, obj)) do fig, env
        plot_all_subtypes(fig, env; env...)
    end
end

function cluster_arbor_whist(arbors, df)
    cortarbors = leftjoin(arbors, df, on=:path) |> dropmissing!
    filter!(cortarbors) do r
        !(r.sample in ("234197", "232717")) && return r.cortexrat >= 0.7
        sum(r.fv_whist) / r.f_totlen >= 0.7
    end
    filter!(r -> r.sample in ("234197", "232717"), cortarbors)
    function depth_hist_dist(x)
        Random.seed!(0)
        # silent NaNs
        x = x .+ rand(size(x)...) * 0.01
        x = x ./ sum(x, dims=1)
        x = vcat(zeros(1, size(x, 2)), x, zeros(1, size(x, 2)))
        x0 = deepcopy(x)
        x[2:end, :] .+= 0.3 .* x0[1:end-1, :]
        x[1:end-1, :] .+= 0.3 .* x0[2:end, :]
        dpdist(x, y) = begin
            norm(x .- y)
        end
        N = size(x, 2)
        m = @showprogress [dpdist(view(x, :, i), view(x, :, j)) for i in 1:N, j in 1:N]
        Symmetric(m)
    end
    resample(h, N) = begin
        hh = zeros(N + 1)
        for (i, v) in pairs(h)
            ii = round(Int, (i - 1) / (length(h) - 1) * N + 1)
            hh[ii] += v
        end
        hh
    end
    distmat0, umap1, depths = cached_result(:depthhist0, cortarbors) do cortarbors
        depths = reduce(hcat, resample.(cortarbors.fv_whist, 10))
        distmat0 = depth_hist_dist(depths)
        global xxxdistmap = deepcopy(distmat0)
        umap0 = UMAP.umap(distmat0, 2, metric=:precomputed)
        distmat0, umap0, depths
    end

    distmat1 = cached_result(:depthhist0, distmat0) do distmat0
        ents = [StatsBase.entropy(c ./ (sum(c) + 0.1)) for c in eachcol(depths)]
        distmat1 = [norm(c .- c2) for c in ents, c2 in ents]
        global xxxdistmap1 = deepcopy(distmat1)
        distmat1
    end

    insertcols!(cortarbors, :arborareatag => 0)
    for r in eachrow(cortarbors)
        for (i, t) in enumerate([:LIP, :PCC, :STSd, :TE,])
            if matchhier("$t")(r.arborhier) && r.ipsi
                r.arborareatag = i
                break
            end
        end
        if matchhier("Frontal")(r.arborhier) && r.ipsi
            r.arborareatag = 5
        end
        if matchhier("Frontal")(r.arborhier) && !r.ipsi
            r.arborareatag = 6
        end
    end
    w1, w2 = argmax((0.1:0.1:0.6) .=> (0:0.1:0)') do (w1, w2)
        hclust1 = Clustering.hclust(distmat0 + w1 .* distmat1, linkage=:ward, branchorder=:barjoseph)
        clusters1 = Clustering.cutree(hclust1, k=6)
        aaa = sort(unique(clusters1))
        bbb = sort(unique(cortarbors.arborareatag))
        p = [count(==(a => b), clusters1 .=> cortarbors.arborareatag) for a in aaa, b in bbb] ./ length(clusters1)
        pa = [count(==(a), clusters1) for a in aaa] ./ length(clusters1)
        pb = [count(==(b), cortarbors.arborareatag) for b in bbb] ./ length(clusters1)
        @assert sum(p) ≈ 1 "sum(p)=$(sum(p))"
        @assert sum(pa) ≈ 1
        @assert sum(pb) ≈ 1
        q = [a * b for a in pa, b in pb]
        mi = StatsBase.kldivergence(p, q)
        @info "$w1 $w2 $mi"
        mi
    end
    @warn "max: $w1 $w2"
    # 0.4

    hclust1 = Clustering.hclust(distmat0 + w1 .* distmat1, linkage=:ward, branchorder=:barjoseph)
    clusters1 = Clustering.cutree(hclust1, k=6)
    reorder_cls!(clusters1, hclust1)
    insertcols!(cortarbors, :arborcls => clusters1)
    cortarbors, hclust1
end

import Base64
function optimizepdf(f::AbstractString, fo::AbstractString; q=75, ppi=400)
    hashjs!(h, d::Union{<:AbstractString,<:Number}) = begin
        push!(h, hash(d))
    end
    hashjs!(h, d::AbstractVector) = begin
        for v in d
            v === nothing && continue
            hashjs!(h, v)
        end
    end
    hashjs!(h, d::AbstractDict) = begin
        for k in sort(collect(keys(d)))
            push!(h, hash(k))
            d[k] === nothing && continue
            hashjs!(h, d[k])
        end
    end
    walkrep!(j::Union{<:AbstractString,<:Number}, ff, t) = begin end
    walkrep!(j::AbstractVector, ff, t) = begin
        kk = []
        for (k, v) in enumerate(j)
            if v in ff
                push!(kk, k)
                continue
            end
            v === nothing && continue
            walkrep!(v, ff, t)
        end
        for k in kk
            j[k] = t
        end
    end
    walkrep!(j::AbstractDict, ff, t) = begin
        kk = []
        for (k, v) in j
            if v in ff
                push!(kk, k)
                continue
            end
            v === nothing && continue
            walkrep!(v, ff, t)
        end
        for k in kk
            j[k] = t
        end
    end
    hashjs(jj) = begin
        h = UInt64[]
        hashjs!(h, jj)
        join(h, "/")
    end
    qpdf = expanduser("~/opt/bin/qpdf")
    load_bin_b64(fn) = begin
        open(fn) do io
            Base64.base64encode(read(io))
        end
    end
    opt_iii(tmpdir, tag, jjj, ds, src; q) = begin
        ww = jjj["dict"]["/Width"]
        hh = jjj["dict"]["/Height"]
        pad(v) = ceil(Int, v / 1) * 1
        w = pad(ww / ds)
        h = pad(hh / ds)
        if jjj["dict"]["/BitsPerComponent"] == 8 && jjj["dict"]["/ColorSpace"] == "/DeviceGray" && (jjj["dict"]["/Filter"] == "/FlateDecode" || ds > 1.3)
            img2 = joinpath(tmpdir, "$tag-out.jp2")
            qq = q == 100 ? 0 : q
            run(`magick $src -resize $(w)x$h -quality $qq $img2`)
            h, w = size(load(img2))
            @warn "$tag $w*$h"
            d = load_bin_b64(img2)
            if length(d) < length(jjj["data"])
                jjj["data"] = d
                jjj["dict"]["/Width"] = w
                jjj["dict"]["/Height"] = h
                jjj["dict"]["/Filter"] = "/JPXDecode"
            end
        elseif jjj["dict"]["/BitsPerComponent"] == 8 && jjj["dict"]["/ColorSpace"] == "/DeviceRGB" && (jjj["dict"]["/Filter"] == "/FlateDecode" || ds > 1.3)
            img2 = joinpath(tmpdir, "$tag-out.jp2")
            run(`magick $src -resize $(w)x$h -quality $q $img2`)
            h, w = size(load(img2))
            @warn "$tag $w*$h"
            d = load_bin_b64(img2)
            if length(d) < length(jjj["data"])
                jjj["data"] = d
                jjj["dict"]["/Width"] = w
                jjj["dict"]["/Height"] = h
                jjj["dict"]["/Filter"] = "/JPXDecode"
            end
        else
        end
    end
    rescale_gray(tmpdir, tag, jjj, d; q) = begin
        img1 = joinpath(tmpdir, "$tag-in.bin")
        img2 = joinpath(tmpdir, "$tag-out.bin")
        img = open(img1, "w") do io
            dd = Base64.base64decode(jjj["data"])
            dd = read(CodecZlib.ZlibDecompressorStream(IOBuffer(dd)))
            img = reshape(dd, ww, hh)
            write(io, dd)
            img
        end
        global xxximg
        push!(xxximg, deepcopy(img))
        cp(img1, "/tmp/xyz-$tag.jpg", force=true)
        #d=1
        #pad(v) = (v + 7) ÷ 8 * 8
        @info "$w*$h $ww*$hh "
        #run(`magick $img1 -resize $(w)x$h -background white -gravity center -quality $q $img2`)
        img2 = map(v -> round(UInt8, clamp(v, 0, 255)), Images.imresize(img, (w, h)))

        #run(`file $img1 $img2`)
        e = let
            dd = read(CodecZlib.ZlibCompressorStream(IOBuffer(vec(img2))))
            Base64.base64encode(dd)
        end
        jjj["data"] = e
        #h, w = size(load(img2))
        jjj["dict"]["/Width"] = w
        jjj["dict"]["/Height"] = h
        #cp(img2, "/tmp/xxx-$tag-opt.jpg", force=true)
        #cp(img2, string("/tmp/", basename(img2)), force=true)
    end
    rescale(tmpdir, tag, jjj, d; q) = begin
        img1 = joinpath(tmpdir, "$tag-in.bin")
        img2 = joinpath(tmpdir, "$tag-out.bin")
        open(img1, "w") do io
            write(io, Base64.base64decode(jjj["data"]))
        end
        #d=1
        #pad(v) = (v + 7) ÷ 8 * 8
        pad(v) = ceil(Int, v / 1) * 1
        w = pad(jjj["dict"]["/Width"] / d)
        h = pad(jjj["dict"]["/Height"] / d)
        run(`magick $img1 -resize $(w)x$h -background white -gravity center -quality $q $img2`)
        println(`magick $img1 -resize $(w)x$h -quality $q $img2`)
        run(`file $img1 $img2`)
        e = open(img2) do io
            Base64.base64encode(read(io))
        end
        jjj["data"] = e
        h, w = size(load(img2))
        jjj["dict"]["/Width"] = w
        jjj["dict"]["/Height"] = h
        cp(img1, "/tmp/xxx-$tag.jpg", force=true)
        cp(img2, "/tmp/xxx-$tag-opt.jpg", force=true)
        cp(img2, string("/tmp/", basename(img2)), force=true)
    end

    reduceppi(in, tmpdir, out; ppi) = begin
        mid = joinpath(tmpdir, "reduceppi.json")
        list = open(readlines, `pdfimages -list -all $in`, "r")
        run(`qpdf --json-output $in $mid --warning-exit-0`)
        j = open(mid) do io
            JSON.parse(io)
        end
        pfx = joinpath(tmpdir, "reduceppi-img")
        run(`pdfimages -all $in $pfx`)
        i2fn = map(filter(startswith("reduceppi-img"), readdir(tmpdir))) do fn
            @info "$fn"
            parse(Int, match(r"^reduceppi-img-(\d+)\..*$", fn).captures[1]) => joinpath(tmpdir, fn)
        end
        i2fn = Dict(i2fn)

        done = Set()
        for (ii, ll) in enumerate(list)
            m = match(r"^\s*\d+\s+(\d+)\s+(image|smask)\s+(\d+)\s+(\d+)\s+(rgb|gray)\s+\d+\s+\d+\s+(image|jpeg|jpx)\s+(yes|no)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+[BK])\s+[\d.]+%\s*$", ll)
            if m === nothing
                ii > 2 && @warn "match $ii $ll"
                continue
            end
            ii, ww, hh, o1, o2, xdpi, ydpi = parse.(Int, m.captures[[1; 3:4; 8:11]])
            @info "$ii $ww $hh $o1 $o2"
            ds = 1
            while xdpi / (ds + 1 / 10) >= ppi && ydpi / (ds + 1 / 10) >= ppi
                ds += 1 / 10
            end

            objref = "$o1 $o2 R"
            if m.captures[2] == "image"
                jjj = j["qpdf"][2]["obj:$objref"]["stream"]
                if !in!(objref, done)
                    opt_iii(tmpdir, "$objref", jjj, ds, i2fn[ii]; q)
                end
            elseif m.captures[2] == "smask"
                jjj = j["qpdf"][2]["obj:$objref"]["stream"]
                objref = jjj["dict"]["/SMask"]
                jjj = j["qpdf"][2]["obj:$objref"]["stream"]
                if !in!(objref, done)
                    opt_iii(tmpdir, "$objref", jjj, ds, i2fn[ii]; q)
                end
            end

        end
        open(mid, "w") do io
            write(io, JSON.json(j))
        end
        run(`qpdf --json-input $mid $out --warning-exit-0`)
    end
    optimg(in, out; q) = begin
        run(`$qpdf $in --compress-streams=y --decode-level=generalized --recompress-flate --compression-level=9 --optimize-images --jpeg-quality=$q --object-streams=generate --linearize $out`)
    end
    optimg0(in, out; q) = begin
        run(`$qpdf $in --compress-streams=y --decode-level=generalized --recompress-flate --compression-level=9 --object-streams=generate --linearize $out --warning-exit-0`)
    end
    dedup(in, mid, out) = begin
        run(`qpdf --json-output $in $mid --warning-exit-0`)
        j = open(mid) do io
            JSON.parse(io)
        end
        while true
            h2objs = Dict()
            for (n, jj) in j["qpdf"][2]
                !startswith("obj:")(n) && continue
                jj === nothing && continue
                o = n[5:end]
                h = hashjs(jj)
                push!(get!(() -> Set(), h2objs, h), o)
            end
            hit = 0
            for (h, oo) in h2objs
                if length(oo) > 1
                    hit += 1
                    ooo = sort(collect(oo))
                    pop!(oo, ooo[1])
                    @info "$ooo"
                    for o in ooo[2:end]
                        pop!(j["qpdf"][2], "obj:$o")
                    end
                    walkrep!(j, oo, ooo[1])
                end
            end
            hit == 0 && break
            @warn "Re dedup"
        end

        open(mid, "w") do io
            write(io, JSON.json(j))
        end
        cp(mid, "/tmp/xxx.json", force=true)
        run(`qpdf --json-input $mid $out --warning-exit-0`)
    end

    return mktempdir() do path
        json1 = joinpath(path, "1.json")
        json2 = joinpath(path, "2.json")
        pdf1 = joinpath(path, "1.pdf")
        pdf2 = joinpath(path, "2.pdf")
        pdf3 = joinpath(path, "3.pdf")
        # optimg(f, pdf1; q)
        reduceppi(f, path, pdf1; ppi)
        dedup(pdf1, json2, pdf2)
        optimg0(pdf2, pdf3; q)
        cp(pdf3, fo, force=true)
        run(`ls -lh $fo $f`)
    end
end

function checkpdfsize2(f::AbstractString)
    s1 = filesize(f)
    ll = open(readlines, `pdfimages -list -all $f`, "r")
    s2 = 0
    n2 = 0
    for (ii, ll) in enumerate(ll)
        m = match(r"^\s*\d+\s+\d+\s+(image|smask)\s+\d+\s+\d+\s+(rgb|gray|index)\s+\d+\s+\d+\s+(image|jpeg|jpx)\s+(yes|no)\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+([\d.]+[BK])\s+[\d.]+%\s*$", ll)
        if m === nothing
            ii > 2 && @warn "match $ii $ll"
            continue
        end
        x, y, a = m.captures[5:7]
        if parse(Float64, x) < 300 || parse(Float64, y) < 300
            @warn "$f: DPI $x $y $ll"
        end
        n2 += 1
        s2 += if endswith("K")(a)
            parse(Float64, a[1:end-1]) * 1000
        elseif endswith("B")(a)
            parse(Float64, a[1:end-1]) * 1
        else
            @assert false
        end
    end
    s3 = 0
    n3 = 0
    mktempdir() do tmpdir
        pfx = joinpath(tmpdir, "xxx")
        run(`pdfimages -all $f $pfx`)
        for g in readdir(tmpdir)
            s3 += filesize(joinpath(tmpdir, g))
            n3 += 1
        end
    end
    @info "$f: $s1 $(s2/s1)($n2) $(s3/s1)($n3)"
end

(out::output_fig(:S03))(
    detail::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    nodesuvw::list_nodes_uvw(),
) = begin
    df = JLD2.load(detail.path, "df")
    sample_soma_pos_for_qroot!(df)
    arbors = JLD2.load(arbors.path, "df")

    cortarbors, hclust1 = cluster_arbor_whist(arbors, df)

    obj = load("interm/mesh-nmt-brain-lr.obj")

    arbors_train = JLD2.load("interm/arbor-gnn-train.dat", "df")
    arbors_pred = JLD2.load("interm/arbor-gnn-pred.dat", "df")

    nodesuvw = nodes_info_factory(nodesuvw.path)
    arborssize = cached_result(:arborssize, cortarbors) do cortarbors
        guess_arbor_size_colonor(cortarbors, nodesuvw)
    end

    with_figure(:S05, out.path, (; df, obj, arbors, arbors_train, arbors_pred, cortarbors, hclust1, arborssize)) do fig, env
        plot_find_arbors(fig, env; env...)
    end
end

allen_obj(id) = Make.@x "interm/allen-$id.obj"
(out::allen_obj(id))() where {id} = begin
    tmp = out.path * ".tmp"
    run(`curl -Lo $tmp http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/$id.obj`)
    mv(tmp, out.path)
end



function sortas!(df::DataFrame, dfref::DataFrame, on::Symbol; filter=false)
    f2i = Dict(reverse.(enumerate(dfref[:, on])))
    if filter
        filter!(r -> get(f2i, r[on], nothing) !== nothing, df)
    end
    sort!(df, on, by=r -> f2i[r])
end

function reorder_cls!(cls, h)
    c2c = Dict(reverse.(enumerate(unique(cls[h.order]))))
    for i in axes(cls, 1)
        cls[i] = c2c[cls[i]]
    end
    cls
end

(out::list_neurons_hclust())(
    list::list_neurons(),
    listtyp::list_neurons_itptct(),
    distmat::list_neurons_dist(),
) = begin
    Random.seed!(0)
    df = JLD2.load(list.path, "df")
    filter!(r -> !endswith("X")(String(r.inj)), df)
    dftyp = JLD2.load(listtyp.path, "df")
    leftjoin!(df, dftyp, on=:path) |> dropmissing!
    swcmap, distmat = JLD2.load(distmat.path, "df", "distmat")
    distm = calc_dist_mat(df, swcmap, distmat)
    for idx in CartesianIndices(distm)
        t1, t2 = df[[idx[1], idx[2]], :typ]
        clsdist = t1[1:2] == t2[1:2] ? (t1 == t2 ? 0 : 9000) : 9000
        distm[idx] += clsdist
    end
    # @info "maxdist: $(maximum(distm))"
    #####hclust = myhclust(distm; linkage=:ward, branchorder=:barjoseph, grp=indexin(df.inj, unique(df.inj)))

    hclust = Clustering.hclust(distm; linkage=:ward_presquared, branchorder=:barjoseph)
    #hclust = Clustering.hclust(distm; linkage=:ward , branchorder=:barjoseph)
    #hclust = twopasshclust(distm; K=32, linkage=:ward, branchorder=:barjoseph)
    cls = Clustering.cutree(hclust, k=32)
    reorder_cls!(cls, hclust)
    insertcols!(df, :cls => cls)
    ensure_directory("monkeytemp/clusters") do outdir, outdirtmp
        for r in eachrow(df)
            fn = joinpath(outdirtmp, string(r.cls, '-', neuron_basename(r), ".swc.gz"))
            run(`cp $(r.path) $fn`)
        end
    end
    df = df[:, [:path, :cls]]
    JLD2.jldsave(out.path; df, hclust)
end

(out::list_mus_neurons_hclust())(
    list::list_mus_neurons(),
    distmat::list_mus_neurons_dist(),
) = begin
    Random.seed!(0)
    df = JLD2.load(list.path, "df")
    swcmap, distmat = JLD2.load(distmat.path, "df", "distmat")
    distm = calc_dist_mat(df, swcmap, distmat)
    # for idx in CartesianIndices(distm)
    #     t1, t2 = df[[idx[1], idx[2]], :typ]
    #     clsdist = t1[1:2] == t2[1:2] ? 0 : 3e4
    #     ###distm[idx] += clsdist
    # end
    # @info "maxdist: $(maximum(distm))"
    #####hclust = myhclust(distm; linkage=:ward, branchorder=:barjoseph, grp=indexin(df.inj, unique(df.inj)))
    ##hclust = Clustering.hclust(distm; linkage=:single, branchorder=:barjoseph)
    hclust = twopasshclust(distm; K=64, linkage=:ward, branchorder=:barjoseph)
    cls = Clustering.cutree(hclust, k=64)
    reorder_cls!(cls, hclust)
    insertcols!(df, :cls => cls)
    # ensure_directory("monkeytemp/clusters") do outdir, outdirtmp
    #     for r in eachrow(df)
    #         fn = joinpath(outdirtmp, string(r.cls, '-', neuron_basename(r), ".swc.gz"))
    #         run(`cp $(r.path) $fn`)
    #     end
    # end
    df = df[:, [:path, :cls]]
    JLD2.jldsave(out.path; df, hclust)
end

(out::list_neurons_uvw())(
    list::list_neurons(),
    depthmap::Make.@x("interm/depth-mapping-mac.dat"),
    flatmapdat::flatmap_data(:leftfrontal),
) = begin
    depthimg, depthres = JLD2.load(depthmap.path, "img", "res")
    df = JLD2.load(list.path, "df")
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    uvw = xyz2uvw(df.soma_pos, depthimg, depthres, objflat, objphy)
    insertcols!(df, :somauv => map(v -> SVector(v[1:end-1]...,), uvw), :somaw => last.(uvw))
    df = df[:, [:path, :somauv, :somaw]]
    JLD2.jldsave(out.path; df)
end

function loadhclust!(df::DataFrame, path::AbstractString)
    p2i = Dict(reverse.(enumerate(df.path)))
    @info "$path"
    df2, h = JLD2.load(path, "df", "hclust")
    newid(k) = p2i[df2[k, :path]]
    h.merges .= map(h.merges) do i
        i < 0 ? -newid(-i) : i
    end
    insertcols!(df, :hclustord => 0)
    for (i, k) in enumerate(h.order)
        ii = newid(k)
        h.order[i] = ii
        df[ii, :hclustord] = i
        #XXX #@assert df2[k, :cls] == df[h.order[i], :cls]
    end
    h
end

(out::Make.@x("interm/inj-popu-proj.dat"))(
    detail::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    depthmap::Make.@x("interm/depth-mapping-mac.dat"),
) = begin
    df = JLD2.load(detail.path, "df")
    arbors = JLD2.load(arbors.path, "df")
    depthimg, depthres = JLD2.load(depthmap.path, "img", "res")
    nmrdat = load_nmr_data()
    df = get_injsite_population_data(df, arbors, nmrdat, depthimg, depthres)
    JLD2.jldsave(out.path; df)
end

(out::output_fig(2))(
    neurons::list_neurons_detail(),
    proj::list_neurons_proj(),
    hclust::list_neurons_hclust(),
    dist::list_neurons_dist(),
    rawinfo::list_nodes_rawinfo(),
    arbors::Make.@x("interm/arbors.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    #(out::output_fig(:S12))(
    tabs02::Make.ByProduct{output_tab(:S02)},
) = begin
    df = JLD2.load(neurons.path, "df")
    proj = JLD2.load(proj.path, "df")
    ######sortas!(proj, df, :path, filter=true)
    leftjoin!(df, proj, on=:path) |> dropmissing!
    hclust = loadhclust!(df, hclust.path)
    sort!(df, :hclustord)
    distm = load_dist_mat(df, dist.path)
    umap = @time "umap" UMAP.umap(distm, 2, metric=:precomputed)
    obj = load("interm/mesh-nmt-brain-lr.obj")

    arbors = JLD2.load(arbors.path, "df")
    trunks = JLD2.load(trunks.path, "df")
    xlsfile = "monkeytemp/wyz-upload/macaque_proj_biofunction.xlsx"
    functable = map((; areafunc="biofunction", funcgrp="func_class", proj="proj")) do s
        DataFrame(XLSX.readtable(xlsfile, s))
    end

    mktempdir() do path
        texf = joinpath(path, "temp.tex")
        pdff = joinpath(path, "temp.pdf")
        try
            zz2fig.plot_func_table(texf, functable)
            run(Cmd(`lualatex --halt-on-error --interaction=batchmode temp.tex`, dir=path))
        finally
            cp(texf, "/tmp/xxx.tex", force=true)
            run(Cmd(`cat temp.log`, dir=path))
        end
        cp(pdff, tabs02.path, force=true)
    end

    cannolize_str(t) = begin
        t = strip(t)
        t = replace(t, r"\s+" => " ")
        uppercasefirst(lowercase(t))
    end
    mapcols!(v -> cannolize_str.(v), functable.areafunc; cols=:function)
    mapcols!(v -> cannolize_str.(v), functable.funcgrp; cols=[:function_name, :function_class_small, :function_class_big])
    terms = sort(unique(functable.areafunc[:, :function]))
    funcdist, funcdistcnt = calc_term_dist(terms)

    if false
        rawinfo = nodes_info_factory(rawinfo.path)
        t2c = zz2fig.typ2color_func(df)
        rowcolor(r) = t2c(r.typ)
        img = zz1vis.draw_hires_soma(filter(r -> r.inj == "G97-1bL" && !r.qroot, df), rawinfo; rowcolor)
        save("monkeytemp/soma-G97-1bL-itptct.png", img)
    end

    with_figure(2, out.path, (; df, obj, hclust, umap, arbors, trunks, functable, funcdist, funcdistcnt)) do fig, env
        plot_fig_result(fig, env; env...)
    end
end

(out::output_fig(3))(
    detail::list_neurons_detail(),
    proj::list_neurons_proj(),
    hclust::list_neurons_hclust(),
    flatmapdat::flatmap_data(:leftfrontal),
    arbors::Make.@x("interm/arbors.dat"),
    nmr::Make.@x("interm/inj-popu-proj.dat"),
    nodesuvw::list_nodes_uvw(),
) = begin
    df = JLD2.load(detail.path, "df")
    proj = JLD2.load(proj.path, "df")
    leftjoin!(df, proj, on=:path) |> dropmissing!
    obj = load("interm/mesh-nmt-brain-lr.obj")
    hclust = loadhclust!(df, hclust.path)
    sort!(df, :hclustord)
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))
    arbors = JLD2.load(arbors.path, "df")
    nmr = JLD2.load(nmr.path, "df")
    nodesuvw = nodes_info_factory(nodesuvw.path)

    highlights = zz1data.get_flatmap_highlights()
    nmrdat = load_nmr_data()

    tpaths, ppaths = [unique(filter(r -> contains(kw)(r.arborhier) && r.ipsi, arbors).path) for kw in ("|Temporal|", "|Parietal|")]
    insertcols!(df, :Temporal => in.(df.path, Ref(tpaths)), :Parietal => in.(df.path, Ref(ppaths)))

    dfx = filter(zz1data.select_neurons(:long_it_ipsi), df)

    with_figure(3, out.path, (; df, dfx, nodesuvw, arbors, obj, highlights, objflat, objphy, nmr, nmrdat,)) do fig, env
        plot_fig_longproj(fig, env; env...)
    end

    if false
        rawinfo = nodes_info_factory(rawinfo.path)
        t2c = zz2fig.cls_temp_color(dfx)
        rowcolor(r) = t2c(r.cls)
        img = zz2fig.draw_hires_soma(filter(r -> !r.qroot && Data.select_neurons(:long_it_ipsi_g97)(r), dfx), rawinfo; rowcolor)
        save("monkeytemp/soma-G97-1bL-pari-vs-temp.png", img)
    end
end

function plot_homotopy_associate!(ax, arbors)
    # xXX:
    dfx = filter(r -> !r.ipsi && contains("charm")(r.arborhier), arbors)
    dfy = filter(r -> r.ipsi && contains("charm")(r.arborhier), arbors)
    xx = [NmtDat.mirror(r.center) for r in eachrow(dfx)]
    yy = [r.soma_pos for r in eachrow(dfx)]
    Makie.density!(ax, map(norm ∘ -, xx, yy), boundary=(0, 4e4), label="Matched")
    Makie.density!(ax, map(norm ∘ -, xx, shuffle(yy)), boundary=(0, 4e4), label="Permutated")
    ax.xlabel[] = "Distance between arbors and mirrored soma"
    Makie.axislegend(ax, ax)
end

function calc_qhull_2d(qh, pts)
    q = DirectQhull.qh_new_qhull(qh, Float64.(reduce(hcat, pts)), "FS")
    DirectQhull.qh_get_totvol(qh)
    # 1-rnn.jl:        x = DirectQhull.qh_get_convex_hull_vertices(qh, Val(4))
    # 1-rnn.jl:        ccall((:qh_freeqhull, DirectQhull.qh_lib), Cvoid,
    #
end

function guess_arbor_size_colonor(arbors, nodesuvw)
    arbors = filter(r -> r.sample in ("234197", "232717"), arbors)
    insertcols!(arbors, :arborh => NaN, :arbordiam => NaN)
    qh = DirectQhull.qh_alloc_qh()
    for r in eachrow(arbors)
        swc = Gapr.load_swc(r.arborpath)
        uvwinfo = nodesuvw(r.sample, r.channel)
        n2w(n) = uvwinfo[n.id%100_000_000_000].rawdepth
        swc = reorient_arbor(swc, n2w)
        swc === nothing && continue
        www = [n2w(n) for n in swc.nodes]
        sel = findall(r -> !isnan(r), www)
        #length(sel) < 100 && continue
        vvv = [[n.pos[k] for n in swc.nodes] for k in 1:3]
        h1, h2 = quantile(vvv[2], [0, 1])
        r.arborh = h2 - h1
        a = calc_qhull_2d(qh, [Point2f(n.pos[1], n.pos[3]) for n in swc.nodes])
        @info "$a"
        r.arbordiam = sqrt(a / π) * 2
    end
    filter!(r -> !isnan(r.arborh), arbors)
end

(out::output_fig(5))(
    detail::list_neurons_detail(),
    proj::list_neurons_proj(),
    hclust::list_neurons_hclust(),
    arbors::Make.@x("interm/arbors.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    flatmapdat::flatmap_data(:leftfrontal),
    topo_its_str::topo_proj_data(:ITs_Str),
    nodesuvw::list_nodes_uvw(),
    arbordist::arbor_dist_for(234197),
    subregs::Make.@x("interm/subdomains.dat"),
    s06::Make.ByProduct{output_fig(:S06)},
) = begin
    df = JLD2.load(detail.path, "df")
    proj = JLD2.load(proj.path, "df")
    leftjoin!(df, proj, on=:path) |> dropmissing!
    hclust = loadhclust!(df, hclust.path)
    sort!(df, :hclustord)

    arbors = JLD2.load(arbors.path, "df")
    @warn "aaa $(size(arbors))"
    trunks = JLD2.load(trunks.path, "df")
    leftjoin!(trunks, arbors, on=[:arborpath, :path]) |> dropmissing!
    @warn "aaa $(size(trunks))"

    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))
    topo_its_str = JLD2.load(topo_its_str.path, "df")
    nodesuvw = nodes_info_factory(nodesuvw.path)

    arborsg97 = leftjoin(arbors, df, on=:path, order=:left)
    filter!(r -> r.sample == "234197", arborsg97)
    arbordistg97 = load_dist_mat(arborsg97, arbordist.path; on=:arborpath)
    @info "$(size(arbordistg97)) $(size(arborsg97))"
    arborcluster = zz1data.cached_result(:clustarbor, arborsg97, arbordistg97) do arbors, dist
        h = Clustering.hclust(Symmetric(dist + dist') / 2, linkage=:ward, branchorder=:barjoseph)
        Clustering.cutree(h, h=1600)
    end
    insertcols!(arborsg97, :arborcluster => 0)
    @info "$(size(arbordistg97)) $(size(arborsg97))"
    foreach(eachrow(arborsg97), arborcluster) do r, c
        r.arborcluster = c
    end
    @info "$(size(arbordistg97)) $(size(arborsg97))"

    obj = load("interm/mesh-nmt-brain-lr.obj")

    highlights = zz1data.get_flatmap_highlights()

    subregs = JLD2.load(subregs.path, "df")
    attach_subreg!(df, subregs, :path => :neurons)
    attach_subreg!(arbors, subregs, :arborpath => :arbors)
    filter!(r -> r.frontal && r.charmratio > 0.5, subregs)


    dfxitc = filter(zz1data.select_neurons(:itc_strict), df)
    dfxits = filter(zz1data.select_neurons(:its_strict), df)

    highlights = zz1data.get_flatmap_highlights()
    highlights2 = [
        "nmt|charm|Frontal|ACgG|ACC|area_24|area_24c"
        "nmt|charm|Frontal|ACgG|ACC|area_32"
        "nmt|charm|Frontal|ACgG|MCC|area_24a/b_prime|area_24b_prime"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_11|area_11l"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_13|area_13l"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_13|area_13m"
        "nmt|charm|Frontal|OFC|med_OFC|area_10|area_10mc"
        "nmt|charm|Frontal|OFC|med_OFC|area_10|area_10mr"
    ]
    highlights2 = vcat(highlights, highlights2)

    subregcolors = zz2fig.arrange_color(subregs.centuv, Makie.to_colormap(:tab10) |> zz2fig.remove_gray)
    insertcols!(subregs, :color => subregcolors)

    with_figure(5, out.path, (; df, dfxitc, dfxits, arbors, obj, objflat, objphy, highlights, highlights2, topo_its_str, arborsg97, arbordistg97, trunks, subregs)) do fig, env
        plot_fig_it_contra_str(fig, env; env...)
    end
    with_figure(:S06, s06.path, (; df, arborsg97, arbordistg97, obj, dfxitc, dfxits, arbors, objflat, objphy, highlights, highlights2, topo_its_str, trunks, subregs)) do fig, env
        plot_fig_arbor_patches(fig, env; env...)
    end
end

function calc_pathways(pathways, trunks, depthimg, depthres, objflat, objphy)
    pathways = deepcopy(pathways)
    soma_uvws = xyz2uvw(trunks.soma_pos, depthimg, depthres, objflat, objphy)

    foreach(eachrow(trunks), soma_uvws) do t, uvw
        t.somauv = uvw[1:2]
        t.somaw = uvw[3]
    end

    segs = zz1data.load_swcs_as_segs(trunks.arborpath, trunks.soma_pos)
    xyzms = xyz2dxyz([s.pos for s in segs], depthimg, depthres)
    uvws = dxyz2uvw(xyzms, objflat, objphy)

    votes = Dict{Int,Vector}()
    nmt = NmtDat.load_images()
    foreach(segs, xyzms, uvws) do seg, xyzm, uvw
        isnan(uvw[3]) && return
        idx = NmtDat.pos2idx(xyzm[2], nmt.res)
        v = nmt.charm[idx..., 1]
        v == 0 && return
        vv = get!(() -> [], votes, trunks[seg.nrn, :trunkcls])
        push!(vv, (; v, uvw))
    end

    votexyz = Dict{Int,Vector}()
    for s in segs
        push!(get!(() -> [], votexyz, trunks[s.nrn, :trunkcls]), (; s.pos, s.ic))
    end

    insertcols!(pathways, :ic => Symbol(), :headuv => Ref(SVector(NaN, NaN)), :headxyz => Ref(SVector(NaN, NaN, NaN)), :headw => Ref(Float32[]),
        :tailuv => Ref(SVector(NaN, NaN)), :tailxyz => Ref(SVector(NaN, NaN, NaN)), :tailw => Ref(Float32[]),
    )
    for r in eachrow(pathways)
        vv = votexyz[r.trunkcls]
        ic = [v.ic for v in vv]
        r.headxyz = SVector(median(reduce(hcat, [v.pos for v in vv]), dims=2)...)
        tt = unique(NamedTuple.(eachrow(filter(r2 -> r2.trunkcls == r.trunkcls, trunks)[:, [:path, :somauv, :somaw, :soma_pos]])))
        r.tailxyz = SVector(median(reduce(hcat, [t.soma_pos for t in tt]), dims=2)...)
        r.ic = argmax(v -> count(==(v), ic), unique(ic))
        vv = get(votes, r.trunkcls, nothing)
        vv === nothing && continue
        count(v -> v.v == 1, vv) / length(vv) < 0.3 && continue
        vv = reduce(hcat, [v.uvw for v in vv])
        r.headw = vv[3, :]
        r.headuv = SVector(median(vv[1:2, :], dims=2)...)
        @assert length(tt) == r.w
        r.tailuv = SVector(median(reduce(hcat, [t.somauv for t in tt]), dims=2)...)
        r.tailw = [t.somaw for t in tt]
    end
    pathways
end

function expand_pathways!(pathways, trunks)
    nmt = NmtDat.load_images()
    getter = NmtDat.getter(nmt)
    filter!(r -> r.w >= 3, pathways)
    insertcols!(pathways, :isfront => [!isnan(v[1]) for v in pathways.headuv], :src_hier => "", :tgt_hier => "",
        :intensitys => Ref(SVector(fill(NaN32, 5)...)),
        :trunkdepths => Ref(SVector(fill(NaN32, 5)...)),
    )
    calc_median(cls, col) = begin
        pernrn = Dict{String,Vector}()
        for r in eachrow(trunks)
            r.trunkcls != cls && continue
            vvv = get!(() -> [], pernrn, r.path)
            push!(vvv, r[col])
        end
        @assert !isempty(pernrn)
        vvvv = map(collect(pernrn)) do (_, vvv)
            @assert !isempty(vvv)
            vvv = map(vvv) do vv
                len = length(vv)
                map(1:5) do j
                    mean(view(vv, (1+(j-1)*len÷5):(j*len÷5)))
                end
            end
            median(reduce(hcat, vvv), dims=2)
        end
        SVector(median(reduce(hcat, vvvv), dims=2)...)
    end
    for r in eachrow(pathways)
        r.src_hier, r.tgt_hier = getter(reduce(hcat, [r.tailxyz, r.headxyz]))[2]
        r.intensitys = calc_median(r.trunkcls, :intensity)
        r.trunkdepths = calc_median(r.trunkcls, :trunkw)
    end
    pathways
end

function dump_pathways(fn::AbstractString, pathways)
    pathways = deepcopy(pathways)
    pathways = sort(pathways, [order(:isfront, rev=true), order(:ic, rev=true), order(:w, rev=true)])
    abbr(v) = replace(v, r"^.*Frontal\|" => "")

    XLSX.openxlsx(fn, mode="w") do xf
        sheet = xf[1]
        colj = 0
        write_col(col, vals) = begin
            colj += 1
            sheet[:, colj] = [col, vals...]
        end
        write_col("Pathway ID", pathways.trunkcls)
        write_col("Weight", pathways.w)
        write_col("Contra", [v == :C ? "Contra" : "Ipsi" for v in pathways.ic])
        write_col("Src", pathways.src_hier)
        write_col("Tgt", pathways.tgt_hier)
        write_col("SrcDepth", [isempty(v) ? NaN : median(v) for v in pathways.tailw])
        write_col("TgtDepth", [isempty(v) ? NaN : median(v) for v in pathways.headw])
        write_col("TrunkIntensity", [join(v, ",") for v in pathways.intensitys])
        write_col("TrunkDepth", [join(v, ",") for v in pathways.trunkdepths])
    end
end

function find_high_signif(mm, k)
    pvals = [HypothesisTests.pvalue(HypothesisTests.ChisqTest(view(mm, k, :), view(mm, l, :), (0:1, 0:1))) for l in axes(mm, 1)]
    pp = sortperm(pvals)[1:20]
    @info "$(pvals[pp])"
    pp
end

function find_subdomains(dfneuron, dfarbor, depthimg, depthres, objflat, objphy; K=200)
    path2inj = Dict(zip(dfneuron.path, dfneuron.inj))
    arb2inj = [path2inj[r.path] for r in eachrow(dfarbor)]
    somapos = [NmtDat.isleft(p) ? p : NmtDat.mirror(p) for p in dfneuron.soma_pos]
    arbpos = [NmtDat.isleft(p) ? p : NmtDat.mirror(p) for p in dfarbor.center]
    xyzs = [somapos; arbpos]
    xyzms = xyz2dxyz(xyzs, depthimg, depthres)
    uvs = [v[1:2] for v in dxyz2uvw(xyzms, objflat, objphy)]

    nmt = NmtDat.load_images()
    hiers = NmtDat.getter(nmt)(reduce(hcat, xyzs))[2]
    charms = contains("charm").(hiers)

    M = size(dfneuron, 1)
    N = size(dfarbor, 1)
    @warn "$M+$N"

    has_charm(i) = charms[i] && !isnan(xyzms[i][1])
    charm_pos(i) = xyzms[i][2]#uvs[i] * 30000
    dist_pt(i, j) = begin
        if has_charm(i) && has_charm(j)
            max(norm(charm_pos(i) - charm_pos(j)) + 1, norm(xyzs[i] - xyzs[j]))
        elseif !has_charm(i) && !has_charm(j)
            norm(xyzs[i] - xyzs[j])
        else
            (norm(xyzs[i] - xyzs[j]) + 1) * 9.01
        end
    end
    dist_soma(i, j) = begin
        d = dist_pt(i, j)
        dfneuron[i, :inj] == dfneuron[j, :inj] ? 0.6d : d
    end
    dist_arb(i, j) = begin
        d = dist_pt(i, j)
        arb2inj[i-M] == arb2inj[j-M] ? 0.9d : d
    end
    dist_xross(i, j) = begin
        dist_pt(i, j)
    end
    isnrn(i) = i <= M
    dist(i, j) = begin
        if isnrn(i) && isnrn(j)
            dist_soma(i, j)
        elseif !isnrn(i) && !isnrn(j)
            dist_arb(i, j)
        else
            @assert i != j
            (dist_xross(min(i, j), max(i, j)) + 1) * 1.01
        end
    end
    dm = Data.cached_result(:dmsubdomain7, dfneuron, dfarbor) do _, _
        @showprogress "calc dist mat" [Float32(dist(i, j)) for i in 1:M+N, j in 1:M+N]
    end

    h = Clustering.hclust(dm, linkage=:ward, branchorder=:barjoseph)
    cls = Clustering.cutree(h, k=K)
    init = [(sel = findall(cls .== k); sel[findmin(vec(maximum(dm[sel, sel], dims=2)))[2]]) for k in 1:K]
    mmm = Clustering.kmedoids(dm, K; init=init)
    cls = Clustering.assignments(mmm)

    df = map(sort(unique(cls))) do subreg
        sel = findall(cls .== subreg)
        neurons = dfneuron[filter(isnrn, sel), :path]
        arbors = dfarbor[filter(!isnrn, sel).-M, :arborpath]
        charmratio = count(charms[sel]) / length(sel)
        centxyz = SVector(median(reduce(hcat, xyzs[sel]), dims=2)...)
        radxyz = quantile(norm.(xyzs[sel] .- Ref(centxyz)), 0.75)
        centuv, raduv = if charmratio > 0.5
            uvssel = filter(v -> !isnan(v[1]), uvs[sel])
            c = SVector(median(reduce(hcat, uvssel), dims=2)...)
            c, quantile(norm.(uvssel .- Ref(c)), 0.75)
        else
            (NaN32, NaN32), NaN32
        end
        centuv = SVector(centuv...)

        hiervotes = Dict{String,Int}()
        for h in hiers[filter(isnrn, sel)]
            hiervotes[h] = get(hiervotes, h, 0) + 3
        end
        for h in hiers[filter(!isnrn, sel)]
            hiervotes[h] = get(hiervotes, h, 0) + 1
        end
        hiervotes = sort(collect(hiervotes), by=last, rev=true)
        xyzvals = xyzs[sel]
        uvvals = filter(v -> !isnan(v[1]), uvs[sel])
        (; subreg, neurons, arbors, charmratio, centxyz, radxyz, centuv, raduv, hiervotes, xyzvals, uvvals)
    end
    fig = Makie.Figure()
    display(fig)
    Makie.scatter(fig[1, 1], xyzs, color=rand(Makie.to_colormap(:tab10), K)[cls], inspector_label=(p, i, x) -> "$(cls[i])")
    a1, _ = Vis.dendrogram(fig[2, 1], h)
    Makie.scatter!(a1, 1:length(xyzs), fill(0, length(xyzs)), color=[(j = h.order[i]; isnrn(j) && dfneuron[j, :inj] == "G97-1bL" ? :red : (isnrn(j) ? :orange : :blue)) for i in 1:length(xyzs)])
    DataFrame(df)
end

function fix_subdomains!(subregs, df)
    #nmt = NmtDat.load_images()
    #bas = NmtDat.getter(nmt)(reduce(hcat, subregs.centxyz))[2]
    bas = [isempty(hv) ? "" : first(first(hv)) for hv in subregs.hiervotes]
    insertcols!(subregs,
        :parhier => bas,
        :frontal => contains("Frontal").(bas),
        :lpfc => contains("lat_PFC").(bas),
        :dlpfc => contains("dlPFC").(bas),
        :vlpfc => contains("vlPFC").(bas),
    )
    for inj in unique(df.inj)
        paths = Set(filter(r -> r.inj == inj, df).path)
        dfx = filter(subregs) do r
            count(n -> n in paths, r.neurons) > 0
        end
        if size(dfx, 1) != 1
            @error "inj site splitted $inj $(dfx.subreg)"
        end
    end
end

function attach_subreg!(df, subregs, (from, to))
    insertcols!(df, :subreg => 0)
    for r in eachrow(df)
        dfx = filter(rr -> r[from] in rr[to], subregs)
        @assert size(dfx, 1) == 1
        r.subreg = dfx[1, :subreg]
    end
end

(out::Make.@x("interm/subdomains.dat"))(
    list::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    depthmap::Make.@x("interm/depth-mapping-mac.dat"),
    flatmapdat::flatmap_data(:leftfrontal),
) = begin
    dflist = JLD2.load(list.path, "df")
    arbors = JLD2.load(arbors.path, "df")
    depthimg, depthres = @time "load depthimg" JLD2.load(depthmap.path, "img", "res")
    objflat, objphy = @time "load flatmap" JLD2.load(flatmapdat.path, "obj", "objphy")
    df = find_subdomains(dflist, arbors, depthimg, depthres, objflat, objphy, K=100)
    global xxxdf = deepcopy(df)
    fix_subdomains!(df, dflist)
    JLD2.jldsave(out.path; df)
end

(out::Make.@x("interm/pathways.dat"))(
    list::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    trunks_cls::Make.@x("interm/arbor-trunk-clu.dat"),
    trunks_dist::Make.@x("interm/arbor-trunk-dist.dat"),
    flatmapdat::flatmap_data(:leftfrontal),
    depthmap::Make.@x("interm/depth-mapping-mac.dat"),
) = begin
    df = JLD2.load(list.path, "df")
    dfarbor = JLD2.load(arbors.path, "df")
    leftjoin!(dfarbor, df, on=:path) |> dropmissing!
    trunks = JLD2.load(trunks.path, "df")
    trunks_cls = JLD2.load(trunks_cls.path, "df")
    leftjoin!(trunks, trunks_cls, on=:arborpath) |> dropmissing!
    leftjoin!(trunks, dfarbor, on=[:arborpath, :path]) |> dropmissing!
    distm = load_dist_mat(trunks, trunks_dist.path, on=:arborpath)

    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))
    depthimg, depthres = JLD2.load(depthmap.path, "img", "res")

    @info "get pathways"
    pathways = get_pathways(trunks, distm)
    @info "calc pathways"
    pathways = calc_pathways(pathways, trunks, depthimg, depthres, objflat, objphy)
    pathways = expand_pathways!(pathways, trunks)

    JLD2.jldsave(out.path; df=pathways)
end

intrinsic_arbor_egs(df, tag) = begin
    egs = (;
        mac=[
            (231623, 2) => [
                "root148-good" => (25662.7, 15571.2, 3202.5)
            ]
            (232717, 2) => [
                "root214-good" => (16843.6, 25185.2, 5014.45)
                "root265-good" => (16742.8, 25218.4, 4982.38)
            ]
            (221747, 1) => [
                "root279-good" => (26257.7402, 13627.7617, 4780.4678)
            ]
            (234197, 1) => [
                "root049-good" => (30129.4, 22738.6, 15437.3)
                "root011-good" => (27360.5, 23075.3, 17195.3)
            ]
        ],
        mus=[
            "196548" => ["030" => [1082403, 1082419, 1082448, 1084276, 1084331, 1084405, 1086008, 1089849]]
            "195789" => ["033" => [444210, 444218, 444324, 445338, 447257, 451715, 451766, 453023, 454895, 456171, 458324]]
            "200295" => ["083" => [106, 112, 213, 707, 733, 1816, 2905, 6487, 7313, 8393, 53, 9741]]
        ]
    )
    find_neurons(samps, df, matchds, matchnrn) = begin
        grp = 0
        dfx = map(samps) do (k, nrns)
            dfx = filter(r -> matchds(r, k), df)
            grp += 1
            insertcols!(dfx, :intrinsic_pts => Ref([]), :intrinsic_grp => 0)
            for r in eachrow(dfx)
                for (a, b) in nrns
                    f = matchnrn(k, a)
                    if f(basename(r.path))
                        r.intrinsic_pts = b isa AbstractVector ? b : [b]
                        r.intrinsic_grp = grp
                    end
                end
            end
            filter!(r -> r.intrinsic_grp != 0, dfx)
        end
        reduce(vcat, dfx)
    end
    ff = (;
        mac=((r, k) -> join((r.sample, r.channel), "/") == join(k, "/"), (k, a) -> contains("-$a")),
        mus=((r, k) -> r.sample == k, (k, a) -> startswith("$(k)_$a"))
    )[tag]
    find_neurons(egs[tag], df, ff...)
end

(out::output_fig(4))(
    list::list_neurons_detail(),
    proj::list_neurons_proj(),
    arbors::Make.@x("interm/arbors.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    trunks_cls::Make.@x("interm/arbor-trunk-clu.dat"),
    pathways::Make.@x("interm/pathways.dat"),
    flatmapdat::flatmap_data(:leftfrontal),
    subregs::Make.@x("interm/subdomains.dat"),
    rawinfo::list_nodes_rawinfo(),
    s04::Make.ByProduct{output_fig(:S04)},
    sd03::Make.ByProduct{output_fig(:SD03)},
) = begin
    df = JLD2.load(list.path, "df")
    proj = JLD2.load(proj.path, "df")
    #leftjoin!(df, proj, on=:path) |> dropmissing!
    dfarbor = JLD2.load(arbors.path, "df")
    leftjoin!(dfarbor, df, on=:path) |> dropmissing!
    trunks = JLD2.load(trunks.path, "df")
    trunks_cls = JLD2.load(trunks_cls.path, "df")
    leftjoin!(trunks, trunks_cls, on=:arborpath) |> dropmissing!
    leftjoin!(trunks, dfarbor, on=[:arborpath, :path]) |> dropmissing!
    pathways = JLD2.load(pathways.path, "df")
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))
    rawinfo = @time "load rawinfo" nodes_info_factory(rawinfo.path)

    subregs = JLD2.load(subregs.path, "df")
    attach_subreg!(df, subregs, :path => :neurons)
    attach_subreg!(dfarbor, subregs, :arborpath => :arbors)
    filter!(r -> r.frontal && r.charmratio > 0.5, subregs)

    obj = load("interm/mesh-nmt-brain-lr.obj")

    highlights = zz1data.get_flatmap_highlights()
    highlights2 = [
        "nmt|charm|Frontal|ACgG|ACC|area_24|area_24c"
        "nmt|charm|Frontal|ACgG|ACC|area_32"
        "nmt|charm|Frontal|ACgG|MCC|area_24a/b_prime|area_24b_prime"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_11|area_11l"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_13|area_13l"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_13|area_13m"
        "nmt|charm|Frontal|OFC|med_OFC|area_10|area_10mc"
        "nmt|charm|Frontal|OFC|med_OFC|area_10|area_10mr"
        "nmt|charm|Frontal|OFC|lat_OFC|area_12m/o|area_12m"
    ]
    highlights = vcat(highlights, highlights2)

    subregcolors = zz2fig.arrange_color(subregs.centuv, Makie.to_colormap(:tab10) |> zz2fig.remove_gray)
    insertcols!(subregs, :color => subregcolors)

    trunk2ft(rr) = begin
        ff = (x = filter(r -> rr.path in r.neurons, subregs); size(x, 1) == 0 ? 0 : x[1, :subreg])
        tt = (x = filter(r -> rr.arborpath in r.arbors, subregs); size(x, 1) == 0 ? 0 : x[1, :subreg])
        if tt != 0 && !rr.ipsi
            tt = -tt
        end
        ff => tt
    end
    dfxreciprocal = let pathways = zz2fig.vote_connections(subregs, dfarbor; thr=3)
        subregrow(reg) = filter(r -> r.subreg == reg, subregs)[1, :]
        pws = Dict([(t.from => t.to) => length(t.neurons) for t in pathways])
        pws = sort(filter(t -> subregrow(t.from).vlpfc && get(pws, (t.to => t.from), nothing) !== nothing && norm(subregrow(t.from).centuv - subregrow(t.to).centuv) > 0.2, pathways), by=r -> min(pws[r.from=>r.to], pws[r.to=>r.from]))[1:end]
        pws = [t.from => t.to for t in pws]
        @warn "reciprocal $pws"
        pws = [22 => 19]
        trunkdir(rr) = begin
            ff, tt = trunk2ft(rr)
            ((ff => tt) in pws) && return 1
            ((tt => ff) in pws) && return -1
            0
        end
        dfx = insertcols(dfarbor, :trunkdir => [trunkdir(r) for r in eachrow(dfarbor)])
        filter!(r -> r.trunkdir != 0, dfx)
    end
    dfxdlvl = let
        subregrow(reg) = filter(r -> r.subreg == reg, subregs)[1, :]
        trunkdir(rr) = begin
            ff, tt = trunk2ft(rr)
            (ff == 0 || tt <= 0) && return 0
            subregrow(ff).dlpfc && subregrow(tt).vlpfc && return 1
            subregrow(ff).vlpfc && subregrow(tt).dlpfc && return -1
            0
        end
        dfx = insertcols(dfarbor, :trunkdir => [trunkdir(r) for r in eachrow(dfarbor)])
        filter!(r -> r.trunkdir != 0, dfx)
    end
    dfxacggofc = let
        subregrow(reg) = filter(r -> r.subreg == reg, subregs)[1, :]
        trunkdir(rr) = begin
            ff, tt = trunk2ft(rr)
            (ff == 0 || tt <= 0) && return 0, 0
            fff = if subregrow(ff).dlpfc
                1
            elseif subregrow(ff).vlpfc
                2
            else
                0
            end
            ttt = if contains("ACgG")(subregrow(tt).parhier)
                1
            elseif contains("OFC")(subregrow(tt).parhier)
                2
            else
                0
            end
            fff, ttt
        end
        ddd = [trunkdir(r) for r in eachrow(dfarbor)]
        dfx = insertcols(dfarbor, :trunkfrom => first.(ddd), :trunkto => last.(ddd))
        filter!(r -> r.trunkfrom * r.trunkto != 0, dfx)
    end

    dfxintrinsic = intrinsic_arbor_egs(df, :mac)
    with_figure(4, out.path, (; df, dfarbor, dfxreciprocal, dfxdlvl, dfxacggofc, trunks, obj, pathways, objflat, objphy, highlights, subregs, dfxintrinsic, rawinfo)) do fig, env
        plot_fig_intra_pfc(fig, env; env...)
    end

    with_figure(:S04, s04.path, (; df, dfarbor, dfxreciprocal, dfxdlvl, dfxacggofc, trunks, obj, pathways, objflat, objphy, highlights, subregs)) do fig, env
        plot_fig_intra_pfc_supp(fig, env; env...)
    end
    with_figure(:SD03, sd03.path, (; df, dfarbor, dfxreciprocal, dfxdlvl, dfxacggofc, trunks, obj, pathways, objflat, objphy, highlights, subregs, dfxintrinsic, rawinfo)) do fig, env
        zz2fig.with_caption("")
        zz2fig.plot_intra_pfc_combo_flatmap(Makie.GridLayout(fig[1, 1]), env; env...)
    end
end

(out::output_fig(:S05))(
    list::list_neurons_detail(),
    listmus::list_mus_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    trunks_cls::Make.@x("interm/arbor-trunk-clu.dat"),
    musarbors::Make.@x("interm/mouse-arbors.dat"),
    rawinfo::list_nodes_rawinfo(),
    flatmapdat::flatmap_data(:leftfrontal),
) = begin
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))

    df = JLD2.load(list.path, "df")
    dfmus = JLD2.load(listmus.path, "df")
    dfarbor = JLD2.load(arbors.path, "df")
    leftjoin!(dfarbor, df, on=:path)
    trunks = JLD2.load(trunks.path, "df")
    trunks_cls = JLD2.load(trunks_cls.path, "df")
    leftjoin!(trunks, trunks_cls, on=:arborpath) |> dropmissing!
    musarbors = JLD2.load(musarbors.path, "df")
    dfxintrinsic = intrinsic_arbor_egs(df, :mac)
    dfxintrinsic_mus = intrinsic_arbor_egs(dfmus, :mus)
    rawinfo = @time "load rawinfo" nodes_info_factory(rawinfo.path)
    obj = load("interm/mesh-nmt-brain-lr.obj")
    highlights = zz1data.get_flatmap_highlights()
    with_figure(:S05, out.path, (; df, dfxintrinsic, dfxintrinsic_mus, dfarbor, trunks, musarbors, rawinfo, obj, objflat, objphy, highlights)) do fig, env
        zz2fig.plot_fig_intrinsic(fig, env; env...)
    end
end

(out::output_fig(:S05xx))(
    list::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    subregs::Make.@x("interm/subdomains.dat"),
    flatmapdat::flatmap_data(:leftfrontal),
) = begin
    df = JLD2.load(list.path, "df")
    dfarbor = JLD2.load(arbors.path, "df")
    leftjoin!(dfarbor, df, on=:path) |> dropmissing!
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))

    subregs = JLD2.load(subregs.path, "df")
    attach_subreg!(df, subregs, :path => :neurons)
    attach_subreg!(dfarbor, subregs, :arborpath => :arbors)

    highlights = Data.get_flatmap_highlights()
    highlights2 = [
        "nmt|charm|Frontal|ACgG|ACC|area_24|area_24c"
        "nmt|charm|Frontal|ACgG|ACC|area_32"
        "nmt|charm|Frontal|ACgG|MCC|area_24a/b_prime|area_24b_prime"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_11|area_11l"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_13|area_13l"
        "nmt|charm|Frontal|OFC|lat_OFC|areas_11/13|area_13|area_13m"
        "nmt|charm|Frontal|OFC|med_OFC|area_10|area_10mc"
        "nmt|charm|Frontal|OFC|med_OFC|area_10|area_10mr"
    ]
    highlights = vcat(highlights, highlights2)

    Fig.with_figure(:S05xx, out.path, (; df, subregs, dfarbor, highlights, objflat, objphy,)) do fig, env
        Fig.with_caption("Neuron subtypes, intra-PFC projection flatmaps.")
        Fig.plot_all_subdomains_intra_pfc_proj_flat(Makie.GridLayout(fig[1, 1]), env; env...)
    end
end


function find_typic_pathways(df, xxxtrunks, xxxpathways, tgtcls)
    df = filter(r -> r.inj == "G97-1bL", df)
    for rr in eachrow(xxxpathways)
        trunkcls = rr.trunkcls
        paths = filter(t -> t.trunkcls == trunkcls, xxxtrunks).path
        vals = map(r -> (tgtcls(r), r.path in paths), eachrow(df))
        xx = Int.(first.(vals))
        yy = Int.(last.(vals))
        npred = sum(xx) / length(xx) * sum(yy) / length(yy) * length(yy)
        nval = count(>=(2), xx .+ yy)
        t = HypothesisTests.ChisqTest(xx, yy, (0:1, 0:1))
        p = HypothesisTests.pvalue(t)
        if p < 0.05 && npred < nval && contains("Frontal")(rr.tgt_hier) && rr.ic === :I
            println("$p $trunkcls $npred $nval")
        end
    end
end

function detect_intrinsic(trunks, rawinfo, depthimg, depthres)
    swccache = Dict{String,Gapr.SwcTable}()
    cached_swc(path) = get!(() -> Gapr.load_swc(path), swccache, path)
    fixid(id) = id % 100_000_000_000

    df = @showprogress map(eachrow(trunks)) do r
        info = rawinfo(r.sample, r.channel)
        swc = cached_swc(r.path)
        idx = swc[!, r.trunkterm].ind
        pts = Vec3f[]
        intensity = Float32[]
        while idx != 0
            n = swc[idx]
            push!(pts, n.pos)
            push!(intensity, info[fixid(n.id)].rawavg)
            idx = n.parind
        end
        (; r.arborpath, pts, intensity, r.sample, r.channel)
    end
    df = df |> DataFrame

    insertcols!(df, :trunkw => [deepcopy(r.intensity) for r in eachrow(df)])
    xyzs = []
    for (i, r) in enumerate(eachrow(df))
        for (j, p) in enumerate(r.pts)
            push!(xyzs, (; i, j, p))
        end
    end
    dxyzs = xyz2dxyz([t.p for t in xyzs], depthimg, depthres)
    foreach(dxyzs, xyzs) do dxyz, xyz
        df[xyz.i, :trunkw][xyz.j] = dxyz[1]
    end

    sampchan(r) = (r.sample, r.channel)
    @showprogress for sc in unique(sampchan.(eachrow(df)))
        qt = 0:0.1:1
        qtv = quantile(reduce(vcat, filter(r -> sampchan(r) == sc, df).intensity), qt)
        for r in eachrow(df)
            sampchan(r) != sc && continue
            r.intensity .= [qt[findmin(q -> abs(q - v), qtv)[2]] for v in r.intensity]
        end
    end

    get_mean_vals(vv; K=5) = begin
        len = length(vv)
        map(1:K) do j
            mean(view(vv, (1+(j-1)*len÷K):(j*len÷K)))
        end
    end
    for r in eachrow(df)
        r.intensity = get_mean_vals(r.intensity; K=10)
        r.trunkw = get_mean_vals(r.trunkw; K=10)
    end
    df[:, [:arborpath, :intensity, :trunkw]]
end

function trunk_from_arbor(swc; soma_pos)
    for n in swc.nodes
        n.typ = 0
    end
    extract(pts, K=100) = begin
        tot = 0.0
        for i in 2:length(pts)
            tot += norm(pts[i] - pts[i-1])
        end
        par = 0.0
        res = Vec3f[]
        push!(res, pts[1])
        for i in 2:length(pts)
            l = norm(pts[i] - pts[i-1])
            lenx = min(K, (par + l) / tot * K)
            while length(res) < lenx
                t = (tot * length(res) / K - par) / l
                push!(res, t * pts[i] + (1 - t) * pts[i-1])
            end
            par += l
        end
        push!(res, pts[end])
    end
    get_pts(swc, idx, soma_pos) = begin
        pts = Vec3f[]
        while idx != 0
            push!(pts, swc[idx].pos)
            idx = swc[idx].parind
        end
        push!(pts, NmtDat.isleft(soma_pos) ? soma_pos : NmtDat.mirror(soma_pos))
        extract(reverse!(pts))
    end
    (ids) -> begin
        if !isempty(ids)
            for id in ids
                swc[!, id].typ = 1
            end
            idx = swc[!, first(ids)].ind
            trunk = get_pts(swc, idx, soma_pos)
            (; trunk, trunkterm=swc[idx].id)
        else
            depths = collect(Gapr.nodedepth(swc).depth)
            idx = findmax(identity, depths)[2]
            if swc[idx].typ == 0
                trunk = get_pts(swc, idx, soma_pos)
                (; trunk, trunkterm=swc[idx].id)
            end
        end
    end
end

(out::Make.@x("interm/arbor-trunk.dat"))(
    detail::list_neurons(),
    arbors::Make.@x("interm/arbors.dat"),
    rawinfo::list_nodes_rawinfo(),
    depthmap::Make.@x("interm/depth-mapping-mac.dat"),
) = begin
    dflist = JLD2.load(detail.path, "df")
    filter!(r -> !endswith("X")(r.inj), dflist)
    arbors = JLD2.load(arbors.path, "df")
    sample_soma_pos_for_qroot!(dflist)

    rawinfo = @time "load rawinfo" nodes_info_factory(rawinfo.path)
    depthimg, depthres = @time "load depthimg" JLD2.load(depthmap.path, "img", "res")

    fixid(id) = id % 100_000_000_000

    df = @showprogress map(eachrow(dflist)) do rr
        swc = Gapr.load_swc(rr.path) |> Data.mirrorswcby(NmtDat)
        func = trunk_from_arbor(swc; rr.soma_pos)
        trunks = map(eachrow(filter(r -> r.path == rr.path, arbors))) do r
            aaa = Gapr.load_swc(r.arborpath)
            trunk = func((fixid(n.id) for n in aaa.nodes))
            (; rr.path, r.arborpath, trunk...)
        end
        trunk = func([])
        if trunk !== nothing
            push!(trunks, (; rr.path, arborpath=rr.path, trunk...))
        end
        trunks
    end
    df = DataFrame(filter!(r -> length(r.trunk) == 101, reduce(vcat, df)))
    df3 = detect_intrinsic(leftjoin(df, dflist, on=:path), rawinfo, depthimg, depthres)
    leftjoin!(df, df3, on=:arborpath) |> dropmissing!
    JLD2.jldsave(out.path; df)
end

(out::Make.@x("interm/arbor-trunk-dist.dat"))(
    trunks::Make.@x("interm/arbor-trunk.dat"),
) = begin
    trunks = JLD2.load(trunks.path, "df")
    procs = addprocs(11)
    distmat = try
        @everywhere procs @eval using ProgressMeter, GeometryBasics, LinearAlgebra
        @everywhere procs global alltrunks = $(trunks.trunk)
        @showprogress "dist" pmap(WorkerPool(procs), axes(trunks, 1)) do i
            map(axes(Main.alltrunks, 1)) do j
                mapreduce(norm ∘ -, +, alltrunks[i], alltrunks[j])
            end
        end
    finally
        rmprocs(procs)
    end
    distmat = reduce(hcat, distmat)
    lens = map(eachrow(trunks)) do r
        mapreduce(norm ∘ -, +, @view(r.trunk[2:end]), @view(r.trunk[1:end-1]))
    end
    distmat = distmat ./ sqrt.((reshape(lens, 1, :) .+ reshape(lens, :, 1) .+ 1))
    #distm = (distm .+ distm') / 2
    df = trunks[:, [:arborpath]]
    JLD2.jldsave(out.path; df, distmat)
end

# Clustering
# cls = cached_result(:clustarbxxxxx) do
#     # hclu=Clustering.hclust(distm, linkage=:complete, branchorder=:barjoseph)
#     #Main.xxxhclu=hclu
#     # cls=Clustering.cutree(hclu, k=60)
#     Clustering.kmedoids(distm, 60).assignments
# end
#
#
# Main.xxxf = Makie.Figure()
# display(Main.xxxf)
# ax = Makie.LScene(Main.xxxf[1, 1])
# collect!(pts, l, v, col) = begin
#     for ll in l
#         push!(pts, (Point3f(ll), col))
#     end
#     push!(pts, (Point3f(NaN, NaN, NaN), v, col))
# end
# pts = []
# @showprogress foreach(cls, eachrow(arbors), lines,) do c, r, l
#     if r.inj == "D17-2bL"
#         collect!(pts, l, c, c % 7 + 1)
#     end
# end
# Makie.lines!(ax, first.(pts), color=last.(pts), colorrange=(1, 7), colormap=:tab10, inspector_label=(p, i, x) -> string(pts[i][2]))
# return
#
# with_figure(out.path) do fig
#     with_subfig(fig[1, 1]) do grid
#         Vis.dendrogram(grid[1, 1], hclu, axis=(; height=200, width=500))
#     end
#     Main.xxxlines = lines
#     with_subfig(fig[2, 1]) do grid
#         for (K, cc) in enumerate(sort(unique(cls)))
#             rowcol = divrem(K - 1, 6) .+ 1
#             ax = Makie.LScene(grid[rowcol...], width=100, height=160)
#             with_mixed_scene(ax, GLMakie) do mix
#                 pts = []
#                 foreach(lines, eachrow(arbors), cls) do l, r, c
#                     if c == cc
#                         col = hash(r.inj) % 7 + 1
#                     end
#                 end
#                 Makie.lines!(mix.scene2, first.(pts), color=last.(pts), colorrange=(1, 7), colormap=:tab10)
#                 setup_camera_nmt_any(ax, :h)
#             end
#         end
#     end
#     ""
# end
#
function test_clustering(distm)
    clu_quality(clu, dm) = begin
        map([(+, :calinski_harabasz), (-, :xie_beni), (-, :davies_bouldin), (+, :dunn), (+, :silhouettes),]) do (s, q)
            q => try
                s(Clustering.clustering_quality(clu, dm; quality_index=q))
            catch
                NaN
            end
        end |> NamedTuple
    end

    cls = nothing
    if false
        h = Clustering.hclust(distm, linkage=:ward_presquared, branchorder=:barjoseph)
        h = Clustering.cutree(h, k=128)
        @info clu_quality(h, distm)
        # ward (calinski_harabasz = NaN, xie_beni = NaN, davies_bouldin = NaN, dunn = 0.00061102444f0, silhouettes = 0.18082348f0)
        # wardps (calinski_harabasz = NaN, xie_beni = NaN, davies_bouldin = NaN, dunn = 7.330131f-5, silhouettes = 0.18551034f0)
        # complete (calinski_harabasz = NaN, xie_beni = NaN, davies_bouldin = NaN, dunn = 0.001174221f0, silhouettes = 0.12631822f0)
        cls = h
    end
    if true
        clu = Clustering.kmedoids(distm, 128; init=Clustering.initseeds_by_costs(:rand, distm, 128))
        @info clu_quality(clu, distm)
        # rand (calinski_harabasz = NaN, xie_beni = NaN, davies_bouldin = NaN, dunn = 9.66981f-7, silhouettes = 0.100278415f0)
        # kmcen (calinski_harabasz = NaN, xie_beni = NaN, davies_bouldin = NaN, dunn = 1.5454698f-6, silhouettes = -0.07857633f0)
        # kmpp (calinski_harabasz = NaN, xie_beni = NaN, davies_bouldin = NaN, dunn = 1.2024633f-6, silhouettes = 0.11408786f0)
        cls = Clustering.assignments(clu)
    end
    if false
        clu = Clustering.affinityprop(-distm)
        @info clu_quality(clu, distm)
        cls = Clustering.assignments(clu)
    end
    if false
        clu = Clustering.mcl(Float32.(distm .< 1))
        @info "$(Clustering.nclusters(clu))"
        @info clu_quality(clu, distm)
        cls = Clustering.assignments(clu)
    end
    return cls



    r2 = Clustering.hclust(distm, linkage=:ward, branchorder=:barjoseph)
    r2ct = Clustering.cutree(r2, k=32)
    ii = [
        begin
            grp = findall(==(g), r2ct)
            grp[argmin(sum(distm[grp, grp], dims=2))]
        end for g in unique(r2ct)
    ]
    r0 = Clustering.kmedoids(distm, 32; init=ii)
    # :rand
    # *:kmpp
    # :kmcen
    r0 = Clustering.kmedoids(distm, 32; init=Clustering.initseeds_by_costs(:kmpp, distm, 32))
    r1 = Clustering.affinityprop(-distm)
    #r3=Clustering.mcl(-distm)
    #return r3
    r4 = Clustering.dbscan(distm, 2000; metric=nothing)
    # r5=Clustering.fuzzy_cmeans(distm, 32, 2; dist_metric=nothing)
    # return r5
    return Clustering.assignments(r0)
    return Clustering.counts(r0, r4)
    return clu_quality(r4, distm)
    return r4
    return Clustering.confusion(r0, r2ct)
    return clu_quality(r0, distm)
    return Clustering.mutualinfo(r0, r2ct)
    return Clustering.randindex(r0, r2ct)
    return Clustering.wcounts(r0)
    return Clustering.varinfo(r0, r2ct)
    return Clustering.vmeasure(r0, r2ct)
    #return clu_quality(r2ct, distm)
    return clu_quality(r0, distm)
    return Clustering.silhouettes(r0, distm)
    return clu_quality(r0, distm)
end

function load_trunks_as_pts(f::Function, trunks)
    ptscol = map(enumerate(trunks)) do (i, t)
        pts = deepcopy(t)
        col = [f(i, j) for j in axes(pts, 1)]
        push!(pts, Vec3f(NaN)), push!(col, f(0, 0))
    end
    reduce(vcat, first.(ptscol)), reduce(vcat, last.(ptscol))
end

function test_clustering_sparse(trunks, distm)
    Random.seed!(0)
    @assert size(trunks, 1) == size(distm, 1)
    insertcols!(trunks, :trunkcls => 0)
    let
        rad_thr = 2000.0
        #clu = Clustering.kmedoids(distm, 256; init=Clustering.initseeds_by_costs(:kmpp, distm, 256))
        #cls = Clustering.assignments(clu)
        hhh = Clustering.hclust(distm, linkage=:complete)
        Vis.dendrogram(hhh) |> display
        cls = Clustering.cutree(hhh, h=rad_thr)
        for K in 1:50
            break
            meds = [(grp = findall(==(c), cls); grp[argmin(sum(view(distm, grp, grp), dims=2))]) for c in sort(unique(cls))]
            clu = Clustering.kmedoids!(distm, meds)
            cls = Clustering.assignments(clu)
            rad_thr > 9 && break

            grps = map(sort(unique(cls))) do c
                c => findall(==(c), cls)
            end
            s = map(grps) do (c, grp)
                c => maximum(view(distm, grp, grp))
            end
            j = [(c1, c2) => (c1 == c2 ? Inf : (grp = [g1; g2]; maximum(view(distm, grp, grp)))) for (c1, g1) in grps, (c2, g2) in grps]
            s = s[findmax(last, s)[2]]
            s1 = s.first
            grpx = findall(==(s1), cls)
            taboo = Set()
            j = sort!(filter!(v -> last(v) < 1000, j[:]), by=last)
            @info "$(length(grps)) $rad_thr split $s[$(length(grpx))] join $(j[1])"
            for ((j1, j2), jd) in j
                jd > rad_thr && break
                (j1 in taboo || j2 in taboo) && continue
                cls[cls.==j2] .= j1
                push!(taboo, j1, j2)
                @info "$(length(grps)) split $s[$(length(grpx))] join $j1 $j2 $jd"
            end
            if isempty(taboo)
                rad_thr += 1
            end
            # ccc=Clustering.cutree(Clustering.hclust(distm[grpx, grpx], linkage=:complete), k=2)
            # cls[grpx[ccc.==1]].=j2
            #.
            #cls[shuffle!(findall(==(s1), cls))[1:end÷2]].=j2
        end
        foreach(eachrow(trunks), cls) do r, c
            r.trunkcls = c
        end
    end
    trunks[:, [:arborpath, :trunkcls]]
end

(out::Make.@x("interm/arbor-trunk-clu.dat"))(
    trunks::Make.@x("interm/arbor-trunk.dat"),
    trunks_dist::Make.@x("interm/arbor-trunk-dist.dat"),
) = begin
    trunks = JLD2.load(trunks.path, "df")
    distm = load_dist_mat(trunks, trunks_dist.path, on=:arborpath)
    trunks = test_clustering_sparse(trunks, distm)
    JLD2.jldsave(out.path; df=trunks)
end

function get_pathways(trunks, distm)
    map(sort(unique(trunks.trunkcls))) do trunkcls
        inds = findall(==(trunkcls), trunks.trunkcls)
        nrn2trunks = Dict()
        for i in inds
            vv = get!(() -> [], nrn2trunks, trunks[i, :path])
            push!(vv, i)
        end
        vv = map(collect(nrn2trunks)) do (_, vv)
            vv[argmin(sum(distm[vv, vv], dims=2)[:])]
        end
        centroid = vv[argmin(sum(distm[vv, vv], dims=2)[:])]
        trunk = trunks[centroid, :trunk]
        (; trunkcls, w=length(nrn2trunks), trunk, centroid)
    end |> DataFrame
end

function select_pathways(pathways, trunks, distm)
    obj = load("interm/mesh-nmt-brain-lr.obj")

    fig = Makie.Figure()
    Main.xxxfig = fig
    display(fig)
    Makie.DataInspector(fig)
    h = Clustering.hclust(distm[pathways.centroid, pathways.centroid], linkage=:ward, branchorder=:barjoseph)
    Vis.dendrogram(fig[1, 1][1, 1], h, axis=(; height=200))
    ax, _ = Makie.mesh(fig[1, 1][2, 1], obj, color=(:gray, 0.05), transparency=true, fxaa=true, inspectable=false)

    pathwayplts = []
    inspector_label = inspect_nmt_hier()
    @showprogress for r in eachrow(pathways)
        #println(r)
        if r.w < 5
            push!(pathwayplts, missing)
            continue
        end
        rad = (grp = findall(==(r.trunkcls), trunks.trunkcls); maximum(view(distm, grp, grp)))
        plt = Makie.lines!(ax, r.trunk, linewidth=1.5, color=(r.w >= 10 ? :orange : :gray), inspector_label=(p, i, x) -> "cls $(r.trunkcls); weight $(r.w); rad $rad; $(inspector_label(p,i,x))", transparency=true, fxaa=true)
        @assert r.trunkcls == length(pathwayplts) + 1
        push!(pathwayplts, plt)
    end
    pickcls = Ref(0)
    selcls = Int[]
    update_display() = begin
        for c in (pickcls[], selcls...)
            if c != 0
                pathwayplts[c].linewidth[] = 5
            end
        end
        for c in selcls
            if c != 0
                pathwayplts[c].color[] = :red
            end
        end
    end
    btn_add = Makie.Button(fig[1, 1][3, 1][1, 1], label="Select")
    txt_sel1 = Makie.Textbox(fig[1, 1][3, 1][1, 2], placeholder="from", validator=Int)
    txt_sel2 = Makie.Textbox(fig[1, 1][3, 1][1, 3], placeholder="to", validator=Int)
    btn_sel = Makie.Button(fig[1, 1][3, 1][1, 4], label="Select 2")
    btn_clr = Makie.Button(fig[1, 1][3, 1][1, 5], label="Clear")
    btn_draw = Makie.Button(fig[1, 1][3, 1][1, 6], label="Draw")
    ax2, _ = Makie.mesh(fig[1, 2][1, 1], obj, color=(:gray, 0.05), transparency=true, fxaa=true, inspectable=false)
    ax3 = Makie.Axis(fig[1, 2][2, 1])
    ax3.height[] = 200
    Makie.on(btn_draw.clicks) do n
        plts = filter(p -> (p isa Makie.Lines), ax2.scene.plots)
        for p in plts
            delete!(ax2, p)
        end
        dfx = filter(r -> r.trunkcls in selcls, trunks)
        draw_trunks!(ax2, dfx)
        empty!(ax3)
        ids = findall(map(r -> r in selcls, trunks.trunkcls))
        hh = Clustering.hclust(distm[ids, ids], linkage=:complete, branchorder=:barjoseph)
        Vis.dendrogram!(ax3, hh)
    end


    Makie.content(fig[1, 1][3, 1]).tellwidth[] = false
    selectcls(ids) = begin
        for cls in ids
            push!(selcls, cls)
        end
    end
    clearsel() = begin
        for prev in selcls
            if prev != 0
                pathwayplts[prev].color[] = pathways[prev, :w] >= 10 ? :orange : :gray
                pathwayplts[prev].linewidth[] = 1.5
            end
        end
        empty!(selcls)
    end
    Makie.on(btn_sel.clicks) do n
        clearsel()
        f = parse(Int, txt_sel1.stored_string[])
        t = parse(Int, txt_sel2.stored_string[])
        selectcls(h.order[f:t])
        update_display()
    end
    Makie.on(btn_clr.clicks) do n
        clearsel()
        update_display()
    end
    Makie.on(btn_add.clicks) do n
        pp = pickcls[]
        @info "sel $pp"
        selectcls([pp])
        update_display()
    end
    Makie.on(Makie.events(fig).mousebutton, priority=2) do event
        if event.button == Makie.Mouse.left && event.action == Makie.Mouse.press
            picks = Makie.pick_sorted(Makie.get_scene(ax), Makie.events(fig).mouseposition[], 3)
            for (plt, plti) in picks

                cls = findfirst(v -> v !== missing && v == plt, pathwayplts)
                if cls !== nothing
                    prev = pickcls[]
                    if prev != 0
                        pathwayplts[prev].linewidth[] = 1.5
                    end
                    @info "pick $cls"
                    pickcls[] = cls
                    update_display()
                    return Makie.Consume(true)
                else
                    @info "pick 0 $(hash(plt)) $(plt), $(plti)"
                    ##pickcls[] = 0
                end
            end
        end
        return Makie.Consume(false)
    end
end

(out::Make.@x("interm/arbor-trunk-test.dat"))(
    list::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    trunks_cls::Make.@x("interm/arbor-trunk-clu.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    trunks_dist::Make.@x("interm/arbor-trunk-dist.dat"),
) = begin
    df = JLD2.load(list.path, "df")[:, Not(:cls)]
    dfarbor = JLD2.load(arbors.path, "df")
    trunks = JLD2.load(trunks.path, "df")
    trunks_cls = JLD2.load(trunks_cls.path, "df")
    trunks = leftjoin(trunks, leftjoin(dfarbor, df, on=:path), on=:arborpath, order=:left) |> dropmissing!
    leftjoin!(trunks, trunks_cls, on=:arborpath) |> dropmissing!
    distm = load_dist_mat(trunks, trunks_dist.path, on=:arborpath)
    pathways = get_pathways()
    if true
    else
    end
    Main.xxxdistm = distm
    Main.xxxtrunks = trunks
end

function nodes_info_factory(path)
    nodeinfo = JLD2.load(path, "df")
    nodeinfo = map(nodeinfo) do (sc, info)
        sc => map(info) do tup
            tup.id => tup
        end |> Dict
    end |> Dict
    (path...) -> begin
        nodeinfo[join(path, '/')]
    end
end

function draw_with_intensity(df, nodeinfo)
    df = filter(r -> r.inj == "G97-1aL", df)
    obj = load("interm/mesh-nmt-brain-lr.obj")
    Makie.mesh(obj, color=(:gray, 0.05), transparency=true, inspectable=false) |> display
    for r in eachrow(df)
        info = nodeinfo(r.sample, r.channel)
        pts, col = load_swc_as_lines(r.path) do n
            info[n.id].rawavg
        end
        Makie.lines!(pts, color=col, transparency=true, inspector_label=(p, i, x) -> string(col[i]))
    end
    Makie.DataInspector()
end

(out::list_nodes_rawinfo())(list::list_neurons()) = begin
    df = JLD2.load(list.path, "df")
    filter!(r -> !endswith("X")(r.inj), df)
    sampchan_of(r) = string(r.sample, "/", r.channel)
    infos = [
        (221059, 1) => ["fmost221059.1-1", "fmost221059.2-1"]
        (221747, 1) => ["fmost221720", "fmost221747"]
        (231623, 1) => ["fmost231623.ch1", "fmost231623-1.ch1"]
        (231623, 2) => ["fmost231623.ch2", "fmost231623-1.ch2"]
        (232717, 1) => ["fmost232717.ch1", "fmost232717-1.ch1"]
        (232717, 2) => ["fmost232717.ch2", "fmost232717-1.ch2"]
        (233001, 1) => ["fmost233001xx1.ch1", "fmost233001xx2.ch1"]
        (233001, 2) => ["fmost233001xx1.ch2", "fmost233001xx2.ch2"]
        (234162, 1) => ["fmost234162.ch1"]
        (234162, 2) => ["fmost234162.ch2"]
        (234197, 1) => ["fmost234197.ch1"]
        (240025, 1) => ["fmost240025.ch1"]
    ]

    procs = addprocs(3)
    df = try
        @everywhere procs @eval using ProgressMeter, Gapr, IniFile, DataFrames, StaticArrays, NRRD, FileIO
        @showprogress pmap(WorkerPool(procs), unique([sampchan_of(r) for r in eachrow(df)])) do sampchan
            hits = filter(i -> join(i.first, "/") == sampchan, infos)
            @assert length(hits) == 1
            info = hits[1].second
            dfx = filter(r -> sampchan_of(r) == sampchan, df)
            nodes = Set{Int}()
            for r in eachrow(dfx)
                swc = Gapr.load_swc(r.path)
                for n in swc.nodes
                    push!(nodes, n.id)
                end
            end
            catfile(tag) = expanduser("~/Desktop/segreg/input/$tag.catalog")
            swcfile(tag) = expanduser("~/Desktop/segreg/input/$tag.swc.gz")
            imgfile(tag, typ) = expanduser("/data/dat/dsimgs/$tag@ch00-$typ.nrrd")
            id_inc = 1_000_000_000
            dfnodes = map(enumerate(info)) do (b, tag)
                ini = read(IniFile.Inifile(), catfile(tag))
                sizresorg = nothing
                parsevec(T, v) = SVector(parse.(T, split(v, r"\s"))...)
                for (sec, kvals) in ini.sections
                    if endswith("_AVG")(sec) || endswith("_MIP")(sec)
                        vv = (parsevec(Int, kvals["size"]), parsevec(Float64, kvals["direction"]), parsevec(Float64, kvals["origin"]))
                        if sizresorg === nothing
                            sizresorg = vv
                        else
                            @assert sizresorg == vv
                        end
                    end
                end
                siz, res, org = sizresorg[1], sizresorg[2][1:4:9], sizresorg[3]
                imgs = map((:avg, :mip)) do typ
                    img = load(imgfile(tag, typ))
                    @assert SVector(size(img)...) == siz "$(size(img))!= $siz"
                    typ => img
                end |> NamedTuple
                getvox(pos, typs...) = begin
                    idx = ceil.(Int, (pos .- org) ./ res)
                    for i in (0, -1, 1)
                        idx2 = idx .+ (0, 0, i)
                        if checkbounds(Bool, imgs[1], idx2...)
                            idx = idx2
                            break
                        end
                    end
                    map(imgs) do img
                        v = img[idx...]
                        @assert v != 0 "$v $idx $tag"
                        v / typemax(v)
                    end
                end
                swc = Gapr.load_swc(swcfile(tag))
                map(swc.nodes) do n
                    id = n.id + id_inc * (b - 1)
                    if id in nodes
                        pop!(nodes, id)
                        rawxyz = n.pos
                        rawavg, rawmip = getvox(rawxyz)[(:avg, :mip)]
                        (; id, rawxyz, rawavg, rawmip)
                    else
                        missing
                    end
                end
            end
            @assert isempty(nodes) "left nodes: $sampchan $(first(nodes)) $(length(nodes))"
            dfnodes = collect(skipmissing(reduce(vcat, filter!(!isempty, dfnodes))))
            sampchan => dfnodes
        end
    finally
        rmprocs(procs)
    end
    JLD2.jldsave(out.path; df)
end

function knn_based_depth(img1, img2, imgmsk, res)
    get_tree(img, res) = begin
        pts = map(findall(>(0), img)) do idx
            NmtDat.idx2pos(SVector(Tuple(idx)), res)
        end
        NearestNeighbors.KDTree(pts)
    end
    tree1 = get_tree(img1, res)
    tree2 = get_tree(img2, res)
    (xyz) -> begin
        d1s = NearestNeighbors.knn(tree1, xyz, 1)[2]
        d2s = NearestNeighbors.knn(tree2, xyz, 1)[2]
        map(reduce(vcat, d1s), reduce(vcat, d2s), xyz) do d1, d2, p
            idx = NmtDat.pos2idx(p, res)
            imgmsk[idx...] > 0 ? d1 / (d1 + d2 + 1) : NaN32
        end
    end
end

function knn_based_depth_allen()
    border = load("/data/dat/isocortex_boundary_10.nrrd").data
    mask = load("/data/dat/isocortex_mask_10.nrrd").data
    #allen = Allen.load_images()
    #@assert SVector(size(border) .* 10.0) == size(allen.seg) .* allen.res "$(size(border).*10.0) $(size(allen.seg).*allen.res)"
    @assert size(border) == size(mask)
    knn_based_depth(border .== 1, border .== 3, mask, (10.0, 10.0, 10.0))
end

function knn_based_depth_any(img, res)
    img1, img2, imgmsk = zz1data.cached_result(:maskimg2, img) do img
        img = @time "load" load(img)
        v1, v2, v3 = @time "sort" sort(unique(vec(img)))
        img3 = @time "mask" img .== v3
        img3 = @time "filter" Images.imfilter(img3, ones(Float32, 3, 3, 3))
        img .== v1 .&& img3 .> 0, img .== v2 .&& img3 .> 0, img3
    end
    knn_based_depth(img1, img2, imgmsk, res)
end

(out::list_nodes_uvw())(
    list::list_neurons(),
    rawinfo::list_nodes_rawinfo(),
    depthmap::Make.@x("interm/depth-mapping-mac.dat"),
    flatmapdat1::flatmap_data(:lefths),
    flatmapdat2::flatmap_data(:leftfrontal),
) = begin
    curated = [
        "234197" => [("monkeytemp/wyz-upload/colored_regions/234197-1-whole-cortex.nrrd", (65.0, 65.0, 60.0))],
        "232717" => [
            ("monkeytemp/wyz-upload/colored_regions/232717-pfc-proj-cortex.nrrd", (65.0, 65.0, 60.0)),
            ("monkeytemp/wyz-upload/colored_regions/232717-1-pfc-proj-cortex.nrrd", (65.0, 65.0, 60.0)),
        ]
    ]
    df = JLD2.load(list.path, "df")
    filter!(r -> !endswith("X")(r.inj), df)
    rawinfo = @time "load rawinfo" nodes_info_factory(rawinfo.path)
    depthimg, depthres = @time "load depthimg" JLD2.load(depthmap.path, "img", "res")
    objflat1, objphy1 = @time "load flatmap" JLD2.load(flatmapdat1.path, "obj", "objphy")
    objflat2, objphy2 = @time "load flatmap" JLD2.load(flatmapdat2.path, "obj", "objphy")

    sampchan(r) = (; r.sample, r.channel)
    sampchans = unique(sampchan.(eachrow(df)))
    calc_depth_nmt(nodes) = begin
        id2xyz = [n.id => n.pos for n in nodes]
        xyzms = xyz2dxyz(last.(id2xyz), depthimg, depthres)
        uvws1 = dxyz2uvw(xyzms, objflat1, objphy1)
        uvws2 = dxyz2uvw(xyzms, objflat2, objphy2)
        map(id2xyz, uvws1, uvws2) do (id, _), uvw1, uvw2
            (; id, uv=SVector(uvw1[1:2]...), depth=uvw1[3], uv2=SVector(uvw2[1:2]...))
        end
    end
    calc_depth_curated(blkinfos) = begin
        blkid(id; id_inc=1_000_000_000) = id ÷ id_inc + 1
        depth_funcs = [knn_based_depth_any(img, res) for (img, res) in blkinfos]
        (nodes, info) -> begin
            id2xyz2 = [n.id => info[n.id].rawxyz for n in nodes]
            blks = unique(blkid.(first.(id2xyz2)))
            blkdata = map(blks) do blk
                id2xyz = filter(==(blk) ∘ blkid ∘ first, id2xyz2)
                xyz = last.(id2xyz)
                map(id2xyz, depth_funcs[blk](xyz)) do (id, _), rawdepth
                    (; id, rawdepth)
                end
            end
            reduce(vcat, blkdata)
        end
    end

    df = @showprogress map(sampchans) do sc
        hits = filter(p -> p.first == sc.sample, curated)
        func = length(hits) == 1 ? calc_depth_curated(first(hits).second) : nothing
        dfx = filter(r -> sampchan(r) == sc, df)
        @info "start $sc"

        @info "load nodes"
        nodes = map(eachrow(dfx)) do r
            swc = Gapr.load_swc(r.path)
            map(swc.nodes) do n
                (; n.id, n.pos)
            end
        end
        nodes = reduce(vcat, nodes) |> sort!
        @info "calc uvw"
        dat = calc_depth_nmt(nodes)
        if func !== nothing
            @info "raw depth"
            dat = map(dat, func(nodes, rawinfo(sc...))) do t1, t2
                @assert t2.id == t1.id
                (; t1..., t2.rawdepth)
            end
        end
        join(sc, "/") => dat
    end
    df = collect(skipmissing(df))
    JLD2.jldsave(out.path; df)
end

(out::output_fig(6))(
    detail::list_neurons_detail(),
    proj::list_neurons_proj(),
    hclust::list_neurons_hclust(),
    flatmapdat::flatmap_data(:leftfrontal),
    topo_ct_md::topo_proj_data(:CT_MD),
    topo_ct_thal::topo_proj_data(:CT_Thal),
    topo_pt_md::topo_proj_data(:PT_MD),
    topo_pt_mid::topo_proj_data(:PT_Mid),
    topo_pt_sc::topo_proj_data(:PT_SC),
    topo_pt_thal::topo_proj_data(:PT_Thal),
    arbors::Make.@x("interm/arbors.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    ###cmpctpt::Make.ByProduct{output_fig(:Sx2)},
    rawinfo::list_nodes_rawinfo(),
    scd17::Make.ByProduct{Make.@x("monkeytemp/sc-layers-fiber-D17.png")},
    scg97::Make.ByProduct{Make.@x("monkeytemp/sc-layers-fiber-G97.png")},
) = begin
    df = JLD2.load(detail.path, "df")
    proj = JLD2.load(proj.path, "df")
    leftjoin!(df, proj, on=:path) |> dropmissing!
    hclust = loadhclust!(df, hclust.path)
    sort!(df, :hclustord)

    dfarbor = JLD2.load(arbors.path, "df")
    trunks = JLD2.load(trunks.path, "df")
    leftjoin!(trunks, dfarbor, on=[:arborpath, :path]) |> dropmissing!

    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))

    topo_ct_md = JLD2.load(topo_ct_md.path, "df")
    topo_ct_thal = JLD2.load(topo_ct_thal.path, "df")
    topo_pt_md = JLD2.load(topo_pt_md.path, "df")
    topo_pt_mid = JLD2.load(topo_pt_mid.path, "df")
    topo_pt_sc = JLD2.load(topo_pt_sc.path, "df")
    topo_pt_thal = JLD2.load(topo_pt_thal.path, "df")
    obj = load("interm/mesh-nmt-brain-lr.obj")

    highlights = zz1data.get_flatmap_highlights()

    dfxpt = filter(zz1data.select_neurons(:pt_strict), df)
    dfxct = filter(zz1data.select_neurons(:ct_strict), df)

    dfxdlpfc = filter(r -> !zz1data.istypit(r) && zz1data.matchhier("dlPFC")(r.somahier), df)

    rowcolor = zz2fig.cls_temp_color(vcat(dfxct, dfxpt))
    rawinfo = nodes_info_factory(rawinfo.path)
    pick_ptsc_neurons(cam, dfxpt) = begin
        scarbors = leftjoin(filter(r -> zz1data.matchhier("SC")(r.arborhier), dfarbor), df, on=:path)
        d = cam.eye .- cam.center
        d = d ./ norm(d)
        nrn2nodes = Dict{String,Float64}()
        for r in eachrow(scarbors)
            swc = Gapr.load_swc(r.arborpath)
            info = rawinfo(r.sample, r.channel)
            n = count(swc.nodes) do r
                abs(dot(info[r.id%100_000_000_000].rawxyz .- cam.center, d)) < 100
            end
            nrn2nodes[r.path] = get(nrn2nodes, r.path, 0) + n
        end
        sort(dfxpt, :path, by=p -> get(nrn2nodes, p, 0), rev=true)
    end
    cam = (;
        center=(28160.2, 28967.2, 35243.1),
        eye=(28128, 28962.2, 34744.2),
        up=(24393.6, 29017.2, 35486),
        zrange=600 / 2,
        xyrange=4800,
    )
    dfxptsc = pick_ptsc_neurons(cam, dfxpt)
    img = draw_hires_sc_fiber((; cam..., zrange=600 / 2), "232717-1", 2, filter(r -> startswith("D17-2")(r.inj), dfxptsc), rawinfo, rowcolor)
    save(scd17.path, rotl90(img))
    cam = (;
        center=(28383.7, 38111, 41963.8),
        eye=(28355.1, 38109.9, 41615),
        up=(24297.9, 38017.2, 42299.5),
        zrange=600 / 2,
        xyrange=4800,
    )
    dfxptsc = pick_ptsc_neurons(cam, dfxpt)
    img = draw_hires_sc_fiber((; cam..., zrange=600 / 2), "234197", 1, filter(r -> startswith("G97-1")(r.inj), dfxptsc), rawinfo, rowcolor)
    save(scg97.path, rotl90(img))

    with_figure(6, out.path, (; trunks, dfarbor, dfxpt, dfxct, dfxdlpfc, obj, objflat, objphy, highlights, topo_pt_mid, topo_pt_thal, topo_ct_md, topo_ct_thal, topo_pt_sc, topo_pt_md)) do fig, env
        plot_fig_ctpt(fig, env; env...)
    end


    # with_figure(:Sx2, cmpctpt.path, (; trunks, dfarbor, dfxpt, dfxct, dfxdlpfc, obj, objflat, objphy, highlights, topo_pt_mid, topo_pt_thal, topo_ct_md, topo_ct_thal, topo_pt_sc, topo_pt_md)) do fig, env
    #     plot_ctpt_cmp(fig, env; env...)
    # end
end

(out::output_fig(:S07))(
    detail::list_neurons_detail(),
    proj::list_neurons_proj(),
    hclust::list_neurons_hclust(),
    arbors::Make.@x("interm/arbors.dat"),
    trunks::Make.@x("interm/arbor-trunk.dat"),
    topo_pt_sc::topo_proj_data(:PT_SC),
    mus::list_mus_neurons_detail(),
    musarbor::Make.@x("interm/mouse-arbors.dat"),
    mus_proj::list_mus_neurons_proj(),
    mus_hclust::list_mus_neurons_hclust(),
    mustopo_ptsc::topo_proj_data_mus(:PT_SC),
    mustopo_ctmd::topo_proj_data_mus(:CT_MD),
) = begin
    df = JLD2.load(detail.path, "df")
    proj = JLD2.load(proj.path, "df")
    leftjoin!(df, proj, on=:path) |> dropmissing!
    hclust = loadhclust!(df, hclust.path)
    sort!(df, :hclustord)
    dfxpt = filter(zz1data.select_neurons(:pt_strict), df)
    dfxct = filter(zz1data.select_neurons(:ct_strict), df)

    topo_pt_sc = JLD2.load(topo_pt_sc.path, "df")

    dfarbor = JLD2.load(arbors.path, "df")
    trunks = JLD2.load(trunks.path, "df")
    leftjoin!(trunks, dfarbor, on=[:arborpath, :path]) |> dropmissing!

    mus = JLD2.load(mus.path, "df")
    musarbor = JLD2.load(musarbor.path, "df")
    mus_proj = JLD2.load(mus_proj.path, "df")
    leftjoin!(mus, mus_proj, on=intersect(names(mus), names(mus_proj))) |> dropmissing!
    insertcols!(mus, :cls => mus.subtyp)
    mus_hclust = loadhclust!(mus, mus_hclust.path)
    sort!(mus, :cls)
    mustopo_ptsc = JLD2.load(mustopo_ptsc.path, "df")
    mustopo_ctmd = JLD2.load(mustopo_ctmd.path, "df")

    obj = load("interm/mesh-nmt-brain-lr.obj")
    # # with_figure(:S10, s10.path, (; mac, macarbor, mus, musarbor, spe2color, musarboregs, mustopo_ctmd, mustopo_ptlhy, mustopo_ptsc, mustopo_ptvta, mustopo_ptpd, mustopo_ptsth, musvol, macvol)) do fig, env #= macvol, musvol, =#
    # # end
    # with_figure(:S10, s10.path, (; trunks, dfarbor, dfxpt, dfxct, dfxdlpfc, obj, objflat, objphy, highlights, topo_pt_mid, topo_pt_thal, topo_ct_md, topo_ct_thal, topo_pt_sc, topo_pt_md)) do fig, env
    # end
    # s10::Make.ByProduct{output_fig(:S10)},
    # s11::Make.ByProduct{output_fig(:S11)},
    with_figure(:S07, out.path, (; dfxpt, dfarbor, trunks, obj, dfxct, topo_pt_sc, mac=df, macarbor=dfarbor, mus, musarbor, mustopo_ptsc, mustopo_ctmd)) do fig, env
        plot_fig_ctpt_supp(fig[1, 1], env; env...)
        #plot_fig_compare_supp(fig[2, 1], env; env...)
    end
end

function calc_arbor_extras(df, dfnrn; segs2uvws=nothing, segs2anns, ann2hier, iscortex, isleft, depthN=100)
    segs = Data.load_swcs_as_segs(df.arborpath)
    uvws = segs2uvws(segs)
    anns = segs2anns(segs)
    path2soma = Dict(zip(dfnrn.path, dfnrn.soma_pos))

    nrninfos = Dict()
    foreach(segs, uvws, anns) do seg, uvw, ann
        nn = get!(() -> [], nrninfos, seg.nrn)
        push!(nn, (; seg.row, seg.len, uvw, ann))
    end

    hist = MVector((0.0 for _ in 1:depthN+1)...)
    ann2len = Dict{Any,Float64}()
    ann2ctx = Dict{Any,Bool}()
    df2 = @showprogress map(enumerate(eachrow(df))) do (i, r)
        hist .= 0
        infos = nrninfos[i]
        sort!(infos, by=r -> r.row)
        @assert [x.row for x in infos] == 1:length(infos)
        uvs1 = [x.uvw.uv for x in infos]
        filter!(x -> !isnan(x[1]), uvs1)
        uvs2 = [x.uvw.uv2 for x in infos]
        filter!(x -> !isnan(x[1]), uvs2)

        empty!(ann2len)
        for x in infos
            ann2len[x.ann] = get(ann2len, x.ann, 0) + x.len
        end
        hiervotes = Dict{String,Float64}()
        cortexlen = 0.0
        for (a, l) in pairs(ann2len)
            h = ann2hier(a)
            if get!(() -> iscortex(a), ann2ctx, a)
                cortexlen += l
            end
            hiervotes[h] = get(hiervotes, h, 0) + l
        end
        hiervotes = sort(collect(hiervotes), by=last, rev=true)
        arborhier = isempty(hiervotes) ? "" : first(hiervotes).first
        cortexrat = cortexlen / (sum(last.(hiervotes)) + 0.001)

        for x in infos
            isnan(x.uvw.depth) && continue
            hist[round(Int, x.uvw.depth * depthN)+1] += x.len
        end
        fv_whist = SVector(hist)

        arboruvcent, f_uvrad = if isempty(uvs1)
            Vec2f(NaN), NaN
        else
            uvs1 = reduce(hcat, uvs1)
            uvc = median(uvs1, dims=2)
            Vec2f(uvc...), quantile(reshape(sqrt.(sum((uvs1 .- uvc) .^ 2, dims=1)), :), 0.75)
        end

        arboruvcent2, f_uvrad2 = if isempty(uvs2)
            Vec2f(NaN), NaN
        else
            uvs2 = reduce(hcat, uvs2)
            uvc = median(uvs2, dims=2)
            Vec2f(uvc...), quantile(reshape(sqrt.(sum((uvs2 .- uvc) .^ 2, dims=1)), :), 0.75)
        end
        ipsi = isleft(r.arborcent) == isleft(path2soma[r.path])
        (; r.arborpath, arborhier, hiervotes, arboruvcent, f_uvrad, arboruvcent2, f_uvrad2, cortexrat, fv_whist, ipsi)
    end
    DataFrame(df2)
end

function transform_feature(x)
    mmd, mmy = (Inf, nothing)
    logx(d) = (x) -> begin
        log.(x .+ d)
    end
    for t in (identity, (logx(exp(i)) for i in -9:5)...,)
        mmm = try
            y = StatsBase.standardize(StatsBase.ZScoreTransform, t(x))
            s = HypothesisTests.ExactOneSampleKSTest(y, Distributions.Gaussian()).δ
            s, y
        catch
            Inf, nothing
        end
        if mmm[1] < mmd
            mmd, mmy = mmm
        end
    end
    mmy
end

function eucdist(x)
    dpdist(x, y) = begin
        norm(x .- y)
    end
    N = size(x, 2)
    m = [dpdist(view(x, :, i), view(x, :, j)) for i in 1:N, j in 1:N]
    (m .+ m') / 2
end

waitclose(fig) = begin
    f = Future()
    Makie.on(v -> put!(f, 0), Makie.events(fig).window_open)
    wait(f)
end

function draw_cls_arbors(arbors, cls)
    for c in sort(unique(cls))
        for r in eachrow(arbors[shuffle(findall(cls .== c))[1:5], :])
            p = replace(r.arborpath, "arbors" => "arbors-predict")
            pts1 = load_swc_as_lines(p)
            pts2 = load_swc_as_lines(r.arborpath)
            f, a = Makie.lines(pts1, color=:gray, axis=(width=800, height=600))
            Makie.lines!(a, pts2, color=:red)
            Makie.Label(f[2, 1], "cls $(c)")
            display(f)
            waitclose(f)
            ##a.title[]=
        end
    end
end

function calc_arbor_type_region(df, dfarbor, t2arbors)
    fig = Makie.Figure()
    display(fig)
    nmt = NmtDat.load_images()
    simplename(t) = replace(t, r"^.*(charm)\|(\w+).*\|" => s"\2|", r"^.*sarm.*\|" => "")
    ax = Makie.LScene(fig[1, 1])
    t2col = Dict()
    obj = load("interm/mesh-nmt-brain-lr.obj")
    Makie.mesh!(ax, obj, color=(:gray, 0.02), transparency=true)
    col = Makie.Palette(:tab10)
    dats = []
    for (iii, (t, ff)) in enumerate(t2arbors)
        ccc = Makie.to_color(col)
        for f in ff
            swc = Gapr.load_swc(f)
            pts = Point3f[]
            for n in swc
                if n.typ == 2 && n.paridx != 0
                    push!(pts, n.pos)
                    push!(pts, swc[n.paridx].pos)
                    push!(pts, Vec3f(NaN))
                end
            end
            #pts=load_swc_as_lines(f)
            Makie.lines!(ax, pts, color=ccc, transparency=true, label="$t")
        end
        continue
        fff = Set(basename.(ff))
        dfx = filter(r -> in(fff)(basename(r.arborpath)), dfarbor)
        leftjoin!(dfx, df, on=:path)
        for r in eachrow(dfx)
            if !isnan(r.somaw)
                push!(dats, (iii, r.somaw))
            end
        end

        # pts = Vec3f[]
        # for f in ff
        #     swc = Gapr.load_swc(f)
        #     for n in swc
        #         if n.typ==2
        #         push!(pts, n.pos)
        #     end
        #     end
        # end
        # hh = NmtDat.getter(nmt)(reduce(hcat, pts))[2]
        # h2c = Dict()
        # for h in hh
        #     h2c[h] = get(h2c, h, 0) + 1
        # end
        # h2c = sort(collect(h2c), by=last)
        # ax, _ = Makie.barplot(fig[iii, 1], axes(h2c, 1), last.(h2c))
        # ax.title[] = t
        # ax.xticks[] = (axes(h2c, 1), simplename.(first.(h2c)))
        # rotatexlabel!(ax)
    end
    # ax,_=Makie.boxplot(fig[1,1], first.(dats), last.(dats))
    # ax.xticks[]=(1:length(t2arbors), first.(t2arbors))
    Makie.axislegend(ax, ax, unique=true)
end

function load_arbor_type_egs(dfarbor, path)
    t2nrns = Dict{String,Vector}()
    for t in readdir(path)
        td = joinpath(path, t)
        if isdir(td)
            for i in readdir(td)
                dfx = filter(r -> contains(string(i))(r.arborpath), dfarbor)

                nrns = get!(() -> String[], t2nrns, t)
                push!(nrns, joinpath(td, i))
                if size(dfx, 1) != 1
                    @warn "$(size(dfx)) $(i)"
                    continue
                end
                push!(nrns, dfx[1, :arborpath])
            end
        end
    end

    # for l in eachline(egspath)
    #     m = match(r"(\d+-\d+-root\d+-\w+-arbor\d+-\w+\.swc)(\.gz)? \w+ (\w+)", l)
    #     @assert m !== nothing
    # end
    sort(collect(t2nrns))
end

function arbor_feature_along(swc, dir)
    dir = dir / norm(dir)
    stats = Dict()
    for n in swc
        n.paridx == 0 && continue
        p = n.pos
        pp = swc[n.paridx].pos
        c = (p + pp) / 2
        d = p - pp
        dn = dot(dir, d)
        l = norm(d)
        w = round(dot(p, dir) / 50)
        vv = get!(() -> [], stats, w)
        push!(vv, (; l, c, dn))
    end
    stats
end

function find_typical_arbors(dfarbor)
    fig = Makie.Figure()
    display(fig)

    feats = names(dfarbor)
    filter!(startswith("f_"), feats)

    t2nrns = load_arbor_type_egs(dfarbor, "monkeytemp/wyz-upload/arbor-check-correct-swc-morph-types")
    Main.xxxt2nrns = t2nrns



    #ax=Makie.LScene(fig[1,1])
    nnn = 1
    for (t, arbors) in t2nrns
        dfx = dfarbor[map(in(Set(arbors)), dfarbor.arborpath), :]
        for r in eachrow(dfx)
            rowcol = divrem(nnn - 1, 8) .+ 1
            aa = Gapr.load_swc(r.arborpath)
            grid = Makie.GridLayout(fig[rowcol...])
            return arbor_feature_along(aa, SVector(0, 1, 0))
            #Makie.hist(grid[1, 1], [n.pos[2] for n in aa], axis=(; title=t))
            # x=reorient_arbor!(aa, r.arborwmap)
            # x===nothing && continue
            # for n in aa
            #     n.pos=n.pos.+(3000nnn,0,0)
            # end
            # nnn+=1
            # pts, col=load_swc_as_lines(n->:black, aa)
            # Makie.lines!(ax, pts, color=rand(Makie.wong_colors()))
            nnn += 1
        end
    end
    return fig

    mat = reduce(hcat, (transform_feature(c) for c in eachcol(dfarbor[:, feats])))
    @info "$(size(mat))"
    dd = 1 .- StatsBase.cor(mat')
    umap = UMAP.umap(dd, 2, metric=:precomputed)

    ax, _ = Makie.scatter(fig[1, 1], umap, color=:gray, markersize=1)
    for (t, arbors) in t2nrns
        Makie.scatter!(ax, umap[:, map(in(Set(arbors)), dfarbor.arborpath)])
    end
end

(out::output_fig(:E11))(
    detail::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    arboruvws::Make.@x("interm/arbors-uvw.dat"),
    ###flatmapdat::flatmap_data(:lefths),
) = begin
    df = JLD2.load(detail.path, "df")
    arbors = JLD2.load(arbors.path, "df")
    arboruvws = JLD2.load(arboruvws.path, "df")
    sample_soma_pos_for_qroot!(df)
    leftjoin!(arbors, df[:, [:path, :soma_pos]], on=:path)
    leftjoin!(arbors, arboruvws, on=:arborpath) |> dropmissing!
    filter!(r -> r.ncharm > 50, arbors)

    with_figure(out.path) do fig
        leftjoin!(arbors, df, on=intersect(names(arbors), names(df))) |> dropmissing!
        plot_row(f, t) = begin
            dfx = filter(f, arbors)
            with_subfig(fig[end+1, 1]) do grid
                ax1 = Makie.Axis(grid[1, 1], title=t, autolimitaspect=1)
                ax2 = Makie.Axis(grid[2, 1])
                #dfx = shuffle(deepcopy(dfx))
                dfx = dfx[1:min(end, 10), :]
                draw_arbor_planar!(ax1, ax2, dfx)
                Makie.rowgap!(grid, 1, 2)
            end
            Makie.rowsize!(fig.layout, size(fig.layout)[1], 80)
        end
        for (j, tgt) in enumerate(("nmt|charm|Frontal|lat_PFC",
            "nmt|charm|Frontal|ACgG",
            "nmt|charm|Frontal|motor",
            "nmt|charm|Frontal|OFC",
            "nmt|charm|Occipital", "nmt|charm|Parietal", "nmt|charm|Temporal",))
            nm = replace(tgt, r".*\|" => "")
            plot_row("lat_PFC -> $nm") do r
                startswith("nmt|charm|Frontal|lat_PFC")(r.somahier) && startswith(tgt)(r.arborhier)
            end
        end
        t2nrns = load_arbor_type_egs(arbors, "monkeytemp/wyz-upload/arbor-check-correct-swc-morph-types")
        for (t, dfxs) in t2nrns
            plot_row("$t") do r
                r.arborpath in dfxs
            end
        end
        Main.xxxt2nrns = t2nrns
        "Example arbors"
    end
end

function calc_population_density_and_entropy(df, dfarbor, tgtinj; mapcoord=nothing)
    df = filter(r -> startswith(tgtinj)(r.inj), df)
    dfarbor = leftjoin(dfarbor, df, on=:path) |> dropmissing!
    segs = Data.load_swcs_as_segs(dfarbor.arborpath)
    arb2nrn = something.(indexin(dfarbor.path, df.path))

    N = size(df, 1)
    accum = Dict{SVector,MVector{N,Float64}}()
    @showprogress "accum" for seg in segs
        pos = mapcoord !== nothing ? mapcoord(seg.pos) : seg.pos
        any(isnan.(pos)) && continue
        idx = NmtDat.pos2idx(pos, (50, 50, 50))
        v = get!(accum, idx) do
            fill!(MVector{N,Float64}(undef), 0)
        end
        i = arb2nrn[seg.nrn]
        v[i] += seg.len
    end

    accum = collect(accum)
    calc_color(v) = begin
        sv = sum(v)
        e = StatsBase.entropy(v ./ sv)
        Makie.RGBf(e, sv, 0)
    end
    siz = (maximum(reduce(hcat, first.(accum)), dims=2)...,)
    img = zeros(Makie.RGBf, siz)
    for (i, c) in accum
        img[i...] = calc_color(c)
    end
    img
end

density_entropy(Inj) = Make.@x("monkeytemp/entropy-density-$Inj.tif")
(out::density_entropy(Inj))(
    detail::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
) where {Inj} = begin
    #  "B47-1cL" => 232
    #"D17-2bL" => 335
    #"G97-1bL" => 447
    df = JLD2.load(detail.path, "df")
    dfarbor = JLD2.load(arbors.path, "df")
    mapcoord(pts) = pts
    img = calc_population_density_and_entropy(df, dfarbor, string(Inj); mapcoord)
    mktemp() do path, io
        save(File{format"TIFF"}(path), img)
        run(`tiffcp -c zip $path $(out.path)`)
    end
end

function arbor_diam(swc)
    maximum(swc) do a
        maximum(swc) do b
            norm(a.pos - b.pos)
        end
    end
end

density_entropy_flat(Inj) = Make.@x("monkeytemp/entropy-density-flat-$Inj.nrrd")
(out::density_entropy_flat(Inj))(
    detail::list_neurons_detail(),
    arbors::Make.@x("interm/arbors.dat"),
    depthmap::Make.@x("interm/depth-mapping-mac.dat"),
    flatmapdat::flatmap_data(:lefths),
) where {Inj} = begin
    df = JLD2.load(detail.path, "df")
    dfarbor = JLD2.load(arbors.path, "df")
    depthimg, depthres = JLD2.load(depthmap.path, "img", "res")
    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    mapcoord(pts) = begin
        return pts
        uvws = xyz2uvw(pts, depthimg, depthres, objflat, objphy)
        [c .* SVector(100000.0f0, 100000.0f0, 1000.0f0) for c in uvws]
    end
    calc_population_density_and_entropy(df, dfarbor, string(Inj); mapcoord)
end

function calc_arbor_dist_spatial(df, dfarbor, tgtsamp)
    pts = SVector[]
    dfarbor = leftjoin(dfarbor, df, on=:path) |> dropmissing!
    filter!(r -> r.sample == tgtsamp, dfarbor)
    insertcols!(dfarbor, :interval => Ref(0:0))
    @showprogress "load swc" for r in eachrow(dfarbor)
        swc = Gapr.load_swc(r.arborpath)
        a = length(pts) + 1
        for n in swc.nodes
            push!(pts, n.pos)
        end
        b = length(pts)
        r.interval = a:b
    end
    pts = reduce(hcat, pts)
    procs = addprocs(13)
    dists = try
        @everywhere procs @eval using ProgressMeter, NearestNeighbors, DataFrames, Statistics, GeometryBasics
        @everywhere procs global pts = $(pts)
        @everywhere procs global dfarbor = $(dfarbor)
        @showprogress "calc dist" pmap(eachrow(dfarbor)) do r
            ####rownumber(r)>100 && return nothing
            tree = NearestNeighbors.KDTree(Main.pts[:, r.interval])
            dd = reduce(vcat, NearestNeighbors.knn(tree, Main.pts, 1)[2])
            map(iv -> exp(median(log.(view(dd, iv)))), Main.dfarbor.interval)
        end
    finally
        rmprocs(procs)
    end
    dists = reduce(hcat, dists)
    dfarbor[:, [:arborpath, :path]], dists
end

(out::arbor_dist_for(samp))(
    detail::list_neurons(),
    arbors::Make.@x("interm/arbors.dat"),
) where {samp} = begin
    df = JLD2.load(detail.path, "df")
    dfarbor = JLD2.load(arbors.path, "df")
    df, distmat = calc_arbor_dist_spatial(df, dfarbor, "$samp")
    JLD2.jldsave(out.path; df, distmat)
end


(out::output_fig(:xx6))(
    flatmapdat::flatmap_data(:lefths),
) = begin





    # # feats = filter(startswith("f_"), names(arbors))
    # # filter!(!endswith("_min"), feats)
    # # filter!(!endswith("_max"), feats)
    # # setdiff!(feats, string.([
    # #     :f_radiusmax
    # #     :f_lm_EucDistance_sd
    # #     :f_lm_PathDistance_sd
    # #     :f_depthmax
    # #     :f_lm_Fragmentation_sd
    # #     :f_lm_Branch_pathlength_sd
    # #     :f_lm_Type_avg
    # #     :f_lm_Type_sd
    # #     :f_lm_N_stems_sum
    # #     :f_lm_EucDistance_sum
    # #     :f_lm_PathDistance_sum
    # #     :f_lm_Terminal_degree_sum
    # #     :f_lm_Branch_Order_sum
    # #     :f_lm_Type_sum
    # #     :f_lm_Fragmentation_sum
    # #     :f_lm_Branch_pathlength_sum
    # #     :f_lm_TerminalSegment_sum
    # #     :f_lm_Bif_ampl_remote_sum
    # #     :f_lm_Bif_ampl_local_sum
    # #     :f_lm_Fractal_Dim_sum
    # #     :f_lm_Pk_sum
    # #     :f_lm_N_branch_sum
    # #     :f_lm_Contraction_sum
    # #     :f_lm_Bif_tilt_local_sum
    # #     :f_lm_Bif_tilt_remote_sum
    # #     :f_lm_Bif_torque_local_sum
    # #     :f_lm_Bif_torque_remote_sum
    # #     :f_lm_Partition_asymmetry_sum
    # #     :f_lm_Branch_Order_avg
    # #     :f_lm_Branch_Order_sd
    # #     :f_lm_Terminal_degree_sd
    # #     :f_lm_Terminal_degree_avg
    # #     :f_lm_Bif_torque_remote_sd
    # #     :f_lm_Bif_torque_local_sd
    # #     :f_lm_Bif_ampl_local_sd
    # #     :f_lm_Bif_tilt_local_sd
    # #     :f_lm_Bif_tilt_remote_sd
    # #     :f_lm_Bif_ampl_remote_sd
    # # ]))
    # # matfeats = reduce(hcat, (transform_feature(x) for x in eachcol(arbors[:, feats])))
    # # distmat2 = 1 .- StatsBase.cor(matfeats')
    # # Main.xxxumap2 = UMAP.umap(distmat2, 2, metric=:precomputed)
    # # hclust2 = Clustering.hclust(distmat2, linkage=:ward, branchorder=:barjoseph)
    # # clusters2 = Clustering.cutree(hclust2, k=2)
    # # reorder_cls!(clusters2, hclust2)
    # # insertcols!(arbors, :arborcls2 => clusters2)
    # #hclust1 = hclust2
    # #
    #
    # Main.xxxarbors = arbors
    # Main.xxxdf = df
    # @info "prep done"
    #
    # arbor_egs = [
    #     "shallow" => [
    #         "232717-1-root1019626624-qroot036-arbor301016378230-7ba3d776479861f5.swc.gz"
    #         "232717-1-root1020808317-root066-arbor301020808307-be95d07bf1835626.swc.gz"
    #     ]
    #     "shallow-patchy" => [
    #         "221747-1-root27793768-root399-arbor100025987379-f864a882b2cf6247.swc.gz"
    #     ]
    # ]

    objflat, objphy = JLD2.load(flatmapdat.path, "obj", "objphy")
    objphy = GeometryBasics.normal_mesh(objphy.position, faces(objphy))


    with_figure(out.path) do fig




        # with_subfig(fig[1, 1][1, 2]) do grid
        #     egs = [
        #         "monkeytemp/split-sample234197-ch1/root53280756-root255-good.swc.gz"
        #         "monkeytemp/split-sample234197-ch1/root52379222-root431-good.swc.gz"
        #         "monkeytemp/split-sample240025-ch1/root932497-root078-good.swc.gz"
        #         "monkeytemp/split-sample234197-ch1/root53280756-root255-good.swc.gz"
        #     ]
        #     for (i, eg) in enumerate(egs)
        #         rowcol = divrem(i - 1, 2) .+ 1
        #         ax = Makie.LScene(grid[rowcol...], show_axis=false)
        #         demostrate_arbor!(ax, arbors, eg)
        #     end
        # end
        # dfx = leftjoin(arbors, df, on=intersect(names(df), names(arbors))) |> dropmissing!
        # with_subfig(fig[3, 1][1, 1]) do grid
        #     hier_of(r) = r.arborhier
        #     name_of(h) = replace(h, r".*\|" => "")
        #     allxvals = sort(unique(hier_of.(eachrow(dfx))))
        #     hier2i = Dict(reverse.(enumerate(allxvals)))
        #     allgrps = sort(unique(dfx.arborcls))
        #     grp2i = Dict(reverse.(enumerate(allgrps)))
        #     cnts = zeros(length(allxvals), length(allgrps))
        #     for r in eachrow(dfx)
        #         cnts[hier2i[hier_of(r)], grp2i[r.arborcls]] += 1
        #     end
        #     cnts = cnts ./ sum(cnts, dims=2)
        #     ax = Makie.Axis(grid[1, 1])
        #     xx = [i[1] for i in CartesianIndices(cnts)][:]
        #     yy = [cnts[i] for i in CartesianIndices(cnts)][:]
        #     gg = [i[2] for i in CartesianIndices(cnts)][:]
        #     colors = Makie.Palette(:tab10).colors
        #     Makie.barplot!(ax, xx, yy, color=colors[gg], stack=gg)
        #     ax.xticks[] = (axes(allxvals, 1), name_of.(allxvals))
        #     rotatexlabel!(ax)
        #
        #     ax = Makie.Axis(grid[1, 2])
        #     draw_pfc_non_pfc_dist!(ax, arbors, df)
        # end
        # Makie.rowsize!(fig.layout, 3, 60)
        # with_subfig(fig[4, 1][1, 1]) do grid
        #     hier_of(r) = (r.arborhier, r.ipsi)
        #     name_of((h, i)) = string(i ? "I" : "C", "-", replace(h, r".*\|" => ""))
        #     allxvals = sort(unique(hier_of.(eachrow(dfx))))
        #     allxvals = filter(h -> count(hh -> hh[1] == h[1], allxvals) > 1, allxvals)
        #     hier2i = Dict(reverse.(enumerate(allxvals)))
        #     allgrps = sort(unique(dfx.arborcls))
        #     grp2i = Dict(reverse.(enumerate(allgrps)))
        #     cnts = zeros(length(allxvals), length(allgrps))
        #     for r in eachrow(dfx)
        #         get(hier2i, hier_of(r), nothing) !== nothing || continue
        #         cnts[hier2i[hier_of(r)], grp2i[r.arborcls]] += 1
        #     end
        #     cnts = cnts ./ sum(cnts, dims=2)
        #     ax = Makie.Axis(grid[1, 1])
        #     xx = [i[1] for i in CartesianIndices(cnts)][:]
        #     yy = [cnts[i] for i in CartesianIndices(cnts)][:]
        #     gg = [i[2] for i in CartesianIndices(cnts)][:]
        #     colors = Makie.Palette(:tab10).colors
        #     Makie.barplot!(ax, xx, yy, color=colors[gg], stack=gg)
        #     ax.xticks[] = (axes(allxvals, 1), name_of.(allxvals))
        #     rotatexlabel!(ax)
        # end
        # Makie.rowsize!(fig.layout, 4, 60)
        LaTeXStrings.L"""
Figure 6: Arbor analysis.
$$"""
    end
end

function log_len_br(x)
    x = insertcols(x, :nbr => 0)
    @info "x"
    @showprogress for r in eachrow(x)
        r.nbr = cached_result(:calcnbr, r.path) do path
            swc = Gapr.load_swc(path)
            swc = Gapr.nodedegree(swc)
            cnt = 0
            for n in swc.nodes
                if Gapr.isbranch(n, n.degree) && n.typ == 2
                    cnt += 1
                end
            end
            cnt
        end
    end
    x
end

function draw_arbors_mus!(ax, dfarbor)
    paths = sort(unique(dfarbor.path))
    for path in paths
        swc = Gapr.load_swc(path)
        for n in swc
            n.typ = 0
        end
        tree = NearestNeighbors.KDTree([n.pos for n in swc.nodes])
        for r in eachrow(dfarbor)
            r.path == path || continue
            swcx = Gapr.load_swc(r.arborpath)
            if !Allen.isleft(swc[1].pos)
                for n in swcx
                    n.pos = Allen.mirror(n.pos)
                end
            end
            nn, nnd = NearestNeighbors.knn(tree, [n.pos for n in swcx], 1)
            pts, cols = load_swc_as_lines(swcx) do n
                :green
            end
            #Makie.lines!(pts, color=cols, linewidth=2)
            foreach(nn, nnd) do (i,), d
                swc[i].typ = rownumber(r)
            end
        end
        t2col = Dict()
        pts, cols = load_swc_as_lines(swc) do n
            n.typ > 0 ? get!(() -> rand(Makie.to_colormap(:tab10)), t2col, n.typ) : :blue
        end
        Makie.lines!(pts, color=cols)
    end
end

function calc_maxdepth(x)
    x = insertcols(x, :maxdepth => 0.0)
    @showprogress for r in eachrow(x)
        swc = Gapr.load_swc(r.path)
        dep = Gapr.nodedepth(swc).depth
        r.maxdepth = maximum(dep)
    end
    x[:, [:path, :maxdepth]]
end

function calc_arborlen(df, dfarbor; fixmouse=false)
    df = insertcols(df, :arborlen => 0.0)
    tasks = Dict([r.path => String[] for r in eachrow(df)])
    for r in eachrow(dfarbor)
        push!(tasks[r.path], r.arborpath)
    end
    tasks = collect(tasks)
    df = let
        @showprogress map(tasks) do (path, arborpaths)
            swc = Gapr.load_swc(path)
            for n in swc.nodes
                n.typ = 0
            end
            for arborpath in arborpaths
                @info "ap $arborpath"
                swcx = Gapr.load_swc(arborpath)
                for n in swcx.nodes
                    nn = swc[!, n.id%100_000_000_000, nothing]
                    if nn !== nothing
                        nn.typ = 3
                    end
                end
            end
            (; path, arborlen=get(Gapr.neurite_length(swc), Symbol(3), 0.0))
        end
    end
    DataFrame(df)
end

function link_mus_arbor_nodes(df; dist_thr=0.1, len_tol=0.91, len_tol2=1.2, outdir="monkeytemp/mouse-single-axon-arbors-fixid")
    fixnodeid(nrn) = begin
        swc = Gapr.load_swc(nrn)
        tree = NearestNeighbors.KDTree([n.pos for n in swc.nodes])
        i2grp = [0 for _ in swc.nodes]
        (arb) -> begin
            if !Allen.isleft(swc[1].pos)
                for n in arb.nodes
                    n.pos = Allen.mirror(n.pos)
                end
            end
            i2grp .= 0
            g2len = Dict{Int,Float64}()
            foreach(NearestNeighbors.inrange(tree, [n.pos for n in arb.nodes], dist_thr)) do ii
                for i in ii
                    i2grp[i] = i
                end
            end
            for n in swc.nodes
                n.parind == 0 && continue
                i2grp[n.ind] == 0 && continue
                g = i2grp[n.parind]
                g == 0 && continue
                i2grp[n.ind] = g
                g2len[g] = get(g2len, g, 0.0) + norm(n.pos - swc[n.parind].pos)
            end
            arblen = 0.0
            for n in arb.nodes
                n.parind == 0 && continue
                arblen += norm(n.pos - arb[n.parind].pos)
            end
            l0, g0 = findmax(g2len)
            rat = (l0 + 1) / (arblen + 10)
            @info "$nrn $rat $l0 $g0 $arblen $g2len"
            @assert len_tol <= rat <= len_tol2 "$rat bad"
            filter(n -> i2grp[n.ind] == g0, swc; toroot=false)
        end
    end
    uniqnames = Set()
    df = ensure_directory(outdir) do _, outdirtmp
        nrns = unique([r.path for r in df])
        @showprogress "fixing arbors" map(nrns) do nrn
            fixer = fixnodeid(nrn)
            map(filter(r -> r.path == nrn, df)) do r
                arb = Gapr.load_swc(r.arborpath)
                arb2 = fixer(arb)
                fn = basename(r.arborpath)
                chk = in!(fn, uniqnames)
                @assert !chk
                Gapr.save_swc(joinpath(outdirtmp, fn), arb2)
                (; r..., arborpath=joinpath(outdir, fn))
            end
        end
    end
    sort(reduce(vcat, df))
end

(out::Make.@x("interm/mouse-arbors.dat"))(
    list::list_mus_neurons(),
) = begin
    dir = "/home/goulf/Desktop/monkeyrec/monkeytemp/wyz-upload/mouse-single-axon-arbors"
    list = JLD2.load(list.path, "df")
    t2p = Dict([mus_neuron_decomp(basename(r.path)) => r.path for r in eachrow(list)])
    df = map(readdir(dir)) do f
        m = match(r"(\d+)_(\d+)_(\d+)\.swc.*$", f)
        @assert m !== nothing
        arborpath = joinpath(dir, f)
        path = get(t2p, (; sample=m.captures[1], name=m.captures[2]), nothing)
        if path === nothing
            @warn "$f missing"
            return missing
        end
        (; arborpath, path, annotator=:liu)
    end
    df = link_mus_arbor_nodes(df |> skipmissing)
    df2 = map(readdir("monkeytemp/wyz-upload/mouse_arbor_recut", join=true)) do dd
        @assert isdir(dd)
        map(readdir(dd)) do f
            m = match(r"^(\d+)_(\d+)_\d+\.swc(\.gz)?$", f)
            @assert m !== nothing
            path = get(t2p, (; sample=m.captures[1], name=m.captures[2]), nothing)
            @assert path !== nothing
            arborpath = joinpath(dd, f)
            (; arborpath, path, annotator=:wang)
        end
    end
    df = DataFrame(skipmissing(vcat(df, reduce(vcat, df2))))
    sort!(df, :arborpath)

    df2 = Data.cached_result(:arbfeat, df) do df
        calc_arbor_feats(df)
    end
    leftjoin!(df, df2, on=:arborpath) |> dropmissing!
    path2soma = Dict(zip(list.path, list.soma_pos))
    for r in eachrow(df)
        if r.annotator === :liu && !Allen.isleft(path2soma[r.path])
            r.center = Allen.mirror(r.center)
        end
    end
    insertcols!(df, :arborcent => df.center)

    depth_func = knn_based_depth_allen()
    mysegs2uvws(segs) = begin
        map(depth_func([n.pos for n in segs])) do depth
            (; uv=Vec2f(NaN32), uv2=Vec2f(NaN32), depth)
        end
    end
    nmt = Allen.load_images()
    mysegs2anns(segs) = begin
        segs2anns(segs, Allen, nmt)
    end
    ann2hier(a) = Allen.ann2hier(nmt, a)
    iscortex(a) = begin
        hier = nmt.hier
        while a != 0
            a == 315 && return true
            a = hier[a].par
        end
        false
    end
    isleft(p) = Allen.isleft(p)
    Npart = 10
    df2s = map(1:Npart) do Ipart
        calc_arbor_extras(df[(1+end*(Ipart-1)÷Npart):end*Ipart÷Npart, :], list; segs2uvws=mysegs2uvws, segs2anns=mysegs2anns, ann2hier, iscortex, isleft)
    end
    leftjoin!(df, reduce(vcat, df2s), on=:arborpath) |> dropmissing!
    JLD2.jldsave(out.path; df)
end

function calc_symmetry(df; spe)
    procs = addprocs(15)
    df = try
        @everywhere procs @eval using ProgressMeter, Gapr, LinearAlgebra
        @everywhere procs @eval using zz0atlas: Allen, NmtDat
        @showprogress pmap(WorkerPool(procs), df.path) do path
            mod = spe === :mus ? Allen : NmtDat
            swc = Gapr.load_swc(path)
            ic0 = mod.isleft(swc[1].pos)
            axonlenC = 0.0
            for n in swc.nodes
                n.paridx == 0 && continue
                if n.typ == 2 && mod.isleft(n.pos) != ic0
                    axonlenC += norm(n.pos - swc[n.paridx].pos)
                end
            end
            (; path, axonlenC)
        end
    finally
        rmprocs(procs)
    end
    DataFrame(df)
end

function calc_term_dist(names; runai=false)
    if runai
        prompt = """Understand the following neuroscience terms, and then group them based on relevance and/or similarity. Output in JSON format like `{"group description": [{index=1, term="term 1"}, {index=7, term="term 7"},...], ...}`, using exactly the same index as my input so that so that spell corrections will not affect parsing.\n{{terms}}"""
        experiments = Data.cached_result(:aiclassify, names) do names
            res = []
            @showprogress for _ in 1:256
                schema = PromptingTools.OllamaSchema()
                names = shuffle(names)
                for i in 1:32:(length(names)-32+1)
                    bat = names[i:i+32-1]
                    r = PromptingTools.aigenerate(schema, prompt, model="deepseek-r1:32b", terms=join(["$i. $s" for (i, s) in enumerate(bat)], "\n"))
                    push!(res, (; terms=collect(bat), response=r.content))
                end
            end
            res
        end
        global xxxexperiments = deepcopy(experiments)
        error("Rerun and load")
    end
    experiments = JLD2.load("interm/aigen.dat", "df")
    checkjson(json, terms) = begin
        !(json isa Dict) && return -1
        wrongterms = Set{Int}()
        for (_, group) in json
            !(group isa AbstractVector) && return -2
            for item in group
                !(item isa AbstractDict) && return -3
                i = get(item, "index", nothing)
                t = get(item, "term", nothing)
                (i === nothing || t === nothing) && return -4
                !(i isa Integer && 1 <= i <= 32) && return -5
                t = uppercasefirst(lowercase(replace(strip(t), r"\s+" => " ")))
                terms[i] != t && push!(wrongterms, i)
            end
        end
        length(wrongterms)
    end
    groupings = map(experiments) do expr
        lines = split(expr.response, "\n")
        aa = findall(==("```json"), lines)
        bb = findall(==("```"), lines)
        !(length(aa) == length(bb) == 1) && return missing
        json = try
            JSON.parse(join(lines[aa[1]+1:bb[1]-1], ""))
        catch
            return missing
        end
        err = checkjson(json, expr.terms)
        !(0 <= err <= 3) && return missing
        [[expr.terms[b["index"]] for b in bb] for (_, bb) in json]
    end
    groupings = collect(skipmissing(groupings))

    cnt1 = Dict{Pair{String,String},Int}()
    cnt2 = Dict{Pair{String,String},Int}()
    for grouping in groupings
        terms = unique(reduce(vcat, grouping))
        for a in terms
            for b in terms
                a == b && continue
                cnt1[a=>b] = get(cnt1, a => b, 0) + 1
            end
        end
        for group in grouping
            for a in group
                for b in group
                    a == b && continue
                    cnt2[a=>b] = get(cnt2, a => b, 0) + 1
                end
            end
        end
    end
    dists = Dict{Pair{String,String},Rational{Int64}}()
    for a in names
        for b in names
            if a == b
                dists[a=>b] = 0
                continue
            end
            c1 = get(cnt1, a => b, 2)
            c2 = get(cnt2, a => b, 1)
            dists[a=>b] = 1 - c2 / c1
        end
    end
    dists, cnt1
end

(out::output_fig(7))(
    mac::list_neurons_detail(),
    macdecomp::list_neurons_decomp(),
    macarbor::Make.@x("interm/arbors.dat"),
    mac_proj::list_neurons_proj(),
    mus::list_mus_neurons_detail(),
    musdecomp::list_mus_neurons_decomp(),
    musarbor::Make.@x("interm/mouse-arbors.dat"),
    mus_proj::list_mus_neurons_proj(),
    mac_hclust::list_neurons_hclust(),
    mus_hclust::list_mus_neurons_hclust(),
    mustopo_ptsc::topo_proj_data_mus(:PT_SC),
    mustopo_ptlhy::topo_proj_data_mus(:PT_LHy),
    mustopo_ptvta::topo_proj_data_mus(:PT_VTA),
    mustopo_ptpd::topo_proj_data_mus(:PT_Pd),
    mustopo_ptsth::topo_proj_data_mus(:PT_STh),
    mustopo_ctmd::topo_proj_data_mus(:CT_MD),
    s08::Make.ByProduct{output_fig(:S08)},
) = begin
    mac = JLD2.load(mac.path, "df")
    macarbor = JLD2.load(macarbor.path, "df")
    mac_proj = JLD2.load(mac_proj.path, "df")
    leftjoin!(mac, mac_proj, on=intersect(names(mac), names(mac_proj))) |> dropmissing!
    mus = JLD2.load(mus.path, "df")
    musarbor = JLD2.load(musarbor.path, "df")
    mus_proj = JLD2.load(mus_proj.path, "df")
    leftjoin!(mus, mus_proj, on=intersect(names(mus), names(mus_proj))) |> dropmissing!
    insertcols!(mus, :cls => mus.subtyp)
    musvol = zz1data.cached_result(:musvol2) do
        allen = Allen.load_images()
        Allen.volume(allen)
    end
    macvol = zz1data.cached_result(:macvol2) do
        nmt = NmtDat.load_images()
        NmtDat.volume(nmt)
    end

    macarborlen = zz1data.cached_result(:macarborlen, mac, macarbor) do mac, macarbor
        calc_arborlen(mac, macarbor)
    end
    leftjoin!(mac, macarborlen, on=:path)
    musarborlen = zz1data.cached_result(:musarborlen, mus, musarbor) do mus, musarbor
        calc_arborlen(mus, musarbor)
    end
    leftjoin!(mus, musarborlen, on=:path)

    mustopo_ptsc = JLD2.load(mustopo_ptsc.path, "df")
    mustopo_ptlhy = JLD2.load(mustopo_ptlhy.path, "df")
    mustopo_ptvta = JLD2.load(mustopo_ptvta.path, "df")
    mustopo_ptpd = JLD2.load(mustopo_ptpd.path, "df")
    mustopo_ptsth = JLD2.load(mustopo_ptsth.path, "df")
    mustopo_ctmd = JLD2.load(mustopo_ctmd.path, "df")

    leftjoin!(mus, zz1data.cached_result(x -> calc_maxdepth(x), :depth1, mus), on=:path)
    leftjoin!(mac, zz1data.cached_result(x -> calc_maxdepth(x), :depth1, mac), on=:path)

    calc_per_arbor_len!(dfx, dfa) = begin
        cnt = Dict([r.path => 0 for r in eachrow(dfx)])
        for r in eachrow(dfa)
            cnt[r.path] += 1
        end
        insertcols!(dfx, :perarborlen => 0.0)
        for r in eachrow(dfx)
            r.perarborlen = r.arborlen / cnt[r.path]
        end
    end
    calc_per_arbor_len!(mac, macarbor)
    calc_per_arbor_len!(mus, musarbor)

    leftjoin!(mac, zz1data.cached_result(x -> calc_symmetry(x; spe=:mac), :contra1, mac), on=:path)
    leftjoin!(mus, zz1data.cached_result(x -> calc_symmetry(x; spe=:mus), :contra1, mus), on=:path)

    mac_hclust = loadhclust!(mac, mac_hclust.path)
    sort!(mac, :hclustord)
    mus_hclust = loadhclust!(mus, mus_hclust.path)
    sort!(mus, :cls)

    leftjoin!(mac, load(macdecomp.path, "df"), on=:path) |> dropmissing!
    leftjoin!(mus, load(musdecomp.path, "df"), on=:path) |> dropmissing!

    spe2color(r::Symbol) = (; mac=:red, mac6=:cyan, mus=:blue)[r]
    spe2color(r) = spe2color(r.spe)
    mac = insertcols(mac, :spe => :mac)
    mus = insertcols(mus, :spe => :mus)

    for df in (mus, mac)
        df2 = zz1data.cached_result(:bn4532, sort(df.path)) do paths
            @showprogress map(paths) do path
                swc = Gapr.load_swc(path)
                swc = Gapr.nodedegree(swc)
                nbranches = 0
                for n in swc.nodes
                    nbranches += Gapr.isbranch(n, n.degree)
                end
                (; path, nbranches)
            end
        end
        leftjoin!(df, DataFrame(df2), on=:path) |> dropmissing!
    end
    # df2 = zz1data.cached_result(:bn4532, sort(df.path)) do paths
    #     @showprogress map(paths) do path
    #         swc = Gapr.load_swc(path)
    #         deg = Gapr.nodedegree(swc)
    #         nbranches = 0
    #         for (i, n) in enumerate(swc)
    #             nbranches += Gapr.isbranch(n, deg[i])
    #         end
    #         (; path, nbranches)
    #     end
    # end
    # leftjoin!(df, DataFrame(df2), on=:path)

    musarboregs = filter(r -> r.annotator === :wang, musarbor)
    musarbor = filter(r -> r.annotator !== :wang, musarbor)

    # macsymdat = zz1data.cached_result(:lencontraxx, mac, filter(r -> false, macarbor)) do mac, macarbor
    #     zz1data.calc_ipsicontra_len(mac, macarbor)
    # end
    # leftjoin!(mac, macsymdat, on=:path) |> dropmissing!
    # mussymdat = zz1data.cached_result(:lencontraxx, mus, filter(r -> false, musarboregs)) do mus, musarbor
    #     zz1data.calc_ipsicontra_len(mus, musarbor; Atlas=Allen)
    # end
    # leftjoin!(mus, mussymdat, on=:path) |> dropmissing!

    with_figure(:S08, s08.path, (; mac, macarbor, mus, musarbor, spe2color, musarboregs, mustopo_ctmd, mustopo_ptlhy, mustopo_ptsc, mustopo_ptvta, mustopo_ptpd, mustopo_ptsth, musvol, macvol)) do fig, env #= macvol, musvol, =#
        plot_fig_compare_supp2(fig, env; env...)
    end

    with_figure(7, out.path, (; mac, macarbor, mus, musarbor, spe2color, musarboregs, mustopo_ctmd, mustopo_ptlhy, mustopo_ptsc, mustopo_ptvta, mustopo_ptpd, mustopo_ptsth, musvol, macvol)) do fig, env #= macvol, musvol, =#
        plot_fig_compare(fig, env; env...)
    end
end

(out::output_fig(:Abs))(
    neurons::list_neurons_detail(),
    ###proj::list_neurons_proj(),
    hclust::list_neurons_hclust(),
    arbors::Make.@x("interm/arbors.dat"),
) = begin
    df = JLD2.load(neurons.path, "df")
    arbors = JLD2.load(arbors.path, "df")
    hclust = loadhclust!(df, hclust.path)
    sort!(df, :hclustord)
    obj = load("interm/mesh-nmt-brain-lr.obj")
    with_figure(:Abs, out.path, (; df, hclust, arbors, obj)) do fig, env #= macvol, musvol, =#
        fig.layout.alignmode[] = Makie.Outside(1, 1, 1, 1)
        zz2fig.plot_fig_abstract(Makie.GridLayout(fig[1, 1]), env; env...)
    end
end

(out::output_fig(:X71))(
    mus::list_mus_neurons_detail(),
    musproj::list_mus_neurons_proj(),
    mushclust::list_mus_neurons_hclust(),
    mac::list_neurons_detail(),
    macproj::list_neurons_proj(),
    machclust::list_neurons_hclust(),
    mustopo_ptsc::topo_proj_data_mus(:PT_SC),
    mustopo_ptlhy::topo_proj_data_mus(:PT_LHy),
    mustopo_ptvta::topo_proj_data_mus(:PT_VTA),
    mustopo_ptpd::topo_proj_data_mus(:PT_Pd),
    mustopo_ptsth::topo_proj_data_mus(:PT_STh),
) = begin
    mus = JLD2.load(mus.path, "df")
    musproj = JLD2.load(musproj.path, "df")
    leftjoin!(mus, musproj, on=:path) |> dropmissing!
    loadhclust!(mus, mushclust.path)
    sort!(mus, :hclustord)

    mac = JLD2.load(mac.path, "df")
    macproj = JLD2.load(macproj.path, "df")
    leftjoin!(mac, macproj, on=:path) |> dropmissing!
    loadhclust!(mac, machclust.path)
    sort!(mac, :hclustord)

    mustopo_ptsc = JLD2.load(mustopo_ptsc.path, "df")
    mustopo_ptlhy = JLD2.load(mustopo_ptlhy.path, "df")
    mustopo_ptvta = JLD2.load(mustopo_ptvta.path, "df")
    mustopo_ptpd = JLD2.load(mustopo_ptpd.path, "df")
    mustopo_ptsth = JLD2.load(mustopo_ptsth.path, "df")

    Fig.with_figure(:X71, out.path, (; mac, mus, mustopo_ptlhy, mustopo_ptsc, mustopo_ptvta, mustopo_ptpd, mustopo_ptsth)) do fig, env
        Fig.plot_mus_pt_bundle(Makie.GridLayout(fig[1, 1]), env; env...)
    end
end

function map_mac_mus_proj(; mac, macdist, mus, musdist, _...)
    prep_feats(df) = begin
        ka = filter(Data.is_projcol, names(df))
        @info "$(length(ka))"
        feat = zeros(Float32, ceil(Int, length(ka) / 512) * 512, size(df, 1))
        feat[1:length(ka), :] = Matrix(df[:, ka])', ka
    end

    featmac, bamac = prep_feats(mac)
    featmus, bamus = prep_feats(mus)
end

(out::output_fig(:X79))(
    mac::list_neurons_detail(),
    macproj::list_neurons_proj(),
    macdist::list_neurons_dist(),
    mus::list_mus_neurons_detail(),
    musproj::list_mus_neurons_proj(),
    musdist::list_mus_neurons_dist(),
) = begin
    mus = JLD2.load(mus.path, "df")
    musproj = JLD2.load(musproj.path, "df")
    leftjoin!(mus, musproj, on=:path) |> dropmissing!
    musdist = load_dist_mat(mus, musdist.path; on=:path)

    mac = JLD2.load(mac.path, "df")
    macproj = JLD2.load(macproj.path, "df")
    leftjoin!(mac, macproj, on=:path) |> dropmissing!
    macdist = load_dist_mat(mac, macdist.path; on=:path)

    Fig.with_figure(:X79, out.path, (; mac, mus, musdist, macdist)) do fig, env
        map_mac_mus_proj(; env...)
        #     Fig.plot_mus_pt_bundle(Makie.GridLayout(fig[1,1]), env; env...)
    end
end

function calc_projection_probs(df, somasubreg, projsubreg, dfsubreg)
    df = leftjoin(df, somasubreg, on=:path) |> dropmissing!
    subreg2vol = map(eachrow(dfsubreg)) do r
        (r.hier, r.subreg) => r.size / 1e9
    end |> Dict
    proj_str(r) = begin
        v = r.projlen / subreg2vol[(r.hier, r.subreg)]
        @assert v >= 0
        ceil(Int, log10(v + 1) - 1e-6)
    end
    src_of(r) = (r.somahier, r.somasubreg)
    tgt_of(r) = (r.ipsi, r.hier, r.subreg)
    allprojs = map(eachrow(projsubreg)) do r
        t = tgt_of(r)
        (r.path => t) => proj_str(r)
    end |> Dict
    allsrcs = map(src_of, eachrow(df))
    projsubreg = leftjoin(projsubreg, df, on=:path) |> dropmissing!
    maxstr = maximum(last.(collect(allprojs)))
    allsrcs = sort(unique(allsrcs))
    alltgts = sort(unique(last.(first.(collect(allprojs)))))
    src2i = Dict(reverse.(enumerate(allsrcs)))
    projdist = zeros(Float64, length(allsrcs), length(alltgts), 1 + maxstr)
    for r in eachrow(df)
        f = src_of(r)
        si = src2i[f]
        for (ti, t) in enumerate(alltgts)
            v = get(allprojs, r.path => t, 0)
            projdist[si, ti, v+1] += 1
        end
    end
    @warn "length $(size(df, 1)) $maxstr"
    projdist, allsrcs => alltgts
end

(out::topo_proj_data(tag))(
    list::list_neurons_detail(),
    proj::list_neurons_proj(),
) where {tag} = begin
    df = JLD2.load(list.path, "df")
    sample_soma_pos_for_qroot!(df)
    pars = (;
        CT_Thal=(;
            filter=:ct_strict,
            tgt="nmt|sarm|di|Thal",
            ipsi=nothing,
        ),
        CT_MD=(;
            filter=:ct_strict,
            tgt="nmt|sarm|di|Thal|MThal|MD",
            ipsi=true,
        ),
        PT_Thal=(;
            filter=:pt_strict,
            tgt="nmt|sarm|di|Thal",
            ipsi=nothing,
        ),
        PT_Mid=(;
            filter=:pt_strict,
            tgt="nmt|sarm|mes|Mid",
            ipsi=nothing,
        ),
        PT_SC=(;
            filter=:pt_strict,
            tgt="nmt|sarm|mes|Mid|DMid|Co|SC",
            ipsi=true,
        ),
        PT_MD=(;
            filter=:pt_strict,
            tgt="nmt|sarm|di|Thal|MThal|MD",
            ipsi=true,
        ),
        ITc_ctx=(;
            filter=:itc_strict,
            tgt="nmt|charm",
            ipsi=false,
        ),
        ITs_Str=(;
            filter=:its_ipsi_strict,
            tgt="nmt|sarm|tel|BG|Str",
            ipsi=true,
        ),
        # PT_Str=(;
        #     filter=:pt_strict,
        #     tgt="nmt|sarm|tel|BG|Str",
        #     ipsi=true,
        # ),
        # CT_Str=(;
        #     filter=:ct_strict,
        #     tgt="nmt|sarm|tel|BG|Str",
        #     ipsi=true,
        # ),
        ITi_TEO=(;
            filter=:iti_teo_strict,
            tgt="nmt|charm|Temporal|ITC|TEO",
            ipsi=true,
        ),
        ITi_ctx=(;
            typ="ITi",
            tgt="nmt|charm",
            ipsi=true,
        ),
        long_ctx_ipsi=(;
            filter=:long_it_ipsi,
            tgt=(h) -> startswith("nmt|charm|Parietal")(h) || startswith("nmt|charm|Temporal")(h),
            ipsi=true,
        )
    )[tag]

    filter!(Data.select_neurons(pars.filter), df)
    df2 = calc_proj_center(df, (pars.tgt, pars.ipsi), dbgpfx=tag)
    global xxxdf = deepcopy(df)
    global xxxdf2 = deepcopy(df2)
    leftjoin!(df, df2, on=intersect(names(df), names(df2))) |> dropmissing!
    sort!(df, :nnodes)
    filter!(r -> !isnan(r.proj_pos[1]), df)
    filter!(r -> r.nnodes > 500, df)

    insertcols!(df, :srcpos => df.soma_pos, :tgtpos => df.proj_pos, :tgtlen => df.projlen)
    for r in eachrow(df)
        if !NmtDat.isleft(r.soma_pos)
            r.srcpos = NmtDat.mirror(r.srcpos)
            r.tgtpos = NmtDat.mirror(r.tgtpos)
        end
    end
    df = df[:, [:path, :srcpos, :tgtpos, :tgtlen]]
    JLD2.jldsave(out.path; df)
end

(out::topo_proj_data_mus(tag))(
    list::list_mus_neurons_detail(),
    proj::list_mus_neurons_proj(),
) where {tag} = begin
    df = JLD2.load(list.path, "df")
    pars = (;
        PT_SC=(;
            filter=:pt_mus,
            tgt="allen|root|grey|BS|MB|MBmot|SCm",
            ipsi=true,
        ),
        PT_LHy=(;
            filter=:pt_mus,
            tgt="allen|root|grey|BS|IB|HY|LZ|LHA",
            ipsi=true,
        ),
        PT_VTA=(;
            filter=:pt_mus,
            tgt="allen|root|grey|BS|MB|MBmot|VTA",
            ipsi=true,
        ),
        PT_Pd=(;
            filter=:pt_mus,
            tgt="allen|root|grey|CH|CNU|PAL",
            ipsi=true,
        ),
        PT_STh=(;
            filter=:pt_mus,
            tgt="allen|root|grey|BS|IB|HY|LZ|STN",
            ipsi=true,
        ),
        CT_MD=(;
            filter=:ct_mus,
            tgt="allen|root|grey|BS|IB|TH|DORpm|MED|MD",
            ipsi=true,
        ),
    )[tag]

    filter!(Data.select_neurons(pars.filter), df)
    df2 = calc_proj_center(df, (pars.tgt, pars.ipsi), mod=Allen)
    global xxxdf = deepcopy(df)
    global xxxdf2 = deepcopy(df2)
    leftjoin!(df, df2, on=intersect(names(df), names(df2))) |> dropmissing!
    sort!(df, :nnodes)
    filter!(r -> !isnan(r.proj_pos[1]), df)
    filter!(r -> r.nnodes > 500, df)

    insertcols!(df, :srcpos => df.soma_pos, :tgtpos => df.proj_pos, :tgtnid => df.proj_nid, :tgtlen => df.projlen)
    for r in eachrow(df)
        if !Allen.isleft(r.soma_pos)
            r.srcpos = Allen.mirror(r.srcpos)
            r.tgtpos = Allen.mirror(r.tgtpos)
        end
    end
    df = df[:, [:path, :srcpos, :tgtpos, :tgtlen, :tgtnid]]
    JLD2.jldsave(out.path; df)
end

function find_arbor_root(swc, sel, degrees; backtrace=6000.0, dbgfn=nothing)::Vector
    msk1 = falses(length(swc))
    msk2 = falses(length(swc))
    for s in sel
        ll = 0.0
        ii = s
        while ll < backtrace
            msk1[ii] = true
            pp = swc[ii].paridx
            pp == 0 && break
            ll += norm(swc[ii].pos .- swc[pp].pos)
            ii = pp
        end
    end
    firsti = nothing
    for (i, n) in enumerate(swc)
        n.paridx == 0 && continue
        if msk1[n.paridx]
            msk1[i] = true
        end
        if firsti === nothing && msk1[i] && degrees[i] != 2
            firsti = i
        end
    end
    firsti === nothing && return []
    msk2[firsti] = true
    for (i, n) in enumerate(swc)
        n.paridx == 0 && continue
        if msk2[n.paridx]
            msk2[i] = true
        end
    end
    if !isnothing(dbgfn)
        outfn = joinpath("monkeytemp/find-topo-arbors", dbgfn)
        for (i, n) in enumerate(swc)
            n.typ = msk2[i] ? 2 : 0
        end
        Gapr.save_swc(outfn, swc)
    end
    findall(msk2)
end

function calc_proj_center(df, (tgt, ipsi); dbgpfx=nothing, mod=NmtDat)
    nmt = mod.load_images()
    getter = mod.getter(nmt)
    procs = addprocs(5)
    checkipsi = if ipsi === nothing
        (sl, tl) -> true
    elseif ipsi
        (sl, tl) -> sl == tl
    else
        (sl, tl) -> sl == !tl
    end
    try
        @everywhere procs @eval using ProgressMeter, Gapr, StaticArrays
        map(eachrow(df)) do r
            swc = Gapr.load_swc(r.path)
            depth = Gapr.nodedepth(swc)
            maxdepth = maximum(depth)
            lr = mod.isleft(r.soma_pos)
            nodes = map(enumerate(swc)) do (i, n)
                c = depth[i] > maxdepth / 3 && checkipsi(lr, mod.isleft(n.pos))
                i, c
            end
            nodes = filter(last, nodes)
            ann2nodes = Dict()
            map(nodes) do (i, c)
                ii = mod.pos2idx(swc[i].pos, nmt.res)
                ann = mod.idx2ann(nmt, ii)
                nn = get!(() -> Int[], ann2nodes, ann)
                push!(nn, i)
            end
            ann2nodes = collect(ann2nodes)
            ###@info "xxx $ann2nodes"
            grppos = reduce(hcat, map(i -> swc[i].pos, first.(last.(ann2nodes))))
            hitgrps = filter(collect(zip(ann2nodes, getter(grppos)[2]))) do (a2nn, h)
                if tgt isa AbstractString
                    match(Regex("^\\Q$tgt\\E(|\\|.*)\$"), h) !== nothing
                else
                    tgt(h)
                end
            end
            hitnodes = reduce(vcat, last.(first.(hitgrps)), init=Int[])
            nnodes = length(hitnodes)
            projlen = 0.0
            for i in hitnodes
                n = swc[i]
                n.paridx == 0 && continue
                projlen += norm(n.pos - swc[n.paridx].pos)
            end
            if dbgpfx !== nothing
                fn = joinpath("monkeytemp/find-topo-arbors", string(dbgpfx, "-", neuron_basename(r), ".swc.gz"))
                for n in swc
                    n.typ = 0
                end
                for i in hitnodes
                    swc[i].typ = 2
                end
                Gapr.save_swc(fn, swc)
            end
            proj_pos = if isempty(hitnodes)
                SVector{3,Float64}(NaN, NaN, NaN)
            else
                SVector{3,Float64}(median(reduce(hcat, map(i -> swc[i].pos, hitnodes)), dims=2)...)
            end
            proj_nid = argmin(n -> norm(n.pos - proj_pos), [n for n in swc]).id
            (; path=r.path, nnodes, proj_pos, projlen, proj_nid)
        end
    finally
        rmprocs(procs)
    end |> DataFrame
end


(out::Make.@x("interm/reduced-swc.dat"))(
    list::list_neurons(),
    nodes_info::list_nodes_rawinfo(),
) = begin
    df = JLD2.load(list.path, "df")
    filter!(r -> !endswith("X")(r.inj), df)
    procs = addprocs(15)
    swcs = try
        func = generate_dendrogram_factory()
        @everywhere procs @eval using ProgressMeter, Gapr, DirectQhull, DataFrames, LinearAlgebra, Random, Clustering, StaticArrays
        @showprogress pmap(WorkerPool(procs), df.path) do path
            swc = Gapr.load_swc(path)
            dend = func(swc)
            (; path, dend)
        end
    finally
        rmprocs(procs)
    end
    leftjoin!(df, DataFrame(swcs), on=:path)

    nodes_info = nodes_info_factory(nodes_info.path)
    for rr in eachrow(df)
        rawinfo = nodes_info(rr.sample, rr.channel)
        for r in eachrow(rr.dend)
            r.intensity1 = rawinfo[r.rootid].rawavg
            r.intensity2 = mean(map(i -> rawinfo[i].rawavg, r.nodeids))
        end
    end
    df = df[:, [:path, :dend]]
    JLD2.jldsave(out.path; df)
end

function prob_hyper_params_count_errs(dfx, dfxx)
    nerr = 0
    foreach(eachrow(dfx), eachrow(dfxx)) do r1, r2
        foreach(eachrow(r1.dend), eachrow(r2.dend)) do rr1, rr2
            @assert rr2.tgt != 0
            if rr1.tgt != 0 && rr1.tgt != rr2.tgt
                nerr += 1
            end
        end
    end
    nerr
end
function prob_hyper_params(df)
    df = filter(r -> !isempty(setdiff(r.dend.tgt, 0)), df) |> deepcopy
    shuffle!(df)
    featsiz = length(df.dend[1].feat[1])

    for gnnconv in [
        GraphNeuralNetworks.CGConv
        GraphNeuralNetworks.GATv2Conv
        #GraphNeuralNetworks.TransformerConv
        GraphNeuralNetworks.NNConv
        #GraphNeuralNetworks.GMMConv
        #GraphNeuralNetworks.GATConv
    ]
        for xxsiz in [8, 12, 16]
            for batsiz in [12], celsiz in [xxsiz * 2], bnsiz in [xxsiz], encsiz in [xxsiz * 2], encdecdepth in [2, 3], nrecurr in [2, 4]
                pars = (; gnnconv, batsiz, celsiz, bnsiz, encsiz, encdecdepth, nrecurr)
                pars_str = join((join(v, "=") for v in pairs(pars)), ", ")
                @info "test $pars_str"
                res = try
                    mytrain, mypredict = rnn_factory(featsiz; pars...)
                    dfx = deepcopy(df)
                    dfx1 = dfx[1:end*9÷10, :]
                    model = mytrain(dfx1)
                    dfxx = mypredict(deepcopy(dfx1), model)
                    train_err = prob_hyper_params_count_errs(dfx1, dfxx)
                    dfxx = mypredict(deepcopy(dfx), model)
                    train_err, prob_hyper_params_count_errs(dfx, dfxx)
                catch e
                    (e isa InterruptException) && rethrow(e)
                    "error $e"[1:min(end, 200)]
                end
                open("/tmp/gnn-probe.txt", "a") do io
                    println(io, pars_str, ": ", res)
                end
                GC.gc()
            end
        end
    end
end

(out::Make.@x("interm/arbor-gnn.dat"))(
    list::list_neurons(),
    swcs::Make.@x("interm/reduced-swc.dat"),
) = begin
    df = JLD2.load(list.path, "df")
    filter!(r -> !endswith("X")(r.inj), df)
    swcs = JLD2.load(swcs.path, "df")
    leftjoin!(df, swcs, on=:path) |> dropmissing!

    let
        depths = []
        for r in eachrow(df)
            depth = zeros(size(r.dend, 1))
            for rr in reverse(eachrow(r.dend))
                i = rownumber(rr)
                if rr.paridx == 0
                    depth[i] = 0
                else
                    depth[i] = depth[rr.paridx] + 1
                end
            end
            mdep = maximum(depth)
            push!(depths, (mdep, r.path))
        end
        sort!(depths)
        for (mdep, p) in depths
            @info "maxdepth $mdep $p ..."
        end
    end

    all_nodes = Set{Int}()
    nodespace(r) = (r.sample, r.channel)
    nodespaces = sort(unique(map(nodespace, eachrow(df))))
    @info "$nodespaces"
    nnn = 100_000_000_000
    idoffset(ns) = findfirst(==(ns), nodespaces) * nnn
    insertcols!(df, :nodes => similar(axes(df, 1), Gapr.SwcTable))
    @showprogress "load swcs" for r in eachrow(df)
        kkk = idoffset(nodespace(r))
        swc = Gapr.load_swc(r.path; idoffset=kkk)
        for rr in swc.nodes
            @assert rr.id < nnn + kkk
            push!(all_nodes, rr.id)
        end
        r.nodes = swc
        for rr in eachrow(r.dend)
            rr.rootid += kkk
        end
    end
    node2type = Dict()

    add_arbor_eg(ns, n) = begin
        v = sign(n)
        kkk = idoffset(ns)
        n = abs(n) % nnn + kkk
        # @assert n in all_nodes "$n oob"
        prevv = get!(node2type, n, v)
        @assert prevv == v "$ns: $prevv $v"
    end
    egs_dir = "monkeytemp/wyz-upload/arbor-check-record"
    for egs_file in readdir(egs_dir)
        m = match(r"^(\d+)-(\d+).txt$", egs_file)
        m === nothing && continue
        @info "loading $egs_file"
        ns = string.((m.captures[1:2]...,))
        for l in lstrip.(eachline(joinpath(egs_dir, egs_file)))
            (isempty(l) || startswith("#")(l)) && continue
            n = parse(Int, l)
            add_arbor_eg(ns, n)
        end
    end
    swc_egs_dir = "monkeytemp/wyz-upload/arbor-check-correct-swc"
    for swc_file in readdir(swc_egs_dir)
        m = match(r"^(\d+)-(\d+)-root\d+-q?root\d+-arbor(\d+)-\w+.swc(.gz)?$", swc_file)
        m === nothing && continue
        ns = string.((m.captures[1:2]...,))
        n1 = parse(Int, m.captures[3])
        n2, nested = nothing, false
        swc = Gapr.load_swc(joinpath(swc_egs_dir, swc_file))
        for n in swc
            if n2 === nothing && n.typ != 0
                n2 = n.id
            end
            if n.typ == 0 && n.paridx != 0 && swc[n.paridx].typ != 0
                nested = true
            end
        end
        @assert n1 == n2
        if nested
            @warn "nested pos eg $swc_file"
        else
            add_arbor_eg(ns, n2)
        end
    end

    insertcols!(df, :example => 0)
    for r in eachrow(df)
        r.example = count(eachrow(r.dend)) do rr
            get(node2type, rr.root, nothing) !== nothing
        end
    end

    for r in eachrow(df)
        r.dend = rnn_prepare_xyvals(r.dend, node2type)
    end

    #prob_hyper_params(df)

    featsiz = length(df.dend[1].feat[1])
    rnn_pars = (; gnnconv=GraphNeuralNetworks.NNConv, batsiz=12, celsiz=32, bnsiz=16, encsiz=32, encdecdepth=3, nrecurr=4)
    mytrain, mypredict = rnn_factory(featsiz; rnn_pars...)

    model = let dfx = df
        dfx = deepcopy(filter(r -> !isempty(setdiff(r.dend.tgt, 0)), dfx))
        JLD2.jldsave("interm/arbor-gnn-train.dat"; df=dfx[:, Not(:nodes)])
        ensure_directory("monkeytemp/arbors-train") do debugdir, debugdirtmp
            for r in eachrow(dfx)
                arbors = segment_arbors(r.nodes, r.dend)
                swc2 = deepcopy(r.nodes)
                highlight_arbor!(swc2, arbors...)
                Gapr.save_swc(joinpath(debugdirtmp, neuron_basename(r) * ".swc.gz"), swc2)
            end
        end
        model = mytrain(dfx)
        dfxx = mypredict(deepcopy(dfx), model)
        train_err = prob_hyper_params_count_errs(dfx, dfxx)
        @warn "train_err: $train_err"
        model
    end
    df = mypredict(df, model)
    JLD2.jldsave("interm/arbor-gnn-pred.dat"; df=df[:, Not(:nodes)])

    df2 = ensure_directory("monkeytemp/arbors") do outdir, outtmp
        ensure_directory("monkeytemp/arbors-predict") do debugdir, debugdirtmp
            @showprogress "save" map(eachrow(df)) do r
                fnbase = neuron_basename(r)
                arbors = segment_arbors(r.nodes, r.dend)
                swc2 = deepcopy(r.nodes)
                highlight_arbor!(swc2, arbors...)
                Gapr.save_swc(joinpath(debugdirtmp, fnbase * ".swc.gz"), swc2)
                map(arbors) do a
                    fn = string(fnbase, '-', arbor_name(a), ".swc.gz")
                    Gapr.save_swc(joinpath(outtmp, fn), a)
                    (; r.path, arborpath=joinpath(outdir, fn))
                end
            end
        end
    end
    df = reduce(vcat, df2) |> DataFrame
    JLD2.@save out.path df
end

arbor_features_factory(; qhull=qhull_factory()) = (swc) -> begin
    pts = map(n -> n.pos, swc.nodes)
    push!(pts, swc[1].pos + SVector(0.001, 0, 0))
    push!(pts, swc[1].pos + SVector(0, 0.001, 0))
    push!(pts, swc[1].pos + SVector(0, 0, 0.001))
    qh = qhull.init()
    pts = reduce(hcat, pts)
    f_volume = qhull.run(qh, pts)[1]
    qhull.free(qh)
    swc = Gapr.nodedepth(swc)
    swc = Gapr.nodedegree(swc)
    f_depthmax, f_depth75 = quantile(last.(collect(swc.depth)), (1, 0.75))
    f_numterms = 0
    f_numbranches = 0
    f_totlen = 0.0
    foreach(swc.nodes) do n
        d = n.degree
        if n.parind != 0
            if d == 1
                f_numterms += 1
            elseif d == 2
            else
                f_numbranches += 1
            end
            pn = swc[n.parind]
            f_totlen += norm(pn.pos .- n.pos)
        end
    end
    f_density = f_totlen / f_volume
    center = SVector(median(pts, dims=2)...,)
    dists = map(swc.nodes) do n
        norm(n.pos .- center)
    end
    f_radiusmax, f_radius75 = quantile(dists, (1, 0.75))
    # **translate xyz
    # **rotate xyz
    # ???scale xyz
    (; f_volume, f_depthmax, f_depth75, f_numterms, f_numbranches, f_totlen, f_density, center, f_radiusmax, f_radius75)
end

function remove_redundant_reatures(df)
    feats = filter(startswith("f_"), names(df))
    torm = filter(feats) do f
        length(unique(df[:, f])) == 1
    end
    setdiff!(feats, torm)
    corr = [StatsBase.cor(df[:, a], df[:, b]) for a in feats, b in feats]
    for (f1i, f1) in enumerate(feats), (f2i, f2) in enumerate(feats)
        c = corr[f1i, f2i]
        if f1i < f2i && (abs(c - 1) < 1e-9 || abs(c + 1) < 1e-9)
            @warn "$f1 <-> $f2 $c"
            push!(torm, f1)
        end
    end
    if isempty(unique(torm))
        df[:, :]
    else
        df[:, Not(unique(torm)...)]
    end
end

function calc_arbor_feats_impl()
end

function calc_dendrite_feats(df)
    arbor_features = arbor_features_factory()
    arbor_features_lm = get_arbor_feature_lm_factory()

    df2 = @showprogress map(eachrow(df)) do row
        swc = Gapr.load_swc(row.path)
        ff = arbor_features(swc)
        swc4 = filter(swc) do n
            n.type == 4
        end
        swc0 = filter(swc) do n
            n.type == 0
        end
        lm = map([:swc4 => swc4, :swc0 => swc0]) do (sn, s)
            lm0 = try
                arbor_features_lm(Gapr.ignore(s, annots=true, loops=true))
            catch e
                @error "error calc $(row.path)"
                rethrow(e)
            end
            map(eachrow(lm0)) do r
                n = r.lm
                map([:sum, :nnc, :nnd, :min, :avg, :max, :sd,]) do f
                    fn = Symbol(sn, "_f_lm_", n, "_", f)
                    fn => r[f]
                end
            end
        end
        lm = reduce(vcat, reduce(vcat, reduce(vcat, lm)))
        (; row.path, ff..., NamedTuple(lm)...)
    end
    leftjoin(df, DataFrame(df2), on=:path)
end

function calc_arbor_feats(df)
    procs = addprocs(13)
    df2 = try
        arbor_features = arbor_features_factory()
        arbor_features_lm = get_arbor_feature_lm_factory()
        @everywhere procs @eval using ProgressMeter, Gapr, DirectQhull, StaticArrays, LinearAlgebra, Statistics, DataFrames
        @showprogress pmap(WorkerPool(procs), df.arborpath) do arborpath
            swc = Gapr.load_swc(arborpath)
            lm = try
                arbor_features_lm(swc)
            catch e
                @error "error calc $arborpath"
                rethrow(e)
            end
            lm = map(eachrow(lm)) do r
                n = r.lm
                map([:sum, :nnc, :nnd, :min, :avg, :max, :sd,]) do f
                    fn = Symbol("f_lm_", n, "_", f)
                    fn => r[f]
                end
            end
            lm = NamedTuple(reduce(vcat, lm))
            (; arborpath, arbor_features(swc)..., lm...)
        end
    finally
        rmprocs(procs)
    end
    df2 = DataFrame(df2)
    df2 = df2[:, Not(filter(n -> endswith("_nnd")(n) || endswith("_nnc")(n), names(df2))...)]
end

(out::Make.@x("interm/arbors.dat"))(
    list::list_neurons(),
    raw::Make.@x("interm/arbor-gnn.dat"),
    nodesuvw::list_nodes_uvw(),
) = begin
    df = JLD2.load(raw.path, "df")
    dflist = JLD2.load(list.path, "df")
    filter!(r -> !endswith("X")(r.inj), dflist)

    df2 = calc_arbor_feats(df)
    leftjoin!(df, df2, on=:arborpath) |> dropmissing!

    df = remove_redundant_reatures(df)
    selected = Set(dflist.path)
    df = filter(r -> in(selected)(r.path), df)

    insertcols!(df, :arborcent => df.center)
    path2sampchan = Dict(zip(dflist.path, eachrow(dflist)))
    nodesuvw = nodes_info_factory(nodesuvw.path)
    mysegs2uvws(segs) = begin
        curnrn, curuvw = 0, nothing
        fixid(id) = id % 100_000_000_000
        map(segs) do seg
            if seg.nrn != curnrn
                r = path2sampchan[df[seg.nrn, :path]]
                curnrn, curuvw = seg.nrn, nodesuvw(r.sample, r.channel)
            end
            uvinfo = curuvw[fixid(seg.id)]
            depth = get(() -> uvinfo.depth, uvinfo, :rawdepth)
            (; uvinfo.uv, uvinfo.uv2, depth)
        end
    end
    nmt = NmtDat.load_images()
    mysegs2anns(segs) = segs2anns(segs, NmtDat, nmt)
    ann2hier(a) = NmtDat.ann2hier(nmt, a)
    iscortex(a) = begin
        a.charm[1] > 0 && return true
        false
    end
    isleft(p) = NmtDat.isleft(p)
    # (out::Make.@x("interm/arbors-uvw.dat"))(
    # ) = begin
    #     #leftjoin!(df, dflist, on=:path)
    #
    #
    #
    #     #df = df[:, [:arborpath, :arboruvcent, :f_uvrad, :arboruvcent2, :f_uvrad2, :fv_whist, :ncharm, :arborwmap]]
    #     #JLD2.jldsave(out.path; df)
    # #end

    df2 = calc_arbor_extras(df, dflist; segs2uvws=mysegs2uvws, segs2anns=mysegs2anns, ann2hier, iscortex, isleft)
    leftjoin!(df, df2, on=:arborpath) |> dropmissing!
    JLD2.jldsave(out.path; df)
end

cached_tsne(mat) = begin
    h = hash(mat)
    try
        Main.cached_tsne_[h]
    catch
        tmp = TSne.tsne(mat)
        try
            Main.cached_tsne_[h] = tmp
        catch
            Main.cached_tsne_ = Dict()
            Main.cached_tsne_[h] = tmp
        end
        tmp
    end
end

pairwise_eucdist(mat) = begin
    N = size(mat, 2)
    d = Matrix{Float64}(undef, N, N)
    @showprogress for i in 1:N, j in 1:N
        ci = @view mat[:, i]
        cj = @view mat[:, j]
        s = 0.0
        foreach(ci, cj) do v1, v2
            s += (v1 - v2)^2
        end
        d[i, j] = sqrt(s)
    end
    d
end


function show_neurons_with_ba(df, objs...)
    obj = load("interm/mesh-nmt-brain-lr.obj")
    fig = Makie.Figure()
    display(fig)
    nmt = NmtDat.load_images()
    getter = NmtDat.getter(nmt)
    ax, _ = Makie.mesh(fig[1, 1], obj, color=(:gray, 0.02), transparency=true, inspectable=false)
    Makie.scatter!(ax, [NmtDat.isleft(p) ? p : NmtDat.mirror(p) for p in df.soma_pos], color=[Makie.wong_colors()[c%7+1] * 0.7 for c in df.cls], transparency=true, fxaa=true)
    @showprogress for r in eachrow(df)
        pts = load_swc_as_lines(r.path, mirrorby=NmtDat)
        Makie.lines!(ax, pts, transparency=true, fxaa=true, color=Makie.wong_colors()[r.cls%7+1], inspector_label=(plt, idx, pos) -> begin
            h = getter(reshape(pos, 3, 1))[2][1]
            "$h"
        end)
    end
    Makie.DataInspector()
end

function show_neurons_with_ba_mus(files)
    obj = load("interm/allen-997.obj")
    allen = Allen.load_images()
    getter = Allen.getter(Allen.get_hier(), allen)
    fig = Makie.Figure()
    display(fig)
    ax, _ = Makie.mesh(fig[1, 1], obj, color=(:gray, 0.02), transparency=true, inspectable=false)
    for r in files
        pts = load_swc_as_lines(r; mirrorby=Allen)
        Makie.lines!(ax, pts, transparency=true, inspector_label=(plt, idx, pos) -> begin
            h = getter(reshape(pos, 3, 1))[2][1]
            "$h"
        end)
    end
    Makie.DataInspector()
end


function browser_monkey_to_mouse_hits(monkey, mouse, mdist)
    obj = load("interm/allen-997.obj")
    fig = Makie.Figure()
    monkey = filter(r -> r.typ == "PT", monkey) |> shuffle!
    mouse = filter(r -> r.typ == "PT", mouse) |> shuffle!

    ax, _ = Makie.mesh(fig[1, 1], obj, transparency=true, color=(:gray, 0.05))
    for r in eachrow(monkey)
        rownumber(r) > 50 && break
        Makie.lines!(ax, load_swc_as_lines(r.allenpath), color=:red)
    end
    for r in eachrow(mouse)
        rownumber(r) > 50 && break
        Makie.lines!(ax, load_swc_as_lines(r.path), color=:green)
    end
    display(fig)

    return


    selection = filter(r -> !startswith("IT")(monkey[r, :typ]), axes(monkey, 1))
    for r in eachrow(monkey)[selection]
        fig = Makie.Figure()
        display(fig)
        ax, _ = Makie.mesh(fig[1, 1], obj, color=(:gray, 0.05), transparency=true)
        Makie.lines!(ax, load_swc_as_lines(r.allenpath), label="MAC $(r.typ)", transparency=true, color=:black)
        @info "$(r.path)"
        for k in sortperm(mdist[rownumber(r), :])[1:3]
            rr = mouse[k, :]
            Makie.lines!(ax, load_swc_as_lines(rr.path), label="MUS $(rr.typ)", transparency=true)
            @info "$(rr.path)"
        end
        Makie.Legend(fig[1, 2], ax)
        readline()
    end
end



(out::Make.@x("interm/show-case.repo"))() = begin
    rempath = expanduser("~/Desktop/gapr-mirror/confdir/REPO/fmost234197.ch1.repo")
    run(`rsync -acz goulf-ts:$rempath $(repo.path)`)
end

(out::output_vid(:S01))(repo::Make.@x("interm/show-case.repo")) = begin
    catfile = abspath("precious/fmost234197.catalog")
    cmd = cmd_gapr(`show --playback $(repo.path) --script precious/gen-vid.lua --image file://$catfile`)
    cmd = addenv(cmd, [
        "LD_PRELOAD=/usr/lib/libjemalloc.so.2",
        "AUTOCLOSE=1",
        "VIDEO_OUTPUT=$(out.path)",
        "VIDEO_STAGES=889667",
        "VIDEO_MESH=interm/mesh-sample-fmost234197.ch1-lr.obj",
        "GAPR_DEBUG=",
        #"DISPLAY=:1",
    ])
    run(cmd)
end

(out::output_figs_opt())(
    supp::Make.ByProduct{Make.@x("monkeytemp/00-upload/figures-supp-opt.pdf")},
    abs::Make.ByProduct{Make.@x("monkeytemp/00-upload/figAbs-opt.pdf")},
    resp::Make.ByProduct{Make.@x("monkeytemp/00-upload/Response_to_Reviewers-opt.pdf")},
) = begin
    mktempdir() do path
        list1 = sort(filter(r -> match(r"fig\d\.pdf$", r) !== nothing, readdir("output", join=true)), by=p -> (contains("Abs")(p) ? 0 : 1, p))
        list2 = sort(filter(r -> match(r"figS\d+\.pdf$", r) !== nothing, readdir("output", join=true)))
        list1 = map(list1) do f
            fo = joinpath(path, string("lll-", basename(f)))
            optimizepdf(f, fo; q=100, ppi=300)
            fo
        end
        pdf1 = joinpath(path, "1.pdf")
        pdf2 = joinpath(path, "2.pdf")
        run(`pdfunite $list1 $pdf1`)
        optimizepdf(pdf1, out.path; q=100, ppi=300)
        list2 = map(list2) do f
            fo = joinpath(path, string("ll2-", basename(f)))
            optimizepdf(f, fo; q=100, ppi=300)
            fo
        end
        run(`pdfunite $list2 $pdf2`)
        optimizepdf(pdf2, supp.path; q=100, ppi=300)
        #cp(pdf2, supp.path, force=true)

        # zz2fig.join_pdf(list1, joinpath(path, "1.pdf"); hiq=true)
        # zz2fig.join_pdf(list2, joinpath(path, "2.pdf"))
        # run(`pdfunite $(joinpath(path, "1.pdf")) $(joinpath(path, "2.pdf")) $(out.path)`)
    end
    optimizepdf("monkeytemp/00-upload/figAbs.pdf", abs.path; q=100, ppi=300)
    optimizepdf("monkeytemp/00-upload/Response_to_Reviewers.pdf", resp.path; q=100, ppi=300)
end


(out::output_figs())(
    list::Make.list(
        output_fig,
        [
            #:Abs,
            (1:7)...,
            Symbol.(:S, string.(1:8, pad=2))...,
            Symbol.(:SD, string.(1:3, pad=2))...,
        ]
    ),
    _::Make.list(
        output_tab,
        [
            :S02,
        ]
    )
) = begin
    if true
        figs = collect(map(p -> p[2], list.iter))
        for f in figs
            #checkpdfsize(f)
        end
        return run(`pdfunite $figs $(out.path)`)
    end
    genpage(i, t, p) = begin
        p = realpath(p)
        """
        \\bookmark[page=$i,view={FitH}]{Figure $t}
        \\begin{tikzpicture}
        \\node {\\includegraphics{$p}};
        \\end{tikzpicture}
        """
    end
    genpages() = join([genpage(i, t[1], p) for (i, (t, p)) in enumerate(list.iter)])
    wrappertex = """
      \\documentclass[multi=tikzpicture]{standalone}
      \\usepackage{graphicx}
      \\usepackage{bookmark}
      \\usepackage{tikz}
      \\begin{document}
          $(genpages())
      \\end{document}
      """

    compile_pdf(path) = begin
        f = joinpath(path, "output.tex")
        open(f, "w") do io
            write(io, wrappertex)
        end
        Draft.runlatex(f, path)
        Draft.runlatex(f, path)
        Draft.runlatex(f, path)
        joinpath(path, "output.pdf")
    end
    pdf = compile_pdf(mktempdir(prefix="xxxlatex"))
    mv(pdf, out.path, force=true)
end

output_neurons_zip() = Make.@x "monkeytemp/00-upload/neurons.zip"
output_code_zip() = Make.@x "monkeytemp/00-upload/code.zip"

(out::output_neurons_zip())(
    list::list_neurons_detail(),
) = begin
    df = JLD2.load(list.path, "df")
    out = abspath(out.path)
    mktempdir() do path
        path2 = joinpath(path, "neurons")
        mkdir(path2)
        ofs = map(eachrow(df)) do r
            cmd, fn = if lowercase(splitext(r.path)[2]) == ".gz"
                `gunzip -c $(r.path)` => basename(splitext(r.path)[1])
            else
                `cat $(r.path)` => basename(r.path)
            end
            m = match(r"^root\d+-(.+)$", fn)
            @assert m !== nothing
            fn = m.captures[1]
            s = read(cmd, String)
            of = joinpath(path2, string(r.inj, "-", fn))
            open(of, "w") do io
                write(io, s)
            end
            of
        end
        run(setenv(`zip -r $out neurons`, deepcopy(ENV); dir=path))
    end
end


(out::output_code_zip())(
) = begin
    copydir(rd, path2) = begin
        for (r, dd, ff) in walkdir(rd)
            d = relpath(r, rd)
            for f in ff
                lowercase(splitext(f)[2]) != ".jl" && continue
                mkpath(joinpath(path2, d))
                cp(joinpath(rd, d, f), joinpath(path2, d, f))
                @warn "$rd $d $f"
            end
        end
    end
    out = abspath(out.path)
    mktempdir() do path
        path2 = joinpath(path, "code")
        mkdir(path2)
        copydir(@__DIR__, path2)
        copydir(joinpath(@__DIR__, "../segreg"), joinpath(path2, "segreg"))
        run(setenv(`zip -r $out code`, deepcopy(ENV); dir=path))
    end
end
function migrate_jld2(fn::AbstractString)
    read_as_tuple(fn) = begin
        v = JLD2.jldopen(fn) do f
            map(keys(f)) do k
                Symbol(k) => f[k]
            end
        end
        NamedTuple(v)
    end
    prevfn = fn
    try
        for iter in 1:3
            x = read_as_tuple(prevfn)
            fnnext = "$fn.migrate-$iter"
            JLD2.jldsave(fnnext; x...)
            run(`touch -r $prevfn $fnnext`)
            prevfn = fnnext
        end
    catch e
        open("$fn.error", "w") do io
            Base.showerror(io, e)
        end
    end
end
