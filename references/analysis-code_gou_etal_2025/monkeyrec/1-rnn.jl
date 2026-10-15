to_swc(df::DataFrame) = begin
    nodes = map(eachrow(df)) do r
        Gapr.SwcNodeCore((; r.pos, r.id, typ=0, rad=0), r.par_idx)
    end
    Gapr.SwcTable(nodes, [], [], [], Dict{Int,Int}())
end

function qhull_factory(; skip_qhull=false)
    init_qhull() = skip_qhull ? 0 : DirectQhull.qh_alloc_qh()
    free_qhull(qh) = 0
    function run_qhull(qh, pts; skip_qhull=false)
        skip_qhull && return 0, []
        DirectQhull.qh_new_qhull(qh, pts, "FS")
        x = DirectQhull.qh_get_convex_hull_vertices(qh, Val(4))
        pts = pts[:, setdiff(x, 0)]
        vol = DirectQhull.qh_get_totvol(qh)
        ccall((:qh_freeqhull, DirectQhull.qh_lib), Cvoid,
            (Ptr{DirectQhull.qhT}, Cuint), qh, true)
        return vol, pts
    end
    (; init=init_qhull, free=free_qhull, run=run_qhull)
end

generate_dendrogram_factory(; qhull=qhull_factory()) = (swc::Gapr.SwcTable) -> begin
    # advantages with dendrogram:
    # * arbor in arbor scenario, better tree cut
    # * automatic reduction of tree structure
    # * ensured binary tree
    swc = Gapr.nodedegree(swc)
    swc = Gapr.nodedepth(swc)
    br_angles_from_root = Tuple[() for _ in 1:length(swc.nodes)]
    calc_angle(swc, n) = begin
        n.parind == 0 && return ()
        pn = swc[n.parind]
        pangs = br_angles_from_root[n.parind]
        pn.parind == 0 && return (pangs...,)
        d = pn.degree
        @assert d > 1
        d <= 2 && return (pangs...,)
        ppn = swc[pn.parind]
        d1 = (ppn.pos .- pn.pos)
        d2 = (pn.pos .- n.pos)
        a = dot(d1, d2) / (norm(d1) * norm(d2))
        (pangs..., a)
    end
    allterms = Int[]
    for n in swc.nodes
        br_angles_from_root[n.ind] = calc_angle(swc, n)
        if n.degree == 1 && n.parind != 0
            push!(allterms, n.ind)
        end
    end
    m = Gapr.distmat(swc)
    m = m[allterms, allterms]
    dend = Clustering.hclust(m; linkage=:average, branchorder=:barjoseph)
    term2anc = Dict()
    for idx in allterms
        vec = get!(term2anc, idx, Set())
        idx = swc[idx].parind
        while idx != 0
            n = swc[idx]
            if n.degree != 2 || n.parind == 0
                push!(vec, idx)
            end
            idx = n.parind
        end
    end
    commonanc = Dict()
    for i in allterms, j in allterms
        a = argmax(k -> swc[k].depth, intersect(term2anc[i], term2anc[j]))
        commonanc[(i, j)] = a
    end
    tree = Vector{NamedTuple}(undef, length(dend.heights) + length(allterms))
    branch(r) = r + length(allterms)
    foreach(enumerate(allterms)) do (r, t)
        tree[r] = (; root=t, terms=Int[t], height=0, left=0, right=0, xpos=findfirst(==(r), dend.order))
    end
    foreach(enumerate(dend.heights), eachrow(dend.merges)) do (r, height), (a, b)
        terms = Int[]
        rowintree(x) = x < 0 ? -x : branch(x)
        addterms(x) = append!(terms, tree[rowintree(x)].terms)
        addterms(a)
        addterms(b)
        xpos = (tree[rowintree(a)].xpos + tree[rowintree(b)].xpos) / 2
        root = nothing
        rootdepth = Inf
        for s in terms, t in terms
            anc = commonanc[(s, t)]
            rootdepth, root = min((rootdepth, root), (swc[anc].depth, anc))
        end
        tree[branch(r)] = (;
            root,
            terms,
            height,
            left=rowintree(a),
            right=rowintree(b),
            xpos,
        )
    end
    tree = DataFrame(tree)
    nangles = 7
    insertcols!(tree, :ds_len => 0.0, :ds_vol => 0.0, :ds_seg => 0, :radius => 0.0, :angles => Ref(SVector([0.0f0 for _ in 1:nangles]...)), :intensity1 => 0.0, :intensity2 => 0.0)
    insertcols!(tree, :depth => 0.0, :paridx => 0, :rootid => 0, :nodeids => Ref(Int[]))
    qh = qhull.init()
    cache_angs = []
    for r in eachrow(tree)
        r.rootid = swc[r.root].id
        depth0 = swc[r.root].depth
        selnodes = Set{Int}()
        for t in r.terms
            push!(selnodes, t)
            while t != r.root
                t = swc[t].parind
                push!(selnodes, t)
            end
        end
        selnodes = sort(collect(selnodes))
        r.nodeids = map(t -> swc[t].id, selnodes)
        ds_len = 0.0
        radius = 0.0
        pts = SVector[]
        for i in selnodes
            n = swc[i]
            push!(pts, n.pos)
            if i != r.root
                ds_len += norm(n.pos .- swc[n.parind].pos)
                radius = max(radius, n.depth - depth0)
            end
        end
        push!(pts, swc[r.root].pos .+ SVector(0.001, 0, 0))
        push!(pts, swc[r.root].pos .+ SVector(0, 0.001, 0))
        push!(pts, swc[r.root].pos .+ SVector(0, 0, 0.001))
        r.ds_len = ds_len
        r.radius = radius
        r.ds_vol = qhull.run(qh, Matrix(reduce(hcat, pts)))[1]
        r.ds_seg = length(r.terms)
        r.depth = depth0
        empty!(cache_angs)
        append!(cache_angs, fill((1.0, 10000), nangles), reverse.(enumerate(br_angles_from_root[r.root])))
        r.angles = SVector(first.(view(cache_angs, sort(sortperm(cache_angs)[1:nangles])))...)
        ###@info "angle $(r.angles) $(br_angles_from_root[r.root])"
        for x in (r.left, r.right)
            if x != 0
                tree[x, :paridx] = rownumber(r)
            end
        end
    end
    qhull.free(qh)
    tree
end



rnn_gen_reduced_swc_factory(skip_qhull=false; keep_nodes=[]) = function (swc_lines)
    df = DataFrame(swc_lines)
    insertcols!(df, :depth => 0.0, :radius => 0.0, :ds_len => 0.0, :ds_vol => 0.0, :ds_seg => 0)
    insertcols!(df, :ppar => -1)
    insertcols!(df, :nbr => 0, :br_idx => [Int[] for _ in axes(df, 1)])
    insertcols!(df, :seg_len => 0.0, :ds_pol => [[] for xx in 1:size(df, 1)])

    for row in eachrow(df)
        row.paridx != 0 || continue
        prow = df[row.paridx, :]
        row.seg_len = l = norm(row.pos .- prow.pos)
        row.depth = prow.depth + l
    end

    qh = init_qhull()
    for row in Iterators.reverse(eachrow(df))
        poses = [[row.pos...,],]
        for xx in 1:9
            del = randn(3)
            s = 0.1 * min(1.0, 3.0 / norm(del))
            push!(poses, poses[1] .+ (del * s))
        end

        nseg = row.ds_seg
        for ds_pol in row.ds_pol
            for pt in ds_pol
                push!(poses, pt)
            end
        end
        if length(row.ds_pol) >= 2
            nseg += length(row.ds_pol)
        end
        vol, poses = run_qhull(qh, poses)
        row.ds_vol = vol
        row.ds_seg = nseg

        row.paridx != 0 || continue
        prow = df[row.paridx, :]
        l = row.seg_len
        prow.radius = max(row.radius + l, prow.radius)
        prow.ds_len += row.ds_len + l
        push!(prow.ds_pol, poses)
        prow.ds_seg += nseg
    end
    free_qh(qh)

    sel_ids = shuffle(MersenneTwister(df.id[1]), df.id)
    sel_ids = sel_ids[1:min(length(sel_ids), df[1, :ds_seg] ÷ 3)]
    # if length(sel_ids) > 4000
    #     sel_ids = sel_ids[1:500]
    # else
    #     sel_ids = sel_ids[1:ceil(Int64, length(sel_ids) / 8)]
    # end
    sel_ids = Set(sel_ids)
    for n in keep_nodes
        push!(sel_ids, n)
    end
    for row in eachrow(df)
        n = length(row.ds_pol)
        row.nbr = n
        sel = false
        if n == 1
            row.ds_seg += 1
        else
            sel = true
        end
        if row.paridx == 0
            sel = true
        elseif df[row.paridx, :nbr] >= 2
            sel = true
        end
        if sel
            push!(sel_ids, row.id)
        end
        if row.paridx != 0
            prow = df[row.paridx, :]
            if prow.paridx != 0 && !in(prow.id, sel_ids)
                row.ppar = prow.ppar
            else
                row.ppar = prow.id
            end
        end
    end
    df = filter(r -> (r.id in sel_ids), df)
    id2idx = Dict(map(x -> reverse(x), pairs(df.id) |> collect))
    for row in eachrow(df)
        row.paridx = p = get(id2idx, row.ppar, 0)
        if p != 0
            push!(df[p, :br_idx], rownumber(row))
        end
    end
    return df[:, Not([:seg_len, :ds_pol, :ppar])]
end

function highlight_arbor!(swc::Gapr.SwcTable, arbors...)
    for r in swc.nodes
        r.typ = 0
    end
    for (k, arbor) in enumerate(arbors)
        for n in arbor.nodes
            nn = swc[!, n.id]
            @assert nn.id == n.id
            nn.typ = (k - 1) % 5 + 1
        end
    end
end

function ensure_directory(f::Function, outdir)
    outdirtmp = outdir * ".temp"
    rm(outdir; force=true, recursive=true)
    rm(outdirtmp; force=true, recursive=true)
    mkdir(outdirtmp)
    res = f(outdir, outdirtmp)
    mv(outdirtmp, outdir)
    res
end
function neuron_basename(r)
    m = match(r"/(root\d+-q?root\d+).*$", r.path)
    if m !== nothing
        return string(r.sample, '-', r.channel, '-', m.captures[1])
    end
    m = match(r"/(\d+)_(\d+)\.swc(\.gz)?$", r.path)
    @assert m !== nothing "error file path $(r.path)"
    string(r.sample, '-', m.captures[1], '-', m.captures[2])
end
function arbor_name(arbor::Gapr.SwcTable)
    h = string(hash(sort!([n.id for n in arbor.nodes])), base=16)
    string("arbor", arbor[1].id, '-', h)
end

function segment_arbors(swc, dend; debugout=nothing)
    dend = deepcopy(dend)
    insertcols!(dend, :allred => true,)
    for r in eachrow(dend)
        if r.tgt != 2
            r.allred = false
        end
        if r.paridx != 0 && !r.allred
            dend[r.paridx, :allred] = false
        end
    end
    taboo = Set{Int}()
    arbors = []
    for r in Iterators.reverse(eachrow(dend))
        if r.tgt == 2 && (r.paridx == 0 || dend[r.paridx, :tgt] != 2) && !(r.root in taboo)
            length(r.terms) < 3 && continue
            #if r.allred && (r.paridx == 0 || !dend[r.paridx, :allred])
            ##@info "arbor $(swc[r.root].id)"
            ##@info "arbor $(r.terms)"
            allnodes = Set()
            for i in r.terms
                while true
                    #@info "$(r.root) $i $(swc[i].id)"
                    push!(allnodes, i)
                    i == r.root && break
                    i = swc[i].parind
                    i == 0 && break
                end
            end
            allnodes = sort(collect(allnodes))
            for n in allnodes
                push!(taboo, n)
            end

            n2i = Dict()
            subtree = map(enumerate(allnodes)) do (i, ni)
                n2i[ni] = i
                n = if i == 1
                    Gapr.SwcNode(swc[ni].node; parind=0)
                else
                    Gapr.SwcNode(swc[ni].node; parind=n2i[swc[ni].parind])
                end
            end
            push!(arbors, Gapr.SwcTable(subtree))
        end
    end
    arbors
end

function get_arbor_feature_lm_factory()
    (swc) -> begin
        mktempdir() do path
            swc = deepcopy(swc)
            for (i, n) in enumerate(swc.nodes)
                n.id = i
                n.rad = 0
                if n.typ == 0
                    n.typ = 3
                end
            end
            swcfile = joinpath(path, "in.swc")
            lmout = joinpath(path, "lmout.txt")
            lmexe = expanduser("~/opt/Lmv5.3_64bit/lmeasure")
            Gapr.save_swc(swcfile, swc)
            args = ["-f$i,0,0,10.0" for i in setdiff(0:44, (17, 42))]
            run(pipeline(`$lmexe $args -C -s$lmout $swcfile`; stderr=devnull))
            map(filter(!isempty, strip.(eachline(lmout)))) do l
                _, lm, sum, nnc, nnd, min, avg, max, sd = strip.(split(l, "\t"))
                m = match(r"\((\d+)\)", nnd)
                @assert m !== nothing
                nnd = m.captures[1]
                lm = string(lm)
                sum, nnc, nnd, min, avg, max, sd = parse.(Float32, (sum, nnc, nnd, min, avg, max, sd))
                (; lm, sum, nnc, nnd, min, avg, max, sd)
            end |> DataFrame
        end
    end
end

function rnn_prepare_xyvals(swc, node2type)
    function get_arbor_feature(res, i, r)
        @assert res[i, :root] == r.root
        pr = r
        if r.paridx != 0
            pr = res[r.paridx, :]
        end
        feats = Float64[]
        push!(feats, r.depth / 1e3, r.radius / 1e3, r.ds_len / 1e3, r.ds_vol / 1e6, r.ds_seg, r.angles..., r.intensity1, r.intensity2) #= , r.nbr =#
        push!(feats, log(r.depth + 1), log(r.radius + 1), log(r.ds_len + 1), log(r.ds_vol + 1), log(r.ds_seg + 1))
        push!(feats, (r.depth - pr.depth) / 100, (pr.radius - r.radius) / 100, (pr.ds_len - r.ds_len) / 100, (pr.ds_vol - r.ds_vol) / 1e5, (pr.ds_seg - r.ds_seg))
        quots = [(pr.depth + 1) / (r.depth + 1), (r.radius + 1) / (pr.radius + 1), (r.ds_len + 1) / (pr.ds_len + 1), (r.ds_vol + 1) / (pr.ds_vol + 1), (r.ds_seg + 1) / (pr.ds_seg + 1),
            1e3 * (1 + r.ds_len) / (1 + r.radius)^2,
            1e6 * (1 + r.ds_len) / (1 + r.radius)^3,
            1e6 * (1 + r.ds_seg) / (1 + r.radius)^2,
            1e9 * (1 + r.ds_seg) / (1 + r.radius)^3,
            (1 + r.ds_vol) / (1 + r.radius)^2,
            1e3 * (1 + r.ds_vol) / (1 + r.radius)^3,
            1e3 * (1 + r.ds_len) / (1 + r.ds_vol),
            1e6 * (1 + r.ds_seg) / (1 + r.ds_vol),
            1e3 * (1 + r.ds_seg) / (1 + r.ds_len),
        ]
        for q in quots
            push!(feats, q)
            push!(feats, -log(q))
        end
        while length(feats) % 8 != 0
            push!(feats, 0.0)
        end
        feats
    end
    swc = deepcopy(swc)
    insertcols!(swc, :tgt => 0, :feat => [Float32[] for _ in 1:size(swc, 1)], :ups => 0, :ds => 0)

    firsthit = Dict()
    for (i, r) in Iterators.reverse(enumerate(eachrow(swc)))
        ups = get(node2type, r.rootid, 0)
        if ups != 0
            get!(firsthit, r.rootid, i)
        end
    end
    for (_, i) in firsthit
        i = swc[i, :paridx]
        while i != 0
            swc[i, :ds] = -1
            i = swc[i, :paridx]
        end
    end
    for (i, r) in Iterators.reverse(enumerate(eachrow(swc)))
        r.feat = Float32.(get_arbor_feature(swc, rownumber(r), r))
        ups = get(node2type, r.rootid, 0)
        if ups == 0
            if r.paridx != 0
                ups = swc[r.paridx, :ups]
                if ups > 0 && r.ds < 0
                    @assert false
                    ups = 0
                end
            end
        else
            if ups > 0 && r.ds < 0
                ups = 0
            end
        end
        r.ups = ups
    end
    for r in eachrow(swc)
        r.tgt = if r.ups > 0
            r.ds < 0 ? 1 : 2
        elseif r.ups < 0
            r.ds < 0 ? 1 : 1
        else
            r.ds < 0 ? 1 : 0
        end
    end
    swc
end

function draw_dend!(ax, dend)
    col(v) = (:black, :green, :red)[v+1]
    segs = []
    cols = []
    for r in eachrow(dend)
        for c in (r.left, r.right)
            if c != 0
                cr = dend[c, :]
                push!(segs, SVector(cr.xpos, cr.height))
                push!(segs, SVector(cr.xpos, r.height))
                push!(cols, col(cr.tgt))
                push!(segs, SVector(cr.xpos, r.height))
                push!(segs, SVector(r.xpos, r.height))
                push!(cols, col(r.tgt))
            end
        end
    end
    Makie.scatter!(ax, dend.xpos, dend.height, color=col.(dend.tgt), inspector_label=(p, i, v) -> "$i ups$(dend[i, :ups]) ds$(dend[i, :ds]) id$(dend[i, :rootid])")
    plt = Makie.linesegments!(ax, Point2f.(segs), color=cols, inspectable=false, linecap=:round)
    Makie.translate!(plt, Vec3f(0, 0, 10))
    Makie.DataInspector()
end
function draw_dend(dend)
    fig = Makie.Figure()
    display(fig)
    ax = Makie.Axis(fig[1, 1])
    draw_dend!(ax, dend)
end

function rnn_factory(featsiz; epochs=3000, batsiz=24, celsiz=32, bnsiz=16, encsiz=64, encdecdepth=3, gnnconv=nothing, nrecurr=3)
    device = Flux.gpu
    classes = 1:2
    ein = 1

    relu = Flux.relu
    resblk(in, hidden) = Flux.SkipConnection(Flux.Chain(Flux.Dense(in => hidden, relu), Flux.Dense(hidden => in)), +)
    gnnconvlayer(in, ein, out) = begin
        if false
            # elseif gnnconv === GraphNeuralNetworks.A3TGCN
            #     #GraphNeuralNetworks.A3TGCN(in => out)
        elseif gnnconv === GraphNeuralNetworks.AGNNConv
            #GraphNeuralNetworks.AGNNConv()
        elseif gnnconv === GraphNeuralNetworks.CGConv
            GraphNeuralNetworks.CGConv((in, ein) => out, Flux.tanh_fast)
        elseif gnnconv === GraphNeuralNetworks.ChebConv
            #GraphNeuralNetworks.ChebConv(in => out, 3)
        elseif gnnconv === GraphNeuralNetworks.DConv
            #GraphNeuralNetworks.DConv(in => out, 3)
        elseif gnnconv === GraphNeuralNetworks.EGNNConv
            # ???GraphNeuralNetworks.EGNNConv((in, ein) => out)
        elseif gnnconv === GraphNeuralNetworks.GATConv
            GraphNeuralNetworks.GATConv((in, ein) => out, Flux.tanh_fast; add_self_loops=false)
        elseif gnnconv === GraphNeuralNetworks.GATv2Conv
            GraphNeuralNetworks.GATv2Conv((in, ein) => out, add_self_loops=false)
        elseif gnnconv === GraphNeuralNetworks.GCNConv
            #GraphNeuralNetworks.GCNConv(in => out, Flux.tanh_fast; use_edge_weight=true)
        elseif gnnconv === GraphNeuralNetworks.GMMConv
            GraphNeuralNetworks.GMMConv((in, ein) => out, Flux.tanh_fast)
        elseif gnnconv === GraphNeuralNetworks.GatedGraphConv
            #GraphNeuralNetworks.GatedGraphConv(out, 5)
        elseif gnnconv === GraphNeuralNetworks.GraphConv
            #GraphNeuralNetworks.GraphConv(in => out, Flux.tanh_fast)
        elseif gnnconv === GraphNeuralNetworks.MEGNetConv
            #GraphNeuralNetworks.MEGNetConv(in => out)
        elseif gnnconv === GraphNeuralNetworks.ResGatedGraphConv
            #GraphNeuralNetworks.ResGatedGraphConv(in => out, Flux.tanh_fast)
        elseif gnnconv === GraphNeuralNetworks.SAGEConv
            #GraphNeuralNetworks.SAGEConv(in => out, Flux.tanh_fast)
        elseif gnnconv === GraphNeuralNetworks.SGConv
            #GraphNeuralNetworks.SGConv(in => out, 1; use_edge_weight=true)
        elseif gnnconv === GraphNeuralNetworks.TAGConv
            #GraphNeuralNetworks.TAGConv(in => out, 3; use_edge_weight=true)
        elseif gnnconv === GraphNeuralNetworks.TransformerConv
            GraphNeuralNetworks.TransformerConv((in, ein) => out)
        elseif gnnconv === GraphNeuralNetworks.NNConv
            nn = Flux.Dense(ein, out * in)
            GraphNeuralNetworks.NNConv(in => out, nn, Flux.relu)
        else
            @assert false
        end
    end
    build_model() = begin
        enc = Flux.Chain(Flux.Dense(featsiz => encsiz),
            (Flux.Chain(resblk(encsiz, bnsiz), Flux.BatchNorm(encsiz)) for _ in 1:encdecdepth)...,
            Flux.Dense(encsiz => celsiz, relu),
        )
        gnn1 = gnnconvlayer(celsiz + celsiz, ein, celsiz)
        gnn2 = gnnconvlayer(celsiz + celsiz, ein, celsiz)
        gnn3 = gnnconvlayer(celsiz + celsiz, ein, celsiz)
        dec = Flux.Chain(Flux.Dense(celsiz => encsiz),
            (Flux.Chain(resblk(encsiz, bnsiz), Flux.BatchNorm(encsiz)) for _ in 1:encdecdepth)...,
            Flux.Dense(encsiz => length(classes)),
        )
        (; enc, gnn1, gnn2, gnn3, dec)
    end

    get_output(m, g) = begin
        x = g.ndata.x
        h = m.enc(x)
        e = g.edata.x
        for _ in 1:nrecurr
            h0 = h
            h = m.gnn1(g, vcat(h0, h), e)
            h = m.gnn2(g, vcat(h0, h), e)
            h = m.gnn3(g, vcat(h0, h), e)
        end
        m.dec(h)
    end
    get_loss(m, g) = begin
        y = g.ndata.y
        yl = g.ndata.yl
        yy = get_output(m, g)
        ylnz = yl .!= 0
        l1 = Flux.logitcrossentropy(yy[:, ylnz], y[:, ylnz])

        s, t = GraphNeuralNetworks.edge_index(g)
        topar = g.edata.x[1, :] .> 0
        s = s[topar]
        t = t[topar]
        smyy = Flux.softmax(yy)
        #########
        #l2 = Flux.mean(Flux.leakyrelu(smyy[2, t] .- smyy[2, s]))
        # yl2 = yl .== 2
        # l2 = Flux.binarycrossentropy(Flux.softmax(yy)[2, ylnz], yl2[ylnz])
        l1 #+ l2
    end
    decode_swc(swc) = begin
        e = Float32[]
        s = Int[]
        t = Int[]
        for r in eachrow(swc)
            if r.paridx != 0
                a, b = rownumber(r), r.paridx
                push!(s, a)
                push!(t, b)
                push!(e, 1.0f0)
                push!(s, b)
                push!(t, a)
                push!(e, -1.0f0)
            end
        end
        GraphNeuralNetworks.GNNGraph(s, t;
            num_nodes=size(swc, 1),
            ndata=(;
                x=reduce(hcat, swc.feat),
                y=Flux.label_smoothing(Flux.onehotbatch(swc.tgt, [0, classes...]), 0.01)[2:end, :],
                yl=swc.tgt,
            ),
            edata=(;
                x=reshape(e, 1, :),
            ),
        )
    end
    prepare_batch(dfx) = begin
        gg = map(eachrow(dfx)) do r
            decode_swc(r.dend)
        end
        Flux.batch(gg)
    end

    function mytrain(df)
        Random.seed!(0)
        model = build_model() |> device

        losses = Float64[]
        for eee in 1:3
            opt = Flux.setup(Flux.RMSProp(0.001 / eee), model)
            sample_rows = Int[]
            batsiz = 4
            while length(sample_rows) < epochs * batsiz
                append!(sample_rows, shuffle(axes(df, 1)))
                ########@info "append $(length(sample_rows))"
            end
            sample_rows = reshape(resize!(sample_rows, epochs * batsiz), batsiz, :)
            @showprogress "train" for K in 1:epochs
                g = prepare_batch(df[sample_rows[:, K], :]) |> device
                l, grad = Flux.withgradient(m -> get_loss(m, g), model)
                Flux.update!(opt, model, grad[1])
                push!(losses, l)
                if K % 100 == 0
                    @info "$K $l"
                end
            end
        end
        display(Makie.lines(losses))
        return model |> Flux.cpu
    end

    function mypredict(df, model)
        model = model |> device

        @showprogress "predict" for r in eachrow(df)
            if size(r.dend, 1) < 3
                for rr in eachrow(r.dend)
                    rr.tgt = 1
                end
                continue
            end
            g = decode_swc(r.dend) |> device
            yy = get_output(model, g) |> Flux.cpu
            for rr in eachrow(r.dend)
                rr.tgt = Flux.onecold(view(yy, :, rownumber(rr)), classes)
            end
        end
        return df
    end
    mytrain, mypredict
end
