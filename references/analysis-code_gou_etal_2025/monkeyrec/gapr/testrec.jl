import Gapr
using Graphs
import GLMakie
import Random
using StaticArrays
using LinearAlgebra
using ProgressMeter

import Flux
import CUDA
import NearestNeighbors
import DataStructures
import JLD2

# XXX:
# *1. preserve proofread nodes @10945040
# ?2. longer dir
# ?3. longer adj
# ?4. pruned (31763.6,25670.3,9665.36) 
# ?5. xross (19580.7,45303.9,13053.7) (20448.6,49929,30464.6)
#     (19366.5,45341,13308.3) 
# 6. split long cc
# ?7. pred long links / long gaps

function jlgetpass(n::Cstring)::Nothing
    Gapr.print_thread_id("in jl code")
    nn = unsafe_string(n)
    tp = Threads.threadpool()
    #throw("asdf")
    println("hellojl $nn $tp ")
end

function plot_rec(ax, g)
    lbls = connected_components(g)
    v2c = Vector{GLMakie.RGBf}(undef, nv(g))
    for vv in lbls
        c = rand(GLMakie.RGBf)
        for v in vv
            v2c[v] = c
        end
    end
    pts = [Gapr.node_attr(g, v).pos for v in vertices(g)]
    GLMakie.scatter!(ax, pts, color=v2c)
    pts = GLMakie.Point3f[]
    col = GLMakie.RGBf[]
    for e in edges(g)
        c = v2c[src(e)]
        for n in e.points
            push!(pts, n.pos)
            push!(col, c)
        end
        push!(pts, GLMakie.Point3f(NaN))
        push!(col, c)
    end
    GLMakie.lines!(ax, pts, color=col)

    ncc = length(lbls)
    nvv = nv(g)
    nee = ne(g)
    npp = length(g.props)
    sss = GLMakie.campixel(ax.scene)
    GLMakie.text!(sss, 0, 0, text="nv $nvv; ne $nee; np $npp; ncc $ncc.")
end

const Vec3f = Gapr.GeometryBasics.Vec3f
mutable struct NodeInfo
    pos::Gapr.NodePos
    cov::Int
    fvecs::MVector{3,Vec3f}
    bvecs::MVector{3,Vec3f}
end

normvec(a, b, s) = begin
    d = a .- b
    l = norm(d) / s
    Vec3f(d .* ((l < 0.001 ? 1 : tanh(l) / l) / s))
end
function get_data_for_train(g; vec_ds=(8.0, 16.0, 32.0))
    nodeinfos = Dict{UInt32,NodeInfo}()
    nodeinfo(nid, attr) =
        get!(nodeinfos, nid) do
            ni = NodeInfo(attr.pos, attr.coverage, MVector{3,Vec3f}(undef), MVector{3,Vec3f}(undef))
            fill!(ni.fvecs, Vec3f(0))
            fill!(ni.bvecs, Vec3f(0))
            ni
        end
    for v in vertices(g)
        nodeinfo(Gapr.node_id(g, v), Gapr.node_attr(g, v))
    end
    edgeseqs = []
    edgepartlens = Float64[]
    interpol(a, b, x, y) = begin
        b > 0.1 ? (a / b) .* y .+ ((b - a) / b) .* x : x
    end
    for e in edges(g)
        n = e.n_nodes
        push!(edgeseqs, [e.nodes[i] for i in 1:n])
        empty!(edgepartlens)
        for i in 1:n
            if i == 1
                push!(edgepartlens, 0.0)
            else
                push!(edgepartlens, edgepartlens[i-1] + norm(e.points[i].pos - e.points[i-1].pos))
            end
        end
        for i in 1:n
            ni = nodeinfo(e.nodes[i], e.points[i])
            if i > 1
                ii = i - 1
                for (k, dd) in enumerate(vec_ds)
                    while ii > 1 && edgepartlens[ii] + dd > edgepartlens[i]
                        ii -= 1
                    end
                    a = max(edgepartlens[i] - dd - edgepartlens[ii], 0.0)
                    b = edgepartlens[ii+1] - edgepartlens[ii]
                    ni.fvecs[k] = normvec(e.points[i].pos, interpol(a, b, e.points[ii].pos, e.points[ii+1].pos), dd)
                end
            end
            if i < n
                ii = i + 1
                for (k, dd) in enumerate(vec_ds)
                    while ii < n && edgepartlens[ii] - dd < edgepartlens[i]
                        ii += 1
                    end
                    a = max(edgepartlens[ii] - edgepartlens[i] - dd, 0.0)
                    b = edgepartlens[ii] - edgepartlens[ii-1]
                    ni.bvecs[k] = normvec(e.points[i].pos, interpol(a, b, e.points[ii].pos, e.points[ii-1].pos), dd)
                end
            end
        end
    end
    for v in vertices(g)
        if outdegree(g, v) != 1
            ni = nodeinfos[Gapr.node_id(g, v)]
            fill!(ni.fvecs, Vec3f(0))
            fill!(ni.bvecs, Vec3f(0))
        end
    end
    (; nodeinfos, edgeseqs)
end

function vis_train_data(nodeinfos, edgeseqs)
    pts = [ni.pos for (_, ni) in pairs(nodeinfos)]
    display(GLMakie.scatter(pts, markersize=1))
    GLMakie.arrows!(pts, [ni.fvecs[1] .* 10 for (_, ni) in pairs(nodeinfos)], color=:red, linewidth=1)
    GLMakie.arrows!(pts, [ni.fvecs[2] .* 10 for (_, ni) in pairs(nodeinfos)], color=:green, linewidth=1)
    GLMakie.arrows!(pts, [ni.fvecs[3] .* 10 for (_, ni) in pairs(nodeinfos)], color=:blue, linewidth=1)
end

function vis_conn_data(nodeinfos, conn)
    pts = GLMakie.Point3f[]
    col = Float32[]
    for ((a, b), c) in pairs(conn)
        if a < b
            push!(pts, nodeinfos[a].pos)
            push!(pts, nodeinfos[b].pos)
            push!(col, c)
        end
    end
    display(GLMakie.linesegments(pts, color=col))
end

function get_model(; insiz=32)
    resblk(n) = Flux.SkipConnection(
        Flux.Chain(
            Flux.Dense(n => n ÷ 2),
            Flux.softplus,
            Flux.BatchNorm(n ÷ 2),
            Flux.Dense(n ÷ 2 => n)),
        +)
    Flux.Chain(
        resblk(32),
        Flux.Dense(32 => 16),
        resblk(16),
        resblk(16),
        Flux.Dense(16 => 8),
        resblk(8),
        resblk(8),
        Flux.Dense(8 => 1),
    )
end

link_input(nodeinfos, a, b) = begin
    nia = nodeinfos[abs(a)]
    nib = nodeinfos[abs(b)]
    dira = a > 0 ? nia.fvecs : nia.bvecs
    dirb = b > 0 ? nib.fvecs : nib.bvecs
    dirx = normvec(nia.pos, nib.pos, 16)
    dirs = (dira..., dirb..., dirx)
    SVector((dot(dirs[i], dirs[j]) for i in 1:length(dirs) for j in 1:i)..., 0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

function pred_adj_conn(nodeinfos, alllinks, model)
    topred = collect(alllinks)
    get_batch_pred(; n_eg=1024) = begin
        nn = Pair{UInt32,UInt32}[]
        xx = SVector{32,Float32}[]
        while !isempty(topred) && length(nn) < n_eg
            a00, b00 = pop!(topred)
            push!(nn, a00 => b00)
            for rrr in 0:7
                a, b = iseven(rrr >> 0) ? (b00, a00) : (a00, b00)
                if iseven(rrr >> 1)
                    a = -Int(a)
                end
                if iseven(rrr >> 2)
                    b = -Int(b)
                end
                push!(xx, link_input(nodeinfos, a, b))
            end
        end
        nn, reduce(hcat, xx)
    end

    conn = Dict{Pair{UInt32,UInt32},Float32}()
    Flux.testmode!(model)
    while !isempty(topred)
        left = length(topred) / length(alllinks)
        @info "left: $left"
        nn, x = get_batch_pred()
        y = model(x |> Flux.gpu)
        y = Flux.sigmoid(vec(maximum(reshape(vec(y), 8, :), dims=1))) |> Flux.cpu
        foreach(nn, y) do (a, b), c
            conn[a=>b] = max(get(conn, a => b, 0.0), c)
        end
    end
    conn
end

#function train_adj_classifier(nodeinfos, edgeseqs; thr=16.0, thr2=32.0, termrng=16.0)
function train_adj_classifier(nodeinfos, edgeseqs; thr=6.0, thr2=9.0, termrng=6.0)
    nodeinfos2 = collect(pairs(nodeinfos))
    nodeinfos3 = let
        node2term = Set{UInt32}()
        for es in edgeseqs
            p1 = nodeinfos[es[1]].pos
            for n in es
                norm(nodeinfos[n].pos .- p1) > termrng && break
                push!(node2term, n)
            end
            pn = nodeinfos[es[end]].pos
            for n in Iterators.reverse(es)
                norm(nodeinfos[n].pos .- pn) > termrng && break
                push!(node2term, n)
            end
        end
        filter(p -> p.first in node2term, nodeinfos2)
    end
    @info "knn $(length(nodeinfos2))"
    tree = NearestNeighbors.KDTree([ni.pos for (_, ni) in nodeinfos2])
    @info "knn 1 done"
    indices = NearestNeighbors.inrange(tree, [ni.pos for (_, ni) in nodeinfos2], thr)
    @info "knn 2 done"
    indices3 = NearestNeighbors.inrange(tree, [ni.pos for (_, ni) in nodeinfos3], thr2)
    @info "knn 3 done"
    alllinks = Set{Pair{UInt32,UInt32}}()
    for (i, jj) in enumerate(indices)
        a = nodeinfos2[i].first
        for j in jj
            b = nodeinfos2[j].first
            a == b && continue
            push!(alllinks, a => b)
        end
    end
    for (i, jj) in enumerate(indices3)
        a = nodeinfos3[i].first
        for j in jj
            b = nodeinfos2[j].first
            a == b && continue
            push!(alllinks, a => b)
        end
    end
    lset1 = length(alllinks)
    lset2 = 0
    lset12 = 0
    countset(a, b) = begin
        lset2 += 1
        if (a => b) in alllinks
            lset12 += 1
        end
    end
    alllinks2 = Dict{Pair{Int,Int},Float32}()
    calc_prop(dd) = begin
        exp(-dd^2 / 48^2)
    end
    for es in edgeseqs
        for (i, b) in enumerate(es)
            if i > 1
                ii = i - 1
                a = es[ii]
                countset(a, b)
                countset(b, a)
                bb = -Int(b)
                dd = norm(nodeinfos[a].pos .- nodeinfos[b].pos)
                while true
                    pp = calc_prop(dd)
                    push!(alllinks2, (a => bb) => pp)
                    push!(alllinks2, (bb => a) => pp)
                    !(ii > 1 && i - ii <= 8) && break
                    ii -= 1
                    a = es[ii]
                    dd = norm(nodeinfos[a].pos .- nodeinfos[b].pos)
                    dd > thr2 && break
                end
                # ll = norm(nodeinfos[a].pos - nodeinfos[b].pos)
                # ll > thr && @warn "long $a $b $ll"
            end
        end
    end
    @info "set1 $lset1"
    @info "set2 $lset2"
    @info "set12 $lset12"

    pos_shuff = []
    neg_shuff = []
    get_batch(; npos=16, nneg=16,) = begin
        if length(pos_shuff) < 100 || length(neg_shuff) < 100
            pos_shuff = Random.shuffle!(collect(alllinks2))
            neg_shuff = Random.shuffle!(collect(alllinks))
        end
        xx = SVector{32,Float32}[]
        yy = Float32[]
        while length(yy) < nneg
            a00, b00 = pop!(neg_shuff)
            rrr = Random.rand(Int)
            a, b = iseven(rrr >> 0) ? (b00, a00) : (a00, b00)
            if iseven(rrr >> 1)
                a = -Int(a)
            end
            if iseven(rrr >> 2)
                b = -Int(b)
            end
            haskey(alllinks2, a => b) && continue
            push!(xx, link_input(nodeinfos, a, b))
            push!(yy, 0)
        end
        for _ in 1:npos
            (a, b), y = pop!(pos_shuff)
            push!(xx, link_input(nodeinfos, a, b))
            push!(yy, y)
        end
        reduce(hcat, xx), yy
    end

    loss3(m, x, y) = begin
        ŷ = m(x)
        ###@info "$(size(ŷ)) $(size(x))"
        Flux.logitbinarycrossentropy(vec(ŷ), y)
    end

    model = get_model() |> Flux.gpu
    Flux.trainmode!(model)
    opt_state = Flux.setup(Flux.Adam(), model)
    @showprogress for iter in 1:99_000
        x, y = get_batch() |> Flux.gpu
        #@info "type $(typeof(x))"
        #@info "type $(typeof(y))"
        l, grad = Flux.withgradient(loss3, model, x, y)
        Flux.update!(opt_state, model, grad[1])
        if iter % 1000 == 0
            println("iter $iter $l")
        end
    end
    global trainalllinks = deepcopy(alllinks)
    global trainedmodel = Flux.cpu(model)
    pred_adj_conn(nodeinfos, alllinks, model)
end

function delta_init(props)
    (;
        del_nodes=UInt32[],
        nodesxxx=Set{UInt32}(),
        new_links=Pair[],
        new_nodes=Pair[],
        props,
    )
end
function delta_get!(delta)
    if !isempty(delta.del_nodes) && delta.del_nodes[end] == 0
        pop!(delta.del_nodes)
    end
    del_props = map(p -> (; NamedTuple(p)..., val=nothing), filter(p -> p.id in delta.nodesxxx, delta.props))
    sort!(del_props)
    (; del_props, delta.del_nodes, delta.new_links, delta.new_nodes,)
end
function del_one_node!(delta, a)
    push!(delta.nodesxxx, a)
    push!(delta.del_nodes, a, 0)
end
function del_one_link!(delta, a, b)
    @assert a != b
    push!(delta.del_nodes, a, b, 0)
end
function del_one_edge!(delta, itr, del_first, del_last)
    n = length(itr)
    for (i, nn) in enumerate(itr)
        push!(delta.del_nodes, nn)
        if i == 1
            if del_first
                push!(delta.del_nodes, nn)
                push!(delta.nodesxxx, nn)
            end
        elseif i == n
            if del_last
                push!(delta.del_nodes, nn)
                push!(delta.nodesxxx, nn)
            else
                push!(delta.del_nodes, 0)
            end
        else
            push!(delta.nodesxxx, nn)
        end
    end
end
function add_one_link!(delta, a, b)
    i = length(delta.new_nodes)
    push!(delta.new_nodes, nothing => 0, nothing => i + 1)
    push!(delta.new_links, i + 1 => a, i + 2 => b)
end

function enque_conn_probs(nodeinfos, edgeseqs, conndat)
    elinks = Set{Pair{UInt32,UInt32}}()
    for es in edgeseqs
        for (i, b) in enumerate(es)
            if i > 1
                a = es[i-1]
                push!(elinks, a => b, b => a)
            end
        end
    end
    @info "push pq"
    pq = DataStructures.PriorityQueue{Pair{UInt32,UInt32},Float32}()
    for ((a, b), c) in pairs(conndat)
        a, b = min(a, b), max(a, b)
        c = min(-c, get(pq, a => b, 0))
        if nodeinfos[a].cov + nodeinfos[b].cov > 0 && !in(a => b, elinks)
            c = 0
        end
        pq[a=>b] = c
    end
    for es in edgeseqs
        for (i, b) in enumerate(es)
            if i > 1
                a = es[i-1]
                if nodeinfos[a].cov > 0 && nodeinfos[b].cov > 0
                    a, b = min(a, b), max(a, b)
                    pq[a=>b] = -2.0
                end
            end
        end
    end
    @info "push pq done $(length(pq))"
    pq
end

function calc_delta2(nodeinfos, edgeseqs, conndat, g)
    pq = enque_conn_probs(nodeinfos, edgeseqs, conndat)
    GC.gc()
    n2cc = Dict{UInt32,UInt32}()
    optstk = UInt32[]
    opt_cc() = begin
        kkk = keys(n2cc)
        for a in kkk
            empty!(optstk)
            while true
                a2 = get(n2cc, a, a)
                if a2 == a
                    if length(optstk) > 1
                        for a0 in optstk
                            n2cc[a0] = a2
                        end
                    end
                    break
                end
                push!(optstk, a)
                a = a2
            end
        end
    end
    get_cc(a) = begin
        cnt = 0
        a0 = a
        while true
            a2 = get(n2cc, a, a)
            if a2 == a
                if cnt > 64
                    @time opt_cc()
                elseif cnt > 1
                    ###cnt > 3 && @info "$a0 $cnt-> $a2"
                    n2cc[a0] = a2
                end
                return a2
            end
            cnt += 1
            a = a2
        end
    end
    cc2len = Dict{UInt32,Float64}()
    newlinks = Set{Pair{UInt32,UInt32}}()
    @info "pop pq"
    while !isempty(pq)
        (a, b), c = DataStructures.dequeue_pair!(pq)
        c > -0.6 && break
        a, b = min(a, b), max(a, b)
        a2 = get_cc(a)
        b2 = get_cc(b)
        a2 == b2 && continue
        l1 = get(cc2len, a2, 1)
        l2 = get(cc2len, b2, 1)
        l1 + l2 > 3000 && c > -2.0 && continue
        # @info "$c"
        n2cc[a2] = b2
        cc2len[a2] = l1 + l2
        cc2len[b2] = l1 + l2
        push!(newlinks, a => b)
    end
    oldlinks = Set{Pair{UInt32,UInt32}}()
    for es in edgeseqs
        for (i, b) in enumerate(es)
            if i > 1
                a = es[i-1]
                a, b = min(a, b), max(a, b)
                push!(oldlinks, a => b)
            end
        end
    end
    oldlinks, newlinks

    delta = delta_init(g.props)
    for (a, b) in setdiff(oldlinks, newlinks)
        del_one_link!(delta, a, b)
    end

    for (a, b) in setdiff(newlinks, oldlinks)
        add_one_link!(delta, a, b)
    end
    delta_get!(delta)
end


### function calc_delta_trim(nodeinfos, edgeseqs, g::Gapr.EdgeModel; thr_d=24.0, thr_l=196.0)
function calc_delta_trim(nodeinfos, edgeseqs, g::Gapr.EdgeModel; thr_d=9.0, thr_l=64.0)
    simpverts = Set{UInt32}()
    for v in vertices(g)
        if outdegree(g, v) == 0 && Gapr.node_attr(g, v).coverage == 0
            push!(simpverts, Gapr.node_id(g, v))
        end
    end
    @info "n simpverts $(length(simpverts))"
    seedroots = Set{UInt32}()
    for p in g.props
        if (p.id in simpverts)
            p = NamedTuple(p)
            if p.key == "root" && p.val == "seed"
                push!(seedroots, p.id)
            end
        end
    end

    delta = delta_init(g.props)
    for s in seedroots
        del_one_node!(delta, s)
    end

    for v in setdiff(simpverts, seedroots)
        del_one_node!(delta, v)
    end

    nodeinfos2 = collect(pairs(nodeinfos))
    nodepos = map(p -> p.second.pos, nodeinfos2)
    nodeids = map(p -> p.first, nodeinfos2)
    @info "knn"
    tree = NearestNeighbors.KDTree(nodepos)
    indices = NearestNeighbors.inrange(tree, nodepos, thr_d)
    @info "knn done"
    node2edg = Dict{UInt32,UInt32}()
    for (ei, e) in enumerate(edges(g))
        for i in 1:e.n_nodes
            node2edg[e.nodes[i]] = ei
        end
    end
    node2inds = Dict{UInt32,Vector}()
    foreach(indices, nodeids) do inds, n
        node2inds[n] = inds
    end

    calc_l_and_cov(e, n) = begin
        l = 0.0
        nc = 0
        for i in 1:n
            if e.points[i].coverage > 0
                nc += 1
            end
            if i > 1
                l += norm(e.points[i-1].pos .- e.points[i].pos)
            end
        end
        l, nc
    end
    can_delete(e, s0, d0, n, ei) = begin
        if (s0 >= 3 && d0 >= 3) && n <= 3
            l, nc = calc_l_and_cov(e, n)
            nc > 0 && return false
            return true
        elseif (s0 == 1 || d0 == 1) && n <= 32
            l, nc = calc_l_and_cov(e, n)
            nc > 0 && return false
            if l <= thr_d
                return true
            end
            if l <= thr_l
                mmm = 0.0
                for i in 1:n
                    mm = Inf
                    n1 = e.nodes[i]
                    for ii in node2inds[n1]
                        n2 = nodeids[ii]
                        if n2 != n1 && get(node2edg, n2, 0) != ei
                            mm = min(mm, thr_d)
                        end
                    end
                    mmm = max(mmm, mm)
                end
                if mmm <= thr_d
                    return true
                end
            end
        end
        return false
    end

    for (ei, e) in enumerate(edges(g))
        s0 = outdegree(g, src(e))
        d0 = outdegree(g, dst(e))
        n = e.n_nodes
        if can_delete(e, s0, d0, n, ei)
            @info "del edge"
            del_one_edge!(delta, (e.nodes[i] for i in 1:n), s0 == 1, d0 == 1)
        end
    end

    delta_get!(delta)
end

function calc_delta(g)
    global datafortrain = get_data_for_train(g)
    # pq = enque_conn_probs(datafortrain.nodeinfos, datafortrain.edgeseqs,
    #     Dict(
    #         (12936905 => 12936904) => 0.2,
    #         (12936905 => 12938286) => 0.2,
    #         (12936904 => 12936902) => 0.2,
    #         (48127170 => 17209815) => 0.2,
    #     )
    # )
    # global xxxpq = deepcopy(pq)
    # error("asdf")
    ##return (; del_props=[], del_nodes=[], new_links=[], new_nodes=[],)

    if true
        if isfile("./conndata.jld2")
            conndat = JLD2.load("./conndata.jld2", "dat")
            global predconn = deepcopy(conndat)
            return calc_delta2(datafortrain.nodeinfos, datafortrain.edgeseqs, conndat, g)
        else
            ###conndat = JLD2.load("./conndata.jld2", "dat")
            conndat = train_adj_classifier(datafortrain.nodeinfos, datafortrain.edgeseqs)
            global predconn = deepcopy(conndat)
            return calc_delta2(datafortrain.nodeinfos, datafortrain.edgeseqs, conndat, g)
        end
    end
    return (; del_props=[], del_nodes=[], new_links=[], new_nodes=[],)

    # k = 2
    # acc1(e, l2) = begin
    #     p = e.nodes
    #     (Base.unsafe_load(p, i) for i in l2:l2+k)
    # end
    # acc2(e, l2) = begin
    #     v = Base.unsafe_wrap(Array, e.nodes, e.n_nodes)
    #     (v[i] for i in l2:l2+k)
    # end
end

function main()
    with_data = get(ENV, "WITH_DATA", nothing)
    with_data = isempty(with_data) ? nothing : with_data

    looper = Gapr.looper_create()
    try
        Gapr.looper_start(looper)
        try
            host = gethostname() == "goulf-nuc" ? "localhost" : "10.10.42.8:8006"
            engine = if with_data === nothing
                @info "testdata"
                Gapr.engine_create(looper, "bot1@$host/fmost240273.cortex.1", jlgetpass)
            else
                @warn "realdata $with_data"
                Gapr.engine_create(looper, "bot1@$host/$with_data", jlgetpass)
            end
            #engine = Gapr.engine_create(looper, "bot1@localhost:8002/fmost241349.ch1.test2", jlgetpass)
            #engine = Gapr.engine_create(looper, "bot1@yanlab.org.cn:30003/fmost18925", jlgetpass)
            try
                Gapr.engine_start(engine)
                @info "engine prepared"

                nnn = Gapr.engine_load_commits(engine, UInt32(0))
                @info "engine loaded $nnn"

                f = GLMakie.Figure()
                ax1 = GLMakie.LScene(f[1, 1])
                ax2 = GLMakie.LScene(f[1, 2], scenekw=(; camera=GLMakie.camera(ax1)))
                display(f)

                delta = Gapr.with_edge_model(engine) do g
                    @time "dijkstra" dijkstra_shortest_paths(g, UInt32[1, 2, 3])
                    #modularity(g, c, 
                    @time "rich_club" rich_club(g, 1)
                    ##cycle_basis(g)
                    @time "plotrec" plot_rec(ax1, g)

                    calc_delta(g)
                end

                global xxxdelta = deepcopy(delta)

                if !isempty(delta.del_nodes)
                    Gapr.engine_del_patch(engine, delta.del_props, delta.del_nodes, with_data === nothing ? 100_000_000 : 0)
                end
                if !isempty(delta.new_nodes)
                    Gapr.engine_add_patch(engine, delta.new_links, [], delta.new_nodes, with_data === nothing ? 200_000_000 : 0)
                end

                GC.gc()
                delta = Gapr.with_edge_model(engine) do g
                    datafortrain = get_data_for_train(g)
                    calc_delta_trim(datafortrain.nodeinfos, datafortrain.edgeseqs, g)
                end
                global xxxdelta2 = deepcopy(delta)

                if !isempty(delta.del_nodes)
                    Gapr.engine_del_patch(engine, delta.del_props, delta.del_nodes, with_data === nothing ? 300_000_000 : 0)
                end

                GC.gc()
                delta = Gapr.with_edge_model(engine) do g
                    datafortrain = get_data_for_train(g)
                    calc_delta_trim(datafortrain.nodeinfos, datafortrain.edgeseqs, g)
                end
                global xxxdelta3 = deepcopy(delta)

                if !isempty(delta.del_nodes)
                    Gapr.engine_del_patch(engine, delta.del_props, delta.del_nodes, with_data === nothing ? 400_000_000 : 0)
                end

                Gapr.with_edge_model(engine) do g
                    plot_rec(ax2, g)
                end
                delta
            catch e
                Base.showerror(stderr, e, Base.catch_backtrace())
            finally
                Gapr.engine_destroy(engine)
            end
        catch e
            @warn "error2 $e"
        finally
            Gapr.looper_stop(looper)
        end
    catch e
        @warn "error3 $e"
    finally
        Gapr.looper_destroy(looper)
    end
end
