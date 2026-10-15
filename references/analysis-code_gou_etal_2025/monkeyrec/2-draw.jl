function reorient_arbor(swc, n2w)
    pos = reduce(hcat, [n.pos for n in swc.nodes])
    pos = pos .- median(pos, dims=2)
    wmap = [n2w(n) for n in swc.nodes]
    sel = map(v -> !isnan(v) && v < 1 && v > 0, wmap)
    if length(findall(sel)) < 10
        @warn "$wmap"
        return nothing
    end
    www = wmap
    global xxxa, xxxb = reshape(www[sel], 1, :), pos[:, sel]
    loss_cor(w) = StatsBase.cor(reshape(reshape(w, 1, 3) * pos[:, sel], :), www[sel])
    #w = reshape(www[sel], 1, :) / pos[:, sel]
    w = argmax(loss_cor, eachcol(randn(3, 3000)))[:]
    w = w[:] ./ norm(w)
    ortho(u, w) = begin
        u = u - w * dot(w, u)
        u ./ norm(u)
    end
    u = argmax(u -> StatsBase.var(reshape(ortho(u, w), 1, 3) * pos), eachcol(randn(3, 1000)))
    u = ortho(u, w)
    v = ortho(cross(u, w), w)
    pp = hcat(u, w, v)' * pos
    foreach(eachcol(pp), swc.nodes, wmap) do pp, n, w
        n.pos = pp
    end
    swc = Gapr.addattr(swc; w=Float64)
    for n in swc.nodes
        n.w = n2w(n)
    end
    return swc
    c = loss_cor(w)
    #@warn " $c $w $u $v"
end

function draw_arbor_planar!(ax1, ax2, dfarbors; kwargs...)
    ticks = []
    xdelta = 0
    eo = true
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            for r in eachrow(dfarbors)
                aa = Gapr.load_swc(r.arborpath)
                y2y = reorient_arbor!(aa, r.arborwmap)
                y2y === nothing && continue
                pts, col = load_swc_as_lines(aa) do n
                    r.arborwmap[n.row]
                end
                ll, rr = extrema(first.([n.pos for n in aa])) ./ 1000
                xdelta += -ll
                plt = Makie.vspan!(ax1, xdelta + ll, xdelta + rr, color=eo ? :gray90 : :white)
                Makie.translate!(plt, Vec3f(0, 0, -100))
                eo = !eo
                @info "xdelta $xdelta"
                pts2 = map(p -> Point2f(xdelta + p[1] / 1000, p[3] / 1000), pts)
                Makie.lines!(mix1.scene2, pts2, color=col; inspector_label=(plt, i, v) -> string(col[i]), colormap=[:red, :blue], kwargs...)
                pts = map(p -> Point2f(xdelta + p[1] / 1000, y2y(p[2])), pts)
                plt = Makie.hlines!(ax2, [0, 0.5, 1])
                Makie.translate!(plt, Vec3f(0, 0, -10))
                Makie.lines!(mix2.scene2, pts, color=col; inspector_label=(plt, i, v) -> string(col[i]), colormap=[:red, :blue], kwargs...)
                plt = Makie.density!(ax2, filter(!isnan, last.(pts)), color=(:gray, 0.3), direction=:y, offset=xdelta + ll)
                Makie.translate!(plt, Vec3f(0, 0, 5))
                Main.xxxxvals = filter(!isnan, first.(pts))
                plt = Makie.density!(ax1, filter(!isnan, first.(pts)), color=(:gray, 0.3), offset=maximum(filter(!isnan, last.(pts2))))
                plt = Makie.density!(ax1, filter(!isnan, last.(pts2)), color=(:gray, 0.3), direction=:y, offset=xdelta + ll)
                Makie.translate!(plt, Vec3f(0, 0, 5))
                push!(ticks, (xdelta, replace(basename(r.arborpath), r".*(arbor\d+).*" => s"\1")))
                xdelta += rr
            end
        end
    end
    ax2.xticks[] = (first.(ticks), string.(last.(ticks)))
    Makie.linkxaxes!(ax1, ax2)
    #calc_arbor_align
end

function draw_with_arbor(df, dfarbor)
    fig = Makie.Figure()
    ax = Makie.LScene(fig[1, 1])
    for r in eachrow(df)
        demostrate_arbor!(ax, dfarbor, r.path)
    end
    fig
end






function plot_its_subtyps!(ax, ax2, df, umap, obj)
    typ2c = typ_color(df)
    cents = [SVector(median(umap[:, df[:, :typ].=="ITi"], dims=2)...) for t in ("ITc", "ITi")]
    pts = umap[:, df[:, :typ].=="ITs"]
    init = [argmin(j -> norm(c .- pts[:, j]), axes(pts, 2)) for c in cents]
    cls = Clustering.kmeans(pts, 2; init).assignments
    for a in (ax, ax2)
        Makie.mesh!(a, obj, color=(:gray, 0.05), transparency=true)
    end
    foreach(cls, eachrow(filter(r -> r.typ == "ITs", df))) do c, r
        Makie.lines!((ax, ax2)[c], load_swc_as_lines(r.path, mirrorby=NmtDat), color=typ2c(r.typ), transparency=true)
    end
    setup_camera_nmt_any(ax, :h)
    setup_camera_nmt_any(ax2, :h)
end

function select_target_auto(df, K)
    iii = [findall(==(c), df.cls) for c in unique(df.cls)]
    allprojs = filter!(is_projcol, names(df))
    filter!(p -> count('|', p) <= K, allprojs)
    filter!(allprojs) do p
        vals = df[:, p]
        t = HypothesisTests.OneWayANOVATest((vals[ii] for ii in iii)...)
        if endswith("|ic")(p) && startswith("nmtI")(p)
            Main.xxxt = t
        end
        if HypothesisTests.pvalue(t) < 0.05
            Main.xxxttt = t
        end
        HypothesisTests.pvalue(t) < 0.05
    end
    prev = deepcopy(allprojs)
    filter!(p -> !any([startswith("$p|")(q) for q in prev]), allprojs)
    # filter!(keys) do k
    #     maximum(projs[:, k]) > 2000.0f0
    # end
    sort!(unique!(allprojs))
end

function plot_its_g97_inj!(mix, df, obj)
    dfx = filter(r -> r.typ == "ITs" && r.sample == "234197", df)
    injs = sort(unique(dfx.inj))
    Makie.mesh!(mix.scene2, obj, color=(:gray, 0.05), transparency=true)
    for r in eachrow(dfx)
        Makie.lines!(mix.scene2, load_swc_as_lines(r.path; mirrorby=NmtDat), color=Makie.wong_colors()[findfirst(==(r.inj), injs)], transparency=true, label=r.inj)
    end
    Makie.axislegend(mix.axis, mix.scene2, unique=true)
    setup_camera_nmt_any(mix.axis, :hr)
end

function plot_all_arbors!(ax, df, dfarbor)
    for r in eachrow(df)
        swc = Gapr.load_swc(r.path)
        colors = shuffle!(Makie.Palette(:tab10).colors)
        ai = 0
        for rr in eachrow(dfarbor)
            rr.path != r.path && continue
            aa = Gapr.load_swc(r.arborpath)
            #for a
        end
    end
end


# Topo map related
# function plot_fibers(mix1, mix2, dfx, (xpars, ypars))
#     #col0, col1=extrema(dot.(skipmissing(dfx.srcpos), Ref(xpars)))
#     for r in eachrow(dfx)
#         pts = load_swc_as_lines(r.path)
#         #col=r.srcpos===missing ? NaN : dot(r.srcpos, xpars)
#         for mix in (mix1, mix2)
#             #Makie.lines!(mix.scene2, pts, color=col, colorrange=(col0, col1), transparency=true, nan_color=:black)
#             Makie.lines!(mix.scene2, pts, transparency=true)
#         end
#     end
#     for mix in (mix1, mix2)
#         Makie.mesh!(mix.scene2, obj, color=(:gray, 0.02), transparency=true)
#     end
#     setup_camera_nmt(mix1.axis => :h, mix2.axis => :s)
# end
# function plot_topo_coor(ax1, ax2, ax3, dfxx, (xpars, ypars))
#     cols = [dot(r.srcpos, xpars) for r in eachrow(dfxx)]
#     for ax in (ax1, ax2)
#         Makie.scatter!(ax, dfxx.srcpos, color=cols, marker=:circle, transparency=true)
#         Makie.scatter!(ax, dfxx.tgtpos, color=cols, marker=:+, transparency=true)
#     end
# end
#
# function plot_fiber_and_topo(gridpos, sel, topodat)
#     with_subfig(gridpos) do grid
#         ax1 = Makie.LScene(grid[1:2, 1], width=80, height=100, show_axis=false)
#         ax2 = Makie.LScene(grid[1, 2], width=100, height=80, show_axis=false)
#         ax3 = Makie.Axis(grid[2, 2], height=40)
#         plot_topo_coor(ax1, ax2, ax3, dfxx, (xpars, ypars))
#         with_mixed_scene(ax1, GLMakie) do mix1
#             with_mixed_scene(ax2, GLMakie) do mix2
#                 plot_fibers(mix1, mix2, dfx, xpars => ypars)
#             end
#         end
#     end
# end



function draw_pfc_non_pfc_dist!(ax, arbors, df)
    arbors = leftjoin(arbors, df, on=intersect(names(arbors), names(df))) |> dropmissing!
    filter!(r -> startswith("nmt|charm|Frontal|lat_PFC|")(r.somahier), arbors)
    #fig = Makie.Figure()
    #ax = Makie.Axis(fig[1, 1])

    @warn "$(size(arbors, 1))"
    ticks = []
    for (j, tgt) in enumerate(("nmt|charm|Frontal|lat_PFC" => "latPFC",
        "nmt|charm|Frontal|ACgG" => "ACgG(ACC/MCC)",
        "nmt|charm|Frontal|motor" => "motor",
        "nmt|charm|Frontal|OFC" => "OFC",
        "nmt|charm|Occipital", "nmt|charm|Parietal", "nmt|charm|Temporal",))
        tgt, tgtn = if tgt isa Pair
            tgt
        else
            tgt, replace(tgt, r".*\|" => "")
        end
        dfx = filter(r -> startswith(tgt)(r.arborhier), arbors)
        dist = reduce(hcat, dfx.fv_whist)
        dist = dist ./ sum(dist, dims=1) .* 3
        distm = median(dist, dims=2)
        Main.xxxq = quant = [quantile(r, (0.25, 0.75)) for r in eachrow(dist)]
        Makie.band!(ax, ([Vec2f(j + p[q], k) for (k, p) in enumerate(quant)] for q in 1:2)..., alpha=0.5f0)
        Makie.lines!(ax, j .+ distm[:], 1:10)
        push!(ticks, [j, tgtn])


        @warn "$tgt $(size(dfx, 1))"
    end
    ax.xticks[] = (first.(ticks), last.(ticks))
    ax.yticks[] = ([1, 5.5, 10], string.(["shallow", "middle", "deep"]))
    ax.title[] = "from latPFC"
    ax.xlabel[] = "Target"
    #fig
end

function plot_arbor_depth_vs_region!(ax, tgts, segs, uvws, hiers)
    labelof((t, ic)) = string(replace(t, r".*\|(\w+)$" => s"\1"), " ($ic)")
    for (i, (t, ic)) in enumerate(tgts)
        Makie.lines!(ax, i .+ stats ./ maximum(stats), (0:100) ./ 100, color=:black)
    end
    ax.xticks[] = (axes(tgts, 1), labelof.(tgts))
    rotatexlabel!(ax)
    ax.xlabel[] = "Target region"
    ax.ylabel[] = "Arbor depth"
    ax.yreversed[] = true
    ax.width[] = 160
    ax.height[] = 80
end

dist_of(v) = begin
    d = 0
    s = 0
    foreach(v) do (a, b)
        d += abs(a - b)
        s += a + b
    end
    d / s
end
function calc_cluster_relation(arbors, clusters)
    arbor2nrn = Dict([r.arborpath => r.path for r in eachrow(arbors)])
    dist(a, b) = begin
        s = Dict()
        for (c, d) in [(a, (1, 0)), (b, (0, 1))]
            for x in c
                x = arbor2nrn[x]
                s[x] = get(s, x, (0, 0)) .+ d
            end
        end
        pp = last.(collect(s))
    end

    [dist(a, b) for a in clusters, b in clusters]
end
function draw_clusters_link(arbors, clusters, cls)
    arbor2pos = Dict([r.arborpath => r.center for r in eachrow(arbors)])
    arbor2i = Dict([r.arborpath => rownumber(r) for r in eachrow(arbors)])
    g = Graphs.SimpleGraph(length(clusters) + size(arbors, 1))
    for c in unique(cls)
        ccc = findall(==(c), cls)
        for i in ccc, j in ccc
            if i != j
                Graphs.add_edge!(g, i, j)
            end
        end
    end
    clucents = [Point3f(median(reduce(hcat, [arbor2pos[a] for a in c]), dims=2)...) for c in clusters]
    for (ci, c) in enumerate(clusters)
        for a in c
            Graphs.add_edge!(g, arbor2i[a] + length(clusters), ci)
        end
    end
    GraphMakie.graphplot(g, layout=[clucents; arbors.center])
end

function draw_soma_dist_flat!(ax, df, objflat, objphy; rowcolor)
    highlights = [
        "nmt|charm|Frontal" => 0.5
        "nmt|charm|Occipital" => 0.5
        "nmt|charm|Parietal" => 0.5
        "nmt|charm|Temporal" => 0.5
        "nmt|charm|Frontal|lat_PFC" => (:red, 0.09)
        "nmt|charm|Frontal|OFC" => (:green, 0.09)
        "nmt|charm|Frontal|motor" => (:blue, 0.09)
    ]
    plot_flatmap_base!(ax, highlights, objflat, objphy)
    Makie.limits!(ax, -0.35, 0.0, -0.05, 0.25)
    Makie.scatter!(ax, df.somauv, markersize=1.0, color=rowcolor.(eachrow(df)), inspector_label=(plt, idx, pos) -> Tuple(df[idx, :]))
end

function draw_neurons_quick(df; rowcolor=nothing, mouse=false, kw...)
    fig = Makie.Figure()
    ax = Makie.LScene(fig[1, 1])
    Fig.draw_neurons(df, ax => :h, ; rowcolor, mirrorby=mouse ? Allen : NmtDat, kw...)
    obj = load(mouse ? "interm/allen-997.obj" : "interm/mesh-nmt-brain-lr.obj")
    Makie.mesh!(ax, obj, color=(:gray, 0.05), transparency=true)
    fig
end

function draw_fisher_test!(ax, a, b, c, d, xvals, yvals)
    stack = [2, 1, 2, 1]
    color = Makie.wong_colors()[stack]
    Makie.barplot!(ax, [1, 1, 2, 2], [a / c, 1 - a / c, b / d, 1 - b / d], stack=stack, color=color, colormap=:viridis, label=yvals[stack])
    Makie.text!(string.([a, c - a, b, d - b]), position=Makie.Point2f.([1, 1, 2, 2], [1, 0, 1, 0]), color=:black, align=(:center, :center))
    ax.xticks[] = (axes(xvals, 1), xvals)
    #ax.xticks[]=(axes(xvals, 1), xvals)
    unique(zip([Makie.PolyElement(polycolor=i) for i in color], yvals[stack]))
end

function pathway_heatmap(pathways, trunks)
    pathways = filter(r -> r.w >= 5, pathways)
    paths = sort(unique(trunks.path))
    p2i = Dict(reverse.(enumerate(paths)))
    c2i = Dict(reverse.(enumerate(pathways.trunkcls)))
    m = zeros(length(p2i), length(c2i))
    for r in eachrow(trunks)
        i = get(c2i, r.trunkcls, 0)
        i == 0 && continue
        m[p2i[r.path], i] = 1
    end
    inspector_label(p, (i, j), x) = begin
        string(paths[i], pathways[j, :trunkcls])
    end
    Makie.heatmap(m; inspector_label) |> display
    Makie.DataInspector()
    Main.xxxcls = deepcopy(pathways.trunkcls)
    m
end




