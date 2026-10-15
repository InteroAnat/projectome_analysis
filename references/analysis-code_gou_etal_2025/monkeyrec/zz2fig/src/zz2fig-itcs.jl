function plot_itc_topo_demo(grid, env; trunks, dfxitc, obj, _...)
    dfxitc = filter(r -> zz1data.matchhier("lat_PFC")(r.somahier) && !r.qroot, dfxitc)
    rowcolor(r) = zz1data.matchhier("dlPFC")(r.somahier) ? :red : :blue
    dfx = leftjoin(trunks, dfxitc, on=:path) |> dropmissing!
    filter!(r -> zz1data.matchhier("charm", "Frontal")(r.arborhier) && !r.ipsi, dfx)
    select_one_arbor_per_neuron!(dfx, orderby=r -> norm(r.center - NmtDat.mirror(r.soma_pos)))

    ax4 = Makie.LScene(grid[1, 1], width=pt(86), height=pt(70), show_axis=false)
    ax5 = Makie.LScene(grid[2, 1], width=pt(86), height=pt(50), show_axis=false)
    rowgap!(grid, 1, -2)
    #rowgap!(content(grid[1, 1]), 1, 0)
    setup_camera(ax4 => :hf, ax5 => :cf)
    with_mixed_scene(ax4, GLMakie) do mix
        with_mixed_scene(ax5, GLMakie) do mix2
            trunkcolor(rr) = begin
                rr === nothing && return :transparent
                return rowcolor(rr)
            end
            arborcolor(rr) = begin
                return :transparent
            end
            draw_subtrees(dfx, mix, mix2; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...,)

            midpts, miduvs = calc_trunks_midpoints_cc(dfx)
            for m in (mix, mix2)
                scatter!(m.scene2, midpts, markersize=2, color=:black, transparency=true, fxaa=true)
            end
        end
    end
    draw_mesh_overlay([obj], ax4, ax5)

    ax00 = Axis(grid[3, 1][1, 1])
    text!(ax00, 0.5, 1, text="Cross-section", space=:relative, align=(:center, :top), offset=(0, -2))
    ax00.autolimitaspect[] = 1
    content(grid[3, 1]).alignmode[] = Outside()
    rowgap!(grid, 2, 4)
    #filter!(r->r.soma_pos[2]<58000, dfx)
    midpts, miduvs = calc_trunks_midpoints_cc(dfx)
    scatter!(ax00, miduvs, color=rowcolor.(eachrow(dfx));)
    setup_axis_by_orient!(ax00, :x, first.(miduvs) ./ 1000, SVector(0, -1, 0))
    setup_axis_by_orient!(ax00, :y, last.(miduvs) ./ 1000, SVector(0, 0, 1))
    sel = rowcolor.(eachrow(dfx)) .== :red
    ttt = HypothesisTests.MannWhitneyUTest(first.(miduvs[sel]), first.(miduvs[.!sel]))
    @warn "t1 $ttt"
    ttt = HypothesisTests.MannWhitneyUTest(last.(miduvs[sel]), last.(miduvs[.!sel]))
    @warn "t2 $ttt"

    Legend(grid[3, 1][1, 2], [PolyElement(color=c) for c in (:red, :blue)], ["dlPFC", "vlPFC"])
    colgap!(content(grid[3, 1]), 1, 2)

    #f1, (a, b, c) = calc_classify_f1_score(miduvs, [Data.matchhier("dlPFC")(r.somahier) for r in eachrow(dfx)])
    #@warn "f1 $f1 $((a,b,c))"
    #ablines!(ax00, -c / b, -a / b, color=:black)
    #hidedecorations!(ax00)
    #f1 = round(f1, sigdigits=3)
    #text!(ax00, 1, 0, text="F1=$f1", space=:relative, align=(:right, :bottom))
end

function plot_itc_topo_summ(grid, env, topodir=2; trunks, dfxitc, _...)
    dfxitc = filter(r -> zz1data.matchhier("lat_PFC")(r.somahier) && !r.qroot, dfxitc)
    dfxitc = filter(r -> zz1data.matchhier("lat_PFC")(r.somahier) && !r.qroot, dfxitc)
    rowcolor(r) = zz1data.matchhier("dlPFC")(r.somahier) ? :red : :blue
    dfx = leftjoin(trunks, dfxitc, on=:path) |> dropmissing!
    filter!(r -> zz1data.matchhier("charm", "Frontal")(r.arborhier) && !r.ipsi, dfx)
    select_one_arbor_per_neuron!(dfx, orderby=r -> norm(r.center - NmtDat.mirror(r.soma_pos)))

    ax = Axis(grid[1, 1], height=80)
    #rowgap!(grid, 1, 2)

    aaa = last.(filter(r -> rowcolor(r) === :red, dfx).soma_pos)
    bbb = last.(filter(r -> rowcolor(r) === :blue, dfx).soma_pos)
    ttt = HypothesisTests.MannWhitneyUTest(aaa, bbb)
    @warn "$ttt"
    midpts, miduvs = calc_trunks_midpoints_cc(dfx)

    xpars, ypars = zz1data.find_best_coor_axis(dfx.soma_pos, midpts, aligned_x=topodir, aligned_y=topodir)
    @warn "$xpars, $ypars"
    @warn "$xpars $ypars"
    if sum(ypars) > 0
        ypars, xpars = -ypars, -xpars
    end
    ##xpars = (1, 0, 0)
    insertcols!(dfx, :xx => dot.(dfx.soma_pos, Ref(xpars)), :yy => dot.(midpts, Ref(ypars)))
    ttt = HypothesisTests.MannWhitneyUTest(dfx.xx, dfx.yy)
    @warn "ttt $ttt"
    linear_reg_scatter!(ax, dfx; rowx=r -> r.xx / 1000, rowy=r -> r.yy / 1000, rowcolor, force=true, legpos=grid[1, 1, Top()])
    #rowgap!(grid, 1, 2)
    ax.xlabel[] = "Soma position"
    ax.ylabel[] = "Cross-section pos."
    setup_axis_by_orient!(ax, :x, dfx.xx ./ 1000, xpars)
    setup_axis_by_orient!(ax, :y, dfx.yy ./ 1000, ypars)
    Label(grid[1, 1, Top()], "", height=pt(10))
end

function plot_arbor_cluster_all(grid, env; df, arborsg97, obj, arbordistg97, _...)
    ax1 = LScene(grid[1, 1], height=pt(192), show_axis=false)
    ax2 = LScene(grid[2, 1], height=pt(170), show_axis=false)
    rowgap!(grid, 1, 0)
    setup_camera(ax1 => :hr, ax2 => :s)
    clu2col = Dict()
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            dfx = filter(r -> r.inj == "G97-1bL", arborsg97)
            per_arbor = (_, rr, cond) -> begin
                cond && return to_color(:transparent)
                rr === nothing && return to_color(:transparent)
                get!(clu2col, rr.arborcluster) do
                    to_colormap(:tab10)[length(clu2col)%10+1]
                end
            end
            per_node = (pn, pa, n) -> pa => pt(0.3)
            draw_subtrees(dfx, mix1, mix2; autotrunk=false, per_arbor, per_node, mirrorby=NmtDat, drawsoma=false)
        end
    end
    draw_mesh_overlay([obj], ax1, ax2)
end

function draw_arbors_mip(dfx, axspecs...; get_entropy, scale=6)
    p2aa = Dict()
    for r in eachrow(dfx)
        aa = get!(() -> String[], p2aa, r.path)
        push!(aa, r.arborpath)
    end
    anodes = Set{Int}()
    cams = [camera(ax) for ax in axspecs]
    canvases = [fill((0.0f0, 0.0f0), ax.width[] * scale, ax.height[] * scale) for ax in axspecs]
    @showprogress "draw" for (p, aa) in pairs(p2aa)
        empty!(anodes)
        for a in aa
            swcx = Gapr.load_swc(a)
            fixid(id) = id % 100_000_000_000
            for n in swcx.nodes
                push!(anodes, fixid(n.id))
            end
        end
        swc = Gapr.load_swc(p)
        for n in swc.nodes
            n.paridx == 0 && continue
            !(n.id in anodes) && continue
            ent = get_entropy(n.pos)
            pn = swc[n.parind]
            for ttt in 0:0.1:10
                pos = ttt * n.pos + (1 - ttt) * pn.pos
                foreach(cams, canvases) do cam, canvas
                    x, y = round.(Int, Makie.project(cam, :data, :pixel, pos)[1:2] .* scale)
                    if checkbounds(Bool, canvas, x, y)
                        canvas[x, y] = max(canvas[x, y], (ent...,))
                    end
                end
            end
        end
    end
    foreach(axspecs, canvases) do ax, canvas
        plt = image!(Makie.camrelative(ax.scene), (0, 1), (1, 0), exp.([v[2] > 0 ? v[1] : NaN for v in canvas[1:end, end:-1:1]]), nan_color=:black, colormap=:inferno, colorrange=(1, 5))
        translate!(plt, 0, 0, -0.5)
    end
end

function plot_arbor_cluster_all_entropy(grid, env; df, arborsg97, obj, arbordistg97, _...)
    idx2col = zz1data.cached_result(:calcentmap2) do
        res = Dict{CartesianIndex,Tuple{Float32,Float32}}()
        tif = load("./monkeytemp/entropy-density-G97-1bL.tif")
        for (i, v) in pairs(tif)
            !isnan(v.g) && v.g > 0 && push!(res, i => (v.r, v.g))
        end
        res
    end
    get_entropy(pos) = begin
        idx = NmtDat.pos2idx(pos, (50, 50, 50))
        get(idx2col, CartesianIndex(idx...), (0, 0))
    end

    ax1 = LScene(grid[1, 1], width=pt(240), height=pt(192), show_axis=false)
    ax2 = LScene(grid[2, 1], width=pt(240), height=pt(170), show_axis=false)
    rowgap!(grid, 1, 0)
    setup_camera(ax1 => :hr, ax2 => :s)
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            dfx = filter(r -> r.inj == "G97-1bL", arborsg97)
            draw_arbors_mip(dfx, mix1.axis, mix2.axis; get_entropy)

            cb = Colorbar(grid[2, 1, BottomLeft()], limits=(1, 5), vertical=false, tellwidth=false, tellheight=false, flipaxis=true, colormap=:inferno, halign=:left, valign=:bottom)
            cb.ticklabelsvisible = false
            cb.width = pt(80)
            cb.label = "Entropy"
            cb.labelcolor = :white
        end
    end
    draw_mesh_overlay([obj => (; color=:white)], ax1, ax2)
end

function plot_fig_it_contra_str(fig, env; _...)
    with_caption(L"Contralateral and striatal projections of macaque PFC IT neurons.")

    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"Subytpes of contra-lateral projecting neurons.")
        plot_itc_demo(grid, env; env...)
    end
    with_subfig(grid[1, 2][1, 1], env) do grid, env
        with_caption(L"Subytpes of contra-lateral projecting neurons.")
        plot_itc_demo_hm(grid, env; env...)
    end
    with_subfig(grid[1, 2][2, 1], env) do grid, env
        with_caption(L"Subytpes of contra-lateral projecting neurons.")
        plot_itc_homotopy(grid, env; env...)
    end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"Correlation between ispi- and contra-lateral axon lengths.")
        plot_itc_topo_demo(grid, env; env...)
    end
    with_subfig(grid[1, 4][1, 1][1, 1], env) do grid, env
        with_caption(L"Correlation between ispi- and contra-lateral axon lengths.")
        plot_itc_topo_summ(grid, env; env...)
    end
    with_subfig(grid[1, 4][1, 1][1, 2], env) do grid, env
        with_caption(L"Subytpes of contra-lateral projecting neurons.")
        plot_itc_ipsicontra_len(grid, env; env...)
    end
    with_subfig(grid[1, 4][2, 1], env) do grid, env
        with_caption(L"Correlation between ispi- and contra-lateral axon lengths.")
        plot_itc_ipsicontra_arbors(grid, env; env...)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1][1, 1], env) do grid, env
        with_caption(L"Correlation between ispi- and contra-lateral axon lengths.")
        plot_subreg_flat_conn_contra(grid, env; env...)
    end
    with_subfig(grid[1, 1][2, 1], env) do grid, env
        with_caption(L"Correlation between ispi- and contra-lateral axon lengths.")
        #plot_itc_acgg_demo(grid, env; env...)
        plot_itc_acgg_combo(grid, env; env...)
    end

    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"Subytpes of Striatum projecting neurons.")
        plot_its_demo(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1][1, 1][1, 1], env) do grid, env
        with_caption(L"Subytpes of Striatum projecting neurons.")
        plot_its_demo_hm(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1][1, 1][2, 1], env) do grid, env
        plot_its_topo(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1][1, 2], env) do grid, env
        with_caption("largest distance between these neurons <= 3mm")
        plot_arbor_cluster2(grid, env; env...)
    end
    with_subfig(grid[1, 3][2, 1][1, 1], env) do grid, env
        with_caption("largest distance between these neurons <= 3mm")
        plot_arbor_cluster_conn(grid, env; env...)
    end
    with_subfig(grid[1, 3][2, 1][1, 2], env) do grid, env
        with_caption("largest distance between these neurons <= 3mm")
        plot_arbor_cluster_numtgt(grid, env; env...)
    end
    GridLayout(grid[1, 3][2, 1][1, 3])
    with_subfig(grid[1, 3][2, 1][1, 3][1, 1], env; tellheight=false, height=pt(75), valign=:bottom) do grid, env
        with_caption("largest distance between these neurons <= 3mm")
        plot_arbor_cluster_diagram(grid, env; env...)
    end
end

function plot_fig_arbor_patches(fig, env; _...)
    with_caption(L"Systematic identification of arbor patches and stripes in the macaque brain (dataset G97-1).")

    grid = fig[1, 1]
    with_subfig(grid[1, 1][1, 1], env) do grid, env
        plot_itc_topo_summ(grid, env, 3; env...)
    end
    with_subfig(grid[1, 1][1, 2], env) do grid, env
        plot_itc_str_combo(grid, env; env...)
    end
    prev_grid = grid

    grid = fig[2, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_arbor_cluster_all(grid, env; env...)
    end
    GridLayout(grid[1, 2])
    with_subfig(prev_grid[1, 2], env) do grid, env
        grid.tellheight = false
        grid.valign = :top
        plot_arbor_cluster_all_entropy(grid, env; env...)
    end

    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        LTeX(grid[1, 1], PDFDocument(read("monkeytemp/wyz-upload/figs8-arbor-stripe-patch-display.pdf", String)), scale=0.9667)
    end
end
