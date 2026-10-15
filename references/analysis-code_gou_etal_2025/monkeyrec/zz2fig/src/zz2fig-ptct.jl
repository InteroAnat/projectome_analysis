function draw_cross_section_impl(xlims, ylims, uv2pt, axspecs...; label=nothing, addrat=0.09, addfix=999, linewidth=300, arrowsize=1000, color=:black)
    pts = Makie.Observable(fill(Point3f(NaN), 4))
    Makie.onany(xlims, ylims; update=true) do xl, yl
        addx = (xl[2] - xl[1]) * addrat + addfix
        addy = (yl[2] - yl[1]) * addrat + addfix
        p0 = uv2pt((xl[1] - addx, yl[1] - addy))
        p1 = uv2pt((xl[2] + addx, yl[1] - addy))
        p2 = uv2pt((xl[2] + addx, yl[2] + addy))
        p3 = uv2pt((xl[1] - addx, yl[2] + addy))
        pts[] .= (p0, p1, p2, p3)
        notify(pts)
    end
    for mix in axspecs
        arrows!(mix.gl, map(v -> v[[1, 1]], pts), map(v -> v[[2, 4]] .- v[[1, 1]], pts); linewidth, arrowsize, color)
        poly!(mix.gl, pts, [1, 2, 3, 1, 3, 4], color=(:black, 0.5), strokecolor=:black, strokewidth=0, transparency=true, fxaa=true)
        if label !== nothing
            text!(mix.ax, map(v -> v[1], pts), text="$label", align=(:left, :top))
        end
    end
end
function draw_cross_section(miduvs, uv2pt, ax, axspecs...; color2d=nothing, color3d=nothing, labelf1=nothing, calcf1=nothing, label=nothing)
    axspecs = parseaxspec.(axspecs)
    scatter!(ax, miduvs, color=color2d)

    for mix in axspecs
        scatter!(mix.gl, uv2pt.(miduvs), markersize=2, color=@something(color3d, :black))
    end
    draw_cross_section_impl(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits, uv2pt, axspecs...; label)
    if calcf1 !== nothing
        f1, (a, b, c) = calc_classify_f1_score(miduvs, calcf1)
        @warn "f1 $f1 $((a,b,c))"
        ablines!(ax, -c / b, -a / b, color=:black)
        if labelf1 !== nothing
            Label(labelf1, L"\fontfamily{Arial}$F_1$=%$(compact_str(f1))")
        end
    end
    if label !== nothing
        ax.ylabel = "$label"
    end
    hidedecorations!(ax, label=false)
end

function plot_pt_bundle_topo(grid, env; dfxct, dfxpt, trunks, obj, _...)
    dfx = leftjoin(trunks, dfxpt, on=:path) |> dropmissing!
    filter!(r -> zz1data.select_neurons(:pt_strict)(r) && zz1data.matchhier("sarm", "SC")(r.arborhier), dfx)
    filter!(r -> zz1data.matchhier("dlPFC")(r.somahier) && !r.qroot, dfx)
    select_one_arbor_per_neuron!(dfx)
    ttt = 0:0.1:1

    ax2 = Axis(grid[1, 1])
    rats = 0.0:0.05:1.0
    vlines = map(rats) do rat
        miduvs, _ = calc_trunks_midpoints2(dfx; rat, initu=(1, 0, 0))
        calc_topo_map_index(miduvs, dfx.soma_pos)
    end
    mvals = [v[2] for v in vlines]
    lines!(rats, mvals, color=:black)
    errorbars!(ax2, rats, mvals, mvals .- first.(vlines), last.(vlines) .- mvals, whiskerwidth=2, color=:black)
    #ax2.height[] = 60
    #ax2.width[] = 90
    ax2.xlabel[] = "Cross-section position"
    ax2.ylabel[] = "Topography index"
    minf1, _ = calc_classify_f1_score(fill(SVector(0, 0), size(dfx, 1)), axes(dfx, 1) .> size(dfx, 1) ÷ 2)
    hlines!(ax2, [minf1], color=:gray, linestyle=:dot)
    ax2
end

function plot_pt_topo_demo(grid, env; trunks, dfxpt, dfxct, topo_pt_sc, obj, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(vcat(dfxct, dfxpt))
    dfx = leftjoin(trunks, dfxpt, on=:path) |> dropmissing!
    filter!(r -> zz1data.select_neurons(:pt_strict)(r) && contains("SC")(r.arborhier), dfx)
    ttt = [0.3, 0.5, 0.7]

    ax4 = Makie.LScene(grid[1, 1][1, 1], width=pt(95), height=pt(75), show_axis=false)
    ax5 = Makie.LScene(grid[1, 1][2, 1], width=pt(95), height=pt(60), show_axis=false)
    rowgap!(content(grid[1, 1]), 1, -4)
    setup_camera(ax4 => :hr, ax5 => :s)
    with_mixed_scene(ax4, GLMakie) do mix
        with_mixed_scene(ax5, GLMakie) do mix2
            trunkcolor(rr) = begin
                rr === nothing && return :transparent
                return rowcolor(rr)
            end
            arborcolor(rr) = begin
                return :transparent
            end
            draw_subtrees(dfx, mix, mix2; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...)

            gridx = GridLayout(grid[1, 2], alignmode=Outside())
            for (i, t) in enumerate(ttt)
                miduvs, uv2pt = calc_trunks_midpoints2(dfx; rat=t, initu=(1, 0, 0))
                ax = Axis(gridx[i, 1], width=pt(40))
                i > 1 && rowgap!(gridx, i - 1, 2)
                draw_cross_section(miduvs, uv2pt, ax, mix, mix2; calcf1=dfx.cls .== 11, labelf1=grid[1, 2][i, 1, Top()], color2d=rowcolor.(eachrow(dfx)), label='α' + i - 1)
                #i == 2 && (ax.ylabel = "Cross-sections")
            end
        end
    end
    draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:SC => :red]], ax4, ax5)
    draw_mesh_overlay([obj], ax4, ax5)
    colgap!(grid, 1, -4)
end

function plot_pt_topo_summ(grid, env; trunks, dfxpt, dfxct, topo_pt_sc, obj, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(vcat(dfxct, dfxpt))
    dfx = leftjoin(trunks, dfxpt, on=:path) |> dropmissing!
    filter!(r -> zz1data.select_neurons(:pt_strict)(r) && contains("SC")(r.arborhier), dfx)
    tttt = [0.3, 0.5, 0.7]

    ax = Axis(grid[1, 1][1, 1])
    content(grid[1, 1]).alignmode = Outside()
    ttt = 0:0.05:1
    scores = map(ttt) do t
        miduvs, _ = calc_trunks_midpoints2(dfx; rat=t, initu=(1, 0, 0))
        f1, _ = calc_classify_f1_score(miduvs, dfx.cls .== 11)
        f1
    end
    lines!(ax, ttt, scores, color=:black)
    ax.ylabel[] = L"\fontfamily{Arial}$F_1$ score"
    hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    minf1, _ = calc_classify_f1_score(fill(SVector(0, 0), size(dfx, 1)), dfx.cls .== 11)
    hlines!(ax, [minf1], color=:gray, linestyle=:dot)
    ylims!(ax, minf1 - 0.05, 1)
    xlims!(ax, 0, 1)
    ax.xlabel[] = "Cross-section position"

    for (i, t) in enumerate(tttt)
        vlines!(ax, [t], color=:gray, linestyle=:dot)
        text!(ax, t, 1, text=string('α' + i - 1), align=(:left, :top), offset=(1, 0))
    end
    ax
end

function plot_ct_topo_demo(grid, env; trunks, dfxct, topo_pt_sc, obj, _...)
    rowlabel(r) = string("#", r.cls)
    #rowcolor = cls_temp_color(dfxct)
    rowcolor(r) = contains("vlPFC")(r.somahier) ? :red : :blue
    dfx = leftjoin(trunks, dfxct, on=:path) |> dropmissing!
    filter!(r -> zz1data.select_neurons(:ct_strict)(r) && contains("MD")(r.arborhier), dfx)
    ttt = [0.3, 0.5, 0.7]
    tgtcls = (r -> contains("vlPFC")(r.somahier)).(eachrow(dfx))

    ax4 = Makie.LScene(grid[1, 1][1, 1], width=pt(95), height=pt(70), show_axis=false)
    ax5 = Makie.LScene(grid[1, 1][2, 1], width=pt(95), height=pt(55), show_axis=false)
    rowgap!(content(grid[1, 1]), 1, -2)
    setup_camera(ax4 => :hr, ax5 => :s)
    with_mixed_scene(ax4, GLMakie) do mix
        with_mixed_scene(ax5, GLMakie) do mix2
            trunkcolor(rr) = begin
                rr === nothing && return :transparent
                return rowcolor(rr)
            end
            arborcolor(rr) = begin
                return :transparent
            end
            draw_subtrees(dfx, mix, mix2; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...)

            gridx = GridLayout(grid[1, 2], alignmode=Outside())
            for (i, t) in enumerate(ttt)
                miduvs, uv2pt = calc_trunks_midpoints2(dfx; rat=t, initu=(1, 0, 0))
                ax = Axis(gridx[i, 1], width=pt(40))
                i > 1 && rowgap!(gridx, i - 1, 2)
                draw_cross_section(miduvs, uv2pt, ax, mix, mix2; calcf1=rowcolor.(eachrow(dfx)) .== :red, labelf1=grid[1, 2][i, 1, Top()], color2d=rowcolor.(eachrow(dfx)), label='α' + i - 1)
            end
        end
    end
    draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:MD => :red]], ax4, ax5)
    draw_mesh_overlay([obj], ax4, ax5)
    axislegend(ax5, [PolyElement(color=c) for c in [:blue, :red]], ["from dlPFC", "from vlPFC"], position=:lb)
    colgap!(grid, 1, 1)
end
function plot_ct_topo_summ(grid, env; trunks, dfxct, topo_pt_sc, obj, _...)
    rowlabel(r) = string("#", r.cls)
    #rowcolor = cls_temp_color(dfxct)
    rowcolor(r) = contains("vlPFC")(r.somahier) ? :red : :blue
    dfx = leftjoin(trunks, dfxct, on=:path) |> dropmissing!
    filter!(r -> zz1data.select_neurons(:ct_strict)(r) && contains("MD")(r.arborhier), dfx)
    tttt = [0.3, 0.5, 0.7]
    tgtcls = (r -> contains("vlPFC")(r.somahier)).(eachrow(dfx))

    ax = Axis(grid[1, 1])
    ttt = 0:0.05:1
    scores = map(ttt) do t
        miduvs, _ = calc_trunks_midpoints2(dfx; rat=t, initu=(0, 0, 1))
        calc_topo_map_index(miduvs, dfx.soma_pos)
        f1, _ = calc_classify_f1_score(miduvs, tgtcls)
        f1
    end
    lines!(ax, ttt, scores, color=:black)
    #mvals = [v[2] for v in scores]
    #lines!(ttt, mvals, color=:black)
    #errorbars!(ax, ttt, mvals, mvals .- first.(scores), last.(scores) .- mvals, whiskerwidth=2, color=:black)

    ax.ylabel[] = L"\fontfamily{Arial}$F_1$ score"
    hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    minf1, _ = calc_classify_f1_score(fill(SVector(0, 0), size(dfx, 1)), tgtcls)
    hlines!(ax, [minf1], color=:gray, linestyle=:dot)
    ylims!(ax, minf1 - 0.05, 1)
    xlims!(ax, 0, 1)
    ax.xlabel[] = "Cross-section position"

    for (i, t) in enumerate(tttt)
        vlines!(ax, [t], color=:gray, linestyle=:dot)
        text!(ax, t, 1, text=string('α' + i - 1), align=(:left, :top), offset=(1, 0))
    end
    ax
end

function plot_ptct_topo_demo(grid, env; trunks, dfxpt, dfxct, obj, _...)
    dfxpt = filter!(r -> contains("dlPFC")(r.somahier), vcat(dfxpt, dfxct))
    rowlabel(r) = string("#", r.cls)
    rowcolor(r) = r.typ == "CT" ? :red : :blue
    dfx = leftjoin(trunks, dfxpt, on=:path) |> dropmissing!
    filter!(r -> (zz1data.select_neurons(:pt_strict)(r) || zz1data.select_neurons(:ct_strict)(r)) && zz1data.matchhier("sarm", "Thal")(r.arborhier), dfx)
    ttt = [0.3, 0.5, 0.7]
    select_one_arbor_per_neuron!(dfx)
    @warn "nrows $(size(dfx, 1))"
    @warn "nrows2 $(length(unique(dfx.path)))"

    ax4 = Makie.LScene(grid[1, 1][1, 1], width=pt(95), height=pt(70), show_axis=false)
    ax5 = Makie.LScene(grid[1, 1][2, 1], width=pt(95), height=pt(55), show_axis=false)
    rowgap!(content(grid[1, 1]), 1, -2)
    setup_camera(ax4 => :hr, ax5 => :s)
    with_mixed_scene(ax4, GLMakie) do mix
        with_mixed_scene(ax5, GLMakie) do mix2
            trunkcolor(rr) = begin
                rr === nothing && return :transparent
                return rowcolor(rr)
            end
            arborcolor(rr) = begin
                return trunkcolor(rr)
                return :transparent
            end
            draw_subtrees(dfx, mix, mix2; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...)

            gridx = GridLayout(grid[1, 2], alignmode=Outside())
            for (i, t) in enumerate(ttt)
                miduvs, uv2pt = calc_trunks_midpoints2(dfx; rat=t, initu=(1, 0, 0))
                ax = Axis(gridx[i, 1], width=pt(40))
                i > 1 && rowgap!(gridx, i - 1, 2)
                draw_cross_section(miduvs, uv2pt, ax, mix, mix2; calcf1=dfx.typ .== "CT", labelf1=grid[1, 2][i, 1, Top()], color2d=rowcolor.(eachrow(dfx)), label='α' + i - 1)

            end
        end
    end
    draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:MD => :red]], ax4, ax5)
    draw_mesh_overlay([obj], ax4, ax5)
    axislegend(ax5, [PolyElement(color=c) for c in [:blue, :red]], ["PT", "CT"], position=:lb)
    colgap!(grid, 1, 1)
end

function plot_ptct_topo_summ(grid, env; trunks, dfxpt, dfxct, topo_pt_sc, obj, _...)
    dfxpt = filter!(r -> contains("dlPFC")(r.somahier), vcat(dfxpt, dfxct))
    rowlabel(r) = string("#", r.cls)
    rowcolor(r) = r.typ == "CT" ? :red : :blue
    rowcolor = cls_temp_color(dfxpt)
    dfx = leftjoin(trunks, dfxpt, on=:path) |> dropmissing!
    filter!(r -> (zz1data.select_neurons(:pt_strict)(r) || zz1data.select_neurons(:ct_strict)(r)) && zz1data.matchhier("sarm", "Thal")(r.arborhier), dfx)
    select_one_arbor_per_neuron!(dfx)
    tttt = [0.3, 0.5, 0.7]

    ax = Axis(grid[1, 1])
    ax2 = Axis(grid[2, 1])
    rowgap!(grid, 1, 2)
    ttt = 0:0.1:1
    scores = map(ttt) do t
        miduvs, _ = calc_trunks_midpoints2(dfx; rat=t, initu=(1, 0, 0))
        f1, (a, b, c) = calc_classify_f1_score(miduvs, dfx.typ .== "CT")
        f1
    end
    lines!(ax, ttt, scores, color=:black)
    ax.ylabel[] = "F1 Score"
    hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    minf1, _ = calc_classify_f1_score(fill(SVector(0, 0), size(dfx, 1)), dfx.typ .== "CT")
    hlines!(ax, [minf1], color=:gray, linestyle=:dot)
    ylims!(ax, nothing, 1)
    xlims!(ax, 0, 1)
    ax.xlabel[] = "Cross-section position"
    for (i, t) in enumerate(tttt)
        vlines!(ax, [t], color=:gray, linestyle=:dot)
        text!(ax, t, 1, text=string('α' + i - 1), align=(:left, :top), offset=(1, 0))
    end

    for (dfx1, lbl, col) in [(dfx, "CT+PT", :black), (filter(r -> r.typ == "CT", dfx), "CT", :red), (filter(r -> r.typ == "PT", dfx), "PT", :blue)]
        vlines = map(ttt) do t
            miduvs, _ = calc_trunks_midpoints2(dfx1; rat=t, initu=(1, 0, 0))
            calc_topo_map_index(miduvs, dfx1.soma_pos)
        end
        mvals = [v[2] for v in vlines]
        lines!(ax2, ttt, mvals, color=col, label=lbl)
        errorbars!(ax2, ttt, mvals, mvals .- first.(vlines), last.(vlines) .- mvals, whiskerwidth=2, color=col)
        minf1, _ = calc_classify_f1_score(fill(SVector(0, 0), size(dfx, 1)), axes(dfx, 1) .> size(dfx, 1) ÷ 2)
        hlines!(ax2, [minf1], color=:gray, linestyle=:dot)
        break
    end
    #axislegend(ax2, ax2, position=:lt) |> fix_line_legs
    hideydecorations!(ax2, label=false, ticks=false, ticklabels=false)
    hidexdecorations!(ax, ticks=false)
    ax2.ylabel = "Topo. index"
    ax2.xlabel = "Cross-section position"
    linkxaxes!(ax, ax2)
    align_axis_label(:y, ax, ax2)
    #ax2.yaxisposition[] = :right
end

function plot_ctpt_diff_arbors(grid, env; dfxdlpfc, dfarbor, topo_ct_thal, topo_pt_thal, obj, highlights, objflat, objphy, _...)
    dfx = leftjoin(dfarbor, dfxdlpfc, on=:path) |> dropmissing!
    filter!(r -> zz1data.matchhier("sarm", "Thal")(r.arborhier), dfx)
    rowcolor(r) = r.typ == "CT" ? :red : :blue

    for n in names(dfx)
        startswith("f_")(n) || continue
        a = filter(r -> r.typ == "CT", dfx)[:, n]
        b = filter(r -> r.typ == "PT", dfx)[:, n]
        t = HypothesisTests.MannWhitneyUTest(a, b)
        p = HypothesisTests.pvalue(t)
        if p < 0.05
            @warn "$n $p $(quantile(a, 0:.2:1)), $(quantile(b, 0:.2:1)), "
        end
    end

    sample_avg_arbor(dfx, f, n) = begin
        shuffle!(sort!(filter(f, dfx), :f_totlen)[end÷5*2:end÷5*3, :])[1:n, :]
    end

    for (r, typ) in enumerate(("CT", "PT"))
        ax = LScene(grid[1, 1][1, r], height=pt(100), width=pt(35), show_axis=false)
        if r > 1
            colgap!(content(grid[1, 1]), r - 1, 0)
        end
        dfx1 = sample_avg_arbor(dfx, r -> r.typ == typ, 9)
        cam = (fov=9.0, cent=(6500.56, -900.47, 0), deye=(0.0, 0.0, 1.0) .* 25826.494335753636 .* 3.8, up=(1.00, 0.00, -0.00))
        setup_camera(ax => cam)
        with_mixed_scene(ax, GLMakie) do mix
            draw_arbors_sidebyside!(mix.scene2, dfx1; shift=0.95 + (r - 1) * 0.9, rowcolor, dx0=(r - 1) * 1000)
            #center!(ax.scene)
        end
        r == 2 && draw_scale_bar!(ax, 1000, [cameracontrols(ax).lookat[]],)
    end
    Legend(grid[1, 1, Bottom()], [PolyElement(color=c) for c in [:red, :blue]], ["CT", "PT"], orientation=:horizontal)

    specs = [
        (; name="Total axon\nlength (mm)", rowy=r -> r.f_totlen / 1000, ylim=50),
        (; name="Arbor size\n(mm)", rowy=r -> zz1data.arbor_size(r.f_volume) / 1000, ylim=3),
        (; name="No. of\nbranches", rowy=r -> r.f_numbranches,),
        # (; xname=x -> "Average branch\nlength (mm)", rowy=r -> r.f_totlen / 1000 / r.f_numbranches, ylim=5),
        # (; xname=x -> "Fractal\ndimension", rowy=r -> r.f_lm_Fractal_Dim_avg, ylim0=nothing),
    ]
    ax2s = Axis[]
    for (c, spec) in enumerate(specs)
        ax31 = Axis(grid[2, 1][c, 1][1, 1])
        ax2 = Axis(grid[2, 1][c, 1][2, 1])
        push!(ax2s, ax2)
        rowgap!(content(grid[2, 1][c, 1]), 1, 0)
        violin_and_compare!(ax2, dfx, ax31; rowx=r -> r.typ, spec.rowy, xname=x -> "$x", rowcolor, statif=(a, b) -> a[1] != b[1], datalimits=(-0.001, Inf), compact=false, scale=:width)
        #hideydecorations!(ax2, label=false, tick)
        hidexdecorations!(ax2, label=false, ticklabels=false)
        ax2.ylabel = "$(spec.name)"
        ax2.yminorticksvisible = false
        ylims!(ax2, get(spec, :ylim0, -0.001), get(spec, :ylim, nothing))
        if c > 1
            rowgap!(content(grid[2, 1]), c - 1, 2)
        end
        if c < length(specs)
            hidexdecorations!(ax2)
        end
    end
    align_axis_label(:y, ax2s...)
    content(grid[2, 1]).alignmode = Outside()
    rowgap!(grid, 1, 2)
    rowgap!(content(grid[2, 1]), 1, 0)
end

function plot_fig_ctpt(fig, env; _...)
    with_caption(L"Subcortical projections of macaque PFC PT/CT neurons.")

    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"PT neurons.")
        plot_pt_demo(grid, env; env...)
    end
    with_subfig(grid[1, 2][1, 1], env) do grid, env
        with_caption(L"PT neurons.")
        plot_pt_flatmap(grid, env; env...)
    end
    with_subfig(grid[1, 2][2, 1], env) do grid, env
        with_caption(L"PT neurons.")
        plot_pt_heatmap(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1][1, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_pt_topo_demo(grid, env; env...)
    end
    ax1 = with_subfig(grid[1, 3][1, 1][1, 2][1, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_pt_topo_summ(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1][1, 2][2, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        ax2 = plot_pt_topo_term(grid, env; env...)
        align_axis_label(:y, ax1, ax2)
    end
    with_subfig(grid[1, 3][2, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_pt_sc_arbors(grid, env; env...)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"CT neurons and topographic mapping to MD.")
        plot_ct_topo_demo(grid, env; env...)
    end
    ax1 = with_subfig(grid[1, 2][1, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_ct_topo_summ(grid, env; env...)
    end
    with_subfig(grid[1, 2][2, 1], env) do grid, env
        with_caption(L"CT neurons and topographic mapping to MD.")
        ax2 = plot_ct_topo(grid, env; env...)
        align_axis_label(:y, ax1, ax2)
    end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_ct_md_arbors(grid, env; env...)
    end

    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_ptct_topo_demo(grid, env; env...)
    end
    with_subfig(grid[1, 2][1, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_ptct_topo_summ(grid, env; env...)
    end

    with_subfig(grid[1, 3][1, 1], env) do grid, env
        with_caption(L"")
        plot_ctpt_diff(grid, env; env...)
    end

    with_subfig(grid[1, 3][2, 1], env) do grid, env
        with_caption(L"")
        plot_ctpt_diff_demo(grid, env; env...)
    end

    GridLayout(grid[1, 4][1, 1], width=pt(85))
    with_subfig(fig[2, 1][1, 4], env; height=pt(260), tellheight=false) do grid, env
        with_caption(L"")
        plot_ctpt_diff_arbors(grid, env; env...)
    end
end

function plot_pt_targets(grid, env; dfxpt, dfarbor, obj, _...)
    for (iii, tgt) in enumerate((:SC, :MD, :SN, :Pn, :Pd, "LC-Me5"))
        row, col = divrem(iii - 1, 2) .+ 1
        gridx = GridLayout(grid[row, col])
        row > 1 && rowgap!(grid, row - 1, 6)
        col > 1 && colgap!(grid, col - 1, 4)
        ax = LScene(gridx[1, 1], width=pt(116), height=pt(82), show_axis=false)
        translate!(ax.scene, 0, 0, 5)
        ax2 = LScene(gridx[2, 1], width=pt(116), height=pt(95), show_axis=false)
        rowgap!(gridx, 1, -21)
        dfx = filter(r -> zz1data.matchhier("sarm", "$tgt")(r.arborhier), dfarbor)
        sort!(dfx, :f_totlen, rev=true)
        leftjoin!(dfx, dfxpt, on=:path) |> dropmissing!
        src = areaname(dfx[1, :somahier])
        Label(grid[row, col, Top()], "$src → $tgt"; set_font_h2x()...)
        nrns = dfx.path[1:1]
        arbors = Set(dfx.arborpath)
        setup_camera(ax => :s, ax2 => :hr)
        per_neuron(r) = begin
            :black
        end
        per_arbor(pn, rr, cond) = begin
            if rr !== nothing && rr.arborpath in arbors
                return pn => pt(1.0)
            end
            Makie.to_color(:gray) => pt(0.7)
        end
        per_node(pn, pa, n) = pa
        with_mixed_scene(ax, GLMakie) do mix
            with_mixed_scene(ax2, GLMakie) do mix2
                draw_subtrees(filter(r -> r.path in nrns, dfx), mix, mix2; autotrunk=false, mirrorby=NmtDat, per_neuron, per_arbor, per_node)
            end
        end
        draw_contours([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [tgt => :red]], ax, ax2)
        draw_mesh_overlay([obj], ax, ax2)
    end
end

