function plot_arbor_typ_dist(f::Function, ax, cortarbors, x=1)
    dfxx = filter(f, cortarbors)
    aclss = sort(unique(cortarbors.arborcls))
    cnt = [count(dfxx.arborcls .== c) for c in aclss]
    barplot!(ax, fill(x, length(aclss)), cnt ./ sum(cnt) * 100, stack=aclss, color=to_colormap(:Set1_8)[1:length(aclss)])
    #hidedecorations!(ax)
    #hidespines!(ax)
    #
    # ax = Axis(gridx[1, 2])
    # plot_arbor_typ_dist(ax, cortarbors) do r
    #     r.ipsi && matchhier("$tgt")(r.arborhier) && zz1data.select_neurons(Symbol(:long_it_ipsi_, subtgt))(r)
    # end
    # colgap!(gridx, 1, 1)
end

function plot_arbor_size_compare(grid, env; arborssize, _...)
    draw_planar_dist(ax, dfx, color) = begin
        draw_density!(ax, dfx.arbordiam; rowx=r -> 0, rowy=r -> r, rowcolor=r -> color, datalimits=(-Inf, Inf))
    end
    prevax = nothing
    for (kkk, tag) in enumerate([:LIP, :PCC, :STSd, :TE])
        dfx = filter(r -> matchhier("$tag")(r.arborhier) && r.ipsi, arborssize)
        ax = Axis(grid[1, kkk])
        draw_planar_dist(ax, dfx, :blue)
        if kkk > 1
            colgap!(grid, kkk - 1, 8)
            linkaxes!(ax, prevax)
        end
        N = size(dfx, 1)
        ax.title = "$tag ($N arbors)"
        prevax = ax
    end
    #return
    prevax = nothing
    for (kkk, tag) in enumerate((true, false))
        dfx = filter(r -> matchhier("Frontal")(r.arborhier) && r.ipsi == tag, arborssize)
        ax = Axis(grid[2, kkk])
        draw_planar_dist(ax, dfx, :blue)
        if kkk > 1
            colgap!(grid, kkk - 1, 8)
            linkaxes!(ax, prevax)
        end
        N = size(dfx, 1)
        ax.title = "$tag ($N arbors)"
        prevax = ax
    end
end

function plot_arbor_compare_tgt_depth(grid, env; cortarbors, _...)
    ax1 = Axis(grid[1, 1])
    for (kkk, tag) in enumerate(sort(unique(cortarbors.arborcls)))
        dfx = filter(r -> r.arborcls == tag, cortarbors)
        xx, yy = calc_fvhist(dfx.fv_whist)
        lines!(ax1, xx, yy, color=to_colormap(:Set1_8)[kkk], label="#$tag")
    end
    xaxis_as_layer(ax1, 0, 1)
    Legend(grid[2, 1], ax1, orientation=:horizontal) |> fix_line_legs
    ax1.width = 100
    ax1.height = 60
    ax1.ylabel = "Density"
    ylims!(ax1, 0, nothing)
    hideydecorations!(ax1, ticks=false, ticklabels=false, label=false)
    rowgap!(grid, 1, 2)
end

function plot_arbor_compare_tgt_entropy(grid, env; cortarbors, _...)
    prevax = nothing
    for (kkk, tag) in enumerate(sort(unique(cortarbors.arborcls)))
        dfx = filter(r -> r.arborcls == tag, cortarbors)
        ax = Axis(grid[kkk, 1])
        dd = reduce(hcat, dfx.fv_whist)
        dd = dd ./ (sum(dd, dims=1) .+ 1)
        ents = StatsBase.entropy.(eachcol(dd))
        hist!(ax, ents, bins=0.4:0.2:maximum(ents)+1, direction=:y, normalization=:probability, color=to_colormap(:Set1_8)[kkk])
        ax.ylabel = "Density"
        ax.xlabel = "Entropy"
        ax.yticks = 0:0.1:0.2
        ylims!(ax, 0, nothing)
        hideydecorations!(ax, ticks=false, ticklabels=false, label=false)
        if kkk > 1
            rowgap!(grid, kkk - 1, 4)
            linkaxes!(ax, prevax)
            hidexdecorations!(prevax, ticks=false, minorticks=false)
        end
        prevax = ax
    end
end

function plot_arbor_compare_planar_size(grid, env; arborssize, _...)
    draw_planar_dist(ax, dfx, color) = begin
        draw_density!(ax, dfx.arbordiam; rowx=r -> 0, rowy=r -> r, rowcolor=r -> color, datalimits=(0, 3000))
    end
    prevax = nothing
    for (kkk, tag) in enumerate(sort(unique(arborssize.arborcls)))
        dfx = filter(r -> r.arborcls == tag, arborssize)
        ax = Axis(grid[kkk, 1])
        draw_planar_dist(ax, dfx, :blue)
        ax.ylabel = "Density"
        ax.xlabel = "Arbor size (μm)"
        hideydecorations!(ax, ticks=false, ticklabels=false, label=false)
        if kkk > 1
            rowgap!(grid, kkk - 1, 8)
            linkaxes!(ax, prevax)
            hidexdecorations!(prevax, ticks=false, minorticks=false)
        end
        N = size(dfx, 1)
        ax.title = "#$tag ($N arbors)"
        xlims!(ax, 0, nothing)
        ylims!(ax, 0, nothing)
        prevax = ax
    end

    # *contra *front 
    dfx = filter(r -> matchhier("Frontal")(r.arborhier) && !r.ipsi, arborssize)
    write_stat(:ContraFrontalArborPlanarSize, meanandsem(dfx.arbordiam, "um"))
end

function plot_arbor_group_hist(grid, env; cortarbors, _...)
    cfgs = [
        (; ipsi=true, tgt=:LIP,);
        (; ipsi=true, tgt=:PCC,);
        (; ipsi=true, tgt=:STSd,);
        (; ipsi=true, tgt=:TE,);
        (; ipsi=true, tgt=:Frontal,);
        (; ipsi=false, tgt=:Frontal,)
    ]
    prevax = nothing
    for (kkk, cfg) in enumerate(cfgs)
        tgt = cfg.tgt
        dfx = filter(r -> matchhier("$tgt")(r.arborhier) && r.ipsi == cfg.ipsi, cortarbors)
        ax = Axis(grid[kkk, 1])
        cnts = zz1data.vote(dfx, :arborcls, :arborpath)
        s = sum(last.(cnts))

        tests = map(first.(cnts)) do cls
            n1 = count(dfx.arborcls .== cls)
            n2 = count(cortarbors.arborcls .== cls)
            a, b, c, d = n1, s - n1, n2, size(cortarbors, 1) - n2
            pval, odds = RCall.R"""t=fisher.test(matrix(c($a,$b,$c,$d),nrow=2),alternative="greater"); c(t$p.value,t$estimate)"""
            (; cls, pval, odds, vals=(; a, b, c, d))
        end
        pvals = [t.pval for t in tests]
        write_stat(:ArborClassEnrichMethod, :FisherExactTestGreater)
        lbls = map(tests, collect(RCall.R"""p.adjust($pvals, "fdr")""")) do t, padj
            write_stat(Symbol(:ArborClassEnrich, cfg.ipsi ? :Ipsi : :Contra, tgt, :With, t.cls, :Pval), padj)
            write_stat(Symbol(:ArborClassEnrich, cfg.ipsi ? :Ipsi : :Contra, tgt, :With, t.cls, :Cnts), t.vals)
            compact_pval(padj, ns="")
        end
        barplot!(ax, first.(cnts), 100 * last.(cnts) ./ s,)
        ax.xticks = (1:6, ["#$t" for t in 1:6])
        ax.xlabel = "Arbor type"
        ax.ylabel = "Proportion (%)"
        hidedecorations!(ax, ticks=false, ticklabels=false, label=false)
        ylims!(ax, 0, 60)
        if kkk > 1
            rowgap!(grid, kkk - 1, 8)
            linkaxes!(ax, prevax)
            hidexdecorations!(prevax, ticks=false, minorticks=false)
        end
        N = size(dfx, 1)
        ic = (cfg.ipsi ? "Ipsi" : "Contra")
        ax.title = "$tgt $(ic)lateral ($N arbors)"
        prevax = ax
    end
end

function plot_fig_longproj(fig, env; _...)
    with_caption(L"Long-range cortico-cortical projections of macaque PFC ITi neurons.")

    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"Four subytpes with long cortical projection.")
        plot_longproj_demo(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_flatmap(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_heatmap(grid, env; env...)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1][1, 1][1, 1], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_diverg(grid, env; env...)
    end
    with_subfig(grid[1, 1][1, 1][1, 2], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_diverg_soma(grid, env; env...)
    end
    with_subfig(grid[1, 1][2, 1], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_diverg_soma_demo(grid, env; env...)
    end
    with_subfig(grid[1, 2][1, 1], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_converg_temp(grid, env; env...)
    end
    with_subfig(grid[1, 2][2, 1], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_converg_temp_topo(grid, env; env...)
    end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"Divergence projection from G97-1bL.")
        plot_longproj_nmr(grid, env; env...)
    end

    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_longproj_converg(grid, env; env...)
    end
    ax1 = with_subfig(grid[1, 2][1, 1], env) do grid, env
        plot_longproj_converg_topo(grid, env; env...)
    end
    ax2 = with_subfig(grid[1, 2][2, 1], env) do grid, env
        plot_longproj_arbors(grid, env; env...)
    end
    ax2.ylabelpadding[] = 9.0
    #align_axis_label(:y, ax1, ax2; fix=pt(0))

    with_subfig(grid[1, 3], env) do grid, env
        plot_longproj_arbors_demo(grid, env; env...)
    end
    with_subfig(grid[1, 4], env) do grid, env
        plot_longproj_diagram(grid, env; env...)
    end
end

function plot_arbor_detect_res1(grid, env; df, arbors, obj, _...)
    demo_sel = r -> startswith("G97-1")(r.inj)
    dfx = filter(demo_sel, leftjoin(arbors, df, on=:path))

    ax = Makie.LScene(grid[1, 1], height=pt(130), width=pt(100), show_axis=false)
    setup_camera(ax => :h)
    with_mixed_scene(ax, GLMakie) do mix
        arb2col = Dict()
        per_arbor(pn, r, cond) = begin
            cond && return :transparent => 0.0
            r === nothing && return :transparent => 0.0
            c = get!(arb2col, r.arborpath) do
                rand(to_colormap(:Set1_8))
            end
            c => 0.3
        end
        per_node(pn, pa, n) = pa
        draw_subtrees(dfx, mix; per_arbor, per_node, drawsoma=false) #= , mix2, mix3 =#
    end
    draw_mesh_overlay([obj], ax)
end


function plot_find_arbors(fig, env; _...)
    with_caption(L"Segmentation of axon arbors by a machine learning algorithm.")
    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_arbor_detect_meth(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"Arbors identified in dataset G97-1")
        plot_arbor_detect_res1(grid, env; env...)
    end
    grid = fig[2, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_arbor_depth_dist(grid, env; env...)
    end
    grid = fig[3, 1]
    with_subfig(grid[1, 1][1, 1], env) do grid, env
        with_caption(L"Clustering of arbors based on laminar depth distribution.")
        plot_arbor_compare_tgt_depth(grid, env; env...)
    end
    with_subfig(grid[1, 1][2, 1], env) do grid, env
        with_caption(L"Clustering of arbors based on laminar depth distribution.")
        plot_arbor_compare_tgt_entropy(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"Arbors identified in dataset G97-1")
        plot_arbor_by_cls(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1], env) do grid, env
        with_caption(L"Clustering of arbors based on laminar depth distribution.")
        plot_arbor_group_hist(grid, env; env...)
    end
end
