function plot_func_table(texf, functable)
    fixstr(t) = begin
        replace(t, "_" => "\\_")
    end
    count_rows(tbl, row, f::Function) = begin
        n = 0
        for k in row:size(tbl, 1)
            f(tbl[k, :]) != f(tbl[row, :]) && break
            n += 1
        end
        n
    end
    count_rows(tbl, row, col) = count_rows(tbl, row, r -> r[col])
    open(texf, "w") do io
        println(
            io,
            raw"""
                %\documentclass[preprint,10pt]{elsarticle}
                \documentclass[a4paper,10pt]{article}
                \usepackage{tabularray}
                \usepackage{xcolor}

\NewTblrTheme{fancy}{
\DefTblrTemplate{contfoot-text}{default}{\fontsize{8}{0}\selectfont\itshape (Continued on next page)}
\SetTblrTemplate{firsthead}{empty}
\DefTblrTemplate{middlehead,lasthead}{default}{
\begin{tblr}{width=1.0\textwidth,colspec={XlX},row{1-Z}={font=\fontsize{8}{0}\selectfont}}
\hline[1pt]
\SetCell[c=3]{l} {\textbf{\textit{Continued}}} &x&x\\
\end{tblr}\vspace{-1.5ex}%
}
}

                \begin{document}
                \sloppy
            """
        )
        println(io, raw"""A. Projection pattern of neuron clusters.""")
        println(
            io,
            raw"""
\begin{longtblr}[theme = fancy,]{
baseline=T,
colspec = {Q X X}, width = 1.0\linewidth,
rowhead = 1, rowfoot = 0,
row{odd} = {gray9},,
row{1-1} = {white},
hline{1} = {2pt},
row{1-Z}={font=\fontsize{8}{0}\selectfont}
}
Cluster & Soma Region & Target Region \\
\hline[1pt]""")
        preva = nothing
        cc = 0
        for r in eachrow(functable.proj)
            a, b, c, d, e, f = r[[:cluster, :neuron_type, :soma_region_name, :target_region_name, :if_target_in_ipsi, :if_target_in_contra]]
            ef = ["XXX", " (Contralateral)", "", " (Bilateral)"][(; yes=2, no=0)[Symbol(e)]+(; yes=1, no=0)[Symbol(f)]+1]
            if (a, b) != preva
                rownumber(r) != 1 && println(io, "\\hline")
                n = count_rows(functable.proj, rownumber(r), r -> (r.cluster, r.neuron_type))
                println(io, "\\SetCell[r=$n]{h,bg=white}{\\#$a ($b)} & $(areaname(c)) & $(areaname(d))$ef \\\\")
                cc += 1
            else
                println(io, " & $(areaname(c)) & $(areaname(d))$ef \\\\")
            end
            preva = (a, b)
        end
        println(io, raw"""\hline[2pt]\end{longtblr}""")
        println(io, raw"""B. Functions of brain areas.""")
        println(
            io,
            raw"""
\begin{longtblr}[theme = fancy,]{
baseline=T,
colspec = {Q X}, width = 1.0\linewidth,
rowhead = 1, rowfoot = 0,
row{odd} = {gray9},,
row{1-1} = {white},
hline{1} = {2pt},
row{1-Z}={font=\fontsize{8}{0}\selectfont}
}
Brain Area & Function \\
\hline[1pt]""")
        # \multirow{2}{*}{2 Rows}
        # & \multicolumn{2}{c|}{2 Columns}
        # & \multicolumn{2}{c|}{\multirow{2}{*}{2 Rows 2 Columns}} \\
        preva = nothing
        cc = 0
        for r in eachrow(functable.areafunc)
            a, b = r.target_region_name, r.function
            if a != preva
                n = count_rows(functable.areafunc, rownumber(r), :target_region_name)
                rownumber(r) != 1 && println(io, "\\hline")
                c = iseven(cc) ? "gray9" : "white"
                println(io, "\\SetCell[r=$n]{h,bg=white}{$(fixstr(a))} & $(fixstr(b)) \\\\")
                cc += 1
            else
                println(io, " & $(fixstr(b)) \\\\")
            end
            preva = a
        end
        println(io, raw"""\hline[2pt]\end{longtblr}""")
        println(io, raw"""C. Function categories.""")
        println(
            io,
            raw"""
\begin{longtblr}[theme = fancy,]{
baseline=T,
colspec = {X X X}, width = 1.0\linewidth,
rowhead = 1, rowfoot = 0,
row{odd} = {gray9},,
row{1-1} = {white},
hline{1} = {2pt},
row{1-Z}={font=\fontsize{8}{0}\selectfont}
}
Function & Category & Category \\
\hline[1pt]""")
        preva = nothing
        prevb = nothing
        cc = 0
        for r in eachrow(functable.funcgrp)
            a, b, c = r.function_class_big, r.function_class_small, r.function_name
            if a != preva
                @assert b != prevb
                n1 = count_rows(functable.funcgrp, rownumber(r), :function_class_small)
                n2 = count_rows(functable.funcgrp, rownumber(r), :function_class_big)
                rownumber(r) != 1 && println(io, "\\hline")
                println(io, "$(fixstr(c)) & \\SetCell[r=$n1]{h,bg=white}{$(fixstr(b))} & \\SetCell[r=$n2]{h,bg=white}{$(fixstr(a))} \\\\")
                cc += 1
            elseif b != prevb
                println(io, "\\cline[dashed]{1-2}")
                n1 = count_rows(functable.funcgrp, rownumber(r), :function_class_small)
                println(io, "$(fixstr(c)) & \\SetCell[r=$n1]{h,bg=white}{$(fixstr(b))} & \\\\")
            else
                println(io, "$(fixstr(c)) & & \\\\")
            end
            preva = a
            prevb = b
        end
        println(io, raw"""\hline[2pt]\end{longtblr}""")
        println(
            io,
            raw"""
            \end{document}
        """
        )
    end
end

import UMAP
function plot_neuron_func_alg(grid, env; obj, functable, funcdist, _...)
    #grid.width = 156
    grid.height = 101
    eg_funcs = [
        [
            "Scene processing",
            "Face processing",
            "Object recognition",
            "Color perception",
            "Numerosity representation",
        ],
        [
            "Visual processing",
            "Object recognition",
            "Color perception",
        ],
        [
            "Visual processing",
            "Emotional processing",
            #"Non-spatial information processing",
            "Visuoauditory integration",
            "Object recognition",
        ]
    ]
    Random.seed!(7)
    uniqfuncs = unique(functable.areafunc[:, :function])
    uniqfuncs = sort(unique(vcat(shuffle!(uniqfuncs)[1:20], reduce(vcat, eg_funcs))))
    @warn "$uniqfuncs"
    dm = [Float64(get(funcdist, f => g, 1.0)) for f in uniqfuncs, g in uniqfuncs]
    dm = Symmetric((dm + dm') / 2)
    umap = UMAP.umap(dm, 2; metric=:precomputed)
    ax1 = Axis3(grid[2, 1], azimuth=-1.9, aspect=:data, elevation=0.2)
    ax1.protrusions[] = 0
    LTeX(grid[1, 1], compile_tikz_str(raw"\fontsize{6}{0}\selectfont $$f^*=\mathop{\mathrm{arg\,min}}_{f}\sum_j w_j\min_{g\in F_j}|f-g|$$"), tellwidth=false, tellheight=false, halign=:left, valign=:top)
    tgti = indexin(intersect(eg_funcs...), uniqfuncs)[1]
    kscale = 5
    #with_mixed_scene(ax1, GLMakie) do mix1
    (a, b), (c, d) = extrema([umap .- 0.3 umap .+ 0.3], dims=2)
    for (k, eg) in enumerate(eg_funcs)
        kk = k * kscale
        scatter!(ax1, [kk; umap], color=:gray, markersize=pt(1.5), transparency=true)
        ii = something.(filter(!isnothing, indexin(eg, uniqfuncs)))
        @warn "$eg $ii"
        scatter!(ax1, [kk; umap[:, ii]], color=:black, markersize=[i == tgti ? pt(3) : pt(3) for i in ii], strokecolor=[i == tgti ? (:red, 0) : (:red, 0) for i in ii], strokewidth=pt(0.5), transparency=true)
        #poly!(ax1, Point3f[(kk, 0, 0), (kk, 0, 1), (kk, 1, 1), (kk, 1, 0)], color=:gray)
        rect = Rect2f((c, a), (d, b) .- (c, a))
        #p1 = poly!(ax1, rect, color=(:blue, 0.3), transparency=true)
        p1 = band!(ax1, c:0.01:d, a, b, color=((:Red, :Green, :Blue)[k], 0.3))
        #p2 = wireframe!(ax1, rect, color=:black)
        p2 = arrows!(ax1, Point2f[(c, b), (c, b)], Point2f[(d - c, 0), (0, a - b)], color=:black, linewidth=pt(0.5), arrowsize=pt(0))
        p3 = text!(ax1, k < 2 ? Point2f(d, b) : Point2f(c, a), text=k < 2 ? "Soma functions" : "Target functions", fontsize=pt(6), align=(:center, k < 2 ? :bottom : :top))
        k > 0 && (p3.visible[] = false)
        for p in [p1, p2, p3]
            Makie.rotate!(p, 0, -sqrt(1 / 2), 0, sqrt(1 / 2))
            Makie.translate!(p, kk, 0, 0)
        end
    end
    Makie.xlims!(ax1, (1 - 0.2 + 0.1) * kscale, (3 + 0.2 + 0.1) * kscale)
    Makie.ylims!(ax1, a, b)
    Makie.zlims!(ax1, c, d)
    tgt = umap[:, tgti]
    p = arrows!(ax1, Point2f[(0 + 0.5 * kscale, 0)], Point2f[((4 - 1) * kscale, 0)], linewidth=pt(0.7), arrowsize=pt(5), color=:orange)
    Makie.translate!(p, 0, tgt...)
    text!(ax1, (3.5 * kscale, tgt...), text="Neuron\nfunction", align=(:center, :top), offset=(6, -2), color=:orange)
    hidespines!(ax1)
    hidedecorations!(ax1)
    ax2 = Axis(grid[1, 1], aspect=DataAspect())
    ax2.yautolimitmargin = (0, 0.14)
    ax2.xautolimitmargin = (0.02, 0.05)
    relax_clipping(ax2, (30, 30, 30, 30))
    #relax_clipping(ax1, (3, 3, 3, 3))
    rowgap!(grid, 1, 0)
    path1 = "m 12.808125,107.29125 c 36.568123,-1.67063 46.29298,-4.3743 75.734999,12.06562 30.501806,17.03169 44.178746,29.7 100.237506,38.42438"
    path2 = "M 83.96,116.88 C 108.21938,111.74626 104.87813,101.90812 131.60812,95.968124"
    path1, path2 = map(p -> Makie.BezierPath(p, flipy=false), (path1, path2))
    p1 = lines!(ax2, path1, linewidth=pt(2), color=:black, linecap=:round)
    p2 = lines!(ax2, path2, linewidth=pt(2), color=:black, linecap=:round)
    nodes = [path1.commands[1].p, [p.commands[end].p for p in (path2, path1)]...]
    p3 = scatter!(ax2, nodes[1:1], color=:Red, markersize=pt(9))
    p4 = scatter!(ax2, nodes[2:3], color=[(c, 0.3) for c in (:Green, :Blue)], markersize=pt(18))
    hidedecorations!(ax2)
    hidespines!(ax2)
    foreach(enumerate(nodes), ["Soma", "Target 1", "Target 2"]) do (i, p), n
        text!(p, text=n, align=(i < 3 ? :left : :center, i < 3 ? :bottom : :top), offset=(0, [4, 7, -7][i]))
    end
    # for p in [p1, p2, p3, p4]
    #     Makie.rotate!(p,  sqrt(1 / 2), 0, 0, sqrt(1 / 2))
    #     Makie.scale!(p, 0.07, 0.07, -0.07)
    #     Makie.translate!(p, -3, 1,12)
    #     #Makie.translate!(p, )
    #     # Makie.rotate!(p, 0, -sqrt(1 / 2), 0, sqrt(1 / 2))
    #     # Makie.translate!(p, k, 0, 0)
    # end

    #scatter!(ax1, Point3f[(0, 0, 0), (j)])

    #end

    #lines!(Point3f[(0, tgt...), (4, tgt...)], linewidth=pt(1), arrowsize=pt(0.05))
    Legend(grid[3, 1], [MarkerElement(marker=:o, markersize=pt(1.5), color=:gray), MarkerElement(marker=:o, markersize=pt(3), color=:black),], ["Function", "Brain area associated function"], tellwidth=false, tellheight=true, orientation=:horizontal)
    rowgap!(grid, 2, 0)
end

function plot_neurons_func(grid, env; obj, _...)
    ax1x = LScene(grid[1, 1]; height=pt(160), show_axis=false)
    ax2x = LScene(grid[2, 1]; height=pt(100), width=pt(160), halign=:left, show_axis=false, tellwidth=false)
    rowgap!(grid, 1, -30)
    setup_camera(ax1x => :h, ax2x => :s)
    func2grp = deepcopy(xxxfunc2grp)
    #@warn "$func2grp"
    #@warn "$xxxgrp2col"
    rowcolor(r) = get(xxxgrp2col, get(func2grp, r.func_samp, ("", ""))[1], nothing)
    dfx = deepcopy(xxxdffunc)
    filter!(r -> !isnothing(rowcolor(r)), dfx)
    @warn "$(size(dfx))"
    with_mixed_scene(ax1x, GLMakie) do mix1
        with_mixed_scene(ax2x, GLMakie) do mix2
            #dfx = shuffle(dfx)[1:10, :]
            draw_neurons(dfx, mix1, mix2; rowcolor, mirrorby=NmtDat)
        end
    end
    translate!(ax2x.scene, 0, 0, 5)
    draw_mesh_overlay([obj], ax1x, ax2x)
end

function plot_fig_subtyp_supp(fig, env; _...)
    with_caption(L"")

    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"")
        plot_result_depth_g97(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"")
        plot_result_umap(grid, env; env...)
    end
    grid = fig[2, 1]
    ax1, ax2 = with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"")
        plot_result_table_inj(grid, env; env...)
    end
    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        ax3, ax4 = plot_result_table(grid, env; env...)
        align_axis_label(:y, ax1, ax2, ax3, ax4)
        linkxaxes!(ax1, ax2, ax3, ax4)
    end
end
