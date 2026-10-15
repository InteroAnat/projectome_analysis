
function plot_compare_lenx2(grid, env; mac, mus, spe2color, macarbor, musarbor, _...)
    mac = deepcopy(mac)
    mus = deepcopy(mus)
    onearbor_per_nrnr!(mac, macarbor, noop=true)
    onearbor_per_nrnr!(mus, musarbor, noop=true)
    calc_arbor_stats!(mac; thr=1900, scale=1)
    calc_arbor_stats!(mus; thr=700, scale=1)

    cols = intersect(names(mac), names(mus))
    df = vcat(mac[:, cols], mus[:, cols])
    cols = intersect(names(macarbor), names(musarbor))
    dfarbor = vcat(macarbor[:, cols], musarbor[:, cols])
    leftjoin!(dfarbor, df, on=:path) |> dropmissing!

    rowx(r) = (!zz1data.istypit(r), r.typ[1:2])
    xname(x) = "$(x[2])"
    rowdodge(r) = r.spe
    statif(a, b) = begin
        a.x == b.x && (a.d === :mus) != (b.d === :mus)
    end

    specs = [
        (; t="Total axon\nlength\n(mm)", f=r -> r.axonlen / 1000,),
        (; t="No. of\nbranches\n", f=r -> r.nbranches,),
        (; t="Axon trunk\nlength\n(mm)", f=r -> r.trunklen / 1000,),
        (; t="No. of\ncollaterals\n", f=r -> r.collcnt,),
        #(; t="Axon length\nper arbor\n(mm)", f=r -> r.f_totlen / 1000, df=dfarbor),
        (; t="Normalized\narbor size", f=r -> zz1data.arbor_size(r.f_volume) / 1000 / (r.spe === :mac ? 5 : 1), df=dfarbor),
        #(; t="Arbor size\n(mm)", f=r -> zz1data.arbor_size(r.f_volume) / 1000  , df=dfarbor),
        #(; t="Total arbor\nlength (mm)" , f =r -> r.arbortot / 1000, ),
        #(; t="Per-arbor\nvolume (mm3)" , f =r -> r.f_volume / 1000^3, ),
        #(; t="A" , f =r -> r.collcnt, ),
    ]
    ax2s = Axis[]
    ax3s = Axis[]
    foldchange = false
    for (i, spec) in enumerate(specs)
        row, col = divrem(i - 1, 5) .+ 1
        gridx = GridLayout(grid[row, col])
        ax2t = Axis(gridx[1, 1])
        ax2 = Axis(gridx[2, 1])
        push!(ax2s, ax2)
        dfx = filter(r -> r.arborlen > 0, df)
        dat = get_vals_x_and_dodge(get(spec, :df, dfx); rowx, rowy=spec.f, rowdodge, rowcolor=spe2color,)
        violin_and_compare!(ax2, dat, ax2t; xname, statif, compact=true, datalimits=(0, Inf), scale=:width)

        ul = let
            mmm = maximum(dat.y)
            mm = maximum(median ∘ last, dat.grps)
            if mmm / mm > 3
                3 * mm
            else
                nothing
            end
        end
        ylims!(ax2, -0.001, ul)
        ax2.yminorticksvisible = false
        ax2.ylabel = spec.t

        rowgap!(gridx, 1, 0)
        row > 1 && rowgap!(grid, row - 1, 2)
        col > 1 && colgap!(grid, col - 1, 4)
        #col < 3 && row < 2 && (ax2.xticklabelsvisible = false)

        if foldchange
            ax3 = Axis(grid[i, 1][2, 1])
            push!(ax3s, ax3)
            plt = plot_foldchange!(ax3, dat, colorA=:gray, colorB=:gray, color=:gray)
            translate!(plt, 0, 0, -1)
            hidespines!(ax3)
            hidexdecorations!(ax3)
            ax3.yminorticksvisible = false
            ax3.yticklabelcolor = :gray
            #ax3.ylabelcolor=:gray
            ax3.yaxisposition = :right
            ax3.rightspinevisible = true
            yticks = [1 / 4, 1 / 2, 1, 2, 4]
            ax3.yticks = (yticks, [t >= 1 ? "$(round(Int,t))" : "1/$(round(Int, 1/t))" for t in yticks])
        end
        ax2.height = 30
    end
    if foldchange
        linkyaxes!(ax3s...)
        ax3s[end÷2+1].ylabel = "Fold change of the average"
    end
    linkxaxes!(ax2s..., ax3s...)
    #align_axis_label(:y, ax2s...)
    Legend(grid[1, 6], [PolyElement(color=c) for c in [:red, :blue]], ["Monkey", "Mouse"], orientation=:vertical, tellwidth=true, tellheight=false,)
    #rowgap!(content(grid[1, 1]), 1, 1)
end

function plot_compare_nrns(grid, env; mac, mus, spe2color, macarbor, musarbor, _...)
    mac = deepcopy(mac)
    mus = deepcopy(mus)
    calc_arbor_stats!(mac; thr=1900, scale=1)
    calc_arbor_stats!(mus; thr=700, scale=1)
    cols = intersect(names(mac), names(mus))
    df = vcat(mac[:, cols], mus[:, cols])

    rowx(r) = (!zz1data.istypit(r), r.typ[1:2])
    xname(x) = "$(x[2])"
    rowdodge(r) = r.spe
    statif(a, b) = begin
        a.x == b.x && (a.d === :mus) != (b.d === :mus)
    end

    specs = [
        (; t="Total axon\nlength\n(mm)", f=r -> r.axonlen / 1000,),
        (; t="No. of\nbranches\n", f=r -> r.nbranches,),
        (; t="Axon trunk\nlength\n(mm)", f=r -> r.trunklen / 1000,),
        (; t="No. of\ncollaterals\n", f=r -> r.collcnt,),
        (; t="Arbor length\nper neuron(mm)", f=r -> r.arborlen / 1000,),
        (; t="Arbor length\n(%)", f=r -> r.arborrat * 100,),
    ]
    ax2s = Axis[]
    ax3s = Axis[]
    foldchange = false
    for (i, spec) in enumerate(specs)
        ax2t = Axis(grid[i, 1][1, 1])
        ax2 = Axis(grid[i, 1][2, 1])
        push!(ax2s, ax2)
        dfx = filter(r -> r.arborlen > 0, df)
        dat = get_vals_x_and_dodge(get(spec, :df, dfx); rowx, rowy=spec.f, rowdodge, rowcolor=spe2color,)
        violin_and_compare!(ax2, dat, ax2t; xname, statif, compact=true, datalimits=(0, Inf), scale=:width)

        ul = let
            mmm = maximum(dat.y)
            mm = maximum(median ∘ last, dat.grps)
            if mmm / mm > 3
                3 * mm
            else
                nothing
            end
        end
        ylims!(ax2, -0.001, ul)
        ax2.yminorticksvisible = false
        ax2.ylabel = spec.t

        rowgap!(content(grid[i, 1]), 1, 0)
        i > 1 && rowgap!(grid, i - 1, 0)
        i < length(specs) && (ax2.xticklabelsvisible = false)

        if foldchange
            ax3 = Axis(grid[i, 1][2, 1])
            push!(ax3s, ax3)
            plt = plot_foldchange!(ax3, dat, colorA=:gray, colorB=:gray, color=:gray)
            translate!(plt, 0, 0, -1)
            hidespines!(ax3)
            hidexdecorations!(ax3)
            ax3.yminorticksvisible = false
            ax3.yticklabelcolor = :gray
            #ax3.ylabelcolor=:gray
            ax3.yaxisposition = :right
            ax3.rightspinevisible = true
            yticks = [1 / 4, 1 / 2, 1, 2, 4]
            ax3.yticks = (yticks, [t >= 1 ? "$(round(Int,t))" : "1/$(round(Int, 1/t))" for t in yticks])
        end
    end
    if foldchange
        linkyaxes!(ax3s...)
        ax3s[end÷2+1].ylabel = "Fold change of the average"
    end
    linkxaxes!(ax2s..., ax3s...)
    align_axis_label(:y, ax2s...)
end

function plot_compare_arbors_feats(gridx, env; mac, mus, spe2color, macarbor, musarbor, _...)
    mac = deepcopy(mac)
    mus = deepcopy(mus)
    onearbor_per_nrnr!(mac, macarbor, noop=true)
    onearbor_per_nrnr!(mus, musarbor, noop=true)
    calc_arbor_stats!(mac; thr=1900, scale=1)
    calc_arbor_stats!(mus; thr=700, scale=1)

    cols = intersect(names(mac), names(mus))
    df = vcat(mac[:, cols], mus[:, cols])
    cols = intersect(names(macarbor), names(musarbor))
    dfarbor = vcat(macarbor[:, cols], musarbor[:, cols])
    leftjoin!(dfarbor, df, on=:path) |> dropmissing!

    rowx(r) = (!zz1data.istypit(r), r.typ[1:2])
    xname(x) = "$(x[2])"
    rowdodge(r) = r.spe
    statif(a, b) = begin
        a.x == b.x && (a.d === :mus) != (b.d === :mus)
    end

    specs = [
        :f_lm_Bif_ampl_local_avg
        :f_lm_Bif_ampl_remote_avg
        :f_lm_Bif_tilt_local_avg
        :f_lm_Bif_tilt_remote_avg
        :f_lm_Bif_torque_local_avg
        :f_lm_Bif_torque_remote_avg
        :f_lm_Branch_Order_avg
        :f_lm_Branch_pathlength_avg
        :f_lm_Branch_pathlength_sum
        :f_lm_Contraction_avg
        :f_lm_EucDistance_avg
        :f_lm_EucDistance_sum
        :f_lm_Fractal_Dim_avg
        :f_lm_Fragmentation_avg
        :f_lm_Helix_avg
        :f_lm_Helix_sum
        :f_lm_Length_avg
        :f_lm_N_branch_sum
        :f_lm_N_stems_sum
        :f_lm_Partition_asymmetry_avg
        :f_lm_PathDistance_avg
        :f_lm_Pk_sum
        :f_lm_Terminal_degree_avg
        :f_lm_Terminal_degree_sum
        :f_lm_TerminalSegment_sum
    ]
    size_releated(dfx) = begin
        dfx = filter(r -> r.typ[1:2] == "IT", dfx)
        perm = sortperm(dfx.f_totlen)
        filter(specs) do s
            a, b = perm[2*end÷6:3*end÷6], perm[3*end÷6:4*end÷6]
            a, b = dfx[a, s], dfx[b, s]
            p = HypothesisTests.pvalue(HypothesisTests.MannWhitneyUTest(a, b))
            @info "sss $s => $p"
            p < 0.01
        end
        # 
    end
    size_releated(leftjoin(macarbor, mac, on=:path))
    grid = gridx
    ax2sss = Dict()
    ax3sss = Dict()
    nrows = 7
    for (iii, sss) in enumerate(specs)
        spec = (; t=replace("$sss", r"^f_lm_" => ""), f=r -> get(r, sss, NaN), df=dfarbor)
        rrr, ccc = divrem(iii - 1, nrows) .+ 1
        ax2s = get!(() -> Axis[], ax2sss, rrr)
        ax3s = get!(() -> Axis[], ax3sss, rrr)
        ax2t = Axis(grid[ccc, rrr][1, 1])
        ax2 = Axis(grid[ccc, rrr][2, 1], height=pt(40))
        rrr > 1 && colgap!(grid, rrr - 1, 6)
        ccc > 1 && rowgap!(grid, ccc - 1, 8)
        push!(ax2s, ax2)
        dfx = filter(r -> r.arborlen > 0, df)
        dat = get_vals_x_and_dodge(get(spec, :df, dfx); rowx, rowy=spec.f, rowdodge, rowcolor=spe2color,)
        violin_and_compare!(ax2, dat, ax2t; xname, statif, compact=false, scale=:width) #= datalimits=(0, Inf), =#

        ul = let
            mmm = maximum(dat.y)
            mm2 = maximum(v -> quantile(last(v), 0.7), dat.grps)
            mm = minimum(median ∘ last, dat.grps)
            if (mmm - mm) / (mm2 - mm) > 3
                3 * (mm2 - mm) + mm
            else
                nothing
            end
        end
        ll = let
            mmm = minimum(dat.y)
            mm2 = minimum(v -> quantile(last(v), 0.3), dat.grps)
            mm = maximum(median ∘ last, dat.grps)
            if (mmm - mm) / (mm2 - mm) > 3
                3 * (mm2 - mm) + mm
            else
                nothing
            end
        end
        ylims!(ax2, ll, ul)
        ax2.yminorticksvisible = false
        ax2t.title = spec.t

        rowgap!(content(grid[ccc, rrr]), 1, 0)
        ccc > 1 && rrr == 1 && rowgap!(grid, ccc - 1, 6)
        !(ccc == nrows || iii == length(specs)) && (ax2.xticklabelsvisible = false)
    end
    for rrr in 1:4
        ax2s = ax2sss[rrr]
        ax3s = ax3sss[rrr]
        linkxaxes!(ax2s..., ax3s...)
        align_axis_label(:y, ax2s...)
    end
end


function plot_mixture_model(grid, env; _...)
    Random.seed!(1)
    grid.width = pt(230)
    grid.height = pt(100)
    binsearch(f, a, b) = begin
        @assert f(a) < 0 && f(b) > 0
        while norm(a - b) > 0.001
            c = (a + b) / 2
            a, b = f(c) > 0 ? (a, c) : (c, b)
        end
        (a + b) / 2
    end
    rounded_rect(w, h) = begin
        (p) -> max(0, p[1] - w / 2, -w / 2 - p[1])^2 + max(0, p[2] - h / 2, -h / 2 - p[2])^2
    end
    ellipse(r1, r2) = begin
        (p) -> (p[1] / r1)^2 + (p[2] / r2)^2
    end
    genshape(f, g, r) = begin
        aa = push!(collect(0:0.01:2π), 2π)
        verts = map(aa) do a
            binsearch(p -> f(p) - r^2, Point2d(0, 0), Point2d(cos(a), sin(a)) * 100)
        end
        ii = sortperm(aa, by=a -> cos(a))
        inds = map(1:length(ii)-2) do i
            iseven(i) ? SVector(ii[i], ii[i+1], ii[i+2]) : SVector(ii[i], ii[i+2], ii[i+1])
        end
        (; verts, colors=Float64[g(p) for p in verts], indices=reduce(vcat, inds))
    end
    ww = 2
    col0, col1 = :blue, :red
    colormaps(a) = (; colormap=[(col0, a), (col1, a)], colorrange=(0, 1), nan_color=(:gray, a))
    gencell1(p) = begin
        0.5
        # rand(Float64) < sqrt((1 - p) * p) * 2 && return 0.5
        # rand(Float64) < p ? 1 : 0
    end
    for (iii, cfg) in enumerate([
        (; n="All-to-all", f=v -> v / ww * 0.001 + 0.5, g=gencell1),
        (; n="One-to-one", f=v -> v / ww + 0.5, g=p -> (rand(Float64) < p ? 1 : 0)),
    ])
        row, col = divrem(iii - 1, 3) .+ 1
        ax = Axis(grid[row, col], aspect=Makie.DataAspect())
        col == 1 && row > 1 && rowgap!(grid, row - 1, pt(2))
        row == 1 && col > 1 && colgap!(grid, col - 1, pt(2))
        Label(grid[row, col, Top()], cfg.n; tellwidth=false, halign=:center, valign=:bottom, set_font_h2x()...)
        hidespines!(ax)
        hidedecorations!(ax)
        g(p) = cfg.f(p[1])

        rectf = rounded_rect(ww, ww * 0.25)
        rect = genshape(rectf, g, 0.6)
        with_mixed_scene(ax, GLMakie) do mix
            poly!(mix.scene2, rect.verts, rect.indices, color=rect.colors; colormaps(0.3)..., transparency=true)
        end
        lines!(ax, rect.verts, color=:black, linewidth=pt(1.2))

        gennrns(n, d, f, nrns=Point2d[]) = begin
            while length(nrns) < n
                p = Point2d(((rand(Float64, 2) .- 0.5) .* 2ww)...)
                f(p) > 0 && continue
                minimum(q -> norm(q - p), nrns; init=1000) < d && continue
                push!(nrns, p)
            end
            nrns
        end
        nrns = gennrns(10, 0.2, p -> max(rectf(p) - 0.3^2, abs(p[2]) - 0.2))
        nrns = gennrns(30, 0.2, p -> max(rectf(p) - 0.3^2, abs(p[2]) - 0.9), nrns)
        nrns = map(nrns) do p
            gp = clamp(g(p), 0, 1)
            x = cfg.g(gp)
            while isnan(x)
                x = cfg.g(gp)
            end
            p => x
        end

        scatter!(ax, first.(nrns), markersize=pt(6), color=last.(nrns), strokewidth=pt(0.7), strokecolor=:black; colormaps(1.0)...)
        col(p) = p.second
        pos(p) = p.first[1]

        samps = [[filter(p -> col(p) == t1 && t2 <= pos(p) <= t2 + 1 / 6 && -0.3 < p.first[2] < 0.3, nrns) for t2 in ((a, b) = extrema(pos.(nrns)); a:1/6:b-0.001)] for t1 in unique(col.(nrns))]
        samps = reduce(vcat, [reduce(vcat, [shuffle!(s)[1:min(end, 1)] for s in ss]) for ss in samps])
        shuffle!(samps)

        for (dx, c) in (-1 => col0, 1 => col1)
            circf = ellipse(0.9, 0.6)
            circ = genshape(circf, g, 1)
            delta = Vec2d(dx * 1.5, -1.7)
            pp = circ.verts .+ Ref(delta)
            poly!(ax, pp, color=(c, 0.3))
            lines!(ax, pp, color=:black, linewidth=pt(1.0))
            nnn = 0
            tgts = gennrns(9, 0.2, p -> max(circf(p) - 0.7^2, abs(p[2]) - 0.2)) .+ Ref(delta)
            scatter!(ax, tgts, markersize=pt(6), color=c, strokewidth=pt(0.7), strokecolor=:black)
            srcs = []
            for (c, p) in samps
                if abs(p * 2 - 1 - dx) < 1.1
                    push!(srcs, c => p)
                    nnn += 1
                    nnn >= 4 && break
                end
            end
            dir(p) = -atan((p.first.-2*delta)[[2, 1]]...)
            dir2(p) = -atan((p.-1.5*delta)[[2, 1]]...)
            sort!(srcs, by=dir)
            a, b = extrema(dir.(nrns))
            tgts = shuffle(tgts)[1:length(srcs)]
            sort!(tgts, by=dir2)
            ppp = [i => i for i in 1:length(srcs)]
            overlap(a, b, c, d) = begin
                u, v = b - a
                vu = Vec2d(v, -u)
                sign(dot(c - a, vu)) * sign(dot(d - a, vu)) < 0
            end
            while true
                hh = 0
                for (i, (a, b)) in enumerate(ppp)
                    pa, pb = srcs[a].first, tgts[b]
                    for (j, (c, d)) in enumerate(ppp)
                        j <= i && continue
                        pc, pd = srcs[c].first, tgts[d]
                        if overlap(pa, pb, pc, pd)
                            hh += 1
                            ppp[i] = a => c
                            ppp[j] = c => b
                            @warn "overlap $a $b $c $d"
                            break
                        end
                    end
                    hh > 0 && break
                end
                hh == 0 && break
            end
            #tgts = tgts[tgtidx]
            #sort!(srcs, by=p -> p.first[1])
            for (a, b) in ppp
                @warn "pair $a $b"
                c, p = srcs[a]
                t = tgts[b]
                #foreach(srcs, tgts) do (c, p), t
                #d = binsearch(p -> circf(p) - 1, Point2d(0, 0), c - delta) + delta
                d = t
                dd = d - c
                dd = dd / norm(dd)
                arrows!(ax, [c + dd * 0.06], [d - c - dd * 0.16], arrowsize=5)
            end
        end
    end
end

using StatsBase
function prepare_function_table(df, arbors, trunks, area2func, funcdist; somaw=1_000_0000)
    arbors = leftjoin(arbors, trunks, on=intersect(names(arbors), names(trunks)))
    insertcols!(arbors, :trunklen => 0.0) |> dropmissing!
    for r in eachrow(arbors)
        r.trunklen = mapreduce(norm ∘ -, +, Iterators.drop(r.trunk, 1), r.trunk)
    end
    gethier(names, hier) = begin
        hier == "" && return nothing
        matches = findall(n -> zz1data.matchhier(n)(hier), names)
        length(matches) == 0 && return nothing
        if length(matches) > 1
            sort!(matches, lt=(a, b) -> begin
                zz1data.matchhier(names[a], names[b])(hier)
            end)
        end
        matches[end]
    end
    distfix(a, b) = (a == b && return 0; lowercase(a) == lowercase(b) ? 0.001 : 0.005)
    funcdist = Dict([(a => b) => Float64(d + distfix(a, b)) for ((a, b), d) in funcdist])

    uniqareas = sort(unique(first.(area2func)))
    uniqfuncs = sort(unique(last.(area2func)))
    func2ind = Dict(reverse.(enumerate(uniqfuncs)))
    dm = [funcdist[a=>b] for a in uniqfuncs, b in uniqfuncs]
    dm = Symmetric(dm + dm') ./ 2

    area2funcs = Dict{String,Set{String}}()
    for (a, f) in area2func
        push!(get!(() -> Set{String}(), area2funcs, a), f)
    end
    nrn2areas = Dict{String,Dict{String,Float64}}()
    addtgt(n, t, w) = begin
        d = get!(() -> Dict{String,Float64}(), nrn2areas, n)
        d[t] = max(get(d, t, 0), w)
    end
    for r in eachrow(df)
        h = gethier(uniqareas, r.somahier)
        h === nothing && continue
        addtgt(r.path, uniqareas[h], somaw)
    end
    for r in eachrow(arbors)
        h = gethier(uniqareas, r.arborhier)
        h === nothing && continue
        addtgt(r.path, uniqareas[h], r.trunklen)
    end
    Random.seed!(0)
    dfx = insertcols(df, :pattern => Ref(String[]), :func_min => "", :func_min_d => 0.0, :func_n_area => 0, :func_min_n => 0, :func_samp => "")
    zzz = zeros(1, length(uniqfuncs))
    for r in eachrow(dfx)
        a2w = get(() -> Dict{String,Float64}(), nrn2areas, r.path) |> collect
        maxw = maximum(filter(!=(somaw), last.(a2w)), init=1)
        getw(w) = w == somaw ? 1.0 : w / maxw
        ddww = map(a2w) do (a, w)
            ff = get(() -> Set{String}(), area2funcs, a) |> collect
            ii = map(f -> func2ind[f], ff)
            minimum(view(dm, ii, :), dims=1), getw(w)
        end
        dd = vec(+(zzz, (first.(ddww) .* last.(ddww))...))
        r.func_min_d = minimum(dd)
        r.func_min = uniqfuncs[argmin(dd)]
        ii = findall(<(r.func_min_d + 0.05), dd) |> shuffle
        r.func_min_n = length(ii)
        r.func_samp = uniqfuncs[ii[1]]
        r.func_n_area = length(ddww)
        r.pattern = map(a2w) do (a, w)
            ff = get(() -> Set{String}(), area2funcs, a) |> collect
            jj = map(f -> func2ind[f], ff)
            j0 = argmin(view(dm, ii[1], jj))
            ff = join([j == j0 ? "*$f" : f for (j, f) in enumerate(ff)], ",")
            w = getw(w)
            "$w * $a[$ff]"
        end
    end
    global xxxfuncdfx = deepcopy(dfx)
    dfx
end

function plot_func_table_per_cls(grid, env; df, arbors, trunks, functable, funcdist, _...)
    areas = functable.areafunc[:, :target_region_name]
    funcs = functable.areafunc[:, :function]
    func2grp = Dict([r.function_name => (r.function_class_big, r.function_class_small) for r in eachrow(functable.funcgrp)])
    dfx = zz2fig.prepare_function_table(df, arbors, trunks, zip(areas, funcs), funcdist)
    global xxxdffunc = deepcopy(dfx)
    global xxxfunc2grp = deepcopy(func2grp)
    foreach(f -> get!(() -> (@warn "unknown $f"; ("Unknow", "Unknown")), func2grp, f), funcs)

    uniqcls = sort(unique(dfx.cls))
    f2stats = map(unique(funcs)) do f
        stats = map(uniqcls) do cls
            dat = map(r -> (f == r.func_samp) => (r.cls == cls), eachrow(dfx))
            v1 = first.(dat)
            v2 = last.(dat)
            a, b, c, d = sum.((v1 .* v2, (.!v1) .* v2, v1 .* (.!v2), (.!v1) .* (.!v2)))
            @assert +(a, b, c, d) == length(dat) "$a+$b+$c+$d =? $(length(dat))"
            pval, odds = RCall.R"""t=fisher.test(matrix(c($a,$b,$c,$d),nrow=2),alternative="greater"); c(t$p.value,t$estimate)"""
            (; pval, odds, a, b, c, d)
        end
        pvals = [s.pval for s in stats]
        pvals = collect(RCall.R"""p.adjust($pvals, "fdr")""")
        f => map((s, pval) -> (; s..., pval), stats, pvals)
    end
    f2stats = Dict(f2stats)

    fixarea(a) = begin
        startswith("area_")(a) && return a[6:end]
        a
    end
    active_targets(clsi) = begin
        cls = uniqcls[clsi]
        tgts = []
        for r in eachrow(functable.proj)
            c, a = string.((r.cluster, r.target_region_name))
            if "$cls" == "$c"
                foreach(areas, funcs) do a1, f1
                    if a == a1
                        stats = get(f2stats, f1, nothing)
                        if stats !== nothing && stats[clsi].pval < 1.05
                            push!(tgts, a)
                        end
                    end
                end
            end
        end
        tgts = sort!(map(fixarea, unique!(tgts)), lt=(a, b) -> lowercase(a) < lowercase(b))
        string(join(tgts, ", "), "")
    end

    selfuncs = findall(stats -> minimum(v -> v.pval, stats) < 0.05, f2stats)
    mm = [-log10(f2stats[f][i].pval) for i in axes(uniqcls, 1), f in selfuncs]
    funcdm = 1 .- cor(mm) .+ [sum((func2grp[f] .!= func2grp[g]) .* (1000, 100)) for f in selfuncs, g in selfuncs]
    h3 = Clustering.hclust(funcdm, linkage=:ward, branchorder=:barjoseph)
    ax3 = Axis(grid[1, 1])
    #heatmap!(ax3, mm[:, h3.order])

    marks = []
    func2row = Dict(reverse.(enumerate(selfuncs[h3.order])))
    markersize(p) = pt(1.6 * sqrt(-log10(p)))
    for (f, stats) in f2stats
        for i in axes(uniqcls, 1)
            p = stats[i].pval
            if p < 0.05
                nstar = (p < 0.001 ? 3 : (p < 0.01 ? 2 : 1))
                push!(marks, (; i, j=func2row[f], p, stats[i].odds, nstar))
            end
        end
    end
    scatter!(ax3, [m.i for m in marks], [m.j for m in marks], markersize=[markersize(m.p) for m in marks], marker=:o, color=:black)
    ax3.yticks = (axes(selfuncs, 1), selfuncs[h3.order])
    hideydecorations!(ax3, label=false, ticklabels=false)
    ax3.ygridvisible = true
    ax3.xautolimitmargin = (0.025, 0.025)
    ax3.xticks = (axes(uniqcls, 1), ["#$c" for c in uniqcls])
    ax3.xticklabelrotation = π / 3
    ax3.xticklabelalign = (:left, :center)
    ax3.xaxisposition = :top
    ax3.topspinevisible = true
    hidexdecorations!(ax3, label=false, ticklabels=false)
    ax3.xgridvisible = true
    ax3.height = pt(250)
    ax3.width = pt(260)
    showpvals = [0.05, 0.01, 0.001]
    Legend(grid[1, 1, Bottom()], [MarkerElement(marker=:o, markersize=markersize(p)) for p in showpvals], ["P=$p" for p in showpvals], tellwidth=false, valign=:bottom, halign=:right, framevisible=true, margin=(0, 0, -pt(80), 0))
    ax3l = labelygroups(ax3, [func2grp[f][1] for f in selfuncs[h3.order]])
    ax3l.yautolimitmargin = (0.01, 0.01)
    translate!(ax3l.scene, 0, 0, 7)
    global xxxselfuncs = deepcopy(selfuncs[h3.order])
    global xxxf2stats = deepcopy(f2stats)

    ax3b = Axis(grid[2, 1], height=1, alignmode=Mixed())
    ax3b.xticks = (axes(uniqcls, 1), [active_targets(ci) for ci in axes(uniqcls, 1)])
    ax3b.xticklabelrotation = 0.25π
    ax3b.xticklabelalign = (:right, :center)
    ax3b.xautolimitmargin = (0.025, 0.025)
    linkxaxes!(ax3, ax3b)
    hideydecorations!(ax3b, label=false)
    hidexdecorations!(ax3b, ticklabels=false)
    hidespines!(ax3b)
    rowgap!(grid, 1, 0)
    #Label(grid[2, 1, Left()], "Target areas", rotation=0.3π, ; set_font_h2x()...)
    Label(grid[2, 1, Left()], "Target areas", rotation=0.25π, halign=:right, valign=:top, padding=(4, 4, 4, 4); set_font_h2x()...)
    ax3
end
function plot_func_assoc(grid, env; df, arbors, trunks, functable, funcdist, _...)
    c2aa = Dict{String,Set{String}}()
    a2ff = Dict{String,Set{String}}()
    for r in eachrow(functable.proj)
        c, a = string.((r.cluster, r.target_region_name))
        push!(get!(() -> Set{String}(), c2aa, c), a)
    end
    for r in eachrow(functable.areafunc)
        a, f = string.((r.target_region_name, r.function))
        push!(get!(() -> Set{String}(), a2ff, a), f)
    end
    global xxxselfuncs
    selfuncs = deepcopy(xxxselfuncs)
    global xxxf2stats
    activeedges = Set{Pair{String,String}}()
    uniqcls = string.(sort(unique(df.cls)))
    f2stats = deepcopy(xxxf2stats)
    for (f, stats) in f2stats
        foreach(stats, uniqcls) do s, c
            if s.pval < 0.05
                for a in get(c2aa, c, ())
                    if f in get(a2ff, a, ())
                        push!(activeedges, c => a, a => f)
                    end
                end
            end
        end
    end
    c2i = Dict(reverse.(enumerate(keys(c2aa))))
    a2i = Dict(reverse.(enumerate(keys(a2ff))))
    f2i = Dict(reverse.(enumerate(selfuncs)))
    mktempdir() do path
        dotf = joinpath(path, "temp.dot")
        pdff = joinpath(path, "temp.pdf")
        open(dotf, "w") do io
            println(io, "strict graph {")
            println(io, "rankdir=LR;")
            println(io, "ranksep=$(40/72);")
            println(io, "nodesep=$(5/72);")
            println(io, "node[fontsize=6,shape=plaintext,height=$(8/72)];")
            nodeready = Set()
            ensure_node() = begin end
            for (c, aa) in c2aa
                ci = c2i[c]
                println(io, "c$ci [label=\"#$c\"];")
                for a in aa
                    !in((c => a), activeedges) && continue
                    ai = get(a2i, a, 0)
                    if ai != 0
                        if !in!("a$ai", nodeready)
                            println(io, "a$ai [label=\"$a\"];")
                        end
                        println(io, "c$ci -- a$ai;")
                    end
                end
            end
            # for (a, ff) in a2ff
            #     ai = a2i[a]
            #     if in("a$ai", nodeready)
            #         for f in ff
            #             !in((a => f), activeedges) && continue
            #             fi = get(f2i, f, 0)
            #             if fi != 0
            #                 if !in!("f$fi", nodeready)
            #                     println(io, "f$fi [label=\"$f\"];")
            #                 end
            #                 println(io, "a$ai -- f$fi;")
            #             end
            #         end
            #     end
            # end
            println(io, "{\nrank=same;")
            foreach(uniqcls, uniqcls[2:end]) do c1, c2
                ci1, ci2 = c2i[c1], c2i[c2]
                println(io, "c$ci1 -- c$ci2;")
            end
            println(io, "}")
            #
            # println(io, "{\nrank=same;")
            # foreach(selfuncs, selfuncs[2:end]) do f1, f2
            #     fi1, fi2 = f2i[f1], f2i[f2]
            #     println(io, "f$fi1 -- f$fi2;")
            # end
            # println(io, "}")

            println(io, "}")
        end
        run(`dot -Tpdf -Kdot -o$pdff $dotf`)
        cp(pdff, "/tmp/xxx.pdf", force=true)
        LTeX(grid[1, 1], PDFDocument(read(pdff, String)), tellwidth=true)
    end

    # areas = functable.areafunc[:, :target_region_name]
    # funcs = functable.areafunc[:, :function]
    # func2grp = Dict([r.function_name => (r.function_class_big, r.function_class_small) for r in eachrow(functable.funcgrp)])
    # foreach(f -> get!(func2grp, f, ("Unknow", "Unknown")), funcs)
    # dfx = prepare_function_table(df, arbors, trunks, zip(areas, funcs), funcdist)
    #
    # uniqcls = sort(unique(dfx.cls))
    # f2stats = map(unique(funcs)) do f
    #     stats = map(uniqcls) do cls
    #         dat = map(r -> (f == r.func_samp) => (r.cls == cls), eachrow(dfx))
    #         v1 = first.(dat)
    #         v2 = last.(dat)
    #         a, b, c, d = sum.((v1 .* v2, (.!v1) .* v2, v1 .* (.!v2), (.!v1) .* (.!v2)))
    #         @assert +(a, b, c, d) == length(dat) "$a+$b+$c+$d =? $(length(dat))"
    #         pval, odds = RCall.R"""t=fisher.test(matrix(c($a,$b,$c,$d),nrow=2),alternative="greater"); c(t$p.value,t$estimate)"""
    #         (; pval, odds, a, b, c, d)
    #     end
    #     pvals = [s.pval for s in stats]
    #     pvals = collect(RCall.R"""p.adjust($pvals, "fdr")""")
    #     f => map((s, pval) -> (; s..., pval), stats, pvals)
    # end
    # f2stats = Dict(f2stats)
    #
    # selfuncs = findall(stats -> minimum(v -> v.pval, stats) < 0.05, f2stats)
    # mm = [-log10(f2stats[f][i].pval) for i in axes(uniqcls, 1), f in selfuncs]
    # funcdm = 1 .- cor(mm) .+ [sum((func2grp[f] .!= func2grp[g]) .* (1000, 100)) for f in selfuncs, g in selfuncs]
    # h3 = Clustering.hclust(funcdm, linkage=:ward, branchorder=:barjoseph)
    # ax3 = Axis(grid[1, 1])
    # #heatmap!(ax3, mm[:, h3.order])
    #
    # marks = []
    # func2row = Dict(reverse.(enumerate(selfuncs[h3.order])))
    # markersize(p) = pt(1.6 * sqrt(-log10(p)))
    # for (f, stats) in f2stats
    #     for i in axes(uniqcls, 1)
    #         p = stats[i].pval
    #         if p < 0.05
    #             nstar = (p < 0.001 ? 3 : (p < 0.01 ? 2 : 1))
    #             push!(marks, (; i, j=func2row[f], p, stats[i].odds, nstar))
    #         end
    #     end
    # end
    # scatter!(ax3, [m.i for m in marks], [m.j for m in marks], markersize=[markersize(m.p) for m in marks], marker=:o)
    # ax3.yticks = (axes(selfuncs, 1), selfuncs[h3.order])
    # hideydecorations!(ax3, label=false, ticklabels=false)
    # ax3.ygridvisible = true
    # ax3.xautolimitmargin = (0.025, 0.025)
    # ax3.yautolimitmargin = (0.025, 0.025)
    # ax3.xticks = (axes(uniqcls, 1), ["#$cc" for cc in uniqcls])
    # ax3.xticklabelrotation = -π / 3
    # ax3.xticklabelalign = (:left, :center)
    # hidexdecorations!(ax3, label=false, ticklabels=false)
    # ax3.xgridvisible = true
    # ax3.height = pt(250)
    # ax3.width = pt(200)
    # showpvals = [0.05, 0.01, 0.001]
    # axislegend(ax3, [MarkerElement(marker=:o, markersize=markersize(p)) for p in showpvals], ["P=$p" for p in showpvals], position=:rt)
    # labelygroups(ax3, [func2grp[f][1] for f in selfuncs[h3.order]])
    # ax3
end
function labelygroups(ax3, grps)
    gc = Makie.gridcontent(ax3)
    gridpos = GridPosition(gc.parent, gc.span, Left())
    ax = Axis(gridpos)
    linkyaxes!(ax3, ax)
    uniqgrps = unique(grps)
    uniqgrps = map(uniqgrps) do grp
        a, b = extrema(findall(==(grp), grps))
        (a, b) => grp
    end
    cols = repeat(to_colormap(:Set1_6), 10)
    grp2col = zip(last.(sort(uniqgrps)), cols)
    global xxxgrp2col = Dict(grp2col)
    foreach(sort(uniqgrps), cols) do ((a, b), grp), col
        @info "$a $b $grp"
        p = hspan!(a - 0.5, b + 0.5, color=Makie.to_color((col, 0.3)))
        translate!(p, (0, 0, -1))
    end
    ax.yautolimitmargin = (0, 0)
    hidedecorations!(ax)
    hidespines!(ax)
    ax
end
function plot_cls_funcs(grid, env; df, arbors, trunks, dffunc1, dffunc2, funcdist, _...)
    return plot_func_table_per_cls(grid, env; env...)
    uniqfuncs = sort(unique(funcs))
    uniqareas = sort(unique(areas))
    area2funcs = Set([a => b for (a, b) in zip(areas, funcs)])
    global xxxfuncdist = deepcopy(funcdist)
    h = Clustering.hclust(dm, linkage=:complete, branchorder=:barjoseph)
    ax = Axis(grid[1, 1])
    ax.width = 1500
    Vis.dendrogram!(ax, h)
    ax.xticks = (axes(uniqfuncs, 1), uniqfuncs[h.order])
    ax.xticklabelrotation = π / 3
    ax.height = 300

    mat = [((a => b) in area2funcs) for a in uniqareas, b in uniqfuncs]
    # ax2 = Axis(grid[2, 1])
    # h2 = Clustering.hclust(1 .- cor(mat'), linkage=:ward, branchorder=:barjoseph)
    # Makie.heatmap!(ax2, mat[h2.order, h.order]')
    # ax2.height = 400
    # ax2.xticks = (axes(funcs, 1), funcs[h.order])
    # ax2.xticklabelrotation = π / 3
    # ax2.yticks = (axes(tgts, 1), tgts[h2.order])


    draw_nrn_diagram(gpos, dfx) = begin
        graphnodes = []
        graphedges = []
        add_node(n, t, hiers...) = begin
            push!(graphnodes, (; n, t, hiers))
        end
        add_link(n1, n2, t) = begin
            push!(graphedges, (; n1, n2, t))
        end
        #####
        # r = "area_$a"
        # if !in(r, seen)
        #     println(io, "$r [label=\"$a\"];")
        # end
        # r
        # if p == maxp
        # else
        #     println(io, "$r [];")
        # end
        #####
        #####
        mktempdir() do path
            seen = Set{String}()
            dotf = joinpath(path, "test.dot")
            pdff = joinpath(path, "test.pdf")
            @warn "$dotf"
            open(dotf, "w") do io
                println(io, "digraph pfc {")
                #println(io, "size=\"2,2\";")
                println(io, "overlap=\"false\";")
                println(io, "splines=\"true\";")
                println(io, "fontsize=\"6\";")
                for (kkk, rrr) in enumerate(eachrow(dfx))
                    swc = Gapr.load_swc(rrr.path)
                    tt = collect(nrn2areas[rrr.path])
                    tmptgts = first.(tt)
                    t = gethier(tmptgts, rrr.somahier)
                    t === nothing && continue
                    rpi = 1 << t
                    ##rp = ensure_area(tt[t][1])
                    @warn "root $rpi $tt"
                    a2t = Dict{Int,Int}()
                    for r in eachrow(filter(r -> r.path == rrr.path, arbors))
                        a = parse(Int, match(r"/\d+-\d+-root\d+-\w+-arbor(\d+)", r.arborpath).captures[1]) % 100_000_000_000
                        t = gethier(tmptgts, r.arborhier)
                        t === nothing && continue
                        a2t[a] = t
                    end
                    maxp = sum(filter(v -> v != rpi, unique([(1 << t) for (a, t) in a2t])))
                    @warn "$a2t"
                    pp = [Set{Int}() for _ in swc.nodes]
                    curseen = Set{Int}()
                    nodeid(p) = begin
                        r = "nrn$(kkk)_$p"
                        if !in(p, curseen)
                            t, hh = if count_ones(p) == 1
                                a = tt[Int(log2(p))][1]
                                :t, (a,)
                            elseif p == maxp
                                a = tt[Int(log2(rpi))][1]
                                :r, (a,)
                            else
                                :b, ()
                            end
                            add_node(r, t, hh...)
                        end
                        r
                    end
                    for n in Iterators.reverse(swc.nodes)
                        p = get(a2t, n.id, 0)
                        p = p == 0 ? 0 : 1 << p
                        p == rpi && (p = 0)
                        pv = push!(pp[n.ind], p)
                        p = |(pv...)
                        if n.parind != 0
                            push!(pp[n.parind], p)
                        end
                        if n.id == 1325013
                            @warn "xxx $p"
                        end
                    end
                    for pv in reverse(pp)
                        p = |(pv...)
                        if p != 0 && !(p in curseen)
                            @warn "ppp $p $pv"
                            np = nodeid(p)
                            if count_ones(p) != 1
                                for x in pv
                                    x == 0 && continue
                                    nx = nodeid(x)
                                    if count_ones(x) == 1
                                        add_link(np, nx, :t)
                                    else
                                        add_link(np, nx, :b)
                                    end
                                end
                            end
                            push!(seen, np)
                            push!(curseen, p)
                        end
                    end
                    maxpp = maximum(curseen)
                    @warn "maxp $maxpp $maxp"
                end
                allhiers = sort(unique([n.hiers for n in graphnodes]))
                handlehier(ch) = begin
                    for nh in allhiers
                        if length(nh) > length(ch) && nh[1:length(ch)] == ch
                            nh = nh[length(ch)+1]
                            println(io, "subgraph cluster_area_$nh {\nlabel=\"$nh\"; ")
                            handlehier((ch..., nh))
                            println(io, "}")
                        end
                    end
                    for n in graphnodes
                        if n.hiers == ch
                            n, t = n.n, n.t
                            s, c = (;
                                t=(:point, :black),
                                b=(:point, :black),
                                r=(:point, :red),
                            )[t]
                            println(io, "$n [shape=$s,label=\"\", color=$c];")
                        end
                    end
                end
                handlehier(())
                for (n1, n2, t) in graphedges
                    a = (;
                        t="shape=point",
                        b="arrowhead=none",
                    )[t]

                    println(io, "$n1 -> $n2[$a];")
                end
                println(io, "}")
            end
            cp(dotf, "/tmp/xxx.dot", force=true)
            #run(`fdp -Tpdf -Kneato -LC0.9 -o$pdff $dotf`)
            run(`dot -Tpdf -Kdot -o$pdff $dotf`)
            #run(`fdp -n2 -Txdot -Kneato -o/tmp/xxx.xdot $dotf`)
            run(`fdp -Txdot -Kneato -o/tmp/xxx.xdot $dotf`)
            cp(pdff, "/tmp/xxx.pdf", force=true)
            LTeX(gpos, PDFDocument(read(pdff, String)), tellwidth=true)
        end
    end
    Random.seed!(6)
    # for j in 1:3
    #     draw_nrn_diagram(grid[3, 1][1, j], shuffle(filter(r -> r.func == "Reaching Movement Guidence" && r.funcm >= 4 && 1 <= r.funcn <= 2, df))[1:9, :])
    # end
end

function plot_fig_compare(fig, env; mac, mus, _...)
    with_caption(L"Comparison of PFC single-neuron projectomes between macaque and mouse.")

    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_compare_pari_bp(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        plot_compare_pari(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1], env) do grid, env
        plot_compare_venn(grid, env; env...)
    end
    with_subfig(grid[1, 3][2, 1], env) do grid, env
        plot_compare_pari_hm(grid, env; env...)
    end
    with_subfig(grid[1, 3][3, 1], env) do grid, env
        plot_compare_arborsx(grid, env; env...)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1][1, 1], env) do grid, env
        plot_compare_symm(grid, env; env...)
    end
    with_subfig(grid[1, 1][2, 1], env) do grid, env
        plot_compare_symm_str(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        plot_compare_ctpt(grid, env; env...)
    end
    with_subfig(grid[1, 3][1, 1], env) do grid, env
        plot_compare_ptct_numtgt(grid, env; env...)
    end
    with_subfig(grid[1, 3][2, 1], env) do grid, env
        plot_compare_ptct_symm(grid, env; env...)
    end

    grid = fig[3, 1]
    with_subfig(grid[1, 1][1, 1][1, 1], env) do grid, env
        plot_compare_lenx2(grid, env; env...)
    end
    # with_subfig(grid[1, 2][1, 1], env) do grid, env
    #     plot_mixture_model(grid, env; env...)
    # end
end

function plot_fig_ctpt_supp(fig, env; _...)
    with_caption("Topography of axons and arbors of macaque dlPFC PT neurons.")
    grid = fig[1, 1]
    with_subfig(grid[1, 1][1, 1], env) do grid, env
        with_caption(L"")
        plot_pt_targets(grid, env; env...)
    end
    ax1 = with_subfig(grid[1, 1][2, 1][1, 1], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        plot_pt_bundle_topo(grid, env; env...)
    end
    with_subfig(grid[1, 1][2, 1][1, 2], env) do grid, env
        with_caption(L"Topographic mapping of dlPFC PT neurons to Mid.")
        ax2 = plot_pt_topo_term(grid, env, true; env...)
        ax2.width = 75
        #ax.height = 70
        align_axis_label(:x, ax1, ax2)
    end
    with_subfig(grid[1, 2][1, 1], env) do grid, env
        grid.tellwidth = false
        grid.halign = :left
        grid.valign = :top
        grid.height = 320
        LTeX(grid[1, 1], PDFDocument(read("monkeytemp/wyz-upload/fig7e-supp-arbor-morph-examples.pdf", String)))
    end
    with_subfig(grid[1, 2][2, 1], env) do grid, env
        plot_compare_arborsx_contra(grid, env; env...)
    end

    with_subfig(grid[1, 2][3, 1], env) do grid, env
        plot_mus_pt_bundle(grid, env; env...)
    end
    with_subfig(grid[1, 2][4, 1], env) do grid, env
        plot_mus_ct_bundle(grid, env; env...)
    end
    # with_subfig(grid[1, 1], env) do grid, env
    #     plot_compare_nrns(grid, env; env...)
    # end
end
function plot_fig_compare_supp(fig, env; mac, mus, _...)
    with_caption(L"Comparison of PFC single-neuron projectomes between macaque and mouse.")
end
function plot_fig_compare_supp2(fig, env; mac, mus, _...)
    with_caption(L"Comparison of PFC single-neuron projectomes between macaque and mouse.")
    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_compare_arbors_feats(grid, env; env...)
    end
end


function plot_abs_nrnegs(grid, env; df, arbors, obj, _...)
    egs = [
        (; nrn="root062",);
        (; nrn="root388",);
        (; nrn="root141", hl=(11198444, 25877, 11186403, 11320278, 11342148, 11331139, 11331660, 10329434));
        (; nrn="root222", hl=(461299, 461328, 496850, 67977976, 590432, 609406));
        (; nrn="root252", hl=(790970, 785644, 788324, 25236268, 797428, 794709))
    ]
    df = filter(r -> startswith("G97-1")(r.inj), df)
    n2c = Dict()
    get_color1(n) = get!(() -> Makie.to_colormap(:Set1_8)[length(n2c)+1], n2c, n)
    draw_once(gridx, dfx, egs) = begin
        ax1 = LScene(gridx[1, 1], width=pt(120), height=pt(120), show_axis=false)
        ax2 = LScene(gridx[1, 2], width=pt(120), height=pt(120), show_axis=false)
        ax3 = LScene(gridx[1, 3], width=pt(120), height=pt(120), show_axis=false)
        ax4 = LScene(gridx[1, 4], width=pt(120), height=pt(120), show_axis=false)
        setup_camera(ax1 => :h, ax2 => :s, ax3 => :c)
        setup_camera(ax4 => (fov=45.0, cent=(16310.86, 44649.5, 38553.18), deye=(-0.1465598002657872, 0.8094462516466938, -0.5686096997424241) .* 26508.974445613698, up=(-0.57, -0.54, -0.62)))
        nnhlx = Dict{String,Dict{Int,Int}}()
        for (r, eg) in zip(eachrow(dfx), egs)
            swc = Gapr.load_swc(r.path)
            nnhl = get!(() -> Dict{Int,Int}(), nnhlx, r.path)
            hl = get(eg, :hl, ())
            for n in swc.nodes
                if n.parind != 0
                    pp = get(nnhl, n.parent.id, -1)
                    pp != -1 && push!(nnhl, n.id => pp)
                end
                if hl === :all || n.id in hl
                    push!(nnhl, n.id => n.id)
                end
            end
            for n in Iterators.reverse(swc.nodes)
                if n.parind != 0
                    pp = get(nnhl, n.parent.id, -1)
                    ppp = get(nnhl, n.id, -1)
                    (ppp != -1 && pp == -1) && push!(nnhl, n.parent.id => 0)
                end
            end
        end
        i2c = Dict()
        get_color2(i) = get!(() -> Makie.to_colormap(:Set2_6)[length(i2c)%6+1], i2c, i)
        with_mixed_scene(ax1, GLMakie) do mix1
            with_mixed_scene(ax2, GLMakie) do mix2
                with_mixed_scene(ax3, GLMakie) do mix3
                    with_mixed_scene(ax4, GLMakie) do mix4
                        per_neuron(r) = begin
                            get_color1(r.path), nnhlx[r.path]
                        end
                        per_arbor(pn, rr, cond) = begin end
                        per_node((cc, nnhl), pa, n) = begin
                            return cc => pt(0.5)
                        end
                        per_node2((cc, nnhl), pa, n) = begin
                            pp = get(nnhl, n.id, -1)
                            if pp == 0
                                return Makie.to_color(:gray) => pt(0.5)
                            elseif pp > 0
                                return cc => pt(0.5)
                            else
                                return Makie.to_color(:transparent) => pt(0.5)
                            end
                        end
                        draw_subtrees(filter(r -> r.path in dfx.path, arbors), mix1, mix2, mix3; per_neuron, per_arbor, per_node,)
                        draw_subtrees(filter(r -> r.path in dfx.path, arbors), mix4; per_neuron, per_arbor, per_node=per_node2,)
                    end
                end
            end
        end
        draw_mesh_overlay([obj], ax1, ax2, ax3)
        draw_mesh_overlay([load("interm/mesh-nmt-$ba-l.obj") for ba in [:area_45b]], ax4)
    end
    getdfx(eg) = begin
        dfx = filter(df) do r
            m = match(r"^root\d+-(q?root\d+)-", basename(r.path))
            m.captures[1] == eg.nrn
        end
        @assert size(dfx, 1) == 1 "non-unique $dfx"
        dfx
    end
    for (i, eg) in enumerate(egs)
        gridx = GridLayout(grid[i, 1])
        dfx = getdfx(eg)
        draw_once(gridx, dfx, [eg])
    end
    draw_once(GridLayout(grid[end+1, 1]), reduce(vcat, getdfx.(egs)), egs)
end

function plot_simple_hclust(grid, env; df, hclust, obj, _...)
    t2c = typ2color_func(df)
    grid.tellwidth = false
    grid.tellheight = false
    grid.alignmode = Outside(140, 5, 165 - 10, 10)
    ax1 = Axis(grid[1, 1], height=30)
    ax2 = Axis(grid[2, 1], height=7)
    ax3 = Axis(grid[3, 1], height=10)
    gridx = GridLayout(grid[4, 1])
    rowgap!(grid, 1, 0)
    rowgap!(grid, 2, 0)
    rowgap!(grid, 3, 2)
    linkxaxes!(ax1, ax2, ax3)
    ylims!(ax1, 0, 13000000)
    zz1vis.dendrogram!(ax1, hclust, color=:black)
    typinfos = map(unique(df.typ)) do typ
        a, b = extrema(findall(df.typ .== typ))
        a, b, typ
    end
    for (a, b, typ) in sort(typinfos)
        vspan!(ax2, a - 0.5, b + 0.5, color=(t2c(typ), 1.0))
    end
    #todemo = (1, 7)
    prevccc = nothing
    for c in (1, 6, 12, 32)
        a, b = extrema(findall(df.cls .== c))
        dfx = filter(r -> r.cls == c && !r.qroot, df)
        col = t2c(dfx.typ[1])
        lines!(ax2, [a - 0.5, b + 0.5], [0, 0] .+ 0.03, color=:black)
        cc = (a + b) / 2
        xali, ccc = c > 1 ? (c < 32 ? :center => cc : :right => b) : :left => a
        text!(ax3, ccc, 0, text="#$c", align=(xali, :center); color=col, fontsize=8)
        if prevccc !== nothing
            text!(ax3, (cc + prevccc) / 2, 0, text="⋯", align=(:center, :center), fontsize=8)
        end
        pars = Dict(
            1 => (; x=0, y=18, w=50, h=30, c=:c),
            6 => (; x=38, y=2, w=50, h=30, c=:c),
            12 => (; x=50, y=33, w=60, h=30, c=:s),
            32 => (; x=100, y=-2, w=45, h=50, c=:h),
        )
        pars = get(pars, c, nothing)
        if pars !== nothing
            Random.seed!(3)
            dfx = shuffle(sort(dfx, :axonlen, rev=true)[1:min(10, end), :])[1:1, :]
            s = Makie.get_top_parent(grid)
            bbox = map(gridx.layoutobservables.suggestedbbox) do bb
                Rect2f(bb.origin .+ (pars.x, -pars.y), bb.widths)
            end
            ax = LScene(s, width=pars.w, height=pars.h; bbox, halign=:left, valign=:top, tellwidth=false, tellheight=false)
            setup_camera(ax => pars.c)
            with_mixed_scene(ax, GLMakie) do mix
                draw_neurons(dfx, mix; rowcolor=(r) -> t2c(r.typ))
                scatter!(ax2, findall(==(dfx.path[1]), df.path), [0.15], marker=:dtriangle, color=:black)
            end
            draw_contours([obj], ax)
        end
        prevccc = cc
    end
    xlims!(ax3, 0, size(df, 1) + 1)
    ylims!(ax2, 0, 1)
    relax_clipping(ax3, (6, 6, 6, 6))
    relax_clipping(ax2, (6, 6, 0, 0))
    for ax in (ax1, ax2, ax3)
        hidedecorations!(ax)
        hidespines!(ax)
    end
    Label(grid[1, 1, Top()], "32 Subtypes"; fontsize=12, tellheight=false, valign=:center)
end

function plot_simple_result(grid, env; df, obj, _...)
    rowcolor(r) = begin
        rand(to_colormap(:Set1_8))
    end
    grid.tellwidth = false
    grid.tellheight = false
    grid.alignmode = Outside(0, 150, 205 - 8, 10)
    ax = LScene(grid[1, 1], show_axis=false)
    setup_camera(ax => :s)
    with_mixed_scene(ax, GLMakie) do mix
        #return
        Random.seed!(0)
        draw_neurons(df, mix; rowcolor)
    end
    draw_mesh_overlay([obj], ax)
    #draw_contours([obj], ax)
    Label(grid[1, 1, Top()], "2,231 PFC Neurons"; fontsize=12, tellheight=false, valign=:center)
end

function plot_fig_abstract(grid, env; _...)
    # grid.width[] = 288
    # grid.height[] = 288

    #LTeX(grid[1, 1], compile_tikz("figs/fig7-mouse-monkey-neurons.tex"))

    # plot_simple_result(GridLayout(grid[1, 1]), env; env...)
    # plot_simple_hclust(GridLayout(grid[1, 1], valign=:top), env; env...)
    plot_abs_nrnegs(GridLayout(grid[1, 1], valign=:top), env; env...)
end
