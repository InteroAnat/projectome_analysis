function get_area_data_for_cmp(nmr, nmrdat, inj="G97-1bL")
    injpos = filter(r -> r.inj == inj, nmr.injs)[1, :injxyzm]
    cols = sortperm([norm(p .- injpos) for p in nmrdat.sites.sitepos])[1:9]
    c = argmax(c -> StatsBase.corspearman(nmrdat.vert_conn[:, nmrdat.sites[c, :site]], nmr.vert_conn[:, inj]), cols)
    site, sitepos = nmrdat.sites[c, [:site, :sitepos]]

    a2v = Dict()
    foreach(nmrdat.vert_conn[:, site] .+ 0.01, nmrdat.verts.areaname) do v, a
        a2v[a] = get(a2v, a, 0.0) + v
    end
    for k in keys(a2v)
        k2 = replace(k, r"_.*$" => "")
        if get(a2v, k2, nothing) === nothing
            a2v[k2] = a2v[k]
        end
    end
    for k in keys(a2v)
        k2 = replace(k, r"'$" => "_prime")
        if get(a2v, k2, nothing) === nothing
            a2v[k2] = a2v[k]
        end
    end
    for k in keys(a2v)
        k2 = "area_$k"
        if get(a2v, k2, nothing) === nothing
            a2v[k2] = a2v[k]
        end
    end
    #nomat = setdiff(cmpdat.a, keys(a2v))
    #@warn "sd $nomat"
    a2v
end
function plot_recon_compare2(grid, env; cmpdat, nmr, nmrdat, _...)
    ax = Makie.Axis(grid[2, 1], width=pt(80))
    ranked_scatter!(ax, cmpdat; rowx=r -> r.proj, rowy=r -> r.v2, rowcolor=r -> :black)
    ax.xlabel[] = "Rank of projection strength"
    ax.ylabel[] = "Rank of GFP volume"

    a2v = get_area_data_for_cmp(nmr, nmrdat,)
    ax = Makie.Axis(grid[1, 1], width=pt(80))
    ranked_scatter!(ax, cmpdat; rowx=r -> r.proj, rowy=r -> get(a2v, r.a, NaN), rowcolor=r -> :black)
    ax.xlabel[] = "Rank of projection strength"
    ax.ylabel[] = "Rank of fMRI activity"

end

function plot_recon_compare_barplot(grid, env; cmpdat, nmr, nmrdat, _...)
    ax1 = Axis(grid[1, 1], height=45)
    ax2 = Axis(grid[3, 1], height=45)
    ax3 = Axis(grid[2, 1], height=45)
    rowgap!(grid, 1, 2)
    rowgap!(grid, 2, 2)
    linkxaxes!(ax1, ax2, ax3)

    barplot!(ax1, axes(cmpdat.a, 1), cmpdat.proj ./ 1000, color=:black)
    ax1.xticks = (axes(cmpdat.a, 1), [areaname(ba) for ba in cmpdat.a])
    rotatexlabel!(ax1)
    ylims!(ax1, 0, nothing)
    hidexdecorations!(ax1, ticks=false)
    ax1.xlabel = "Brain regions"
    ax1.ylabel = "Projection\nstrength (mm)"

    barplot!(ax2, axes(cmpdat.a, 1), cmpdat.v2, color=:black)
    ax2.xticks = (axes(cmpdat.a, 1), [areaname(ba) for ba in cmpdat.a])
    rotatexlabel!(ax2)
    ylims!(ax2, 0, nothing)
    hidexdecorations!(ax3, ticks=false)
    ax2.xlabel = "Brain regions"
    ax2.ylabel = "GFP volume\n(pixels)"

    a2v = get_area_data_for_cmp(nmr, nmrdat,)
    barplot!(ax3, axes(cmpdat.a, 1), [get(a2v, a, NaN) for a in cmpdat.a], color=:black)
    ax3.xticks = (axes(cmpdat.a, 1), [areaname(ba) for ba in cmpdat.a])
    rotatexlabel!(ax3)
    ylims!(ax3, 0, nothing)
    hidexdecorations!(ax2, ticks=false, ticklabels=false, label=false)
    ax3.xlabel = "Brain regions"
    ax3.ylabel = "fMRI"
    ax3.xlabelpadding = -24
    ax3.xautolimitmargin = (0.015, 0.015)
    align_axis_label(:y, ax1, ax2, ax3)
end

function plot_recon_compare_markov(grid, env; conndat, df, cmpdat, nmr, nmrdat, _...)
    tonmt = Dict{String,Set{String}}()
    fromnmt = Dict{String,Set{String}}()
    for r in eachrow(conndat.areamap)
        a, b = r.region_name_markov, r.region_name_nmt
        push!(get!(() -> Set{String}(), tonmt, a), b)
        push!(get!(() -> Set{String}(), fromnmt, b), a)
    end
    tgthiers = Set{String}()
    for t in unique(conndat.flne.TARGET)
        tt = tonmt[t]
        for b in tt
            push!(tgthiers, b)
        end
    end
    genmatchhier(h) = begin
        !contains(",")(h) && return zz1data.matchhier(h)
        hh = zz1data.matchhier.(strip.(split(h, ",")))
        (x) -> any((h(x) for h in hh))
    end
    somahiers = Set{String}()
    for hh in unique(df.somahier)
        hit = 0
        for (h, aa) in fromnmt
            mh = genmatchhier(h)
            if mh(hh)
                hit += 1
                push!(somahiers, h)
            end
        end
        if hit != 1
            @warn "area $hh $hit"
        end
    end
    for h in union(somahiers, tgthiers)
        aa = fromnmt[h]
        for a in aa
            bb = tonmt[a]
            for b in bb
                @assert b == h
            end
        end
    end
    h2cnt(h) = length(filter(hh -> genmatchhier(h)(hh), df.somahier))
    somahiers = sort(collect(somahiers), by=h2cnt, rev=true)
    for h in somahiers
        l = h2cnt(h)
        @info "$h $l"
    end

    dedup!(cols) = begin
        while true
            toremove = []
            for c in cols
                for d in cols
                    c != d && startswith(c)(d) && push!(toremove, d)
                end
            end
            isempty(toremove) && break
            setdiff!(cols, toremove)
        end
    end
    get_data_from_source(aaa) = begin
        mh = genmatchhier(aaa)
        dfx = filter(r -> mh(r.somahier), df)
        datprev = Dict{String,Set{Float64}}()
        for r in eachrow(conndat.flne)
            bb = tonmt[r.SOURCE]
            @assert length(bb) == 1
            for b in bb
                if b == aaa
                    cc = tonmt[r.TARGET]
                    @assert length(cc) == 1
                    for c in cc
                        push!(get!(() -> Set{Float64}(), datprev, c), r.FLNe)
                    end
                end
            end
        end
        datproj = Dict{String,Set{Float64}}()
        for h in tgthiers
            vv = get!(() -> Set{Float64}(), datproj, h)
            mh = genmatchhier(h)
            cols = filter(names(df)) do n
                zz1data.is_projcol(n) && startswith("nmtI")(n) && mh(n)
            end
            dedup!(cols)
            for c in cols
                v = mean(dfx[:, c])
                push!(vv, v)
            end
        end
        areas = collect(union(keys(datproj), keys(datprev)))
        map(areas) do area
            flne = mean(log.(get(datprev, area, (0,))))
            proj = sum(datproj[area])
            (; proj, flne, area)
        end
    end

    selareas = somahiers[1:4]
    prevax = nothing
    gridx = GridLayout(grid[1, 1])
    gridx.height = 85
    for (kkk, area) in enumerate(selareas)
        row, col = divrem(kkk - 1, 4) .+ 1
        ax = Axis(gridx[row, col])

        dfx = get_data_from_source(area)
        N = h2cnt(area)
        ax.title = "Soma in $(areaname(area, capital=false))"
        ranked_scatter!(ax, DataFrame(dfx); rowx=r -> r.proj, rowy=r -> r.flne, rowcolor=_ -> :black, rowtext=r -> areashort(r.area))
        ax.ylabel = "Rank of Retrograde tracing (FLNe)"
        ax.xautolimitmargin = (0.05, 0.15)
        ax.yautolimitmargin = (0.05, 0.1)
        ax.xlabel = "Rank of projection strength ($N neurons)"
        if col > 1
            ax.ylabelvisible = false
            linkaxes!(ax, prevax)
            ax.xlabel = "($N neurons)"
            colgap!(gridx, col - 1, 4)
        end
        prevax = ax
    end
    selareas = filter(h -> h2cnt(h) >= 100, somahiers)
    conndat = map(selareas) do area
        get_data_from_source(area)
    end
    seltgts = sort(unique(reduce(vcat, [[r.area for r in dfx] for dfx in conndat])))
    conndat = [(i = findfirst(r -> r.area == b, dfx); i === nothing ? (; proj=0, flne=-Inf) : dfx[i]) for dfx in conndat, b in seltgts]
    ax1 = Axis(grid[2, 1][1, 1][1, 1])
    ax1.height = 50
    rowgap!(grid, 1, 8)
    fixname = Dict("area_12r/l, area_12m/o" => "area_12")
    ax1.yticks = (axes(selareas, 1), [areaname(get(fixname, a, a)) for a in selareas])
    ax1.ylabel = "Soma region"
    ax1.xlabel = "Target region"
    flnemat = [exp(d.flne) for d in conndat]
    vmin = minimum(flnemat)
    vmax = quantile(vec(flnemat), 0.9)
    p1 = heatmap!(ax1, flnemat', colorrange=(vmin, vmax), nan_color=:black)
    ax2 = Axis(grid[2, 1][1, 2][1, 1])
    projmat = [d.proj for d in conndat] ./ 1000
    vmin = minimum(filter(v -> true, projmat))
    vmax = quantile(vec(projmat), 0.9)
    p2 = heatmap!(ax2, projmat', nan_color=:black, colorrange=(vmin, vmax))
    for ax in (ax1, ax2)
        ax.xticks = (axes(seltgts, 1), [areaname(b) for b in seltgts])
        ax.xticklabelrotation = -pi / 4
        ax.xticklabelalign = (:left, :center)
        ax.xlabelpadding = -8
    end
    hideydecorations!(ax2)
    ax1.yminorticksvisible = false
    ax1.xminorticksvisible = false
    ax2.xminorticksvisible = false
    ax1.title = "Retrograde tracing (FLNe)"
    ax2.title = "Projection strength (mm)"
    ax2.xlabel = "Target region"
    linkaxes!(ax1, ax2)
    colgap!(content(grid[2, 1]), 1, 8)
    Colorbar(grid[2, 1][1, 1][1, 2], p1, tellheight=false, tellwidth=true)
    Colorbar(grid[2, 1][1, 2][1, 2], p2, tellheight=false, tellwidth=true)
    for j in 1:2
        colgap!(content(grid[2, 1][1, j]), 1, 4)
    end
    Label(grid[2, 1][1, 2][1, 2, Right()], "", width=8)
end

function plot_fig_recon(fig, env; _...)
    with_caption(L"Reconstruction of single-neuron projectomes in macaque PFC.")

    grid = fig[1, 1]
    with_subfig(grid[1, 1][1, 1], env) do grid, env
        with_caption(L"")
        plot_recon_imaging(grid, env; env...)
    end

    with_subfig(grid[1, 1][2, 1], env) do grid, env
        with_caption(L"")
        plot_recon_proc(grid, env; env...)
    end

    with_subfig(grid[1, 2][1, 1], env) do grid, env
        with_caption(L"")
        plot_recon_single_neurons(grid, env; env...)
    end

    with_subfig(grid[1, 2][2, 1], env) do grid, env
        with_caption(L".")
        plot_recon_compare(GridLayout(grid[1, 1]), env; env...)
    end

    grid = GridLayout(fig[2, 1])
    with_subfig(grid[1, 1][1, 1][1, 1], env) do grid, env
        with_caption(L"")
        plot_recon_somadist(grid, env; env...)
    end

    with_subfig(grid[1, 1][1, 1][1, 2], env) do grid, env
        with_caption(L"")
        plot_recon_res_by_inj(grid, env; env...)
        GridLayout(grid[end+1, 1])
    end
    with_subfig(grid[1, 1][2, 1], env) do grid, env
        with_caption(L"")
        plot_recon_injtable(grid, env; env...)
    end

    with_subfig(grid[1, 2][1, 1], env) do grid, env
        with_caption(L"")
        plot_result_itptct(grid, env; env...)
    end
    with_subfig(grid[1, 2][2, 1], env) do grid, env
        with_caption(L"")
        plot_result_depth(grid, env; env...)
    end
end

function plot_fig_compare2(fig, env; cmpdat, _...)
    with_caption(L"Comparison of single-neuron projectomes with fMRI data and bulk tracing data.")
    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"Progression timeline of single-neuron reconstruction for macaque dataset G97-1.")
        plot_rec_progress(grid, env; env...)
    end
    grid = fig[2, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_recon_compare2(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        plot_recon_compare_barplot(grid, env; env...)
    end

    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_recon_compare_markov(grid, env; env...)
    end
end
