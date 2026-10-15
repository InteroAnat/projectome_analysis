#!/usr/bin/julia


@everywhere using Distributions;
@everywhere using GLM;
@everywhere using NonparametricRegression;
@everywhere using FileIO;
@info "rss1: $(Sys.maxrss()/1e9)"
using Clustering;
using CategoricalArrays;
using NRRD;
using BeepBeep;
using NIfTI;
using ProgressMeter;
using ColorSchemes;
#import GIFTI;
@info "rss2: $(Sys.maxrss()/1e9)"


using Flux;
@info "rss3: $(Sys.maxrss()/1e9)"
using Zygote;
@info "rss4: $(Sys.maxrss()/1e9)"
@info "rss5: $(Sys.maxrss()/1e9)"
#using CUDA;
import UnicodePlots;
@info "rss6: $(Sys.maxrss()/1e9)"

using DirectQhull;
@info "rss7: $(Sys.maxrss()/1e9)"
using GeometryBasics
@info "rss8: $(Sys.maxrss()/1e9)"
import GLMakie;
@info "rss9: $(Sys.maxrss()/1e9)"

@everywhere GC.gc()

include("1-mods.jl")
include("1-rnn.jl")
include("1-ann.jl")
include("1-ds.jl")
include("2-qc.jl")

##########################################################################
#XXX dont look up
########################################
############################################################
## persist
# /tmp
# ##############################
# 

Make.@target guess_maxdt (df_stats) -> begin
    x = deepcopy(df_stats)
    y = unique(x[:, [:type, :stg]])
    y[:, :stg] = y[:, :stg] .% 10
    y = unique(y)
    sort!(y, [:type, :stg])
    insertcols!(y, :n => 0, :q5 => 0.0, :q9 => 0.0, :q95 => 0.0, :q99 => 0.0)
    for j = 1:nrow(y)
        tt = x[:, :type] .== y[j, :type] .&& (x[:, :stg] .% 10) .== y[j, :stg]
        tt = x[tt, :time_diff]
        y[j, [:q5, :q9, :q95, :q99]] .= quantile(skipmissing(tt), [0.5, 0.9, 0.95, 0.99])
        y[j, :n] = length(tt)
    end
    y
end



Make.@target sparsity_dat (snapshots2) -> begin
    df = deepcopy(snapshots2)
    filter!(r -> r.stg == maximum(df[df[:, :ds].==r.ds, :stg]), df)
    insertcols!(df, :max_density => 0.0, :density_stats => Dict())

    function get_swc_density(files)
        stats = Dict{Any,Float64}()
        nodes = Dict{Int,Any}()
        for fn in files
            open(endswith(fn, ".gz") ? `gzip -cd $fn` : `cat $fn`, "r") do io
                for l in eachline(io)
                    l = lstrip(l)
                    length(l) == 0 && continue
                    if l[1] == '#'
                        ann = parse_gapr_annot(l)
                        if ann !== nothing && ann.key == "root"
                            p = nodes[ann.id]
                            cub = map(v -> round(v / 100) * 100, p)
                            stats[cub] = get(stats, cub, 0.0) + 1
                        end
                    else
                        (id, typ, p, rad, par) = parse_swc_line(l)
                        nodes[id] = p
                        par != -1 || continue
                        d = norm(nodes[par] .- p)
                        cub = map(v -> round(v / 100) * 100, p)
                        #stats[cub]=get(stats, cub, 0.0)+d
                    end
                end
            end
        end
        stats
    end

    for row in eachrow(df)
        fn = cache_file_swcdir(row.ds, row.stg)
        swcs = filter(x -> match(r".*\.swc(\.gz)?$", x) !== nothing, readdir(fn, join=true))
        stats = get_swc_density(swcs)
        row.max_density = maximum(values(stats))
        row.density_stats = stats
    end
    df
end

Make.@target fig_sparsity (sparsity_dat) -> begin
    df = deepcopy(sparsity_dat)
    sparsity_fix = Dict(
        "17391" => "Moderately dense",
        "18133" => "Moderately dense",
        "18708" => "Moderately sparse",
        "18709" => "Moderately sparse",
        "18925" => "Moderately sparse",
        "18926" => "Very sparse",
        "192095" => "Moderately sparse",
        "192096" => "Moderately sparse",
        "192100" => "Moderately sparse",
        "192101" => "Moderately dense",
        "192102" => "Very sparse",
        "202814" => "Moderately dense",
        "202813" => "Moderately dense",
        "202816" => "Moderately dense",
        #"202812"=>"unknown",
        "202815" => "Moderately dense",
        "221059.1-1" => "unknown",
        "221059.2-1" => "unknown",
    )
    filter!(r -> !in(r.ds, ["18926.3", "18926.4", "192101",]), df)
    p = Plots.Plot()
    for sp in ["Moderately dense", "Very sparse", "Moderately sparse"]
        df2 = filter(r -> sparsity_fix[r.ds] == sp, df)
        scatter!(p, [r.len for r in eachrow(df2)], [r.max_density for r in eachrow(df2)],
            label=sp,
            series_annotations=[r.ds for r in eachrow(df2)],
            xscale=:log10,
            yscale=:log10,
        )
    end
    p
end







include("extra.jl")
include("3-mus-hy.jl")
include("cluster.jl")




## plot location change per user, to see if two logins with one account.
Make.@target df_tracks () -> begin
    ds = 18925
    tmp_pth = "/tmp/gapr-stats-tracks-$ds.txt"
    cmd = gapr_cmd(`show --playback $(mirror_files(ds, :repo)) --script misc/gen-stats.lua --image file://$(mirror_files(ds, :cat))`)
    if !isfile(tmp_pth)
        cmd = addenv(cmd, ["AUTOCLOSE=1", "GAPR_DEBUG=", "DISPLAY=:0",
            "LD_PRELOAD=/usr/lib64/libjemalloc.so.2",
            "GEN_TRACKS=$tmp_pth"])
        run(cmd)
    end
    v = readdlm(tmp_pth, ' ')
    p = v[:, [3, 4, 5]]
    q = Vector{String}(v[:, 2])
    names = unique(q)
    for n in names
        idx2 = findall(q .== n)
        for i = 1:5000:length(idx2)
            println(n, " ", i)
            idx = idx2[i:min(i + 5000, length(idx2))]
            x = plot3d(p[idx, 1], p[idx, 2], p[idx, 3], size=(1024, 768))
            display(x)
            xx = readline(stdin)
        end
    end
    q, p
end



function guess_img_size(len_mm)
    inch(x) = 25.4 * x
    pt(x) = inch(x / 72.27)
    bp(x) = inch(x / 72)
    for px in [64, 80, 96, 128, 160, 192, 240, 256, 320, 360, 384, 480, 512, 640, 720, 768, 960, 1024, 1280]
        for pr in [1, 2, 3, 4, 5, 6, 7, 8]
            dpi = px * pr / (len_mm / inch(1))
            dpi >= 300 || continue
            dpi < 500 || continue
            px2bp = (pr / (px * pr)) * len_mm / bp(1)
            px2bp < 0.8 || continue
            px2bp >= 0.4 || continue
            println("px $px ($(px*pr)); pr $pr; dpi $(round(Int, dpi)); lw $(round(px2bp, digits=2))")
        end
    end
end


Make.@target conv16 () -> begin
    function enc16_fp(x::UInt16)
        if x < 1024
            return x
        end
        i = UInt16(2)
        x = x >> 1
        while x >= 1024
            x = x >> 1
            i += 1
        end
        # now 512<=x<1024, only 9 bits can vary
        x = x & 0x1ff
        return UInt16((i << 9) | x)
    end
    function dec16_fp(x::UInt16)
        i = x >> 9
        if i < 2
            return x
        end
        # add 0.5 to compensate truncated bits
        x = ((x & 0x1ff) << 1) | 0x401
        return UInt16(x << (i - 2))
    end

    function enc16_trunc(x::UInt16)
        return UInt16(min(4095, x))
    end
    function dec16_trunc(x::UInt16)
        return x
    end

    #=
    img=load("precious/conv16-eg2.tif")
    img=map(v->v.val.i, img)
    minv, maxv=extrema(img)
    println(minv)
    println(maxv)
    function enc16_lin(x::UInt16)
    return round(UInt16, 4095.0*(clamp(x, minv, maxv)-minv)/(maxv-minv))
    end
    function dec16_lin(x::UInt16)
    return round(UInt16, clamp(x/4095.0*(maxv-minv)+minv, 0, 65535))
    end
    =#

    function enc16_shift(x::UInt16)
        return UInt16(x >> 4)
    end
    function dec16_shift(x::UInt16)
        return UInt16((x * 2 + 1) << 3)
    end

    function contrast(img2, fn, a, b)
        r = map(v -> round(UInt8, 255 * (clamp(v, a, b) - a) / (b - a)), img2)
        save(fn, r)
    end
    function doconv(tag, enc16, dec16)
        df = DataFrame([:in => 0:65535])
        insertcols!(df, :out => 0, :back => 0, :diff => 0, :rdiff => 0.0, :logv => 0.0)
        for i = 1:size(df)[1]
            v = df[i, :in]
            df[i, :out] = vo = Int64(enc16((UInt16(v))))
            df[i, :back] = vback = Int64(dec16((UInt16(vo))))
            df[i, :diff] = dd = vback - v
            df[i, :rdiff] = dd / (v + 1)
            df[i, :logv] = (log(4, v + 1) + 1) * 512 * 2
        end
        x = df
        dd = 64
        err = map(v -> maximum(abs.(x[max(1, v - dd):v, :diff])), 1:dd:65536)
        err = map(v -> min(v, 100), err)

        p = plot(x[1:dd:65536, :in], err,
            ; plotstyle()...,
            seriescolor=palette(:auto)[1],
            yguidefontcolor=palette(:auto)[1],
            leg=false,
            xguide=tex("Input intensity (\\numrange{0}{65535})"),
            yguide="Conversion error",
            ylims=(-2, 102),
        )
        p = plot!(twinx(p), x[1:dd:65536, :in], x[1:dd:65536, :out],
            ; plotstyle()...,
            seriescolor=palette(:auto)[2],
            yguidefontcolor=palette(:auto)[2],
            leg=false,
            yguide=tex("Encoded value (\\numrange{0}{4095})"),
            size=(346, 200),
        )
        savefig(p, "extra/conv16-$tag.tex")
        return

        img2 = map(v -> enc16(v), img)
        if maximum(img2) >= 4096
            error("wrong conv $tag")
        end
        img2 = map(v -> dec16(v), img2)
        contrast(img2, "extra/conv16-$tag-rng1.png", 0, 32)
        contrast(img2, "extra/conv16-$tag-rng2.png", 0, 30000)
    end

    doconv("fp", enc16_fp, dec16_fp)
    doconv("trunc", enc16_trunc, dec16_trunc)
    #doconv("lin", enc16_lin, dec16_lin)
    doconv("shift", enc16_shift, dec16_shift)
end


# Make.@target autorec_cor_dat (snapshots3) -> begin
#     x = unique(snapshots3[:, [:ds, :stg]])
#     dss = x[x[:, :stg].==9, :ds]
#     dss = filter(v -> !in(dataset_from_id(v).mode, ["manual", "selected"]), dss)
#     function read_swcs(fn::String)
#         swcs = filter(x -> match(r".*\.swc(\.gz)?$", x) !== nothing, readdir(fn, join=true))
#         r = Dict{Int64,Tuple{Float64,Float64,Float64,Int64}}()
#         for f in swcs
#             open(`gzip -cd $f`, "r") do io
#                 while !eof(io)
#                     l = readline(io)
#                     if length(l) == 0 || l[1] == '#'
#                         continue
#                     end
#                     v = split(l, " ")
#                     id = parse(Int64, v[1])
#                     x = parse(Float64, v[3])
#                     y = parse(Float64, v[4])
#                     z = parse(Float64, v[5])
#                     p = parse(Int64, v[7])
#                     r[id] = (x, y, z, p)
#                 end
#             end
#         end
#         r
#     end
#
#     dat = DataFrame([:ds => dss])
#     insertcols!(dat, :fp => 0.0, :tp => 0.0, :fn => 0.0, :pp => 0.0, :p => 0.0)
#
#     for dsi in 1:length(dss)
#         ds = dss[dsi]
#         swc0 = read_swcs(cache_file_swcdir(ds, 0))
#         swc9 = read_swcs(cache_file_swcdir(ds, 9))
#         ii = findfirst(snapshots3[:, :ds] .== ds .&& snapshots3[:, :stg] .== 0)
#         jj = findfirst(snapshots3[:, :ds] .== ds .&& snapshots3[:, :stg] .== 9)
#         dat[dsi, :pp] = snapshots3[ii, :len]
#         dat[dsi, :p] = snapshots3[jj, :len]
#         fp = 0.0
#         tp = 0.0
#         fn = 0.0
#         function haslink(swc, a, b)
#             aa = get(swc, a, nothing)
#             if aa !== nothing && aa[4] == b
#                 return true
#             end
#             bb = get(swc, b, nothing)
#             if bb !== nothing && bb[4] == a
#                 return true
#             end
#             return false
#         end
#         for n in keys(swc0)
#             nn = swc0[n]
#             if nn[4] == -1
#                 continue
#             end
#             nnn = swc0[nn[4]]
#             dd = (nn[1], nn[2], nn[3]) .- (nnn[1], nnn[2], nnn[3])
#             d = sqrt(sum(dd .* dd))
#             if haslink(swc9, n, nn[4])
#                 tp += d
#             else
#                 fp += d
#             end
#         end
#         for n in keys(swc9)
#             nn = swc9[n]
#             if nn[4] == -1
#                 continue
#             end
#             nnn = swc9[nn[4]]
#             dd = (nn[1], nn[2], nn[3]) .- (nnn[1], nnn[2], nnn[3])
#             d = sqrt(sum(dd .* dd))
#             if !haslink(swc0, n, nn[4])
#                 fn += d
#             end
#         end
#         dat[dsi, :tp] = tp
#         dat[dsi, :fp] = fp
#         dat[dsi, :fn] = fn
#     end
#     return dat
# end
#
# Make.@target xform_swcs_mirror (xform_swcs) -> begin
#     swcs = deepcopy(xform_swcs)
#     filter!(r -> check_neuron_qc(r.fn) != 0, swcs)
#     insertcols!(swcs, :mirrored => false)
#     outdir = force_dir(cache_dir_swcs_xform_mirror()) do tmpdir
#         fnscopy = []
#         fns = map(eachrow(swcs)) do r
#                   pos = get_soma_pos(r.path)
#                   if pos[1] <= 456 * 25 / 2
#                       push!(fnscopy, r.path)
#                       return missing
#                   end
#                   r.mirrored = true
#                   r.path
#               end |> skipmissing |> collect
#         awk = "awk \"{print 456*25-\$1,\$2,\$3}\""
#         cmd = gapr_cmd(`map $fns -o "$tmpdir/%s%e" -x $awk`)
#         cc = run(cmd)
#         check_exit(cc)
#         cmd = gapr_cmd(`map $fnscopy -o "$tmpdir/%s%e" -x "cat"`)
#         cc = run(cmd)
#         check_exit(cc)
#     end
#     for r in eachrow(swcs)
#         r.dir = outdir
#         r.path = joinpath(outdir, r.fn)
#     end
#     swcs
# end

# Make.@target debug (df_stats) -> begin
#     Plots.gr()
#     function countwt(x, w, t)
#         n = 0
#         for r in eachrow(x)
#             if r.who == w && r.type == t
#                 n += 1
#             end
#         end
#         n
#     end
#     plots = []
#     map([("192100", 1), ("192101", 1), ("192102", 1), ("192102", 2)]) do (ds, stg)
#         x = filter(r -> r.ds == ds && r.stg == stg, deepcopy(df_stats))
#         types = unique(sort(x.type))
#         users = unique(x.who)
#         cnts = sort([(count(x.who .== w), w) for w in users])
#         if length(cnts) > 5
#             cnts = cnts[end-5:end]
#         end
#         for (nnn, w) in cnts
#             lbl = "$ds-$stg-$w-$nnn"
#             @info "$lbl ..."
#             dist = [countwt(x, w, t) for t in types]
#             push!(plots, bar(types, dist, label=lbl))
#         end
#     end
#     plots
# end
#



