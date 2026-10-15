function decimate_swcs_helper(swcs, outdir)
    function count_nodes(fn)
        n = 0
        open(fn, "r") do io
            for l in eachline(io)
                if match(r"^\d+ \d+", l) !== nothing
                    n += 1
                end
            end
        end
        n
    end
    prev_sizes = Dict{String,Int64}()
    outdir = force_dir(outdir) do tmpdir
        function conv_files(d, h, swcs)
            length(swcs) != 0 || return
            cmd = gapr_cmd(`map $swcs -o "$tmpdir/%s%e" --decimate $d:$h`)
            cc = run(cmd)
            check_exit(cc)
        end
        swcs0 = []
        swcs1 = []
        swcs2 = []
        swcs3 = []
        for swc in swcs.path
            n = count_nodes(swc)
            prev_sizes[swc] = n
            if n > 80000
                push!(swcs3, swc)
            elseif n > 10000
                push!(swcs2, swc)
            elseif n > 1000
                push!(swcs1, swc)
            else
                push!(swcs0, swc)
            end
        end
        conv_files(128, 12, swcs3)
        conv_files(64, 6, swcs2)
        conv_files(20, 2, swcs1)
        conv_files(10, 1, swcs0)
    end

    df = map(swcs.fn, swcs.path) do swc, path
        n2 = count_nodes(joinpath(outdir, swc))
        n1 = prev_sizes[path]
        (fn=swc, path=path, nn_resamp=n2, nn_prev=n1, resamp_dir=outdir)
    end
    DataFrame(df)
end


Make.@target cluster_hy_calc (xform_swcs_mirror, neurons_summary) -> begin
    swcs = deepcopy(xform_swcs_mirror)
    swcs = decimate_swcs_helper(swcs, "/tmp/swcs-xform-small-mus-hy")
    nrns = deepcopy(neurons_summary)
    call_fnt_dist_helper(swcs, nrns, npar=7) do row
        match(r"^/root/grey/BS/IB/HY(/.*)?$", row.soma_ba) !== nothing || return false
        @assert row.stg % 10 == 9
        @assert row.ord < 999000
        true
    end
end


Make.@target projections_accum (xform_swcs_mirror) -> begin
    swcs = deepcopy(xform_swcs_mirror)
    projs = map(swcs.fn, swcs.path) do fn, fn_path
        strength = Dict{String,NTuple{2,Float64}}()
        nodes = Dict{Int64,NTuple{3,Float64}}()
        open(fn_path, "r") do io
            for l in eachline(io)
                (!isempty(l) && l[1] != '#') || continue
                (id, typ, pos, rad, p) = parse_swc_line(l)
                nodes[id] = pos
                (p != -1 && typ == 2) || continue
                s = norm(pos .- nodes[p])
                s = pos[1] < 456 * 25 / 2 ? (s, 0.0) : (0.0, s)
                ann = get_allen_hier(pos)
                strength[ann] = get(strength, ann, (0.0, 0.0)) .+ s
            end
        end
        (fn=fn, projraw=strength)
    end

    ann_ids = get_allen_tree(rev=true)
    projs = map(projs) do row
        hh = My.ProjHelper()
        for ann_id in ann_ids
            v = get(row.projraw, ann_id, (0.0, 0.0))
            v != (0.0, 0.0) || continue
            hh.dat[ann_id] = v ./ 1000
            ann_id != "/" || continue
            par_id = get_ann_hier_par(ann_id)
            row.projraw[par_id] = get(row.projraw, par_id, (0.0, 0.0)) .+ v
        end
        (fn=row.fn, mm_proj=hh)
    end
    DataFrame(projs)
end

Make.@target cluster_mus_hy (cluster_hy_calc, neurons_summary, projections_accum) -> begin
    outdir = "/data/tmp/swcs-xform-clustered-mus-hy"
    swcs, mat = deepcopy(cluster_hy_calc)
    nrns = deepcopy(neurons_summary)
    swcs = leftjoin(swcs, nrns, on=:fn)

    seli = map(eachrow(swcs)) do row
        match(r"^/root/grey/BS/IB/HY(|/.*)$", row.soma_ba) !== nothing || return false
        row.mm_axon >= 0.3 || return false
        row.qc || return false
        true
    end
    swcs = swcs[seli, [:fn, :path]]
    mat = mat[seli, seli]

    if false
        projs = leftjoin(swcs, deepcopy(projections_accum), on=:fn)
        anns = get_allen_tree(rev=true)
        projs = map(eachrow(projs)) do row
            va = Float64[]
            for ann in anns
                push!(va, row.mm_proj[ann, :left])
                push!(va, row.mm_proj[ann, :right])
            end
            va
        end
        projs = reduce(hcat, projs)
        mat .= 1 .- cor(projs)
    end

    h = hclust(mat, linkage=:ward, branchorder=:barjoseph)
    cls = cutree(h, k=6)
    cls = invperm(unique(cls[h.order]))[cls]
    insertcols!(swcs, :cls => cls)
    swcs = remap_groups_alt(swcs, outdir)
    (swcs, h)
end

Make.@target pick_brain_area_hy (cluster_mus_hy, projections_accum) -> begin
    swcs = deepcopy(cluster_mus_hy)[1]
    projs = deepcopy(projections_accum)
    anns = get_allen_tree(rev=true)
    leftjoin!(swcs, projs, on=:fn)
    bas = map(anns) do ba
        quant = quantile(map(v -> v.mm_proj[ba, :all], eachrow(swcs)), [0.7, 0.8, 0.9, 0.95, 0.98])
        (ba=ba, q7=quant[1], q8=quant[2], q9=quant[3], q95=quant[4], q98=quant[5])
    end
    function run_anova(ba; filtercls=0:1000)
        y = map(v -> v.mm_proj[ba, :both], eachrow(swcs))
        y = [
            :y => map(v -> v[1] + v[2], y),
            :yl => map(v -> v[1], y),
            :yr => map(v -> v[2], y),
        ]
        res = map(y) do (lr, yvals)
            df = deepcopy(swcs[:, [:cls]])
            insertcols!(df, :y => yvals,
                :c => categorical(df[:, :cls]),
            )
            filter!(r -> in(r.cls, filtercls), df)
            nullmodel = lm(@formula(y ~ 1), df)
            model = lm(@formula(y ~ 1 + c), df)
            (ba, lr, ftest(nullmodel.model, model.model))
        end
    end
    res = map(anns) do ba
        run_anova(ba, filtercls=[4, 5])
    end
    res = reduce(vcat, res)
    res = filter(a -> a[3] < 0.05, map((a) -> (a[1], a[2], a[3].pval[2]), res))
    res = sort(res, lt=(a, b) -> (a[3] < b[3]))
    #return res

    #=
    sel_bas=[
    ("/root/grey/BS/MB/MBmot/PAG", :y, 2.8212177501752686e-72)
    ("/root/grey/BS/MB/MBmot/SCm", :y, 3.5576012048077786e-31)
    ("/root/grey/BS/MB/MBmot/PRT", :y, 1.4427126994771593e-30)
    ("/root/grey/BS/MB/MBmot/MRN", :y, 6.775048787905854e-30)
    ("/root/grey/BS/MB/MBmot/CUN", :y, 2.1062171322315946e-19)
    ("/root/grey/BS/MB/MBmot/VTA", :y, 2.2976683200522113e-16)
    ("/root/grey/BS/MB/MBmot/SNr", :y, 1.618407857267906e-11)

    ("/root/grey/BS/IB/HY/LZ/LHA", :y, 1.9255429107324548e-40)
    ("/root/grey/BS/IB/HY/LZ/PeF", :y, 9.122509290146693e-20)
    ("/root/grey/BS/IB/HY/LZ/LPO", :y, 5.522716855851647e-18)
    ("/root/grey/BS/IB/HY/LZ/TU", :y, 2.0753916552834686e-16)
    ("/root/grey/BS/IB/HY/LZ/ZI", :y, 2.4762592290430323e-16)
    ("/root/grey/BS/IB/HY/LZ/RCH", :y, 1.9426165455627653e-15)
    ("/root/grey/BS/IB/HY/LZ/PSTN", :y, 6.140180083001339e-13)
    ("/root/grey/BS/IB/HY/LZ/STN", :y, 2.1254788352984333e-7)

    ("/root/grey/BS/IB/HY/MEZ/PH", :y, 5.262426634467602e-36)
    ("/root/grey/BS/IB/HY/MEZ/MBO", :y, 4.8378454259002784e-17)
    ("/root/grey/BS/IB/HY/MEZ/AHN", :y, 8.655805457885938e-17)
    ("/root/grey/BS/IB/HY/MEZ/VMH", :y, 2.0703158741021697e-16)
    ("/root/grey/BS/IB/HY/MEZ/PMv", :y, 3.726972340242101e-13)
    ("/root/grey/BS/IB/HY/MEZ/PMd", :yl, 6.748643399498813e-13)
    ("/root/grey/BS/IB/HY/MEZ/MPN", :y, 1.7386802280988614e-11)

    ("/root/grey/BS/IB/TH/DORsm/SPF", :y, 7.836771626172802e-20)
    ("/root/grey/BS/IB/TH/DORsm/SPA", :y, 2.184710011426947e-19)
    ("/root/grey/BS/IB/TH/DORpm/ILM", :y, 5.565672398206688e-18)
    ("/root/grey/BS/IB/TH/DORsm/VENT", :y, 6.899631034783815e-14)
    ("/root/grey/BS/IB/TH/DORpm/MED", :y, 3.4624527759949815e-9)
    ("/root/grey/BS/IB/TH/DORpm/MTN", :y, 2.1849228553042938e-8)

    ("/root/grey/BS/HB/P", :y, 2.1914012724393614e-29)
    ("/root/grey/BS/HB/P", :yl, 2.456039438661621e-25)
    ("/root/grey/BS/HB/P/P-mot", :y, 2.833628651378571e-20)
    ("/root/grey/BS/HB/P/P-sen", :y, 1.3664407200695879e-18)
    ("/root/grey/BS/HB/P/P-mot", :yl, 1.903420390464073e-17)
    ("/root/grey/BS/HB/P/P-sen", :yl, 5.524350439040781e-15)
    ("/root/grey/BS/HB/P/P-sat", :y, 1.9333239873826954e-9)
    ("/root/grey/BS/HB/MY/MY-mot", :yl, 4.20279757074755e-8)
    ("/root/grey/BS/HB/MY/MY-mot", :y, 4.609321448248434e-8)
    ("/root/grey/BS/HB/MY", :yl, 6.68132762655782e-8)
    ("/root/grey/BS/HB/P", :yr, 1.7225861813760587e-7)
    ("/root/grey/BS/HB/P/P-sat", :yl, 2.460094315046818e-7)
    ("/root/grey/BS/HB/MY", :y, 3.5140806284701607e-7)

    ("/root/grey/BS/MB/MBsta", :y, 7.12743086059861e-33)
    ("/root/grey/BS/MB/MBsta", :yl, 1.427404331237034e-26)
    ("/root/grey/BS/MB/MBsta/PPN", :y, 4.752873267110996e-25)
    ("/root/grey/BS/MB/MBsta/PPN", :yl, 5.657421374287777e-19)
    ("/root/grey/BS/MB/MBsen", :y, 1.7857985946260944e-18)
    ("/root/grey/BS/MB/MBsen/IC", :y, 5.779911461328651e-16)
    ("/root/grey/BS/MB/MBsen", :yl, 2.1347271991158037e-13)
    ("/root/grey/BS/MB/MBsta", :yr, 7.620240395752537e-13)
    ("/root/grey/BS/MB/MBsen", :yr, 2.7932123288960943e-12)
    ("/root/grey/BS/MB/MBsta/RAmb", :y, 1.2411482069587542e-10)
    ("/root/grey/BS/MB/MBsta/SNc", :y, 1.64761063677551e-9)
    ("/root/grey/BS/MB/MBsta/SNc", :yl, 2.5782914256491588e-8)

    ("/root/fiber tracts/cm", :y, 3.85902456450259e-38)
    ("/root/fiber tracts/cm/IIIn", :y, 7.32301288135482e-23)
    ("/root/fiber tracts/mfbs", :y, 1.6960674212009447e-21)
    ("/root/fiber tracts/cm/IIn", :y, 4.754169635820695e-20)
    ("/root/fiber tracts/mfbs", :yl, 2.81965928080484e-19)
    ("/root/fiber tracts/cbf", :y, 2.7309033804240234e-18)
    ("/root/fiber tracts/cbf/cbp", :y, 3.376215473948888e-18)
    ("/root/fiber tracts/eps", :y, 5.94334697488923e-18)
    ("/root/fiber tracts/mfbs/mfsbshy", :y, 8.222081110031291e-17)
    ("/root/fiber tracts/cbf", :yl, 1.1666148943156708e-16)
    ("/root/fiber tracts/cm/drt", :y, 9.07071932500655e-16)
    ("/root/fiber tracts/mfbs/mfbc", :y, 2.312562780602531e-14)
    ("/root/fiber tracts/eps/rust", :y, 1.1815330716076864e-13)
    ("/root/fiber tracts/eps", :yl, 1.8135140483880295e-13)
    ("/root/fiber tracts/cm/VIIIn/cVIIIn", :y, 3.935089607123446e-12)
    ("/root/fiber tracts/cm/VIIIn", :y, 3.935089607123446e-12)
    ("/root/fiber tracts/cm", :yr, 1.2467309256386896e-10)
    ("/root/fiber tracts/eps/rust", :yl, 6.614897518575485e-10)
    ("/root/fiber tracts/cm/VIIIn/cVIIIn", :yl, 1.6685170858296115e-8)
    ("/root/fiber tracts/cm/VIIIn", :yl, 1.6685170858296115e-8)
    ("/root/fiber tracts/lfbs/cst", :y, 2.14179100254543e-7)
    ("/root/fiber tracts/eps", :yr, 2.364701995030253e-7)
    ("/root/fiber tracts/cm/IVn", :y, 5.939910817200014e-7)
    ("/root/fiber tracts/cm/VIIIn/cVIIIn/cic", :y, 1.9916948910391737e-6)
    ("/root/fiber tracts/cbf/arb", :y, 2.162111633213838e-6)

    ("/root/grey/BS/IB/HY/PVR/MPO", :y, 5.968425749673856e-24)
    ("/root/grey/BS/IB/HY/PVZ/ARH", :y, 8.535455502455247e-16)
    ("/root/grey/BS/IB/HY/PVR/DMH", :y, 7.17572500162393e-15)
    ("/root/grey/BS/IB/HY/PVR/PS", :y, 1.3515955078645436e-13)
    ("/root/grey/BS/IB/HY/PVR/AVP", :y, 4.4625513468233644e-10)
    ("/root/grey/BS/IB/HY/PVZ/PVi", :y, 1.7322005477280358e-9)
    ("/root/grey/BS/IB/HY/PVR/PD", :yl, 1.6031333723766817e-8)
    ("/root/grey/BS/IB/HY/PVZ/PVH", :y, 1.4422152505070805e-7)
    ("/root/grey/BS/IB/HY/PVR/ADP", :y, 1.5629495035427297e-7)
    ("/root/grey/BS/IB/HY/PVR/SBPV", :y, 1.901473360859124e-7)
    ("/root/grey/BS/IB/HY/PVR/VLPO", :y, 9.99366019982411e-7)
    ("/root/grey/BS/IB/HY/PVR/VMPO", :y, 2.7273544318970645e-6)

    ("/root/grey/CH/CNU/PAL", :y, 6.5057681894683035e-24)
    ("/root/grey/CH/CNU/STR", :y, 4.859131716631904e-14)


    ("/root/grey/CB/CBX", :y, 4.50504757628557e-24)
           ]
           =#
    sel_bas = [
        "/root/fiber tracts/cbf", #:y, 2.7309033804240234e-18)
        "/root/fiber tracts/cm", #:y, 3.85902456450259e-38)
        "/root/fiber tracts/eps", #:y, 5.94334697488923e-18)
        "/root/fiber tracts/mfbs", #:y, 1.6960674212009447e-21)
        "/root/grey/BS/HB/P/P-mot", #:y, 2.833628651378571e-20)
        "/root/grey/BS/HB/P/P-sen", #:y, 1.3664407200695879e-18)
        "/root/grey/BS/IB/HY/LZ/LHA", #:y, 1.9255429107324548e-40)
        "/root/grey/BS/IB/HY/LZ/LPO", #:y, 5.522716855851647e-18)
        "/root/grey/BS/IB/HY/LZ/PeF", #:y, 9.122509290146693e-20)
        "/root/grey/BS/IB/HY/LZ/TU", #:y, 2.0753916552834686e-16)
        "/root/grey/BS/IB/HY/LZ/ZI", #:y, 2.4762592290430323e-16)
        "/root/grey/BS/IB/HY/ME",
        "/root/grey/BS/IB/HY/MEZ/AHN", #:y, 8.655805457885938e-17)
        "/root/grey/BS/IB/HY/MEZ/MBO", #:y, 4.8378454259002784e-17)
        "/root/grey/BS/IB/HY/MEZ/MPN",
        "/root/grey/BS/IB/HY/MEZ/PH", #:y, 5.262426634467602e-36)
        "/root/grey/BS/IB/HY/MEZ/VMH", #:y, 2.0703158741021697e-16)
        "/root/grey/BS/IB/HY/PVR/AVP",
        "/root/grey/BS/IB/HY/PVR/MPO", #:y, 5.968425749673856e-24)
        "/root/grey/BS/IB/HY/PVR/PS",
        "/root/grey/BS/IB/HY/PVZ/ARH", #:y, 8.535455502455247e-16)
        "/root/grey/BS/IB/TH/DORpm/ILM", #:y, 5.565672398206688e-18)
        "/root/grey/BS/IB/TH/DORsm/SPA", #:y, 2.184710011426947e-19)
        "/root/grey/BS/IB/TH/DORsm/SPF", #:y, 7.836771626172802e-20)
        "/root/grey/BS/MB/MBmot/CUN", #:y, 2.1062171322315946e-19)
        "/root/grey/BS/MB/MBmot/MRN", #:y, 6.775048787905854e-30)
        "/root/grey/BS/MB/MBmot/PAG", #:y, 2.8212177501752686e-72)
        "/root/grey/BS/MB/MBmot/PRT", #:y, 1.4427126994771593e-30)
        "/root/grey/BS/MB/MBmot/SCm", #:y, 3.5576012048077786e-31)
        "/root/grey/BS/MB/MBmot/VTA", #:y, 2.2976683200522113e-16)
        "/root/grey/BS/MB/MBsen/IC", #:y, 5.779911461328651e-16)
        "/root/grey/BS/MB/MBsta/PPN", #:y, 4.752873267110996e-25)
        "/root/grey/CB/CBX", #:y, 4.50504757628557e-24)
        "/root/grey/CH/CNU/PAL", #:y, 6.5057681894683035e-24)
    ]
    #=
    sel_bas2=[
    "AHN", "ARH", "BST", "DMH", "LHA", "LSv", "ME", "MEA", "MPN",
    "MPO", "MRN", "OT", "PA", "PAG", "PH", "PMv", "RCH", "TMv",
    "TU", "VMH",
    ]
    =#
    #=
    	swcs2=leftjoin(swcs, projs, on=:fn)
    	sel_bas2=[
    		 "AHN", "ARH", "BST", "DMH", "LHA", "LSv", "ME", "MEA", "MPN",
    		 "MPO", "MRN", "OT", "PA", "PAG", "PH", "PMv", "RCH", "TMv",
    		 "TU", "VMH",
    		 ]
    	@assert sort(sel_bas)==sort(map(v->get_allen_hier(v), sel_bas2))
    =#
    sort(map(v -> (v.q95, v.ba), bas), rev=true)[1:60], sel_bas
end

Make.@target fig_soma_dist (cluster_mus_hy, neurons_summary) -> begin
    swcs = deepcopy(cluster_mus_hy)[1]
    nrns = deepcopy(neurons_summary)
    leftjoin!(swcs, nrns, on=:fn)
    p = violin(swcs[:, :cls], map(v -> v[3], swcs[:, :soma_pos]),
        #label=map(c->tex("#$c"), swcs[:, :cls]),
        #seriescolor=map(c->palette(:auto)[c], swcs[:, :cls]),
        yticks=([5000, 7500], ["Anterior", "Posterior"]),
        ytickfontrotation=90,
        size=(210, 180),
        xguide="Cluster #",
        yguide="Soma position",
        guidefontsize=7,
        tickfontsize=5,
        legend=:none,
        linewidth=0.5,
    )
    plot!(p,
        seriescolor=map(c -> palette(:auto)[c], swcs[:, :cls]),
    )
    savefig(p, "extra/soma-dist-hy.tex")
    1
end

function clustering_plot_func(swcs, h, projs, sel_bas, texfile; reorder=false)
    swcs = leftjoin(swcs, projs, on=:fn)
    p1 = plot([0, 1], color=:transparent, label=tex("{}"))
    cls = unique(swcs[h.order, :cls])
    cls == 1:length(cls) || error("order")

    for c = cls
        l, r = findall(swcs[h.order, :cls] .== c) |> extrema
        vspan!(p1, [l - 0.5, r + 0.5],
            label=tex("#$c"),
            color=palette(:auto)[c],
        )
    end
    plot!(p1, h,
        extra_kwargs=Dict(:subplot => Dict("scale only axis" => true,)),
        label="",
        linewidth=0.5,
        framestyle=:none,
        ylim=(0, maximum(h.heights) * 1.05),
        xlim=(0, length(h.order) + 1),
    )
    plot!(p1,
        legend_title="Clusters",
        legend=:inline,
        #legend=:inside,
        #legend=:legend,
        legend_title_font_pointsize=6,
        legend_font_pointsize=5,
        legend_foreground_color=:transparent,
        legend_column=2,
    )
    projs_mat = swcs[:, [:fn]]
    for ba in sel_bas
        insertcols!(projs_mat, ba => 0.0)
        for i in 1:size(swcs)[1]
            projs_mat[i, ba] = swcs[i, :mm_proj][ba, :all]
        end
    end
    sel_bas = filter(sel_bas) do ba
        maximum(projs_mat[:, ba]) >= 0.05
    end
    if reorder
        m = cor(Matrix(projs_mat[:, sel_bas]))
        size(m) == (length(sel_bas), length(sel_bas)) || error("err")
        sel_bas = sel_bas[hclust(1 .- m, linkage=:ward, branchorder=:barjoseph).order]
    end
    cticks = 0:0.25:1.0
    cticks = (cticks, join(map(v -> "\$$v\$", cticks), ","))
    function normalize_heat(mat)
        mat = Matrix(mat)
        mat ./ maximum(mat, dims=1)
    end
    function map_ba(v)
        v = replace(String(v), "_" => "\\_")
        v = split(v, "/")[end]
        tex(v)
    end
    p3 = heatmap(normalize_heat(projs_mat[h.order, sel_bas])',
        #fillcolor=cgrad(:OrRd_9),
        fillcolor=cgrad(ColorScheme([Colors.RGB(0, 0, 1), Colors.RGB(1, 0, 0)])),
        extra_kwargs=Dict(:subplot => Dict(
            "scale only axis" => true,
            "every colorbar/.append style" => "height=20mm,width=2mm,at={(1.08,0.5)},anchor=west,xticklabel pos={lower}, xticklabels={{$(cticks[2])}}",
            "xlabel style" => "{anchor=west, at={{(ticklabel cs:0.0)}}}",
        )),
        ymirror=true,
        grid=false,
        yticks=(1:length(sel_bas), map(map_ba, sel_bas)),
        foreground_color_border=:white,
        showaxis=:y,
        tick_direction=:none,
        colorbar_title=tex("Normalized projection strength"),
        colorbar_titlefontsize=6,
        colorbar_ticks=cticks[1],
        legend=:right,
        colorbar_tickfontsize=5,
        clims=(0, 1),
        #tickfontsize=6,
        tickfontsize=5,
        colorbar=:right,
        xlim=(0, length(h.order) + 1),
        xguide="Neurons",
        guidefontsize=7,
        xguide_position=:left,
        #xticks=(1:length(h.order), map(v->fns[v], h.order)),
        yguide="Brain areas",
    )
    p = plot(p1, p3, layout=grid(2, 1, heights=[0.2, 0.6]),
        link=:x,
        size=(600, 480),
    )
    savefig(p, texfile)
    p
end

Make.@target cluster_plot_mus_hy (cluster_mus_hy, projections_accum, pick_brain_area_hy) -> begin
    swcs, h = deepcopy(cluster_mus_hy)
    projs = deepcopy(projections_accum)
    sel_bas = deepcopy(pick_brain_area_hy)[2]
    clustering_plot_func(swcs, h, projs, sel_bas, "extra/clustering-mus-hy.tex", reorder=true)
    1
end

Make.@target pick_mus_neurons_hy (xform_swcs, neurons_summary) -> begin
    swcs = deepcopy(xform_swcs)
    filter!(r -> check_neuron_qc(r.fn) != 0, swcs)
    nrns = deepcopy(neurons_summary)
    outdir = "/data/tmp/swcs-random-pick-hy"

    swcs = leftjoin(DataFrame(swcs), nrns, on=:fn)
    insertcols!(swcs, :cls => -1)

    for row in eachrow(swcs)
        row.cls = (row.qc ? 1 : 0)
        @assert row.stg % 10 == 9
        @assert row.ord < 999000
        @assert match(r"@seg\d*", row.fn) === nothing
        @assert match(r"@\.swc$", row.fn) === nothing
        if match(r"^/root/grey/BS/IB/HY(|/.*)$", row.soma_ba) === nothing
            row.cls = -1
        end
        if row.mm_axon < 0.3
            row.cls = -1
        end
        if !row.qc
            row.cls = -1
        end
    end
    swcs = filter(v -> v.cls >= 0, swcs)

    remap_groups_alt(swcs, outdir)
end
########################
###################################################
###################################################
###################################################
##############################################################
####################################################################

Make.@target regression_data (neurons_summary, projections_accum, cluster_mus_hy) -> begin
    swcs = deepcopy(neurons_summary)
    projs = deepcopy(projections_accum)
    cls = deepcopy(cluster_mpo_mpn)[1][:fn, :cls]
    leftjoin!(swcs, cls, on=:fn)
    leftjoin!(swcs, projs, on=:fn)
    insertcols!(swcs, :soma_x => map(v -> v[1], swcs.soma_pos))
    insertcols!(swcs, :soma_y => map(v -> v[2], swcs.soma_pos))
    insertcols!(swcs, :soma_z => map(v -> v[3], swcs.soma_pos))
    insertcols!(swcs, :mm_dend => (swcs.mm_tot .- swcs.mm_axon))

    for col in [:cls, :ds, :dst, :soma_ba]
        swcs[:, col] = categorical(swcs[:, col])
    end
    sel_cols = [:fn, :ds, :dst,
        :cls,
        :mm_axon,
        :mm_dend,
        :mm_tot,
        :qc,
        :soma_ba,
        :soma_x,
        :soma_y,
        :soma_z,
        #*pl_
        #*pr_
        #*pt_
    ]

    ann_ids = reverse(allen_ids_rev())
    for id in ann_ids
        p = map(eachrow(swcs)) do row
            row.mm_proj[id, :both]
        end
        quantile(map(v -> sum(v), p), 0.99) >= 0.01 || continue
        sym = _get_allen_annot()[id][3]
        new_cols = [Symbol("pl_" * sym) => map(v -> v[1], p),
            Symbol("pr_" * sym) => map(v -> v[2], p),
            Symbol("pt_" * sym) => map(v -> sum(v), p),
        ]
        insertcols!(swcs, new_cols...)
        for p in new_cols
            push!(sel_cols, p.first)
        end
    end
    swcs[:, sel_cols]
end

Make.@target regression_test (regression_data) -> begin
    swcs = deepcopy(regression_data)
    swcs = swcs[swcs.qc, :]
    npregress(hcat(swcs.mm_dend, swcs.cls), swcs.mm_axon), swcs.mm_axon
end

# XXX diff length/with mirror
function length_dist_hy()
    sel_clu = Set()
    for fn in deepcopy(cluster_mus_hy)[1].fn
        fn = basename(fn)
        m = match(r"^(\d+)-(.*)$", fn)
        m !== nothing || error("nomat")
        push!(sel_clu, fn)
    end
    x = filter(x) do r
        r.fn in sel_clu
    end
    p3 = histogram(log10.(x[:, :mm_axon_raw] .* 1000),
        extra_kwargs=Dict(:subplot => Dict(
            "scale only axis" => true,
        )),
        bins=range(1, 6, length=30),
        legend=:none,
        ylim=(0, 150),
        tickfontsize=5,
        yguide="Count",
        #xguide=tex("Neurite length (\\si{mm})"),
        guidefontsize=6,
        xticks=:none,
        grid=false,
        linewidth=0.5,
    )
    annotate!(p3, [(1.0, 130, ("Hypothalamic projection neurons (N=$(size(x)[1]))", 6, :left))])

    yticks = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    yticks = (yticks, map(v -> (2.00^float(v)), yticks))
    p4 = scatter(log10.(x[:, :mm_axon_raw] .* 1000),
        log2.(x[:, :len] ./ 1000 .- x[:, :mm_axon_raw]),
        extra_kwargs=Dict(:subplot => Dict(
            "scale only axis" => true,
        )),
        #bins=range(1, 6, length = 30),
        legend=:none,
        #ylim=(0, 150),
        tickfontsize=5,
        yguide=tex("Dendrite length (\\si{mm})"),
        xguide=tex("Axon length (\\si{mm})"),
        guidefontsize=6,
        xticks=xticks,
        yticks=yticks,
        grid=false,
        linewidth=0.5,
        markersize=1,
        markerstrokealpha=0,
    )
    l = lm(hcat(log10.(x[:, :mm_axon_raw] .* 1000), ones(size(x[:, :mm_axon_raw])...)), log2.(x[:, :len] ./ 1000 .- x[:, :mm_axon_raw]))
    plot!(p4, 2:0.3:6.3, map(x -> x * l.pp.beta0[1] + l.pp.beta0[2], 2:0.3:6.3), color=:black, z_order=:back)
    annotate!(p4, 1.2, -0.3, ("p<1e-6", 6, :left))
    println(l)
    px = plot(p3, p4, layout=grid(2, 1, heights=[0.25, 0.65]),
        link=:x,
        size=(190, 180),
    )
    savefig(px, "extra/length-dist-hy.tex")
    PGFPlotsX.push_preamble!(px.o.the_plot, "\\usepackage{siunitx}")
    1
end

Make.@target soma_dist_plot (neurons_summary, cluster_mus_hy) -> begin
    swcs = deepcopy(cluster_mus_hy)[1]
    nrns = deepcopy(neurons_summary)
    leftjoin!(swcs, nrns, on=:fn)
    bas = [
        "/root/grey/BS/IB/HY/LZ"
        "/root/grey/BS/IB/HY/ME"
        "/root/grey/BS/IB/HY/MEZ"
        "/root/grey/BS/IB/HY/PVR"
        "/root/grey/BS/IB/HY/PVZ"
    ]
    all_cls = sort(unique(swcs[:, :cls]))
    cnts = map(bas) do ba
        function inba(x)
            ba == x || (startswith(ba)(x) && x[length(ba)+1] == '/')
        end
        map(all_cls) do c
            count(swcs[:, :cls] .== c .&& inba.(swcs[:, :soma_ba])), c
        end
    end
    cnts = mapreduce(x -> reshape(x, 1, :), vcat, cnts)
    unique(map(x -> replace(x, r"/HY/(\w+)/.*" => s"/HY/\1"), swcs[:, :soma_ba]))
    #return cnts
    p = groupedbar(map(x -> replace(x, r".*/" => ""), bas), map(v -> v[1], cnts), bar_position=:stack,
        label=reshape(map(c -> tex("#$c"), all_cls), 1, :),
        #group=1:6,
        size=(240, 180),
        tickfontsize=5,
        guidefontsize=6,
        yguide=tex("Count"),
        xguide=tex("Soma Region"),
        legend_title="Clusters",
        #legend=:inline,
        #legend=:inside,
        #legend=:legend,
        legend_title_font_pointsize=6,
        legend_font_pointsize=5,
        legend_foreground_color=:transparent,
        linewidth=0.5,
    )
    savefig(p, "extra/soma-ba-dist-hy.tex")
    1
end


##############################dont look up


#regression

#=
	res=pmap(names(df)) do col
		match(r"^p_", col)!==nothing || return nothing
##################################
#p_PAG~1+strain+cls+sx+sy+sz
	#####
	#
	#######
		res=glm(@formula(proj~1+strain+ba+sx+sy+sz), df, Normal())
		(bas, res, deviance(res)/sum(yv.^2))
	end
	filter(v->v!==nothing, res)
end
=#


#=
		 =#
