draw_single_neuron_intrinsic(ax1s, ax2s, rawinfo, dfx, arbors, spe) = begin
    highlights = Dict{String,Dict{Int,Int}}()
    add_hl(node_pos, path, pts) = begin
        hh = get!(() -> Dict{Int,Int}(), highlights, path)
        swc = Gapr.load_swc(path)
        p2i = [node_pos(n) => n.ind for n in swc.nodes]
        pts = map(pt -> pt isa Number ? swc[!, pt].pos : pt, pts)[1:1]
        nns = last.([minimum([(norm(p .- pt), i) for (p, i) in p2i]) for pt in pts]) |> Set
        for n in swc.nodes
            n.ind in nns && push!(hh, n.id => n.ind)
            if n.parind != 0
                pp = get(hh, swc[n.parind].id, -1)
                pp != -1 && push!(hh, n.id => pp)
            end
        end
        for n in Iterators.reverse(swc.nodes)
            if get(hh, n.id, -1) != -1 && n.parind != 0
                pn = swc[n.parind]
                ppp = get(hh, pn.id, -1)
                ppp == -1 && push!(hh, pn.id => 0)
            end
        end
    end
    for r in eachrow(dfx)
        rawpos = spe === :mac ? (info = rawinfo(r.sample, r.channel); n -> info[n.id].rawxyz) : n -> n.pos
        add_hl(rawpos, r.path, r.intrinsic_pts)
    end
    per_neuron(r) = begin
        highlights[r.path]
    end
    per_arbor(pnrn, rr, cond) = begin
        nothing
    end
    per_node(pnrn, parb, n) = begin
        #n.typ!=2 && return :transparent => pt(0.0)
        i = get(pnrn, n.id, -1)
        i > 0 && n.typename === :axon && return :red => pt(0.9)
        n.typename === :axon && return :black => pt(0.5)
        :transparent => pt(0.5)
    end
    per_node2(pnrn, parb, n) = begin
        #n.typ!=2 && return :transparent => pt(0.0)
        i = get(pnrn, n.id, -1)
        i > 0 && n.typename === :axon && return :red => pt(0.9)
        i == 0 && n.typename === :axon && return :gray => pt(0.9)
        :transparent => pt(0.5)
    end
    nrns = Set([r.path for r in eachrow(dfx)])
    draw_subtrees(filter(r -> r.path in nrns, arbors), ax1s...; inspect=spe === :mac ? :nmt : :allen, autotrunk=false, per_neuron, per_arbor, per_node)
    draw_subtrees(filter(r -> r.path in nrns, arbors), ax2s...; inspect=spe === :mac ? :nmt : :allen, autotrunk=false, per_neuron, per_arbor, per_node=per_node2)
end

function plot_intrinsic_egs(grid, env; dfxintrinsic, dfarbor, rawinfo, obj, _...)
    let dfx = dfxintrinsic[[1,], :]
        ax3 = LScene(grid[1, 1], width=pt(80), height=pt(40), show_axis=false)
        ax2 = LScene(grid[2, 1], width=pt(80), height=pt(60), show_axis=false)
        rowgap!(grid, 1, 0)
        setup_camera(ax2 => :hr)
        setup_camera(ax3 => (fov=6.0, cent=(24215.43, 64331.48, 37861.17), deye=(0.4541725997717278, 0.0334976278671345, 0.8902837517015937) .* 100453.49284658181, up=(-0.78, 0.5, 0.38)))
        with_mixed_scene(ax2, GLMakie) do mix2
            with_mixed_scene(ax3, GLMakie) do mix3
                draw_single_neuron_intrinsic((mix2,), (mix3,), rawinfo, dfx, dfarbor, :mac)
            end
        end
        draw_mesh_overlay([obj => :ivory], ax2)
        draw_mesh_overlay([load("interm/mesh-nmt-$ba-l.obj") => :red for ba in [:area_9d,]], ax3)
        draw_contours([load("interm/mesh-nmt-$ba-l.obj") => :red for ba in [:area_9d,]], ax2)
    end
end

function plot_intrinsic_fibers(grid, env, tag; dfxintrinsic, dfxintrinsic_mus, dfarbor, musarbors, rawinfo, obj, _...)
    load_obj(spec) = begin
        if spec isa Pair
            o = load_obj(spec.first)
            if spec.second === :C && tag == :mac
                o = Main.mirror_mesh(NmtDat.mirror)(o)
            end
            return o
        end
        n2i = (; ACAd=39, CP=672, RSP=254, PL=972, HY=1097, OLF=698, AI=95, STR=477, P=771)
        tag === :mus && return load("interm/allen-$(n2i[spec]).obj")
        load("interm/mesh-nmt-$spec-l.obj")
    end
    nrn_pars = (;
        mac=[
            (; srcs=[:area_9d], tgts=[:area_8Bd => :C, :area_46d => :C], cam=(fov=6.0, cent=(25308.94, 66314.13, 38403.39), deye=(-0.7061358509561285, 0.29233066496403454, -0.6449146783227606) .* 143889.76266029108, up=(-0.32, 0.68, 0.66)));
            (; srcs=[:area_8Bd], tgts=[:area_8Ad, :SMA], cam=(fov=6.0, cent=(28186.8, 58480.58, 42489.41), deye=(0.25964061725440823, -0.9182538349079825, 0.29899271654763326) .* 100651.0471140713, up=(-0.22, 0.25, 0.94))); #= :area_12l,  =#
            (; srcs=[:area_9d], tgts=[:area_23, :V2], cam=(fov=6.0, cent=(15491.23, 53713.23, 44652.52), deye=(0.7604059172375915, 0.5583502705669728, -0.3317044111673001) .* 145524.56456677546, up=(-0.03, 0.54, 0.84)));
            (; srcs=[:area_46d => :C], tgts=[:area_9d => :I, :area_8Bd => :C, :area_8Bm => :I], cam=(fov=7.21, cent=(40609.17, 67157.11, 37082.97), deye=(-0.18789692692593885, -0.8182493341241359, 0.5432888477202495) .* 61942.627054496385, up=(0.55, 0.37, 0.75)));
            (; srcs=[:area_45b], tgts=[:PMd, :area_8Bm, :area_45b => :C], cam=(fov=6.0, cent=(15641.05, 56178.83, 31108.91), deye=(-0.49348600249821967, 0.6440364356489101, -0.5845413885217834) .* 45873.39840521955, up=(-0.36, -0.76, -0.53)));
            (; srcs=[:area_45b], tgts=[:PMd, :area_8Bs], cam=(fov=6.0, cent=(14214.41, 53194.47, 34297.01), deye=(0.5485382224048196, 0.46910153540933813, -0.6921340679648431) .* 33741.71433096724, up=(-0.26, -0.69, -0.67)))
        ],
        mus=[
            (; srcs=[:PL], tgts=[:HY, :P], cam=(fov=6.0, cent=(2980.63, 2690.71, 5672.04), deye=(-0.8963925777515793, -0.44325249269598016, 0.0027521392913189948) .* 21383.365364401834, up=(0.44, -0.9, -0.0)));
            (; srcs=[:PL => :C], tgts=[:STR, :OLF], cam=(fov=6.0, cent=(2980.63, 2690.71, 5672.04), deye=(-0.8963925777515793, -0.44325249269598016, 0.0027521392913189948) .* 21383.365364401834, up=(0.44, -0.9, -0.0))); #= :area_12l,  =#
            (; srcs=[:PL], tgts=[:STR, :AI], cam=(fov=6.0, cent=(2980.63, 2690.71, 5672.04), deye=(-0.8963925777515793, -0.44325249269598016, 0.0027521392913189948) .* 21383.365364401834, up=(0.44, -0.9, -0.0))) #= :area_12l,  =#
        ]
    )[tag]
    if tag === :mus
        obj = load("interm/allen-997.obj")
    end
    let dfx = (; mac=dfxintrinsic, mus=dfxintrinsic_mus)[tag]
        for r in eachrow(dfx)
            pars = nrn_pars[rownumber(r)]
            cam = get(pars, :cam, :hr)
            src = areaname((s = only(pars.srcs); s isa Pair ? s.first : s))
            title = "$src"
            if (inj = get(r, :inj, nothing)) != nothing
                title = "$inj $title"
            end
            row, col = divrem(rownumber(r) - 1, 3) .+ 1
            gridx = GridLayout(grid[row, col])
            row > 1 && rowgap!(grid, row - 1, 8)
            col > 1 && colgap!(grid, col - 1, 6)
            ax1 = LScene(gridx[1, 1], width=pt(64), height=pt(44), show_axis=false)
            ax2 = LScene(gridx[2, 1], width=pt(64), height=pt(52), show_axis=false)
            ax3 = LScene(gridx[:, 2], width=pt(96), height=pt(96), show_axis=false)
            if title !== nothing
                Label(gridx[1, 2, Top()], title, tellheight=false, valign=:top; set_font_h1()...)
            end
            rowgap!(gridx, 1, 0)
            colgap!(gridx, 1, 0)
            let cams = (; mac=(:s, :hr, cam), mus=(:allens, :allenhr, cam))[tag]
                setup_camera(ax1 => cams[1], ax2 => cams[2])
                setup_camera(ax3 => cams[3])
            end
            with_mixed_scene(ax1, GLMakie) do mix1
                with_mixed_scene(ax2, GLMakie) do mix2
                    with_mixed_scene(ax3, GLMakie) do mix3
                        draw_single_neuron_intrinsic((mix1, mix2), (mix3,), rawinfo, filter(rr -> rr.path == r.path, dfx), (; mac=dfarbor, mus=musarbors)[tag], tag)
                    end
                end
            end
            #rownumber(r) != 5 && continue
            draw_mesh_overlay([obj => :ivory], ax1, ax2)
            draw_mesh_overlay([load_obj(ba) => :red for ba in get(pars, :srcs, ())], ax1, ax2, ax3)
            draw_contours([load_obj(ba) => :gray for ba in get(pars, :tgts, ())], ax1, ax2)
        end
    end
end

function plot_fig_intra_pfc(fig, env; _...)
    with_caption(L"Subdomain network of macaque intra-PFC axon connections.")

    grid = fig[1, 1]
    # with_subfig(grid[1, 1], env) do grid, env
    #     with_caption(L"Eight example subdomains in frontal cortex.")
    #     plot_subreg_arbors(grid, env; env...)
    # end
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"All subdomains in frontal cortex.")
        plot_subreg_arbors_all(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"All subdomains on flatmap. Red outline emphasizes subdomains that contain at least 10 neurons.")
        plot_subreg_flat(grid, env; env...)
    end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"Connectivity network between subdomains. Red, blue and black lines indicate connection from dlPFC, vlPFC and other PFC areas, respectively. Color circles show two modules of lPFC subdomains by network analysis.")
        plot_subreg_conn_hm(grid, env; env...)
    end
    with_subfig(grid[1, 4], env) do grid, env
        with_caption(L"Connectivity network between subdomains. Red, blue and black lines indicate connection from dlPFC, vlPFC and other PFC areas, respectively. Color circles show two modules of lPFC subdomains by network analysis.")
        plot_subreg_flat_conn(grid, env; env...)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"Reciprocal connection between subdomains in areas 45b and 12l.")
        plot_intra_pfc_reciprocal(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"Segregated dl/vl connection.")
        plot_intra_pfc_dlvl(grid, env; env...)
    end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"Segregated dl/vl connection.")
        plot_intra_pfc_arbor_depth(grid, env; env...)
    end

    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"to acgg/ofc.")
        plot_intra_pfc_acgg_ofc_bp(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"to acgg/ofc.")
        plot_intra_pfc_ofc_acgg_somadepth(grid, env; env...)
    end

    # with_subfig(grid[1, 3], env) do grid, env
    #     with_caption(L"to acgg/ofc.")
    #     #plot_intra_pfc_ofc_acgg(grid, env; env...)
    # end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"to acgg/ofc.")
        plot_intra_pfc_arbor_cond(grid, env; env...)
    end
    with_subfig(grid[1, 4], env) do grid, env
        with_caption(L"to acgg/ofc.")
        plot_intra_pfc_arbor_test(grid, env; env...)
    end
    with_subfig(grid[1, 5], env) do grid, env
        with_caption(L"to acgg/ofc.")
        plot_intra_pfc_dl_acgg_topo(grid, env; env...)
    end

    grid = fig[4, 1]
    with_subfig(grid[1, 1], env) do grid, env
        plot_intra_pfc_tgt_numberx(grid, env; env...)
        # ...: IT, not ITi
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"arbors of dl/vlpfc to acgg/ofc")
        plot_intra_pfc_subdomain_hm(grid, env; env...)
    end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"arbors of dl/vlpfc to acgg/ofc")
        plot_intra_pfc_combo(grid, env; env...)
        #plot_intra_pfc_arbor_conn(grid, env; env...)
        # random color neurons/non-relevent branches gray
    end
    with_subfig(grid[1, 4], env) do grid, env
        with_caption(L"")
        plot_intrinsic_egs(grid, env; env...)
    end
end

function plot_fig_intra_pfc_supp(fig, env; _...)
    with_caption(L"Subdomain network analysis of intra-PFC axon connections.")

    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"PT neurons.")
        plot_intra_network(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"PT neurons.")
        LTeX(grid[1, 1], PDFDocument(read("monkeytemp/wyz-upload/fig4-supp-arbor-morph-examples.pdf", String)), tellheight=false, halign=:right, valign=:top, scale=0.9667)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"PT neurons.")
        plot_acgg_ofc_soma_venn(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"PT neurons.")
        plot_acgg_ofc_soma_flat(grid, env; env...)
    end
    with_subfig(grid[1, 3], env) do grid, env
        with_caption(L"arbors of dl/vlpfc to acgg/ofc")
        plot_intra_pfc_ofc_acgg_arbors(grid, env; env...)
    end
    GridLayout(grid[1, 4]).width[] = pt(120)

    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"PT neurons.")
        LTeX(grid[1, 1], PDFDocument(read("monkeytemp/wyz-upload/figs6-supp-arbor-morph-examples.pdf", String)), scale=0.9667)
    end
end

function plot_intrinsic_flatmap(grid, env; df, highlights, objflat, objphy, dfarbor, trunks, obj, _...)
    dfx = filter(select_intrinsic, trunks)
    a2c = Dict(zip(dfx.arborpath, dfx.trunkcls))
    dfx = filter(r -> r.arborpath in dfx.arborpath, dfarbor) |> dropmissing!
    c2c = Dict()
    arbcolor(r) = begin
        get!(c2c, a2c[r.arborpath]) do
            rand(to_colormap(:Set1_8))
        end
    end

    ax = Axis(grid[1, 1])
    plot_flatmap_base!(ax, highlights, objflat, objphy)
    limits!(ax, -0.55, 0.45, -0.6, 0.35)

    Makie.linesegments!(ax, collect(zip(dfx.somauv, dfx.arboruvcent2)), color=arbcolor.(eachrow(dfx)), colormap=[(:blue, 0.3), (:red, 0.3)])
    normit(v) = v ./ norm(v) / 1000
    Makie.arrows!(ax, Point2f.(dfx.arboruvcent2), (normit ∘ Point2f).(dfx.arboruvcent2 .- dfx.somauv), color=arbcolor.(eachrow(dfx)), colormap=[(:blue, 0.9), (:red, 0.9)], arrowsize=3)
end

function plot_fig_intrinsic(fig, env; _...)
    with_caption(L"")

    grid = fig[1, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"")
        plot_intrinsic_flatmap(grid, env; env...)
    end
    with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"")
        plot_bundle_demo_intrinsic2(grid, env; env...)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"")
        plot_intrinsic_fibers(grid, env, :mac; env...)
    end
    grid = fig[3, 1]
    with_subfig(grid[1, 1], env) do grid, env
        with_caption(L"")
        plot_intrinsic_fibers(grid, env, :mus; env...)
    end
end

