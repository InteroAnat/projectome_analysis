# gou_flatmap_minimal.jl
# Minimal adaptation of Gou et al. 2025 flatmap pipeline for local data.
# Only changes from original:
#   1. File paths point to local atlas / surfaces
#   2. GIFTI loading with fix_mesh_object (instead of .obj)
#   3. Geometric depth model approximation (replaces Flux GPU training)

using Meshes        # needed by refine_mesh
using CairoMakie    # for Figure, Axis, save

const MONKEYREC_ROOT = raw"D:\projectome_analysis\references\analysis-code_gou_etal_2025\monkeyrec"
const LOCAL_NMT_ROOT = raw"D:\projectome_analysis\atlas\NMT_v2.0_sym"
const SURF_ROOT      = joinpath(LOCAL_NMT_ROOT, "NMT_v2.0_sym_surfaces")
const SOMA_TABLE     = raw"D:\projectome_analysis\neuron_tables\251637_INS_HE_inferred.xlsx"
const OUT_ROOT       = raw"D:\projectome_analysis\figures_charts\gou_flatmap_conservative"
const CACHE_DIR      = joinpath(OUT_ROOT, "cache")

for d in [OUT_ROOT, CACHE_DIR]; mkpath(d); end

cd(MONKEYREC_ROOT)
include(joinpath(MONKEYREC_ROOT, "monkeyrec.jl"))

if !isdefined(zz1data, :xxxcached_result)
    @eval zz1data xxxcached_result = Dict{Tuple,Any}()
end

# ── Patch atlas paths to local ──────────────────────────────────────
@eval NmtDat begin
    function load_images()
        root = raw"D:\projectome_analysis\atlas\NMT_v2.0_sym"
        d    = joinpath(root, "NMT_v2.0_sym")
        nmr  = NmtDat.NIfTI.niread(joinpath(d, "NMT_v2.0_sym_SS.nii.gz")).raw
        cb   = NmtDat.NIfTI.niread(joinpath(d, "supplemental_masks",
                   "NMT_v2.0_sym_cerebellum_mask.nii.gz")).raw .> 0
        charm = reshape(NmtDat.NIfTI.niread(
                    joinpath(d, "CHARM_in_NMT_v2.0_sym.nii.gz")).raw, size(nmr)..., :)
        sarm  = reshape(NmtDat.NIfTI.niread(
                    joinpath(d, "SARM_in_NMT_v2.0_sym.nii.gz")).raw, size(nmr)..., :)
        seg  = NmtDat.NIfTI.niread(joinpath(d, "NMT_v2.0_sym_segmentation.nii.gz")).raw
        sarm_key  = DelimitedFiles.readdlm(
                        joinpath(root, "tables_SARM", "SARM_key_all.txt"), '\t', header=true)[1]
        charm_key = DelimitedFiles.readdlm(
                        joinpath(root, "tables_CHARM", "CHARM_key_all.txt"), '\t', header=true)[1]
        res = SVector(250.0f0, 250.0f0, 250.0f0)
        tag2tbl = map((:sarm => sarm_key, :charm => charm_key)) do (tag, tbl)
            tbl = map(eachrow(tbl)) do r; r[1] => r[2]; end |> Dict
            tag => tbl
        end |> NamedTuple
        (nmr=nmr, cb_msk=cb, charm=charm, charm_key=charm_key,
         sarm=sarm, sarm_key=sarm_key, seg=seg,
         res=res, range=SVector(res .* size(nmr)...), tag2tbl)
    end
end

# ── Surface loading (GIFTI → fix_mesh_object) ──────────────────────
function load_surface(kind::Symbol)
    fname = Dict(:gray  => "lh.gray_surface.rsl.gii",
                 :mid   => "lh.mid_surface.rsl.gii",
                 :white => "lh.white_surface.rsl.gii")[kind]
    fix_mesh_object(GIFTI.load(joinpath(SURF_ROOT, fname)))
end

# ── Step 1: Geometric depth model (replaces Flux GPU training) ─────
# For each mid-surface vertex, compute column direction and extents
# from the three co-registered cortical surfaces.
function build_depth_model()
    obj_gray  = load_surface(:gray)
    obj_mid   = load_surface(:mid)
    obj_white = load_surface(:white)
    n = length(obj_mid.position)
    @assert n == length(obj_gray.position) == length(obj_white.position)
    df = map(1:n) do i
        mid   = SVector(obj_mid.position[i]...)
        gray  = SVector(obj_gray.position[i]...)
        white = SVector(obj_white.position[i]...)
        dir_raw = gray - white
        ldir = norm(dir_raw)
        dir = ldir > eps(Float32) ? dir_raw / ldir : SVector(0f0, 0f0, 1f0)
        te = dot(gray  - mid, dir)   # positive (outer toward pial)
        ti = dot(white - mid, dir)   # negative (inner toward white matter)
        (; dir, ti=Float32(ti), te=Float32(te), pos=mid)
    end |> DataFrame
    df, obj_mid
end

# ── Step 2: Depth-volume rasterization ─────────────────────────────
# Gou's exact algorithm from 1-unibrain.jl lines 154-206.
# Traces rays along each cortical column and fills a 3D volume with
# depth (0→1 from outer to inner) and nearest column center position.
function build_depth_volume(model, obj)
    @assert length(obj.position) == nrow(model)
    nmt = NmtDat.load_images()
    charm = nmt.charm[1:end÷2, :, :, 1]
    res   = nmt.res
    ratio = 2
    img = Images.imresize(charm, ratio=ratio, method=Interpolations.Constant())
    res = res ./ ratio
    buf  = map(_ -> (true, Inf32, NaN32, NaN32, NaN32, NaN32), img)
    hits = Tuple[]
    sample_mesh(obj, obj.position, model.dir, model.ti, model.te; perface=48) do vec
        @assert length(vec) == 8
        cent = SVector(vec[1:3]...)
        dir  = SVector(vec[4:6]...)
        ti, te = vec[7:8]
        empty!(hits)
        @assert te >= 0 "$te >=0"
        @assert ti <= 0 "$ti <=0"
        te2, ti2 = map((0:60:te*1.5f0, 0:-60:ti*1.5f0)) do rng
            outside = false
            tt = last(rng)
            for t in rng
                p  = cent + dir * t
                i  = NmtDat.pos2idx(p, res)
                ii = NmtDat.pos2idx(p, nmt.res)
                if !(checkbounds(Bool, charm, ii...) && charm[ii...] != 0) && !outside
                    tt = t
                    outside = true
                end
                pp = NmtDat.idx2pos(i, res)
                dd = norm(pp - cent)
                push!(hits, (outside, i, dd, t))
            end
            tt
        end
        if te2 > ti2
            for (outside, i, dd, t) in hits
                if checkbounds(Bool, buf, i...)
                    buf[i...] = min(buf[i...],
                        (outside, dd, clamp((t - te2) / (ti2 - te2), 0, 1), cent...))
                end
            end
        end
    end
    depthimg = stack((map(v -> v[k], buf) for k in 3:6), dims=4)
    depthimg = cat(depthimg, depthimg[end:-1:1, :, :, :], dims=1)
    (depthimg, res)
end

# ── Step 3: Flatmap (Gou's exact logic, 2-meshes.jl:285-328) ──────
function build_flatmap(; tag::Symbol=:leftfrontal, niter::Int=30000)
    obj = refine_mesh(load_surface(:mid))
    cfg = (;
        lefths=(;
            clipfn=(obj1) -> begin
                nmt = NmtDat.load_images()
                charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res),
                    CartesianIndices(nmt.seg)[nmt.charm[:,:,:,1] .> 0])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj, findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 10000
            end,
            rightfn=p -> p[2],
        ),
        leftfrontal=(;
            clipfn=(obj1) -> begin
                nmt = NmtDat.load_images()
                charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res),
                    CartesianIndices(nmt.seg)[nmt.charm[:,:,:,1] .== 1])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj, findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 60000
            end,
            rightfn=p -> p[1],
        ),
        leftinsula=(;
            clipfn=(obj1) -> begin
                nmt = NmtDat.load_images()
                ins_mask = (nmt.charm[:,:,:,3] .== 224) .|   # floor_of_ls (Ins/Pi/Ri)
                           (nmt.charm[:,:,:,3] .== 37)       # caudal_OFC  (OFa-p/G/PrCO)
                charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res),
                    CartesianIndices(nmt.seg)[ins_mask])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj, findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 25000
            end,
            rightfn=p -> p[2],
        ),
    )[tag]
    obj2 = clip_mesh(cfg.clipfn(obj), obj)
    objphy = GeometryBasics.normal_mesh(obj2.position, GeometryBasics.faces(obj2))
    vmax = findmin(cfg.rightfn, objphy.position)[2]
    objflat = flatten_mesh(objphy, vmax; niter=niter)
    (objflat=objflat, objphy=objphy)
end

# ── Soma loading ───────────────────────────────────────────────────
function load_somata(; path=SOMA_TABLE)
    tbl = XLSX.readtable(path, 1)
    df  = DataFrame(tbl.data, vec(Symbol.(tbl.column_labels)))
    rename!(df, Symbol.(names(df)))
    soma_pos = map(eachrow(df)) do r
        xyz = SVector{3,Float32}(Float32(r.Soma_Phys_X),
                                 Float32(r.Soma_Phys_Y),
                                 Float32(r.Soma_Phys_Z))
        side = strip(uppercase(String(r.Soma_Side)))
        if side == "R" || !NmtDat.isleft(xyz)
            SVector{3,Float32}(Tuple(NmtDat.mirror(xyz)))
        else
            xyz
        end
    end
    original_side = map(r -> strip(uppercase(String(r.Soma_Side))), eachrow(df))
    insertcols!(df, :soma_pos      => soma_pos)
    insertcols!(df, :original_side => original_side)
    insertcols!(df, :typ   => string.(df.Neuron_Type))
    insertcols!(df, :qroot => fill(false, nrow(df)))
    df
end

# ── Pipeline ───────────────────────────────────────────────────────
function run_pipeline(; tag::Symbol=:leftfrontal, niter::Int=30000)
    depth_cache = joinpath(CACHE_DIR, "depth_volume.jld2")
    flat_cache  = joinpath(CACHE_DIR, "flatmap_$(tag)_n$(niter).jld2")

    # 1+2: Depth volume
    depthimg, depthres = if isfile(depth_cache)
        @info "Loading cached depth volume"
        JLD2.load(depth_cache, "depthimg", "depthres")
    else
        @info "Building geometric depth model..."
        model, obj_mid = build_depth_model()
        @info "Rasterizing depth volume (may take several minutes)..."
        di, dr = build_depth_volume(model, obj_mid)
        JLD2.jldsave(depth_cache; depthimg=di, depthres=dr)
        di, dr
    end

    # 3: Flatmap
    objflat, objphy = if isfile(flat_cache)
        @info "Loading cached flatmap"
        of, op = JLD2.load(flat_cache, "objflat", "objphy")
        op = hasproperty(op, :normals) ? op :
             GeometryBasics.normal_mesh(op.position, GeometryBasics.faces(op))
        of, op
    else
        @info "Building flatmap ($tag, niter=$niter)..."
        fm = build_flatmap(; tag, niter)
        JLD2.jldsave(flat_cache; objflat=fm.objflat, objphy=fm.objphy)
        fm.objflat, fm.objphy
    end

    # 4: Map somata via Gou's exact xyz2uvw
    @info "Loading somata and projecting via xyz2uvw..."
    df = load_somata()
    uvw = xyz2uvw(df.soma_pos, depthimg, depthres, objflat, objphy)
    insertcols!(df,
        :somauv => map(v -> SVector(v[1:end-1]...,), uvw),
        :somaw  => last.(uvw))

    # 5: Plot
    highlights = zz1data.get_flatmap_highlights()
    typcolor   = zz2fig.typ2color_func(df)
    rowcolor   = r -> typcolor(r.typ)

    # Base flatmap
    fig1 = Figure(size=(650, 420), backgroundcolor=:white)
    ax1  = Axis(fig1[1, 1])
    zz2fig.plot_flatmap_base!(ax1, highlights, objflat, objphy; label=true)
    save(joinpath(OUT_ROOT, "flatmap_base_$(tag).png"), fig1;
         px_per_unit=2, backend=CairoMakie)
    save(joinpath(OUT_ROOT, "flatmap_base_$(tag).svg"), fig1;
         backend=CairoMakie)
    @info "Saved base flatmap"

    # Soma overlay
    fig2 = Figure(size=(650, 420), backgroundcolor=:white)
    ax2  = Axis(fig2[1, 1])
    zz2fig.plot_flatmap_base!(ax2, highlights, objflat, objphy; label=false)
    zz2fig.plot_flatmap_soma!(ax2, df; rowcolor)
    save(joinpath(OUT_ROOT, "flatmap_soma_$(tag).png"), fig2;
         px_per_unit=2, backend=CairoMakie)
    save(joinpath(OUT_ROOT, "flatmap_soma_$(tag).svg"), fig2;
         backend=CairoMakie)
    @info "Saved soma flatmap"

    @info "Done. $(nrow(df)) somata projected."
    df
end

# ── Insula-specific highlights (CHARM hierarchy paths) ────────────
function get_insula_highlights()
    [
        "nmt|charm|Temporal|floor_of_ls" => :gray80
        "nmt|charm|Frontal|OFC|caudal_OFC" => :gray80
        "nmt|charm|Temporal|floor_of_ls|Ins/Pi" => 0.75
        "nmt|charm|Temporal|floor_of_ls|Ri" => 0.75
        "nmt|charm|Frontal|OFC|caudal_OFC|OFa-p" => 0.75
        "nmt|charm|Frontal|OFC|caudal_OFC|cl_OFC" => 0.75
        "nmt|charm|Temporal|floor_of_ls|Ins/Pi|Ins" => 0.5
        "nmt|charm|Temporal|floor_of_ls|Ins/Pi|Ins|Ia/Id"
        "nmt|charm|Temporal|floor_of_ls|Ins/Pi|Ins|Ig"
        "nmt|charm|Temporal|floor_of_ls|Ins/Pi|Pi"
        "nmt|charm|Frontal|OFC|caudal_OFC|OFa-p|Iam/Iapm"
        "nmt|charm|Frontal|OFC|caudal_OFC|OFa-p|lat_Ia"
        "nmt|charm|Frontal|OFC|caudal_OFC|cl_OFC|G"
        "nmt|charm|Frontal|OFC|caudal_OFC|cl_OFC|PrCO"
    ]
end

# ── Compute bounding box from vertices annotated as insula ────────
function insula_axis_limits(objflat, objphy; pad=0.20)
    nmt = NmtDat.load_images()
    ins_verts = map(objphy.position) do p
        i = NmtDat.pos2idx(p, nmt.res)
        if checkbounds(Bool, nmt.charm, i..., 3)
            v = Int(nmt.charm[i..., 3])
            v == 224 || v == 37   # floor_of_ls or caudal_OFC
        else
            false
        end
    end
    uv = objflat.position[ins_verts]
    isempty(uv) && return (-1f0, 1f0, -1f0, 1f0)
    xs = [p[1] for p in uv]
    ys = [p[2] for p in uv]
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    dx = (xmax - xmin) * pad
    dy = (ymax - ymin) * pad
    (xmin - dx, xmax + dx, ymin - dy, ymax + dy)
end

# ── Compute subregion centroids and place labels on flatmap axis ──
function add_insula_labels!(ax, objflat, objphy)
    nmt = NmtDat.load_images()
    label_defs = [
        # (charm_level, charm_index, display_label, fontsize, font_style)
        (6, 228, "Ia/Id",  9, :italic),
        (6, 229, "Ig",     9, :italic),
        (5, 226, "Pi",     9, :italic),
        (4, 230, "Ri",    10, :bold),
        (5,  39, "Iam",    9, :italic),
        (5,  40, "lat Ia", 8, :italic),
        (5,  48, "G",      9, :italic),
        (5,  49, "PrCO",   9, :italic),
    ]
    for (lev, idx, label, fsz, fstyle) in label_defs
        vals = map(objphy.position) do p
            i = NmtDat.pos2idx(p, nmt.res)
            checkbounds(Bool, nmt.charm, i..., lev) ? Int(nmt.charm[i..., lev]) : 0
        end
        mask = vals .== idx
        count(mask) < 3 && continue
        uv = objflat.position[mask]
        cx = Float32(sum(p[1] for p in uv) / length(uv))
        cy = Float32(sum(p[2] for p in uv) / length(uv))
        Makie.text!(ax, cx, cy; text=label, align=(:center, :center),
                    fontsize=fsz, font=fstyle, color=:gray20)
    end
end

function run_insula_pipeline(; niter::Int=30000)
    tag = :leftinsula
    depth_cache = joinpath(CACHE_DIR, "depth_volume.jld2")
    flat_cache  = joinpath(CACHE_DIR, "flatmap_$(tag)_n$(niter).jld2")
    ins_dir     = joinpath(OUT_ROOT, "insula")
    mkpath(ins_dir)

    depthimg, depthres = if isfile(depth_cache)
        @info "Loading cached depth volume"
        JLD2.load(depth_cache, "depthimg", "depthres")
    else
        @info "Building depth volume..."
        model, obj_mid = build_depth_model()
        di, dr = build_depth_volume(model, obj_mid)
        JLD2.jldsave(depth_cache; depthimg=di, depthres=dr)
        di, dr
    end

    objflat, objphy = if isfile(flat_cache)
        @info "Loading cached insula flatmap"
        of, op = JLD2.load(flat_cache, "objflat", "objphy")
        op = hasproperty(op, :normals) ? op :
             GeometryBasics.normal_mesh(op.position, GeometryBasics.faces(op))
        of, op
    else
        @info "Building insula flatmap (niter=$niter)..."
        fm = build_flatmap(; tag, niter)
        JLD2.jldsave(flat_cache; objflat=fm.objflat, objphy=fm.objphy)
        fm.objflat, fm.objphy
    end

    @info "Projecting somata..."
    df = load_somata()
    uvw = xyz2uvw(df.soma_pos, depthimg, depthres, objflat, objphy)
    insertcols!(df,
        :somauv => map(v -> SVector(v[1:end-1]...,), uvw),
        :somaw  => last.(uvw))

    valid = filter(r -> !any(isnan, r.somauv), df)
    @info "Valid projections: $(nrow(valid)) / $(nrow(df))"

    highlights = get_insula_highlights()
    lims = insula_axis_limits(objflat, objphy; pad=0.20)

    side_color(s) = s == "L" ? Makie.to_color(:steelblue) : Makie.to_color(:firebrick)
    side_marker(s) = s == "L" ? :circle : :utriangle

    # ── Figure 1: base with subregion labels ────────────────────────
    fig_base = Figure(size=(600, 600), backgroundcolor=:white)
    ax_base  = Axis(fig_base[1, 1], title="Insula flatmap — subregions",
                    titlesize=14)
    zz2fig.plot_flatmap_base!(ax_base, highlights, objflat, objphy;
                              label=false, debug=false)
    add_insula_labels!(ax_base, objflat, objphy)
    limits!(ax_base, lims...)
    save(joinpath(ins_dir, "insula_base.png"), fig_base;
         px_per_unit=3, backend=CairoMakie)
    @info "Saved insula base"

    # ── Figure 2: combined L+R, different markers ───────────────────
    fig_lr = Figure(size=(600, 600), backgroundcolor=:white)
    ax_lr  = Axis(fig_lr[1, 1], title="Insula somata — L (●) vs R (▲)",
                  titlesize=14)
    zz2fig.plot_flatmap_base!(ax_lr, highlights, objflat, objphy;
                              label=false)
    add_insula_labels!(ax_lr, objflat, objphy)
    for side in ["L", "R"]
        sub = filter(r -> r.original_side == side && !any(isnan, r.somauv), valid)
        nrow(sub) == 0 && continue
        scatter!(ax_lr, sub.somauv,
                 color=side_color(side), marker=side_marker(side),
                 markersize=7, strokecolor=:black, strokewidth=0.4,
                 label="$(side) (n=$(nrow(sub)))")
    end
    limits!(ax_lr, lims...)
    axislegend(ax_lr, position=:lb, framevisible=true, labelsize=11)
    save(joinpath(ins_dir, "insula_soma_LR_combined.png"), fig_lr;
         px_per_unit=3, backend=CairoMakie)
    save(joinpath(ins_dir, "insula_soma_LR_combined.svg"), fig_lr;
         backend=CairoMakie)
    @info "Saved combined L/R soma plot"

    # ── Figure 3: side-by-side L vs R panels ────────────────────────
    fig_split = Figure(size=(1100, 550), backgroundcolor=:white)
    for (col, side, stitle) in [(1, "L", "Left insula"),
                                (2, "R", "Right insula (mirrored)")]
        ax = Axis(fig_split[1, col], title=stitle, titlesize=13)
        zz2fig.plot_flatmap_base!(ax, highlights, objflat, objphy;
                                  label=false)
        add_insula_labels!(ax, objflat, objphy)
        sub = filter(r -> r.original_side == side && !any(isnan, r.somauv), valid)
        nrow(sub) == 0 && continue
        scatter!(ax, sub.somauv,
                 color=side_color(side), marker=side_marker(side),
                 markersize=7, strokecolor=:black, strokewidth=0.4)
        limits!(ax, lims...)
        text!(ax, lims[2] - 0.02*(lims[2]-lims[1]), lims[3] + 0.02*(lims[4]-lims[3]),
              text="n=$(nrow(sub))", fontsize=11, align=(:right, :bottom))
    end
    Label(fig_split[0, :], "Insula soma laterality", fontsize=15)
    save(joinpath(ins_dir, "insula_soma_LR_split.png"), fig_split;
         px_per_unit=3, backend=CairoMakie)
    save(joinpath(ins_dir, "insula_soma_LR_split.svg"), fig_split;
         backend=CairoMakie)
    @info "Saved split L/R panels"

    # ── Figure 4: by neuron type, colored ───────────────────────────
    typcolor = zz2fig.typ2color_func(valid)
    fig_typ = Figure(size=(600, 600), backgroundcolor=:white)
    ax_typ  = Axis(fig_typ[1, 1], title="Insula somata — by neuron type",
                   titlesize=14)
    zz2fig.plot_flatmap_base!(ax_typ, highlights, objflat, objphy;
                              label=false)
    add_insula_labels!(ax_typ, objflat, objphy)
    for side in ["L", "R"]
        sub = filter(r -> r.original_side == side && !any(isnan, r.somauv), valid)
        nrow(sub) == 0 && continue
        scatter!(ax_typ, sub.somauv,
                 color=map(r -> typcolor(r.typ), eachrow(sub)),
                 marker=side_marker(side),
                 markersize=7, strokecolor=:black, strokewidth=0.3)
    end
    limits!(ax_typ, lims...)
    save(joinpath(ins_dir, "insula_soma_type.png"), fig_typ;
         px_per_unit=3, backend=CairoMakie)
    @info "Saved type-colored soma plot"

    @info "Done. Output in $ins_dir"
    valid
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_insula_pipeline()
end
