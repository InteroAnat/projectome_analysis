using CairoMakie

include(raw"D:\projectome_analysis\gou_julia_scripts\gou_flatmap_minimal.jl")

rotate_uv(p, ::Val{:cw}) = SVector(p[2], -p[1])
rotate_uv(p, ::Val{:ccw}) = SVector(-p[2], p[1])
mirror_uv_x(p) = SVector(-p[1], p[2])

function rotate_flatmesh(objflat, dir::Symbol)
    rot = dir == :cw ? Val(:cw) : Val(:ccw)
    pos = map(objflat.position) do p
        uv = rotate_uv(SVector(p[1], p[2]), rot)
        if length(p) == 3
            Point3f(uv[1], uv[2], p[3])
        else
            Point2f(uv[1], uv[2])
        end
    end
    GeometryBasics.Mesh(pos, GeometryBasics.faces(objflat))
end

function run_insula_lr_mirror(; niter::Int=30000)
    tag = :leftinsula
    depth_cache = joinpath(CACHE_DIR, "depth_volume.jld2")
    flat_cache  = joinpath(CACHE_DIR, "flatmap_$(tag)_n$(niter).jld2")
    out_dir     = joinpath(OUT_ROOT, "insula")
    mkpath(out_dir)

    depthimg, depthres = if isfile(depth_cache)
        JLD2.load(depth_cache, "depthimg", "depthres")
    else
        model, obj_mid = build_depth_model()
        di, dr = build_depth_volume(model, obj_mid)
        JLD2.jldsave(depth_cache; depthimg=di, depthres=dr)
        di, dr
    end

    objflat, objphy = if isfile(flat_cache)
        of, op = JLD2.load(flat_cache, "objflat", "objphy")
        op = hasproperty(op, :normals) ? op :
             GeometryBasics.normal_mesh(op.position, GeometryBasics.faces(op))
        of, op
    else
        fm = build_flatmap(; tag, niter)
        JLD2.jldsave(flat_cache; objflat=fm.objflat, objphy=fm.objphy)
        fm.objflat, fm.objphy
    end

    df = load_somata()
    uvw = xyz2uvw(df.soma_pos, depthimg, depthres, objflat, objphy)
    insertcols!(df,
        :somauv => map(v -> SVector(v[1:end-1]...,), uvw),
        :somaw  => last.(uvw))
    valid = filter(r -> !any(isnan, r.somauv), df)

    highlights = get_insula_highlights()
    objflatL = rotate_flatmesh(objflat, :cw)
    # Right panel: mirror first, then rotate anticlockwise.
    posR = map(objflat.position) do p
        uv = rotate_uv(mirror_uv_x(SVector(p[1], p[2])), Val(:ccw))
        if length(p) == 3
            Point3f(uv[1], uv[2], p[3])
        else
            Point2f(uv[1], uv[2])
        end
    end
    objflatR = GeometryBasics.Mesh(posR, GeometryBasics.faces(objflat))
    limsL = insula_axis_limits(objflatL, objphy; pad=0.20)
    limsR = insula_axis_limits(objflatR, objphy; pad=0.20)

    lside = filter(r -> r.original_side == "L", valid)
    rside = filter(r -> r.original_side == "R", valid)
    lside_uv = map(p -> rotate_uv(p, Val(:cw)), lside.somauv)
    rside_uv = map(p -> rotate_uv(mirror_uv_x(p), Val(:ccw)), rside.somauv)

    fig = Figure(size=(1300, 680), backgroundcolor=:white)

    axL = Axis(fig[1, 2], title="Left insula (90° clockwise)", titlesize=16)
    zz2fig.plot_flatmap_base!(axL, highlights, objflatL, objphy; label=false)
    add_insula_labels!(axL, objflatL, objphy)
    scatter!(axL, lside_uv, color=:steelblue, marker=:circle,
             markersize=7, strokecolor=:black, strokewidth=0.4)
    limits!(axL, limsL...)
    axL.xreversed = true
    text!(axL, (limsL[1] + limsL[2]) / 2, limsL[4] - 0.04*(limsL[4]-limsL[3]),
          text="L n=$(nrow(lside))", fontsize=12, align=(:center, :top))

    axR = Axis(fig[1, 1], title="Right insula (90° anticlockwise)", titlesize=16)
    zz2fig.plot_flatmap_base!(axR, highlights, objflatR, objphy; label=false)
    add_insula_labels!(axR, objflatR, objphy)
    scatter!(axR, rside_uv, color=:firebrick, marker=:utriangle,
             markersize=7, strokecolor=:black, strokewidth=0.4)
    limits!(axR, limsR...)
    axR.xreversed = true
    text!(axR, (limsR[1] + limsR[2]) / 2, limsR[4] - 0.04*(limsR[4]-limsR[3]),
          text="R n=$(nrow(rside))", fontsize=12, align=(:center, :top))

    Label(fig[0, :], "Insula LR mirrored flatmaps", fontsize=18, font=:bold)

    save(joinpath(out_dir, "insula_LR_mirror_flatmaps.png"), fig;
         px_per_unit=3, backend=CairoMakie)
    save(joinpath(out_dir, "insula_LR_mirror_flatmaps.svg"), fig;
         backend=CairoMakie)

    return (; valid=nrow(valid), left=nrow(lside), right=nrow(rside))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_insula_lr_mirror()
end
