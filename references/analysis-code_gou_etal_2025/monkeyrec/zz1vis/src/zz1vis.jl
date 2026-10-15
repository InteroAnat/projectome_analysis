module zz1vis

using ProgressMeter

import zz1data

using RPRMakie
using ColorTypes
using GeometryBasics
using LinearAlgebra
import DataStructures
import NRRD
using FileIO
using StaticArrays
using SharedArrays

import Gapr

function mycolorbuffer(screen::MakieScreen; withalpha=true)
    global xxximg = Makie.colorbuffer(screen)
    if withalpha
        fb_alpha = RPR.FrameBuffer(screen.context, RPRMakie.RGBA, screen.fb_size)
        RPR.set!(screen.context, RPR.RPR_AOV_OPACITY, fb_alpha)
    end
    fb_dep = RPR.FrameBuffer(screen.context, RPRMakie.RGBA, screen.fb_size)
    RPR.set!(screen.context, RPR.RPR_AOV_DEPTH, fb_dep)
    fb_pos = RPR.FrameBuffer(screen.context, RPRMakie.RGBA, screen.fb_size)
    RPR.set!(screen.context, RPR.RPR_AOV_WORLD_COORDINATE, fb_pos)
    if !screen.setup_scene
        ###@info "prev :;:: $(screen.rpr_scene)"
        display(screen, screen.scene)
        #@info "pasdfasf :;:: $(screen.rpr_scene)"
    end

    data_1d = reverse(reshape(RPR.get_data(RPRMakie.render(screen)), screen.fb_size), dims=2)
    data_a = if withalpha
        map(red, reshape(RPR.get_data(fb_alpha), screen.fb_size))
    else
        map(alpha, data_1d)
    end
    max_a = data_a |> maximum
    @info "maxa $max_a"
    r = map(data_1d, data_a) do c, a
        color = RGBA{Float32}(red(c), green(c), blue(c), (0.001 + a) / (0.001 + max_a))
        RGBA{ColorTypes.N0f16}(ColorTypes.mapc(x -> clamp(x, 0, 1), color))
    end
    #depth = reshape(RPR.get_data(fb_pos), screen.fb_size)
    depth = map(red, reshape(RPR.get_data(fb_dep), screen.fb_size))
    (color=rotl90(r), depth=rotl90(depth))
end

function showcontour(screen::MakieScreen; linewidth=2.0f0)
    RPR.rprContextSetParameterByKeyString(screen.context, RPR.RPR_CONTEXT_GPUINTEGRATOR, "gpucontour")
    try
        for meth in [:UV :NORMAL :OBJECTID :MATERIALID]
            RPR.set!(screen.context, getglobal(RPR, Symbol(:RPR_CONTEXT_CONTOUR_USE_, meth)), 1)
            RPR.set!(screen.context, getglobal(RPR, Symbol(:RPR_CONTEXT_CONTOUR_LINEWIDTH_, meth)), linewidth)
        end
        RPR.set!(screen.context, RPR.RPR_CONTEXT_CONTOUR_ANTIALIASING, 1.0f0)
        for aov in [:OBJECT_ID :MATERIAL_ID :SHADING_NORMAL :UV]
            RPR.set!(screen.context, getglobal(RPR, Symbol(:RPR_AOV_, aov)), RPR.FrameBuffer(screen.context, RGBA, screen.fb_size))
        end
        mycolorbuffer(screen; withalpha=false)
    finally
        RPR.rprContextSetParameterByKeyString(screen.context, RPR.RPR_CONTEXT_GPUINTEGRATOR, "")
    end
end

function _get_center_and_range(scene)
    limits = RPRMakie.data_limits(scene)
    rngs = limits.widths ./ 2
    center = limits.origin .+ rngs
    range = max(rngs...)
    center, range
end

function recenter!(screen::MakieScreen)
    center, rad = _get_center_and_range(screen.scene)
    cam = Makie.cameracontrols(screen.scene)
    off = cam.eyeposition[] - cam.lookat[]
    cam.lookat[] = center
    cam.eyeposition[] = center .+ off .* rad
    for l in screen.scene.lights
        if l isa RPR.PointLight
            l.position[] += center
        end
    end
    screen
end

function spherelights!(screen::MakieScreen, posrad; intensity=2.5, center=nothing)
    if center === nothing
        center, rad = _get_center_and_range(screen.scene)
    end
    matlight = RPR.UberMaterial(screen.matsys;
        diffuse_weight=0.0f0,
        reflection_weight=0.0f0,
        emission_color=Vec3f(intensity),
        emission_weight=1.0f0,
        emission_mode=RPR.RPR_UBER_MATERIAL_EMISSION_MODE_DOUBLESIDED,
    )
    pos = map(p -> center .+ Vec3f(first(p)), posrad)
    rad = map(Float32 ∘ last, posrad)
    RPRMakie.meshscatter!(screen.scene, pos, markersize=rad, material=matlight)
end

function mixcontour(img, contour, weight=0.5f0)
    img, imgd = img[(:color, :depth)]
    contour, contourd = contour[(:color, :depth)]
    bg = contour[findmax(ColorTypes.red, contour)[2]]
    fg = contour[findmin(ColorTypes.red, contour)[2]]
    img = map(img, contour, imgd, contourd) do c1, c2, d1, d2
        a1 = c1.alpha
        c1 = ColorTypes.RGB(c1)
        a2 = abs(0.0f0 + c2.r - bg.r) * weight
        c2 = ColorTypes.RGB(fg)
        # if d1 + 3 < d2
        #     a1, a2 = a2, a1
        #     c1, c2 = c2, c1
        # end
        cc = a2 * c2 + (1 - a2) * a1 * c1
        aa = (1.0f0 - a1) * a2 + a1
        ColorTypes.RGBA(cc, aa)
    end
    img
end

function extract_contour(contour, contourd)
    bg = contour[findmax(ColorTypes.red, contour)[2]]
    fg = contour[findmin(ColorTypes.red, contour)[2]]
    value(c2) = abs(0.0f0 + c2.r - bg.r)
    contour = map(value, contour)

    seeds = filter(CartesianIndices(contour)) do ind
        contour[ind] > 0.01
    end
    seeds = filter(seeds) do idx
        c = contourd[idx]
        Vec3f(c.r, c.g, c.b) != Vec3f(0)
    end
    map(seeds) do idx
        c = contourd[idx]
        Vec3f(c.r, c.g, c.b)
    end
end

@recipe(MeshVis, mesh) do scene
    return Attributes(
        primary_only_flag=Makie.automatic,
        diffuse=Makie.automatic,
        glossy_reflection=Makie.automatic,
        glossy_refraction=Makie.automatic,
        light=Makie.automatic,
        reflection=Makie.automatic,
        refraction=Makie.automatic,
        shadow=Makie.automatic,
        transparent=Makie.automatic,
        color=:red,
    )
end

function _set_visibility(F, rpr_mesh, plot)
    ff = getglobal(RPR, Symbol(:rpr, F, :SetVisibilityFlag))
    for (flg, vis) in [
        :primary_only_flag => true
        :diffuse => true
        :glossy_reflection => true
        :glossy_refraction => true
        :light => true
        :reflection => true
        :refraction => true
        :shadow => false
        :transparent => false
    ]
        vis = getproperty(plot, flg)[]
        if !isa(vis, Makie.Automatic)
            flg = getglobal(RPR, Symbol(:RPR_, uppercase(String(F)), :_VISIBILITY_, uppercase(String(flg))))
            ff(rpr_mesh, flg, vis)
        end
    end
end

function RPRMakie.to_rpr_object(context, matsys, scene, plot::MeshVis)
    rpr_mesh = RPR.Shape(context, to_value(plot[1]))
    material = RPRMakie.mesh_material(context, matsys, plot)
    map(plot.model) do m
        RPR.transform!(rpr_mesh, m)
    end
    RPR.set!(rpr_mesh, material)
    _set_visibility(:Shape, rpr_mesh, plot)
    return rpr_mesh
end

Makie.point_iterator(plot::MeshVis) = Makie.point_iterator(plot.mesh[])
function Makie.plot!(plot::MeshVis)
    mesh = plot.mesh[]
    return plot
end

function glass(matsys; kwargs...)
    RPR.Glass(matsys;
        refraction_color=Vec4f(1.0f0),
        refraction_roughness=Vec4f(0.01),
        refraction_caustics=true,
        refraction_ior=1.01f0,
        reflection_ior=1.01f0,
        reflection_mode=RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
        refraction_absorption_color=Vec4f(1, 1, 1, 0),
        refraction_absorption_distance=Vec4f(20.0f0),
        coating_color=Vec4f(1.0f0),
        coating_weight=0.1f0,
        coating_roughness=0.01f0,
        coating_mode=RPR.RPR_UBER_MATERIAL_IOR_MODE_METALNESS,
        coating_metalness=1.0f0,
        coating_transmission_color=Vec4f(1.0f0),
        coating_thickness=0.1f0,
        kwargs...
    )
end
function hair(matsys; color=Vec4f(1, 1, 0, 0))
    aaa = RPR.UberMaterial(matsys;
        diffuse_weight=0.0f0,
        reflection_weight=1.0f0,
        reflection_color=GeometryBasics.Vec4f(0.5f0),
        reflection_roughness=0.25f0,
        reflection_anisotropy=1.0f0,
        reflection_anisotropy_rotation=0.5f0,
        reflection_mode=RPR.RPR_MATERIAL_INPUT_UBER_REFLECTION_METALNESS,
        reflection_metalness=1.0f0,
    )
    bbb = RPR.UberMaterial(matsys;
        diffuse_weight=0.0f0,
        color=Vec4f(0, 1.0f0, 0, 1.0f0),
        diffuse_roughness=0.5,
        reflection_weight=0.0f0,
        backscatter_weight=1.0f0,
        backscatter_color=color,
    )
    RPR.AddMaterial(matsys; color=aaa, color2=bbb)
end

function plastic(matsys; kw...)
    RPRMakie.RPR.UberMaterial(matsys; color=Makie.to_color(:yellow), reflection_color=Vec4f(1),
        reflection_weight=Vec4f(1), reflection_roughness=Vec4f(0),
        reflection_anisotropy=Vec4f(0), reflection_anisotropy_rotation=Vec4f(0),
        reflection_metalness=Vec4f(0), reflection_ior=Vec4f(1.5), refraction_weight=Vec4f(0),
        coating_weight=Vec4f(0), sheen_weight=Vec4f(0), emission_weight=Vec3f(0),
        transparency=Vec4f(0), reflection_mode=UInt(RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR),
        emission_mode=UInt(RPRMakie.RPR.RPR_UBER_MATERIAL_EMISSION_MODE_SINGLESIDED),
        coating_mode=UInt(RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR), sss_multiscatter=false,
        refraction_thin_surface=false, kw...)
end

@recipe(Fibers, positions) do scene
    return Attributes(
        primary_only_flag=Makie.automatic,
        diffuse=Makie.automatic,
        glossy_reflection=Makie.automatic,
        glossy_refraction=Makie.automatic,
        light=Makie.automatic,
        reflection=Makie.automatic,
        refraction=Makie.automatic,
        shadow=Makie.automatic,
        transparent=Makie.automatic,
        color=:red,
    )
end

###Makie.MakieCore.conversion_trait(::Type{<:Fibers}) = PointBased()

function Makie.convert_arguments(::Makie.PointBased, swc::Gapr.SwcTable)
    T = Float32
    return ([Point{3,T}(n.pos) for n in swc],)
end
function _get_all_points(swc::Gapr.SwcTable)
    decompose(Point3f, map(n -> Point3f(n.pos ./ 1.0f0), swc.nodes))
end
function _generate_curves(context, points, plot, segs::AbstractVector{<:AbstractVector{<:Integer}})
    indices = RPR.rpr_int[]
    for seg in segs
        prev = nothing
        for i in seg
            if prev !== nothing
                push!(indices, prev, prev, i - 1, i - 1)
            end
            prev = i - 1
        end
    end
    nsegments = length(indices) ÷ 4
    @info "length $(length(indices))"

    radius = if plot.linewidth[] isa AbstractVector
        @assert false
        # lw = plot.linewidth[]
        # perseg = if length(lw) == length(points)
        #     lw[1:2:end]
        # elseif length(lw) == nsegments
        #     lw
        # else
        #     error("Length $(nsegments) doesn't match $(length(lw))")
        # end
        # Float32.(lw ./ 1000)
    else
        fill(Float32(plot.linewidth[] / 1000), nsegments)
    end
    RPR.Curve(context, points, indices, radius, Vec2f.(0.0, LinRange(0, 1, nsegments)),
        fill(1, nsegments))
    # ragaoa ragaoa
end

function RPRMakie.to_rpr_object(context, matsys, scene, plot::Fibers)
    arg1 = to_value(plot[1])
    points = _get_all_points(arg1)
    isempty(points) && return nothing

    curve = _generate_curves(context, points, plot, [[r.ind for r in seg] for seg in Gapr.eachsegment(arg1)])
    # should use extract_material, but mesh_material handles color
    material = RPRMakie.mesh_material(context, matsys, plot, plot.color)
    #material = RPRMakie.extract_material(matsys, plot)
    RPR.set!(curve, material)
    map(plot.model) do m
        RPR.rprCurveSetTransform(curve, false, convert(Matrix{Float32}, m))
    end
    material.color = Makie.to_color(plot.color[])
    #######_set_visibility(:Curve, curve, plot)
    return curve
end

Makie.point_iterator(plot::Fibers) = Makie.point_iterator(plot.positions[])
Makie.point_iterator(swc::Gapr.SwcTable) = [Makie.Point3f(n.pos) for n in swc]
function Makie.plot!(plot::Fibers)
    positions = plot.positions[]
    return plot
end

@recipe(Dendrogram, hclust) do scene
    a = Attributes(
        trimat=0.0f0
    )
    merge(a, Makie.default_theme(scene, Makie.LineSegments))
end

import Clustering

function _calc_points(f::Function, h::Clustering.Hclust)
    leafs = broadcast(SVector, axes(h.order, 1), 0.0f0)
    branches = similar(leafs, length(h.heights))
    merges = broadcast(something, indexin(-h.merges, h.order), -h.merges)
    for k in axes(merges, 1)
        ai, bi = view(merges, k, :)
        a, b = map((ai, bi)) do i
            if i < 0
                branches[-i]
            else
                leafs[i]
            end
        end
        branches[k] = SVector((a[1] + b[1]) / 2, h.heights[k])
        f(-k => branches[k], ai => a, bi => b)
    end
end
function Makie.point_iterator(plot::Dendrogram{Tuple{Clustering.Hclust{T}}}) where {T}
    points = SVector{2,Float32}[]
    _calc_points(plot[1][]) do (_, pxy), (l, lxy), (r, rxy)
        if l > 0
            push!(points, lxy)
        end
        if r > 0
            push!(points, rxy)
        end
        push!(points, pxy)
    end
    points
end

function Makie.plot!(plot::Dendrogram{Tuple{Clustering.Hclust{T}}}) where {T}
    h = plot[1][]
    linesegs = SVector{2,Float32}[]
    _calc_points(h) do (_, pxy), (l, lxy), (r, rxy)
        push!(linesegs, lxy)
        push!(linesegs, SVector(lxy[1], pxy[2]))
        push!(linesegs, rxy)
        push!(linesegs, SVector(rxy[1], pxy[2]))
        push!(linesegs, SVector(lxy[1], pxy[2]))
        push!(linesegs, SVector(rxy[1], pxy[2]))
    end
    trimat=pop!(plot.attributes, :trimat, 0)
    #xticks!(plot, xtickrange=axes(h.order, 1), xticklabels=h.order)
    return linesegments!(plot, linesegs; plot.attributes...)
end

function RPRMakie.RPR.material_info(::Type{RPRMakie.RPR.WardMaterial})
    return (
        color=RPRMakie.RPR.RPR_MATERIAL_INPUT_COLOR,
        rotation=RPRMakie.RPR.RPR_MATERIAL_INPUT_ROTATION,
        roughness_u=RPRMakie.RPR.RPR_MATERIAL_INPUT_ROUGHNESS_X,
        roughness_v=RPRMakie.RPR.RPR_MATERIAL_INPUT_ROUGHNESS_Y,
        # scale=RPRMakie.RPR.RPR_MATERIAL_INPUT_SCALE,
    )
end
function RPRMakie.RPR.material_info(::Type{RPRMakie.RPR.AddMaterial})
    return (
        color=RPRMakie.RPR.RPR_MATERIAL_INPUT_COLOR0,
        #color1=RPRMakie.RPR.RPR_MATERIAL_INPUT_COLOR0,
        color2=RPRMakie.RPR.RPR_MATERIAL_INPUT_COLOR1,
        # scale=RPRMakie.RPR.RPR_MATERIAL_INPUT_SCALE,
    )
end

function select_channel(sample, channel)
    imgidx = parse(Gapr.ImageIndex, "../segreg/input/fmost$sample.ch$channel.catalog", base="http://bap.cebsit.ac.cn/yanlab/fmost$sample.ch$channel/")
    chansel = []
    for (nam, chan) in imgidx.channels
        if chan isa Gapr.ImageCubes
            push!(chansel, chan)
        end
    end
    chansel = first(chansel)
end

function collect_cubes(pos, chansel)
    cubes = Set()
    invdir = inv(chansel.direction)
    posfw(xyz) = invdir * (xyz .- chansel.origin)
    posbw(xyz) = chansel.direction * xyz .+ chansel.origin
    for xyz in pos
        posv = posfw(xyz)
        cub = floor.(Int, posv ./ chansel.cubesize) .* chansel.cubesize
        push!(cubes, cub)
    end
    if true
        cubes0 = collect(cubes)
        for c in cubes0
            for dx in -1:1, dy in -1:1, dz in -1:1
                push!(cubes, c .+ chansel.cubesize .* (dx, dy, dz))
            end
        end
    end
    cubes = collect(cubes)
    fig = Makie.Figure()
    ax = Makie.LScene(fig[1, 1])
    for o in cubes
        #Makie.lines!(ax, Makie.Rect3f(o, chansel.cubesize))
    end
    Makie.scatter!(ax, map(posbw, cubes))
    fig
    cubes
end

function draw_hires(df, rawinfo, camera; rowcolor)
    sampchans = map(r -> (r.sample, r.channel), eachrow(df)) |> unique!
    @info "$sampchans"
    @assert length(sampchans) == 1
    root_raw_pos = map(eachrow(df)) do r
        rawinfo(r.sample, r.channel)[r.root].rawxyz
    end
    chansel = select_channel(sampchans[1]...)
    cubes = collect_cubes(root_raw_pos, chansel)
    @warn "$(length(cubes)) cubes"
    # for c in cubes
    #     @info "$c"
    # end

    if false
        xrng = median((c[1] for c in cubes))
        yrng = median((c[2] for c in cubes))
        filter!(c -> (c[1] in xrng:xrng+2000) && c[2] in yrng:yrng+2000, cubes)
    end

    colors = [rowcolor(r) for r in eachrow(df)]
    pal = unique(colors)
    typs = something.(indexin(colors, pal))
    proj = zz1data.cached_result(:proj769, cubes) do cubes
        draw_large_cubes(camera, cubes, chansel, root_raw_pos, typs)
    end
    visualize_cubes_proj(proj, pal)
end

function draw_hires_soma(df, rawinfo; rowcolor)
    root_raw_pos = map(eachrow(df)) do r
        rawinfo(r.sample, r.channel)[r.root].rawxyz
    end
    resolution = SVector(4096, 4096)
    cam = simple_ortho_camera(resolution, root_raw_pos)
    canvas = fill((; d=Inf, c=Makie.RGBAf(0, 0, 0, 0)), resolution...)
    draw(canvas, pt, val) = begin
        idx = ceil.(Int, pt)
        if checkbounds(Bool, canvas, idx...)
            ######@warn "aaa $idx $val"
            canvas[idx...] = min(canvas[idx...], val)
        end
    end
    foreach(eachrow(df), root_raw_pos) do r, pos
        p, d = cam(pos)
        for dx in -9:9
            for dy in -9:9
                dx^2 + dy^2 > 49 && continue
                draw(canvas, p .+ (dx, dy), (; d, c=rowcolor(r)))
            end
        end
    end
    global xxximg = deepcopy(canvas)
    map(p -> p.c, canvas)
end

function draw_hires_fibers(df, rawinfo, camera; rowcolor)
    resolution = SVector(4096, 4096)
    funcs = draw_context_funcs((; resolution, root_raw_pos=nothing, camera))
    canvas = fill((; d=Inf, c=Makie.RGBAf(0, 0, 0, 0)), resolution...)
    draw(canvas, pt, val) = begin
        idx = ceil.(Int, pt)
        if checkbounds(Bool, canvas, idx...)
            if canvas[idx...].d > val.d
                canvas[idx...] = val
            end
        end
    end
    for r in eachrow(df)
        swc = Gapr.load_swc(r.path)
        info = rawinfo(r.sample, r.channel)
        for n in swc.nodes
            n.pos = info[n.id].rawxyz
        end
        for n in swc.nodes
            n.parind == 0 && continue
            pxy, pz = funcs.cam(n.pos)
            qxy, qz = funcs.cam(swc[n.parind].pos)
            for t in 0:0.1:1
                xy = t * qxy + (1 - t) * pxy
                d = t * qz + (1 - t) * pz
                if funcs.clampz(d)
                    for dx in -9:9
                        for dy in -9:9
                            dx^2 + dy^2 > 49 && continue
                            draw(canvas, xy .+ (dx, dy), (; d, c=rowcolor(r)))
                        end
                    end
                end
            end
        end
    end
    global xxximg = deepcopy(canvas)
    map(p -> Makie.RGBf(p.c), canvas)
end

function simple_ortho_camera(resolution, root_raw_pos, camera)
    # cent = Vec3f(29080, 24196, 16219)
    # eye = Vec3f(25492, 26195, 18794)
    # view = Makie.lookat(eye, cent, Vec3f(31167, 22721, 20204) .- cent)
    cent = Vec3f(camera.center)
    eye = Vec3f(camera.eye)
    view = Makie.lookat(eye, cent, Vec3f(camera.up) .- cent)
    #toview(p)= view * Vec4f(p..., 1)

    root_raw_pos = @something root_raw_pos begin
        dd = (eye - cent)
        dd = camera.zrange * dd / norm(dd)
        [eye + dd, eye - dd]
    end
    ptsview = map(root_raw_pos) do p
        view * Vec4f(p..., 1)
    end
    ext = extrema(reduce(hcat, ptsview), dims=2)

    rad = if (xyrange = get(camera, :xyrange, nothing)) !== nothing
        xyrange
    else
        norm(eye - cent)
    end
    aspr = Float32(resolution[1] / resolution[2])
    proj = Makie.orthographicprojection(Float32, -rad, rad, -rad / aspr, rad / aspr, ext[3]...)
    projview = proj * view
    tocanvas(xyz) = begin
        p = projview * Vec4f(xyz..., 1)
        (0.5 * (SVector(p[1], p[2]) ./ p[4] .+ 1)) .* (resolution .- 1), p[3]
    end
    tocanvas
end

using Distributed

download_and_load(u) = begin
    mktempdir() do path
        webm = joinpath(path, "xxx.webm")
        nrrd = joinpath(path, "xxx.nrrd")
        try
            run(`curl $u -so $webm -Y1000 -y9`)
            conv = expanduser("~/Desktop/gapr/build/source/convert/gapr-convert")
            run(`$conv --to-nrrd $webm $nrrd`)
            FileIO.load(nrrd)
        catch e
            @warn "err get img $e"
            # XXX: 
            (; data=zeros(UInt16, 400, 400, 80))
        end
    end
    # ext=extrema(reduce(hcat, cubes), dims=2)
    # cub=zeros(eltype(first(cubesout)), (last.(ext).+chansel.cubesize.-first.(ext))...)
    # oo=first.(ext)
    # foreach(cubesout, cubes) do c,o
    #     Rsrc=CartesianIndices(c)
    #     copyto!(cub, Rsrc.+CartesianIndex((o.-oo)...), c, Rsrc)
    # end
    # @info "$(size(cub)) $(oo)"
    # cub,SVector(oo...)
end

draw_context_funcs(context) = begin
    resolution = SVector(4096, 4096)
    cam = simple_ortho_camera(resolution, context.root_raw_pos, context.camera)

    alloc_canvas() = fill((; prox=0.0f0, val=0.0f0, typ=UInt32(0)), resolution...)
    draw(canvas, pt, val) = begin
        idx = ceil.(Int, pt)
        if checkbounds(Bool, canvas, idx...)
            canvas[idx...] = max(canvas[idx...], val)
        end
    end
    posbw(xyz) = context.chansel.direction * xyz .+ context.chansel.origin
    clampz = if (zr = get(context.camera, :zrange, nothing)) !== nothing
        de = context.camera.eye .- context.camera.center
        de = de ./ norm(de)
        z1, z2 = map((-zr, zr)) do z
            cam(z .* de .+ context.camera.center)[2]
        end
        z1, z2 = min(z1, z2), max(z1, z2)
        @info "zr $z1 $z2"
        (z) -> z1 < z < z2
    else
        (_) -> true
    end
    (; posbw, draw, alloc_canvas, cam, clampz)
end
draw_cubes_impl(oo, cubesxout, stc, context) = begin
    funcs = draw_context_funcs(context)
    canvas2 = funcs.alloc_canvas()
    for ci in CartesianIndices(cubesxout)
        val = cubesxout[ci]
        st = stc[ci]
        val = val / typemax(val)
        posxy, posz = funcs.cam(funcs.posbw(SVector(Tuple(ci)) .- 0.5 .+ oo))
        if funcs.clampz(posz)
            funcs.draw(canvas2, posxy, (; prox=exp(-st.dist), val, st.typ))
        end
    end
    canvas2
end


function draw_large_cubes(camera, cubes, chansel, root_raw_pos, typ; dist_thr=30.0f0)
    activecubes = Dict{Vec3i,AbstractArray}()

    context = (; chansel, root_raw_pos, camera)
    funcs = draw_context_funcs(context)

    merge_canvas!(canvas, alt) = begin
        canvas .= max.(canvas, alt)
    end
    canvas = funcs.alloc_canvas()

    sweepdir = argmax(d -> length(unique([c[d] for c in cubes])), 1:3)
    @info "sweep $sweepdir"
    zrange(cubes) = sort(unique([c[sweepdir] for c in cubes]))
    filterz(cubes, z) = filter(c -> c[sweepdir] == z, cubes)


    draw_cubes(procs, cubes) = begin
        @everywhere GC.gc()
        zrng = zrange(cubes)
        canvas2s = @showprogress "draw $zrng $(length(cubes))" pmap(draw_cubes_impl, WorkerPool(procs), cubes, [activecubes[c] for c in cubes], [get_state_cube(c) for c in cubes], [context for _ in cubes])
        for c in canvas2s
            merge_canvas!(canvas, c)
        end
    end

    pq = DataStructures.PriorityQueue{Tuple{Vec3i,UInt32,Float32},Float64}()
    push_pq!(pq, k, v) = begin
        pq[k] = min(get(pq, k, Inf64), v)
    end
    invdir = inv(chansel.direction)
    posfw(xyz) = invdir * (xyz .- chansel.origin)
    state_cubes = Dict{Vec3i,AbstractArray}()
    get_state_cube(c) = begin
        get!(state_cubes, c) do
            #SharedArray(fill((; dist=Inf16, typ=UInt16(0)), chansel.cubesize...))
            SharedArray(fill((; dist=UInt8(0), typ=UInt8(0)), chansel.cubesize...))
        end
    end
    pq_skip = []
    color_cubes(root_raw_pos, typs) = begin
        @info "color"
        for (k, v) in pq_skip
            push_pq!(pq, k, v)
        end
        empty!(pq_skip)
        foreach(root_raw_pos, typs) do xyz, typ
            true && return
            vox = floor.(Int, posfw(xyz))
            cubo = floor.(Int, vox ./ chansel.cubesize) .* chansel.cubesize
            cube = get(activecubes, cubo, nothing)
            if cube !== nothing
                v = cube[(vox .- cubo .+ 1)...]
                push_pq!(pq, (vox, typ, v / typemax(v)), 0.0)
            end
        end
        while !isempty(pq) && DataStructures.peek(pq).second < dist_thr
            curkey, prio = DataStructures.dequeue_pair!(pq)
            vox, voxtyp, voxrootval = curkey
            cubo = floor.(Int, vox ./ chansel.cubesize) .* chansel.cubesize
            cube = get(activecubes, cubo, nothing)
            if cube === nothing
                push!(pq_skip, curkey => prio)
                continue
            end
            cubs = get_state_cube(cubo)
            prevs = cubs[((vox .- cubo) .+ 1)...]
            ss = cube[((vox .- cubo) .+ 1)...]
            ss = ss / typemax(ss)
            xyz = posfw(vox .+ 0.5f0)
            if prevs.dist > prio
                cubs[((vox .- cubo) .+ 1)...] = (; dist=prio, typ=voxtyp)
                for dx in -1:1, dy in -1:1, dz in -1:1
                    vox2 = vox .+ (dx, dy, dz)
                    vox2 == vox && continue
                    xyz2 = posfw(vox2 .+ 0.5)
                    dd = norm(xyz2 - xyz)
                    dist2 = prio + (dd / (ss / voxrootval))
                    push_pq!(pq, (vox2, voxtyp, voxrootval), dist2)
                end
            end
        end
    end
    ##########

    procs = addprocs(16 - 3)
    try
        ###@everywhere procs @eval using SharedArrays
        @everywhere procs @eval using zz1vis
        @info "start draw $(zrange(cubes))"
        for z in zrange(cubes)
            cubesx = filterz(cubes, z)
            loadtask = "$(zrange(cubesx))/$(length(cubesx))"
            @info "loadx $loadtask"
            cubesxout = @time "load" pmap(zz1vis.download_and_load, WorkerPool(procs), [chansel[c...] for c in cubesx])
            foreach(cubesx, cubesxout) do o, c
                push!(activecubes, o => SharedArray(c.data))
            end
            zzz = zrange(keys(activecubes))
            if length(zzz) > 1
                zz = first(zzz)
                tmpcubes = filterz(keys(activecubes), zz)
                @time "color" color_cubes(root_raw_pos, typ)
                @time "draw" draw_cubes(procs, tmpcubes)
                for c in tmpcubes
                    pop!(activecubes, c)
                    pop!(state_cubes, c)
                end
                GC.gc()
            end
            # XXX:
            # for c in collect(keys(activecubes))[1:end-1]
            #     pop!(activecubes, c)
            #     if haskey(state_cubes, c)
            #         pop!(state_cubes, c)
            #     end
            # end
        end
        @time "color" color_cubes(root_raw_pos, typ)
        @time "draw" draw_cubes(procs, keys(activecubes))
    finally
        rmprocs(procs)
    end
    canvas

    #download_and_load(chansel, cubes[1:10])
end

function visualize_cubes_proj(proj, cols)
    vm = maximum([v.val for v in proj])
    colw = Makie.to_color(:white)
    @info "$vm"
    img = map(proj) do p
        v = (p.val / vm)
        c = if p.typ == 0
            colw * v * 1
        else
            cols[p.typ] * clamp(v + 0.3, 0, 1)
        end
        Makie.RGBf(c)
    end
    img
    #Makie.image(img)
end

end # module zz1vis
