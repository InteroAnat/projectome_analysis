module GaprPlotRPRExt

using Gapr
using RPRMakie
import RPRMakie: to_rpr_object
import RPRMakie: RPR
using GeometryBasics

##import GaprPlotExt

function RPRMakie.to_rpr_object(context, matsys, scene, plot::RPRMakie.Plot{Gapr.swcplot})
    swc = plot[1][]
    # #arg1 = to_value(plot[1])
    # arg1 = [Point3f(0, 0, 0), Point3f(1, 0, 0)]
    # isempty(arg1) && return nothing
    # points = decompose(Point3f, arg1)
    # segments = TupleView{2,2}(RPR.rpr_int(0):RPR.rpr_int(length(points) - 1))
    # indices = RPR.rpr_int[]
    #
    # for (a, b) in segments
    #     push!(indices, a, a, b, b)
    # end
    #
    # nsegments = length(indices) ÷ 4
    #
    # radius = if plot.linewidth[] isa AbstractVector
    #     lw = plot.linewidth[]
    #     perseg = if length(lw) == length(points)
    #         lw[1:2:end]
    #     elseif length(lw) == nsegments
    #         lw
    #     else
    #         error("Length $(nsegments) doesn't match $(length(lw))")
    #     end
    #     Float32.(lw ./ 1000)
    # else
    #     fill(Float32(plot.linewidth[] / 1000), nsegments)
    # end
    #
    points = Point3f[]
    indices = RPR.rpr_int[]
    nsegs = RPR.rpr_int[]
    ind2ind = Dict{Int,Int}()
    inds = RPR.rpr_int[]
    ptsbuf = Point3d[]
    putpt!(pt, i) = begin
        if i <= 4
            push!(ptsbuf, pt)
        else
            ptsbuf[(i-1)%4+1] = pt
        end
    end
    getpt(i) = ptsbuf[(i-1)%4+1]
    calc_cp(p1, p2, p3, _) = begin
        d = p3 .- p1
        l = GeometryBasics.norm(d)
        l < 0.01 && return p2, p2
        #return p2, p2
        l1 = GeometryBasics.norm(p2 .- p1)
        l2 = GeometryBasics.norm(p3 .- p2)
        p2 .- d .* (l1 / l / 3), p2 .+ d .* (l2 / l / 3)
    end
    for s in Gapr.eachsegment(swc)
        a = length(indices)
        aa = length(points)
        empty!(ptsbuf)

        ###
        #     s=collect(s)
        #     if length(s)<100
        #         continue
        #     end
        # s=s[[1, end÷4*1, end÷4*2, end÷4*3, end]]
        #     
        ###
        i = 0
        for n in s
            i += 1
            putpt!(n.pos, i)
            if i >= 3
                p1, p2, p3 = getpt(i - 2), getpt(i - 1), n.pos
                p2l, p2r = calc_cp(p1, p2, p3, i)
                push!(points, p2l, p2, p2r)
                ii = (i - 3) * 3 - 1
                push!(indices, max(ii, 0) + aa, ii + 1 + aa, ii + 2 + aa, ii + 3 + aa)
            elseif i >= 2
                push!(points, getpt(i - 1))
            end
        end
        if i >= 3
            push!(points, getpt(i))
            ii = (i - 2) * 3 - 1
            push!(indices, max(ii, 0) + aa, ii + 1 + aa, ii + 2 + aa, ii + 2 + aa)
        elseif i >= 2
            push!(points, getpt(i))
            push!(indices, i - 2 + aa, i - 2 + aa, i - 1 + aa, i - 1 + aa)
        else
            @assert false
        end
        b = length(indices)
        b > a && push!(nsegs, (b - a) ÷ 4)
    end
    @warn "$(length(points)) points"
    radius = fill(Float32(plot.linewidth[] / 1000), length(nsegs))

    # : 1 2 2 3
    # : 1 2 2 3
    # 1 2 3 4
    # : 1 2 3 4
    # 1 2 3 4 5
    # :  ; 1 1 2l 2; 2 2r 3l 3; 3 3r 4l 4; 4 4r 5 5
    # : 1; 2l 2 2r; 3l 3 3r; 4l 4 4r; 5 
    # : 2; 3; 4; 5; E
    # : 1 2 3 4 4 5?
    # 1 2 3 4 5 6
    uvs = length(nsegs) > 1 ? Vec2f.(0.0, LinRange(0, 1, length(nsegs))) : [Vec2f(0, 0)]
    curve = RPR.Curve(context, points, indices, radius, uvs, nsegs)
    # control_pointsf0 = convert(Vector{}, control_points)
    # indices_i = convert(Vector{}, indices)
    # segments_per_curve_i = convert(Vector{}, segments_per_curve)
    # uvs_f0 = convert(Vector{Vec2f}, uvs)
    # radiusf0 = convert(Vector{Float32}, radius)
    # rprContextCreateCurve(context, curve_ref, length(control_pointsf0), control_pointsf0, sizeof(Point3f),
    #                       length(indices_i), length(segments_per_curve_i), indices_i, radiusf0, uvs_f0,
    #                       segments_per_curve_i, creationFlag_tapered)
    # curve = Curve(curve_ref[], context, (control_pointsf0, indices_i, segments_per_curve_i, uvs_f0, radiusf0))
    # add_to_context!(context, curve)
    # return curve
    #
    #material = RPRMakie.extract_material(matsys, plot)
    color = to_color(plot.color[])
    colv = Vec4f(color.r, color.g, color.b, 1)
    # material = RPR.Material(matsys, Dict(pairs((;
    #     color=colv,
    #     reflection_color=colv, reflection_weight=Vec4f(1.0),
    #     reflection_roughness=Vec4f(0.0), reflection_anisotropy=Vec4f(0.0),
    #     reflection_anisotropy_rotation=Vec4f(0.0),
    #     refraction_weight=Vec4f(1.0), refraction_thin_surface=false,
    #     sheen_weight=Vec4f(0),
    #     transparency=Vec4f(0.0),
    #     refraction_color=colv,
    #     refraction_roughness=Vec4f(0.00),
    #     refraction_caustics=true,
    #     refraction_ior=1.6f0,
    #     reflection_ior=1.6f0,
    #     reflection_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
    #     refraction_absorption_color=colv,
    #     refraction_absorption_distance=Vec4f(10.f0),
    #     coating_color=colv,
    #     coating_weight=1.0f0,
    #     coating_roughness=0.01f0,
    #     coating_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_METALNESS,
    #     coating_metalness=1.0f0,
    #     coating_transmission_color=colv,
    #     coating_thickness=0.1f0,
    #     emission_color=colv .* 9,
    #     emission_weight=9.0f0,
    #     emission_mode=UInt(RPR.RPR_UBER_MATERIAL_EMISSION_MODE_DOUBLESIDED),
    #     #         type=:Volume,
    #     # # scattering=colv, absorption=colv,
    #     # #             emission=colv, scatter_direction=colv,
    #     # #             multiscatter=colv,
    #     # #             densitygrid=colv,
    #     # #             density=colv,
    #     # color=colv.*3,
    # ))))

    material = RPR.Plastic(matsys)
    #
    # function set_color!(colorvec)
    #     isempty(colorvec) && return
    #     tex = RPR.ImageTextureMaterial(matsys)
    #     ncols = length(colorvec)
    #     img = RPR.Image(context, reshape(colorvec, (ncols, 1)))
    #     tex.data = img
    #     material.color = tex
    #     return
    # end
    #
    # if color isa AbstractVector{<:Colorant}
    #     set_color!(copy(color))
    # elseif color isa AbstractVector{<:Number}
    #     sampler = Makie.sampler(to_colormap(plot.colormap[]), color; scaling=Makie.Scaling(identity, plot.colorrange[]))
    #     set_color!(collect(sampler))
    # else
    #     material.color = to_color(color)
    # end
    material.color = to_color(color) .* 3
    RPRMakie.set!(curve, material)
    return curve
end

end
