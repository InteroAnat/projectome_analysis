module GaprPlotExt

using Gapr
import Gapr: swcplot, swcplot!
using Makie
using GeometryBasics

println("hello plot ext")

# XXX: naming
# Gapr.swcplot(SwcTable)
"""
    swcplot(swc::SwcTable; kwargs...)

Draws a line plot of the swc table.
"""
Makie.@recipe SwcPlot begin
    "test kw"
    testkw = 0
    rpr = false
    color = @inherit linecolor
    linewidth = @inherit linewidth
    linestyle = :solid
    linecap = @inherit linecap
    joinstyle = @inherit joinstyle
    miter_limit = @inherit miter_limit

    Makie.MakieCore.mixin_colormap_attributes()...
    Makie.MakieCore.mixin_generic_plot_attributes()...
    Makie.MakieCore.mixin_shading_attributes()...
end

function Makie.convert_arguments(::Type{<:SwcPlot}, swc::SwcTable)
    return (swc,)
end

# XXX: plotfunc func in Gapr/Makie/Gapr/Makie/Gapr/Makie

Makie.plottype(p::SwcTable) = SwcPlot

function Makie.plot!(pl::SwcPlot)
    swc = pl[1][]
    @info "type $(typeof(swc))"
    @assert swc isa SwcTable
    pts = Point3d[]
    for s in Gapr.eachsegment(swc)
        for n in s
            push!(pts, n.pos)
        end
        push!(pts, Point3d(NaN))
    end
    # for n in swc.nodes
    #     push!(pts, n.pos)
    # end
    # , color=:blue, linecap=:round,joinstyle=:round))
    if !pl.rpr[]
        lines!(pl, pts; color=pl.color, linewidth=pl.linewidth, linecap=pl.linecap, joinstyle=pl.joinstyle, linestyle=pl.linestyle, miter_limit=pl.miter_limit, transparency=true)
    end

    ## rpr: 1/16
    # scene = parent_scene(pl)
    #
    # textoffset_vec = Observable(Vec2f[])
    # bp = Observable(BezierPath[])
    # text_tuples = Observable(Tuple{Any,Point2f}[])
    #
    # realtextoffset = lift(pl, pl.textoffset, pl.fontsize) do to, fs
    #     return to === automatic ? Float32.(0.75 .* fs) : Float32.(to)
    # end
    #
    # onany(pl, points, scene.camera.projectionview, pl.model, transform_func(pl),
    #       scene.viewport, pl.offset, pl.width, pl.orientation, realtextoffset,
    #       pl.style, pl.text) do points, _, _, _, _, offset, width, orientation, textoff, style, text
    #
    #     empty!(bp[])
    #     empty!(textoffset_vec[])
    #     empty!(text_tuples[])
    #
    #     broadcast_foreach(points, offset, width, orientation, textoff, style, text) do (_p1, _p2), offset, width, orientation, textoff, style, text
    #         p1 = plot_to_screen(pl, _p1)
    #         p2 = plot_to_screen(pl, _p2)
    #
    #         v = p2 - p1
    #         d1 = normalize(v)
    #         d2 = [0 -1; 1 0] * d1
    #         orientation in (:up, :down) || error("Orientation must be :up or :down but is $(repr(orientation)).")
    #         if (orientation == :up) != (d2[2] >= 0)
    #             d2 = -d2
    #         end
    #
    #         off = offset * d2
    #
    #         push!(textoffset_vec[], d2 * textoff)
    #
    #         b, textpoint = bracket_bezierpath(style, p1 + off, p2 + off, d2, width)
    #         push!(text_tuples[], (text, textpoint))
    #         push!(bp[], b)
    #     end
    #
    #     notify(bp)
    #     notify(text_tuples)
    # end
    #
    # notify(points)
    #
    # autorotations = lift(pl, pl.rotation, textoffset_vec) do rot, tv
    #     rots = Quaternionf[]
    #     broadcast_foreach(rot, tv) do rot, tv
    #         r = if rot === automatic
    #             to_rotation(tv[2] >= 0 ? tv : -tv)
    #         else
    #             to_rotation(rot)
    #         end
    #         push!(rots, r)
    #     end
    #     return rots
    # end
    #
    # # Avoid scale!() / translate!() / rotate!() to affect these
    # series!(pl, bp; space = :pixel, solid_color = pl.color, linewidth = pl.linewidth,
    #     linestyle = pl.linestyle, linecap = pl.linecap, joinstyle = pl.joinstyle,
    #     miter_limit = pl.miter_limit, transformation = Transformation())
    # text!(pl, text_tuples, space = :pixel, align = pl.align, offset = textoffset_vec,
    #     fontsize = pl.fontsize, font = pl.font, rotation = autorotations, color = pl.textcolor,
    #     justification = pl.justification, model = Mat4f(I))
    pl
end

# XXX:
Makie.data_limits(pl::SwcPlot) = Rect3d(pl[1][])
# Makie.conversion_trait(::Type{<:SwcPlot}) = Makie.PointBased()
Makie.args_preferred_axis(::SwcTable) = LScene

# *****

#using SomePackage
#import SomePackage: someplot, someplot!

#Makie.convert_single_argument(v::SomeVector) = v.v

#@recipe(SomePlot) do scene
#Theme()
#end

#function Makie.plot!(p::SomePlot)
#lines!(p, p[1])
#scatter!(p, p[1])
#return p
#end

end # module
# function convert_arguments(::Type{<:Bracket}, point1::VecTypes{2, T1}, point2::VecTypes{2, T2}) where {T1, T2}
#     return ([(Point2{float_type(T1)}(point1), Point2{float_type(T2)}(point2))],)
# end
# function convert_arguments(::Type{<:Bracket}, x1::Real, y1::Real, x2::Real, y2::Real)
#     return ([(Point2{float_type(x1, y1)}(x1, y1), Point2{float_type(x2, y2)}(x2, y2))],)
# end
# function convert_arguments(::Type{<:Bracket}, x1::AbstractVector{<:Real}, y1::AbstractVector{<:Real}, x2::AbstractVector{<:Real}, y2::AbstractVector{<:Real})
#     T1 = float_type(x1, y1); T2 = float_type(x2, y2)
#     points = broadcast(x1, y1, x2, y2) do x1, y1, x2, y2
#         (Point2{T1}(x1, y1), Point2{T2}(x2, y2))
#     end
#     return (points,)
# end
#
#
# boundingbox(pl::Bracket, space::Symbol = :data) = apply_transform_and_model(pl, data_limits(pl))
#
#
#
#
#
#
#
# end
# function MakieCore.convert_arguments(P::Type{<:MakieCore.XYBased}, p::AbstractPolynomial)
#     xs = Polynomials.poly_interval(p)
#     return MakieCore.convert_arguments(P, xs, p.(xs))
# end
#
#
#
#
#
#
#
#
#
# **Makie
# !!MakieCore
# !!GLMakie...
# !!RPRMakie




# """
#     pie(values; kwargs...)
#     pie(point, values; kwargs...)
#     pie(x, y, values; kwargs...)
#
# Creates a pie chart from the given `values`.
# """
# @recipe Pie (xs, ys, values) begin
#     "If `true`, the sum of all values is normalized to 2π (a full circle)."
#     normalize = true
#     color = :gray
#     strokecolor = :black
#     strokewidth = 1
#     "Controls how many polygon vertices are used for one degree of rotation."
#     vertex_per_deg = 1
#     "The outer radius of the pie segments."
#     radius = 1
#     "The inner radius of the pie segments. If this is larger than zero, the pie pieces become ring sections."
#     inner_radius = 0
#     "The angular offset of the first pie segment from the (1, 0) vector in radians."
#     offset = 0
#     "The offset of each pie segment from the center along the radius"
#     offset_radius = 0
#     MakieCore.mixin_generic_plot_attributes()...
# end
#
# convert_arguments(PT::Type{<:Pie}, values::RealVector) = convert_arguments(PT, 0.0, 0.0, values)
# convert_arguments(PT::Type{<:Pie}, point::VecTypes{2}, values::RealVector) = convert_arguments(PT, point[1], point[2], values)
# convert_arguments(PT::Type{<:Pie}, ps::AbstractVector{<:VecTypes{2}}, values::RealVector) = convert_arguments(PT, getindex.(ps, 1), getindex.(ps, 2), values)
#
# function convert_arguments(::Type{<:Pie}, xs::Union{Real,RealVector}, ys::Union{Real,RealVector}, values::RealVector)
#     xs = length(xs) == 1 ? fill(only(xs), length(values)) : xs
#     ys = length(ys) == 1 ? fill(only(ys), length(values)) : ys
#     return (float_convert(xs), float_convert(ys), float_convert(values))
# end
#
# function plot!(plot::Pie)
#     xs = plot[1]
#     ys = plot[2]
#     values = plot[3]
#
#     polys = lift(plot, xs, ys, values, plot.vertex_per_deg, plot.radius, plot.inner_radius, plot.offset_radius, plot.offset, plot.normalize) do xs, ys, vals, vertex_per_deg, radius, inner_radius, offset_radius, offset, normalize
#         radius = length(radius) == 1 ? fill(only(radius), length(vals)) : radius
#         inner_radius = length(inner_radius) == 1 ? fill(only(inner_radius), length(vals)) : inner_radius
#         offset_radius = length(offset_radius) == 1 ? fill(only(offset_radius), length(vals)) : offset_radius
#
#         T = eltype(vals)
#
#         # find start and end angles of all pie pieces
#         summed = cumsum([zero(T); vals])
#         boundaries = if normalize
#             summed ./ summed[end] .* 2pi
#         else
#             summed
#         end
#
#         # create vector of a vector of points for each piece
#         vertex_arrays = map(boundaries[1:end-1], boundaries[2:end], xs, ys, radius, inner_radius, offset_radius) do sta, en, x, y, r, inner_r, offset_r
#             x += cos((en + sta) / 2 + offset) * offset_r
#             y += sin((en + sta) / 2 + offset) * offset_r
#             distance = en - sta
#             # how many vertices are needed for the curve?
#             nvertices = max(2, ceil(Int, rad2deg(distance) * vertex_per_deg))
#
#             # curve points
#             points = map(LinRange(sta, en, nvertices)) do rad
#                 Point2(cos(rad + offset) * r + x, sin(rad + offset) * r + y)
#             end
#
#             # add inner points (either curve or one point)
#             if inner_r != 0
#                 inner_points = map(LinRange(en, sta, nvertices)) do rad
#                     Point2(cos(rad + offset) * inner_r + x, sin(rad + offset) * inner_r + y)
#                 end
#                 append!(points, inner_points)
#             else
#                 push!(points, Point2(x, y))
#             end
#
#             points
#         end
#     end
#
#     # plot pieces as polys
#     poly!(
#         plot, polys,
#         color = plot.color, strokewidth = plot.strokewidth,
#         strokecolor = plot.strokecolor, inspectable = plot.inspectable,
#         visible = plot.visible, transparency = plot.transparency
#     )
#
#     plot
# end
