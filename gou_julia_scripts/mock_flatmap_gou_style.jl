using CairoMakie
using Random
using Statistics

"""
Julia-only mock flatmap renderer aligned to Gou's Makie plotting style.

This still uses synthetic data, but the visual grammar is closer to:
- `plot_flatmap_base!`: contourf fills + black contour boundaries
- `plot_flatmap_soma!`: tiny overdraw scatter points
- fixed flatmap limits with hidden decorations
"""

const OUTDIR = raw"D:\projectome_analysis\figures_charts\mock_flatmap_gou_style"
mkpath(OUTDIR)

const TYPE_COLORS = Dict(
    "ITi" => colorant"#1f77b4",
    "ITs" => colorant"#2ca02c",
    "ITc" => colorant"#9467bd",
    "PT"  => colorant"#d62728",
    "CT"  => colorant"#bc8f00",
)

gaussian2d(xx, yy, cx, cy, sx, sy; theta=0.0) = begin
    ct, st = cos(theta), sin(theta)
    xr = ct .* (xx .- cx) .+ st .* (yy .- cy)
    yr = -st .* (xx .- cx) .+ ct .* (yy .- cy)
    exp.(-0.5 .* ((xr ./ sx).^2 .+ (yr ./ sy).^2))
end

function make_flatmap_fields(hemisphere::Symbol)
    x = collect(range(-1.0, 1.0, length=600))
    y = collect(range(-1.0, 1.0, length=600))
    xx = [xx for yy in y, xx in x]
    yy = [yy for yy in y, xx in x]

    shift = hemisphere == :left ? -0.12 : 0.12
    dorsal =
        gaussian2d(xx, yy, -0.10 + shift, -0.04, 0.22, 0.11; theta=0.15) .+
        0.8 .* gaussian2d(xx, yy, -0.02 + shift, 0.08, 0.15, 0.08; theta=-0.20)
    middle =
        gaussian2d(xx, yy, 0.08 + shift, -0.18, 0.18, 0.10; theta=-0.10) .+
        0.5 .* gaussian2d(xx, yy, -0.02 + shift, -0.12, 0.10, 0.07; theta=0.30)
    ventral = gaussian2d(xx, yy, -0.04 + shift, -0.34, 0.16, 0.09; theta=0.18)

    return x, y, [
        (; name="dorsal", field=dorsal, color=colorant"#4f81bd"),
        (; name="middle", field=middle, color=colorant"#f28e2b"),
        (; name="ventral", field=ventral, color=colorant"#59a14f"),
    ]
end

function sample_points_in_ellipse(cx, cy, rx, ry, n, rng)
    pts = Point2f[]
    while length(pts) < n
        x = rand(rng) * (2rx) - rx
        y = rand(rng) * (2ry) - ry
        (x / rx)^2 + (y / ry)^2 <= 1.0 || continue
        push!(pts, Point2f(cx + x, cy + y))
    end
    pts
end

function add_orientation_compass!(ax; origin=Point2f(-0.38, -0.48), len=0.11)
    cx, cy = origin
    lines!(ax, [Point2f(cx, cy - len), Point2f(cx, cy + len)], color=:black, linewidth=1.0)
    lines!(ax, [Point2f(cx - len, cy), Point2f(cx + len, cy)], color=:black, linewidth=1.0)
    scatter!(ax, [Point2f(cx, cy + len)], marker=:utriangle, color=:black, markersize=6)
    scatter!(ax, [Point2f(cx, cy - len)], marker=:dtriangle, color=:black, markersize=6)
    scatter!(ax, [Point2f(cx - len, cy)], marker=:ltriangle, color=:black, markersize=6)
    scatter!(ax, [Point2f(cx + len, cy)], marker=:rtriangle, color=:black, markersize=6)
    text!(ax, cx, cy + len + 0.04, text="A", align=(:center, :center), fontsize=9, font=:bold, color=:black)
    text!(ax, cx, cy - len - 0.04, text="P", align=(:center, :center), fontsize=9, font=:bold, color=:black)
    text!(ax, cx - len - 0.04, cy, text="D", align=(:center, :center), fontsize=9, font=:bold, color=:black)
    text!(ax, cx + len + 0.04, cy, text="V", align=(:center, :center), fontsize=9, font=:bold, color=:black)
end

function style_flatmap_axis!(ax)
    hidedecorations!(ax)
    ax.topspinevisible[] = true
    ax.rightspinevisible[] = true
    ax.leftspinevisible[] = false
    ax.bottomspinevisible[] = false
    ax.autolimitaspect[] = 1
    limits!(ax, -0.47, 0.47, -0.55, 0.35)
end

function plot_mock_flatmap_panel!(ax; hemisphere::Symbol, seed::Int)
    rng = MersenneTwister(seed)
    x, y, fields = make_flatmap_fields(hemisphere)

    style_flatmap_axis!(ax)

    for item in fields
        thr = quantile(vec(item.field), 0.84)
        img = item.field .- thr
        contourf!(ax, x, y, img; levels=[0.0, maximum(img)], colormap=[(item.color, 0.20)])
        contour!(ax, x, y, img; levels=[0.0], color=:black, linewidth=0.9)
        iy, ix = Tuple(argmax(item.field))
        text!(ax, x[ix], y[iy], text=item.name, color=:gray40, fontsize=8, align=(:center, :center))
    end

    xcenter = hemisphere == :left ? -0.08 : 0.08
    type_counts = Dict("ITi" => 60, "ITs" => 35, "ITc" => 25, "PT" => 18, "CT" => 10)
    for (typ, n) in type_counts
        pts = sample_points_in_ellipse(xcenter, -0.18, 0.27, 0.22, n, rng)
        scatter!(ax, pts; color=(TYPE_COLORS[typ], 0.80), markersize=2.0, marker=:circle, overdraw=true, transparency=true)
    end

    add_orientation_compass!(ax)
    ax.title = hemisphere == :left ? "Left flatmap" : "Right flatmap"
    ax.titlealign = :left
    ax.titlegap = 4
    ax.titlefont = :regular
    ax.titlesize = 11
end

function render_mock_figure()
    fig = Figure(size=(1100, 560), backgroundcolor=:white)
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])

    plot_mock_flatmap_panel!(ax1; hemisphere=:left, seed=42)
    plot_mock_flatmap_panel!(ax2; hemisphere=:right, seed=77)

    elems = [MarkerElement(marker=:circle, color=TYPE_COLORS[k], markersize=6) for k in ("ITi", "ITs", "ITc", "PT", "CT")]
    Legend(fig[2, 1:2], elems, ["ITi", "ITs", "ITc", "PT", "CT"]; orientation=:horizontal, tellwidth=false, framevisible=false)
    Label(fig[0, 1:2], "Gou-style flatmap mock (Julia, synthetic)", fontsize=13, font=:regular)

    png_out = joinpath(OUTDIR, "mock_flatmap_gou_style_jl.png")
    svg_out = joinpath(OUTDIR, "mock_flatmap_gou_style_jl.svg")
    save(png_out, fig, px_per_unit=2)
    save(svg_out, fig)
    println("Saved: ", png_out)
    println("Saved: ", svg_out)
end

render_mock_figure()
