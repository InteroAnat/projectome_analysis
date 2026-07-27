# Multi-monkey insula flatmap - adapted from gou_flatmap_minimal.jl.
# Generates one all-monkey combined plot (color by sample) and one
# adaptive per-sample panel grid using the cached leftinsula flatmap.
# Samples / colors / layout are discovered from the soma table at runtime.
# Reuses depth_volume.jld2 + flatmap_leftinsula_n30000.jld2 caches.

using CairoMakie
using Statistics

include(raw"D:\projectome_analysis\gou_julia_scripts\gou_flatmap_minimal.jl")

const COMBINED_XLSX = raw"D:\projectome_analysis\group_analysis\combined\multi_monkey_INS_combined.xlsx"
const MM_OUT_DIR    = raw"D:\projectome_analysis\group_analysis\R_analysis\outputs\figures\flatmap"
mkpath(MM_OUT_DIR)

# ── Sample-aware soma loader ─────────────────────────────────────
# Reads multi_monkey_INS_combined.xlsx Summary sheet and produces a DataFrame
# matching the one expected by xyz2uvw + plot_flatmap_soma!.
function load_combined_somata(; path=COMBINED_XLSX, sheet="Summary")
    df = _read_soma_table(path, sheet)

    # SampleID may be Int or String depending on cell formatting; normalize
    df.SampleID = string.(df.SampleID)

    # Use Soma_Side_Final if present, else Soma_Side
    side_col = :Soma_Side_Final in propertynames(df) ? df.Soma_Side_Final : df.Soma_Side
    df.original_side = map(side_col) do s
        s isa AbstractString ? strip(uppercase(String(s))) : "U"
    end

    # Use Soma_Region_Final if present, else Soma_Region
    region_col = :Soma_Region_Final in propertynames(df) ? df.Soma_Region_Final :
                 (:Soma_Region_Refined in propertynames(df) ? df.Soma_Region_Refined : df.Soma_Region)
    df.region_label = map(region_col) do r
        r isa AbstractString ? String(r) : ""
    end

    soma_pos = map(eachrow(df)) do r
        xyz = SVector{3,Float32}(Float32(r.Soma_Phys_X),
                                  Float32(r.Soma_Phys_Y),
                                  Float32(r.Soma_Phys_Z))
        # Mirror right-hemifield neurons to left for plotting on the leftinsula flatmap
        if r.original_side == "R" || !NmtDat.isleft(xyz)
            SVector{3,Float32}(Tuple(NmtDat.mirror(xyz)))
        else
            xyz
        end
    end
    insertcols!(df, :soma_pos => soma_pos)
    insertcols!(df, :typ => string.(df.Neuron_Type))
    insertcols!(df, :qroot => fill(false, nrow(df)))
    df
end

# Preferred colors for the original four (visual continuity with Apr 2026 plots).
const SAMPLE_COLOR_PREFERRED = Dict(
    "251637" => :steelblue,
    "252383" => :darkorange,
    "252384" => :forestgreen,
    "252385" => :firebrick,
)

# Extension palette for new / future SampleIDs (deterministic by sorted ID).
const SAMPLE_COLOR_POOL = [
    :steelblue, :darkorange, :forestgreen, :firebrick,
    :mediumpurple, :teal, :goldenrod, :deeppink,
    :sienna, :dodgerblue, :olivedrab, :crimson,
]

"""Stable SampleID → color map from whatever IDs are present in the table."""
function sample_color_map(sample_ids)
    ids = sort(unique(string.(sample_ids)))
    used = Set{Symbol}()
    cmap = Dict{String,Any}()

    for sid in ids
        if haskey(SAMPLE_COLOR_PREFERRED, sid)
            sym = SAMPLE_COLOR_PREFERRED[sid]
            cmap[sid] = Makie.to_color(sym)
            push!(used, sym)
        end
    end

    pool_i = 1
    for sid in ids
        haskey(cmap, sid) && continue
        while pool_i <= length(SAMPLE_COLOR_POOL) && SAMPLE_COLOR_POOL[pool_i] in used
            pool_i += 1
        end
        if pool_i <= length(SAMPLE_COLOR_POOL)
            sym = SAMPLE_COLOR_POOL[pool_i]
            pool_i += 1
            push!(used, sym)
            cmap[sid] = Makie.to_color(sym)
        else
            # Exhausted named pool: deterministic HSV from sorted index
            h = ((findfirst(==(sid), ids) - 1) * 0.6180339887) % 1.0
            cmap[sid] = Makie.to_color(Makie.HSVA(h * 360, 0.65, 0.80, 1.0))
        end
    end
    return cmap
end

"""Adaptive panel grid: ncols = ceil(sqrt(N)), nrows = ceil(N / ncols)."""
function sample_panel_grid(n::Int)
    n <= 0 && return (0, 0)
    ncols = ceil(Int, sqrt(n))
    nrows = ceil(Int, n / ncols)
    return nrows, ncols
end

# Side → marker
sample_marker(s) = s == "L" ? :circle : :utriangle

function run_multi_monkey_flatmap(; niter::Int=30000,
                                    soma_table::AbstractString=COMBINED_XLSX,
                                    sheet::AbstractString="Summary",
                                    out_dir::AbstractString=MM_OUT_DIR,
                                    cache_dir::AbstractString=CACHE_DIR)
    # Inner `include()` calls inside zz2fig (e.g. monkeytemp/wyz-upload/...)
    # resolve relative to the cwd at call time, not the file path. Force cwd
    # to MONKEYREC_ROOT so those resolve correctly.
    cd(MONKEYREC_ROOT)

    tag = :leftinsula
    depth_cache = joinpath(cache_dir, "depth_volume.jld2")
    flat_cache  = joinpath(cache_dir, "flatmap_$(tag)_n$(niter).jld2")
    mkpath(out_dir)

    @info "Loading cached depth volume + flatmap..." depth=depth_cache flat=flat_cache
    isfile(depth_cache) || error("Missing depth cache: $depth_cache (run --mode single once to build)")
    isfile(flat_cache)  || error("Missing flatmap cache: $flat_cache (run --mode single once to build)")
    depthimg, depthres = JLD2.load(depth_cache, "depthimg", "depthres")
    objflat, objphy = JLD2.load(flat_cache, "objflat", "objphy")
    objphy = hasproperty(objphy, :normals) ? objphy :
             GeometryBasics.normal_mesh(objphy.position, GeometryBasics.faces(objphy))

    @info "Loading combined somata..." path=soma_table sheet=sheet
    df = load_combined_somata(; path=soma_table, sheet=sheet)
    samples_all = sort(unique(df.SampleID))
    @info "Total neurons in combined table: $(nrow(df))"
    @info "Samples discovered (n=$(length(samples_all))): $(join(samples_all, ", "))"
    @info "Per-sample counts: $(combine(groupby(df, :SampleID), nrow))"

    @info "Projecting somata via xyz2uvw..."
    uvw = xyz2uvw(df.soma_pos, depthimg, depthres, objflat, objphy)
    insertcols!(df,
        :somauv => map(v -> SVector(v[1:end-1]...,), uvw),
        :somaw  => last.(uvw))

    valid = filter(r -> !any(isnan, r.somauv), df)
    @info "Valid projections: $(nrow(valid)) / $(nrow(df))"

    highlights = get_insula_highlights()
    lims = insula_axis_limits(objflat, objphy; pad=0.20)

    # ── Figure 1: ALL monkeys, color = sample, marker = side ────────
    samples_present = sort(unique(valid.SampleID))
    colors = sample_color_map(samples_present)
    counts = Dict(sid => sum(valid.SampleID .== sid) for sid in samples_present)
    @info "Color map" pairs=["$sid => $(colors[sid])" for sid in samples_present]

    fig_all = Figure(size=(700, 700), backgroundcolor=:white)
    ax_all = Axis(fig_all[1, 1],
                  title="Multi-monkey insula somata on combined flatmap",
                  titlesize=14)
    zz2fig.plot_flatmap_base!(ax_all, highlights, objflat, objphy; label=false)
    add_insula_labels!(ax_all, objflat, objphy)

    legend_handles = []
    legend_labels = String[]
    for sid in samples_present
        for side in ["L", "R"]
            sub = filter(r -> r.SampleID == sid && r.original_side == side &&
                              !any(isnan, r.somauv), valid)
            nrow(sub) == 0 && continue
            color = colors[sid]
            mk = sample_marker(side)
            scatter!(ax_all, sub.somauv,
                     color=color, marker=mk,
                     markersize=8, strokecolor=:black, strokewidth=0.4)
            push!(legend_handles,
                  MarkerElement(color=color, marker=mk, markersize=10,
                                strokecolor=:black, strokewidth=0.4))
            push!(legend_labels, "$(sid) $(side) (n=$(nrow(sub)))")
        end
    end
    limits!(ax_all, lims...)
    Legend(fig_all[1, 2], legend_handles, legend_labels;
           framevisible=true, labelsize=10)
    n_total = nrow(valid)
    count_str = join(["$sid=$(counts[sid])" for sid in samples_present], " + ")
    Label(fig_all[0, :],
          "Combined: n=$n_total  ($count_str)",
          fontsize=12, font=:bold)
    save(joinpath(out_dir, "flatmap_all_monkeys_combined.png"), fig_all;
         px_per_unit=3, backend=CairoMakie)
    save(joinpath(out_dir, "flatmap_all_monkeys_combined.svg"), fig_all;
         backend=CairoMakie)
    @info "Saved combined-all flatmap" n_samples=length(samples_present)

    # ── Figure 2: per-sample adaptive panel grid ────────────────────
    n_samp = length(samples_present)
    nrows, ncols = sample_panel_grid(n_samp)
    fig_w = max(600, 500 * ncols)
    fig_h = max(550, 500 * nrows)
    fig_per = Figure(size=(fig_w, fig_h), backgroundcolor=:white)
    for (i, sid) in enumerate(samples_present)
        row = div(i - 1, ncols) + 1
        col = mod(i - 1, ncols) + 1
        sub = filter(r -> r.SampleID == sid && !any(isnan, r.somauv), valid)
        nL = sum(sub.original_side .== "L")
        nR = sum(sub.original_side .== "R")
        ax = Axis(fig_per[row, col],
                  title="Sample $sid (n=$(nrow(sub))  L=$nL  R=$nR)",
                  titlesize=13)
        zz2fig.plot_flatmap_base!(ax, highlights, objflat, objphy; label=false)
        add_insula_labels!(ax, objflat, objphy)
        for side in ["L", "R"]
            sub_side = filter(r -> r.original_side == side, sub)
            nrow(sub_side) == 0 && continue
            color = side == "L" ? Makie.to_color(:steelblue) : Makie.to_color(:firebrick)
            scatter!(ax, sub_side.somauv,
                     color=color, marker=sample_marker(side),
                     markersize=8, strokecolor=:black, strokewidth=0.4)
        end
        limits!(ax, lims...)
    end
    Label(fig_per[0, :],
          "Per-sample insula somata (n=$n_samp: $(join(samples_present, ", ")))",
          fontsize=15, font=:bold)
    save(joinpath(out_dir, "flatmap_per_monkey_panels.png"), fig_per;
         px_per_unit=3, backend=CairoMakie)
    save(joinpath(out_dir, "flatmap_per_monkey_panels.svg"), fig_per;
         backend=CairoMakie)
    @info "Saved per-sample panels" grid="$(nrows)×$(ncols)" n_samples=n_samp

    @info "Done. Output in $out_dir"
    valid
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_multi_monkey_flatmap()
end
