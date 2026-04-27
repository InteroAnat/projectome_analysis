# Multi-monkey insula flatmap - adapted from gou_flatmap_minimal.jl.
# Generates one all-monkey combined plot (color by sample) and one
# 2x2 per-sample panel using the cached leftinsula flatmap.
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
function load_combined_somata(; path=COMBINED_XLSX)
    tbl = XLSX.readtable(path, "Summary")
    df  = DataFrame(tbl.data, vec(Symbol.(tbl.column_labels)))
    rename!(df, Symbol.(names(df)))

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

# Sample → distinctive color (color-blind safe-ish)
const SAMPLE_COLORS = Dict(
    "251637" => Makie.to_color(:steelblue),
    "252383" => Makie.to_color(:darkorange),
    "252384" => Makie.to_color(:forestgreen),
    "252385" => Makie.to_color(:firebrick),
)

# Side → marker
sample_marker(s) = s == "L" ? :circle : :utriangle

function run_multi_monkey_flatmap(; niter::Int=30000)
    # Inner `include()` calls inside zz2fig (e.g. monkeytemp/wyz-upload/...)
    # resolve relative to the cwd at call time, not the file path. Force cwd
    # to MONKEYREC_ROOT so those resolve correctly.
    cd(MONKEYREC_ROOT)

    tag = :leftinsula
    depth_cache = joinpath(CACHE_DIR, "depth_volume.jld2")
    flat_cache  = joinpath(CACHE_DIR, "flatmap_$(tag)_n$(niter).jld2")

    @info "Loading cached depth volume + flatmap..."
    depthimg, depthres = JLD2.load(depth_cache, "depthimg", "depthres")
    objflat, objphy = JLD2.load(flat_cache, "objflat", "objphy")
    objphy = hasproperty(objphy, :normals) ? objphy :
             GeometryBasics.normal_mesh(objphy.position, GeometryBasics.faces(objphy))

    @info "Loading combined somata..."
    df = load_combined_somata()
    @info "Total neurons in combined table: $(nrow(df))"
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
    counts = Dict(sid => sum(valid.SampleID .== sid) for sid in samples_present)
    side_counts = Dict(sid => (sum((valid.SampleID .== sid) .& (valid.original_side .== "L")),
                                sum((valid.SampleID .== sid) .& (valid.original_side .== "R")))
                       for sid in samples_present)

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
            color = get(SAMPLE_COLORS, sid, Makie.to_color(:gray))
            mk = sample_marker(side)
            sc = scatter!(ax_all, sub.somauv,
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
    n_251637 = get(counts, "251637", 0)
    n_252383 = get(counts, "252383", 0)
    n_252384 = get(counts, "252384", 0)
    n_252385 = get(counts, "252385", 0)
    Label(fig_all[0, :],
          "Combined: n=$n_total  (251637=$n_251637 + 252383=$n_252383 + 252384=$n_252384 + 252385=$n_252385)",
          fontsize=12, font=:bold)
    save(joinpath(MM_OUT_DIR, "flatmap_all_monkeys_combined.png"), fig_all;
         px_per_unit=3, backend=CairoMakie)
    save(joinpath(MM_OUT_DIR, "flatmap_all_monkeys_combined.svg"), fig_all;
         backend=CairoMakie)
    @info "Saved combined-all flatmap"

    # ── Figure 2: per-sample 2x2 panel ──────────────────────────────
    fig_per = Figure(size=(1200, 1100), backgroundcolor=:white)
    sample_layout = [("251637", 1, 1),
                     ("252383", 1, 2),
                     ("252384", 2, 1),
                     ("252385", 2, 2)]
    for (sid, row, col) in sample_layout
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
    Label(fig_per[0, :], "Per-sample insula somata", fontsize=15, font=:bold)
    save(joinpath(MM_OUT_DIR, "flatmap_per_monkey_panels.png"), fig_per;
         px_per_unit=3, backend=CairoMakie)
    save(joinpath(MM_OUT_DIR, "flatmap_per_monkey_panels.svg"), fig_per;
         backend=CairoMakie)
    @info "Saved per-sample panels"

    @info "Done. Output in $MM_OUT_DIR"
    valid
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_multi_monkey_flatmap()
end
