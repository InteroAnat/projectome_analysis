using Random
using DataFrames
using StaticArrays
using DelimitedFiles
using GIFTI
using CairoMakie
using Meshes

const MONKEYREC_ROOT = raw"D:\projectome_analysis\references\analysis-code_gou_etal_2025\monkeyrec"
const LOCAL_NMT_ROOT = raw"D:\projectome_analysis\atlas\NMT_v2.0_sym"
const LOCAL_SURF_ROOT = joinpath(LOCAL_NMT_ROOT, "NMT_v2.0_sym_surfaces")
const SOMA_TABLE = raw"D:\projectome_analysis\neuron_tables\251637_INS_HE.xlsx"
const OUTPUT_ROOT = raw"D:\projectome_analysis\figures_charts\gou_flatmap_conservative"
const BASE_OUTDIR = joinpath(OUTPUT_ROOT, "base_only")
const SOMA_OUTDIR = joinpath(OUTPUT_ROOT, "soma_projection")
const DOCS_OUTDIR = joinpath(OUTPUT_ROOT, "docs")
const CACHE_OUTDIR = joinpath(OUTPUT_ROOT, "cache")
const TEST_OUTDIR = joinpath(OUTPUT_ROOT, "tests")
mkpath(BASE_OUTDIR)
mkpath(SOMA_OUTDIR)
mkpath(DOCS_OUTDIR)
mkpath(CACHE_OUTDIR)
mkpath(TEST_OUTDIR)

cd(MONKEYREC_ROOT)
include(joinpath(MONKEYREC_ROOT, "monkeyrec.jl"))

if !isdefined(zz1data, :xxxcached_result)
    @eval zz1data xxxcached_result = Dict{Tuple,Any}()
end

# Conservative patch: keep Gou's code path, only swap atlas file paths.
@eval NmtDat begin
    function load_images()
        root = raw"D:\projectome_analysis\atlas\NMT_v2.0_sym"
        nmt_dir = joinpath(root, "NMT_v2.0_sym")
        nmr = NmtDat.NIfTI.niread(joinpath(nmt_dir, "NMT_v2.0_sym_SS.nii.gz")).raw
        cb_img = NmtDat.NIfTI.niread(joinpath(nmt_dir, "supplemental_masks", "NMT_v2.0_sym_cerebellum_mask.nii.gz")).raw .> 0
        charm = reshape(NmtDat.NIfTI.niread(joinpath(nmt_dir, "CHARM_in_NMT_v2.0_sym.nii.gz")).raw, size(nmr)..., :)
        sarm = reshape(NmtDat.NIfTI.niread(joinpath(nmt_dir, "SARM_in_NMT_v2.0_sym.nii.gz")).raw, size(nmr)..., :)
        seg = NmtDat.NIfTI.niread(joinpath(nmt_dir, "NMT_v2.0_sym_segmentation.nii.gz")).raw
        sarm_key = DelimitedFiles.readdlm(joinpath(root, "tables_SARM", "SARM_key_all.txt"), '\t', header=true)[1]
        charm_key = DelimitedFiles.readdlm(joinpath(root, "tables_CHARM", "CHARM_key_all.txt"), '\t', header=true)[1]
        res = SVector(250.0f0, 250.0f0, 250.0f0)
        tag2tbl = map((:sarm => sarm_key, :charm => charm_key)) do (tag, tbl)
            tbl = map(eachrow(tbl)) do r
                r[1] => r[2]
            end |> Dict
            tag => tbl
        end |> NamedTuple
        (nmr=nmr, cb_msk=cb_img, charm=charm, charm_key=charm_key, sarm=sarm, sarm_key=sarm_key, seg=seg,
            res=res, range=SVector(res .* size(nmr)...), tag2tbl)
    end
end

function local_flatmap_cfg(tag::Symbol, obj_ref)
    (
        test=(;
            clipfn=(obj) -> begin
                vmax = findmin(p -> -p[2], obj.position)[2]
                dist1 = dijkstra_vertex(obj, vmax)
                (v) -> dist1[v][1] < 40000
            end,
            rightfn=p -> p[1],
        ),
        lefths=(;
            clipfn=(obj1) -> begin
                nmt = NmtDat.load_images()
                charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res), CartesianIndices(nmt.seg)[nmt.charm[:, :, :, 1] .> 0])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj_ref[], findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 10000
            end,
            rightfn=p -> p[2],
        ),
        leftfrontal=(;
            clipfn=(obj1) -> begin
                nmt = NmtDat.load_images()
                charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res), CartesianIndices(nmt.seg)[nmt.charm[:, :, :, 1] .== 1])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj_ref[], findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 60000
            end,
            rightfn=p -> p[1],
        ),
    )[tag]
end

surface_filename(kind::Symbol) = Dict(
    :gray => "lh.gray_surface.rsl.gii",
    :mid => "lh.mid_surface.rsl.gii",
    :white => "lh.white_surface.rsl.gii",
)[kind]

function load_local_surface(kind::Symbol; refined::Bool=true)
    surf = GIFTI.load(joinpath(LOCAL_SURF_ROOT, surface_filename(kind)))
    obj = fix_mesh_object(surf)
    refined ? refine_mesh(obj) : obj
end

function build_local_flatmap_bundle(; tag::Symbol=:leftfrontal, niter::Int=30000, refined::Bool=true)
    obj_mid = load_local_surface(:mid; refined=refined)
    obj_gray = load_local_surface(:gray; refined=refined)
    obj_white = load_local_surface(:white; refined=refined)
    obj_ref = Ref{Any}(nothing)
    obj_ref[] = obj_mid
    cfg = local_flatmap_cfg(tag, obj_ref)
    keepfn = cfg.clipfn(obj_mid)
    objphy0 = clip_mesh(keepfn, obj_mid)
    objgray0 = clip_mesh(keepfn, obj_gray)
    objwhite0 = clip_mesh(keepfn, obj_white)
    objphy = GeometryBasics.normal_mesh(objphy0.position, GeometryBasics.faces(objphy0))
    objgray = GeometryBasics.normal_mesh(objgray0.position, GeometryBasics.faces(objgray0))
    objwhite = GeometryBasics.normal_mesh(objwhite0.position, GeometryBasics.faces(objwhite0))
    vmax = findmin(cfg.rightfn, objphy.position)[2]
    objflat = flatten_mesh(objphy, vmax; niter=niter)
    return (; objflat, objphy, objgray, objwhite)
end

flatmap_cache_path(tag::Symbol, niter::Int; refined::Bool=true) = joinpath(
    CACHE_OUTDIR,
    "flatmap_$(tag)_n$(niter)_$(refined ? "refined" : "raw").jld2",
)

function load_or_build_flatmap_bundle(; tag::Symbol=:leftfrontal, niter::Int=30000, refined::Bool=true)
    cache = flatmap_cache_path(tag, niter; refined)
    if isfile(cache)
        objflat, objphy, objgray, objwhite = JLD2.load(cache, "objflat", "objphy", "objgray", "objwhite")
        return (; objflat, objphy, objgray, objwhite)
    end
    bundle = build_local_flatmap_bundle(; tag, niter, refined)
    JLD2.jldsave(cache; bundle...)
    bundle
end

function load_or_build_flatmap(; tag::Symbol=:leftfrontal, niter::Int=30000, refined::Bool=true)
    bundle = load_or_build_flatmap_bundle(; tag, niter, refined)
    bundle.objflat, bundle.objphy
end

function sample_uv_xyz_pairs(objflat, objphy; perface::Int=64)
    uv_samples = SVector{2,Float32}[]
    xyz_samples = SVector{3,Float32}[]
    sample_mesh(objflat, objflat.position, objphy.position; perface=perface) do vec
        push!(uv_samples, SVector{2,Float32}(vec[1], vec[2]))
        push!(xyz_samples, SVector{3,Float32}(vec[3], vec[4], vec[5]))
    end
    uv_samples, xyz_samples
end

function sample_uv_column_pairs(objflat, objphy, objgray, objwhite; perface::Int=64)
    uv_samples = SVector{2,Float32}[]
    mid_samples = SVector{3,Float32}[]
    gray_samples = SVector{3,Float32}[]
    white_samples = SVector{3,Float32}[]
    center_samples = SVector{3,Float32}[]
    sample_mesh(objflat, objflat.position, objphy.position, objgray.position, objwhite.position; perface=perface) do vec
        push!(uv_samples, SVector{2,Float32}(vec[1], vec[2]))
        mid = SVector{3,Float32}(vec[3], vec[4], vec[5])
        gray = SVector{3,Float32}(vec[6], vec[7], vec[8])
        white = SVector{3,Float32}(vec[9], vec[10], vec[11])
        push!(mid_samples, mid)
        push!(gray_samples, gray)
        push!(white_samples, white)
        push!(center_samples, (gray + white) / 2)
    end
    (; uv_samples, mid_samples, gray_samples, white_samples, center_samples)
end

function load_soma_table(; path::AbstractString=SOMA_TABLE)
    tbl = XLSX.readtable(path, 1)
    df = DataFrame(tbl.data, vec(Symbol.(tbl.column_labels)))
    rename!(df, Symbol.(names(df)))
    df
end

function mirror_to_left_hemisphere(xyz::SVector{3,Float32}, side)
    side_norm = strip(uppercase(String(side)))
    if side_norm == "R" || (!ismissing(side) && !NmtDat.isleft(xyz))
        SVector{3,Float32}(Tuple(NmtDat.mirror(xyz)))
    else
        xyz
    end
end

function normalize_soma_region(region)
    r = uppercase(strip(String(region)))
    r = replace(r, r"^(CL|CR|L|R)[_\-]" => "")
    r
end

function insula_hierarchy(region)
    r = normalize_soma_region(region)
    if startswith(r, "IG")
        "Granular"
    elseif startswith(r, "ID")
        "Dysgranular"
    elseif startswith(r, "IA") || startswith(r, "FI")
        "Agranular"
    else
        "Unknown"
    end
end

function point_to_segment_projection(x::SVector{3,Float32}, a::SVector{3,Float32}, b::SVector{3,Float32})
    ab = b - a
    denom = sum(ab .* ab)
    if denom <= eps(Float32)
        return (0.5f0, a, norm(x - a))
    end
    t = clamp(dot(x - a, ab) / denom, 0.0f0, 1.0f0)
    p = a + (ab * t)
    (Float32(t), p, norm(x - p))
end

function nearest_column_match(xyz::SVector{3,Float32}, tree, samples; candidates::Int=24)
    idxs, _ = NearestNeighbors.knn(tree, [xyz], candidates, true)
    best = nothing
    for idx in idxs[1]
        w, anchor, dist = point_to_segment_projection(xyz, samples.gray_samples[idx], samples.white_samples[idx])
        middist = norm(xyz - samples.mid_samples[idx])
        cand = (; idx, w, anchor, dist, middist)
        if best === nothing || cand.dist < best.dist
            best = cand
        end
    end
    best
end

function project_real_somas_to_flatmap(objflat, objphy; somafile::AbstractString=SOMA_TABLE, perface::Int=64, maxdist_um::Real=6000.0)
    src = load_soma_table(; path=somafile)
    uv_samples, xyz_samples = sample_uv_xyz_pairs(objflat, objphy; perface)
    tree = NearestNeighbors.KDTree(xyz_samples)

    rows = NamedTuple[]
    for r in eachrow(src)
        xyz = SVector{3,Float32}(Float32(r.Soma_Phys_X), Float32(r.Soma_Phys_Y), Float32(r.Soma_Phys_Z))
        xyz_left = mirror_to_left_hemisphere(xyz, r.Soma_Side)
        idxs, dists = NearestNeighbors.knn(tree, [xyz_left], 1, true)
        idx = idxs[1][1]
        dist = Float32(dists[1][1])
        dist > maxdist_um && continue
        push!(rows, (
            NeuronID=string(r.NeuronID),
            SampleID=string(r.SampleID),
            typ=string(r.Neuron_Type),
            soma_pos=xyz_left,
            somauv=uv_samples[idx],
            projdist_um=dist,
            qroot=false,
            source_side=string(r.Soma_Side),
            source_region=string(r.Soma_Region),
            source_region_norm=normalize_soma_region(r.Soma_Region),
            hierarchy=insula_hierarchy(r.Soma_Region),
        ))
    end
    DataFrame(rows)
end

function project_real_somas_to_flatmap_column(objflat, objphy, objgray, objwhite;
    somafile::AbstractString=SOMA_TABLE, perface::Int=64, maxdist_um::Real=6000.0, candidates::Int=24)
    src = load_soma_table(; path=somafile)
    samples = sample_uv_column_pairs(objflat, objphy, objgray, objwhite; perface)
    tree = NearestNeighbors.KDTree(samples.center_samples)
    rows = NamedTuple[]
    for r in eachrow(src)
        xyz = SVector{3,Float32}(Float32(r.Soma_Phys_X), Float32(r.Soma_Phys_Y), Float32(r.Soma_Phys_Z))
        xyz_left = mirror_to_left_hemisphere(xyz, r.Soma_Side)
        match = nearest_column_match(xyz_left, tree, samples; candidates)
        match.dist > maxdist_um && continue
        push!(rows, (
            NeuronID=string(r.NeuronID),
            SampleID=string(r.SampleID),
            typ=string(r.Neuron_Type),
            soma_pos=xyz_left,
            somauv=samples.uv_samples[match.idx],
            somaw=match.w,
            projdist_um=Float32(match.dist),
            midsurf_dist_um=Float32(match.middist),
            qroot=false,
            source_side=string(r.Soma_Side),
            source_region=string(r.Soma_Region),
            source_region_norm=normalize_soma_region(r.Soma_Region),
            hierarchy=insula_hierarchy(r.Soma_Region),
            projection_method="column_approx",
        ))
    end
    DataFrame(rows)
end

function apply_consistent_limits!(ax; tag::Symbol)
    if tag == :lefths
        limits!(ax, -0.45, 0.45, -0.55, 0.25)
    else
        limits!(ax, -0.45, 0.45, -0.55, 0.25)
    end
end

function hierarchy_color(h::AbstractString)
    Dict(
        "Agranular" => Makie.to_color("#C6DBEF"),
        "Dysgranular" => Makie.to_color("#6BAED6"),
        "Granular" => Makie.to_color("#08306B"),
        "Unknown" => Makie.to_color("#999999"),
    )[h]
end

function hierarchy_marker(h::AbstractString)
    Dict(
        "Agranular" => :circle,
        "Dysgranular" => :rect,
        "Granular" => :utriangle,
        "Unknown" => :diamond,
    )[h]
end

function select_diagnostic_samples(df; per_group::Int=3)
    keep = NamedTuple[]
    for grp in ("Agranular", "Dysgranular", "Granular", "Unknown")
        dfx = sort(filter(r -> r.hierarchy == grp, eachrow(df)), by=r -> r.projdist_um)
        for r in Iterators.take(dfx, per_group)
            push!(keep, NamedTuple(r))
        end
    end
    DataFrame(keep)
end

function add_hierarchy_annotations!(ax, df_all)
    for grp in ("Agranular", "Dysgranular", "Granular")
        dfx = filter(r -> r.hierarchy == grp, df_all)
        nrow(dfx) == 0 && continue
        uv = reduce(hcat, dfx.somauv)
        cent = SVector{2,Float32}(Float32(median(uv[1, :])), Float32(median(uv[2, :])))
        text!(ax, cent[1], cent[2], text=grp, color=hierarchy_color(grp), align=(:center, :center), fontsize=12, font=:bold)
    end
end

function render_hierarchy_diagnostic(; tag::Symbol=:lefths, niter::Int=30000, per_group::Int=3, somafile::AbstractString=SOMA_TABLE)
    bundle = load_or_build_flatmap_bundle(; tag, niter)
    objflat, objphy, objgray, objwhite = bundle.objflat, bundle.objphy, bundle.objgray, bundle.objwhite
    highlights = zz1data.get_flatmap_highlights()
    df_all = project_real_somas_to_flatmap_column(objflat, objphy, objgray, objwhite; somafile)
    nrow(df_all) == 0 && error("No somata passed projection filters for diagnostic rendering.")
    df_sample = select_diagnostic_samples(df_all; per_group)

    fig = Figure(size=(900, 520), backgroundcolor=:white)
    ax = Axis(fig[1, 1])
    cd(MONKEYREC_ROOT) do
        zz2fig.plot_flatmap_base!(ax, highlights, objflat, objphy; label=false)
    end
    apply_consistent_limits!(ax; tag)
    add_hierarchy_annotations!(ax, df_all)

    for r in eachrow(df_sample)
        scatter!(ax, [r.somauv], color=hierarchy_color(r.hierarchy), marker=hierarchy_marker(r.hierarchy),
            markersize=11, strokecolor=:black, strokewidth=0.8, transparency=true, overdraw=true)
        label = "$(replace(r.NeuronID, ".swc" => "")) $(r.source_region_norm)"
        text!(ax, r.somauv[1] + 0.012f0, r.somauv[2] + 0.01f0, text=label, color=:black, align=(:left, :center), fontsize=9)
    end

    Label(fig[0, 1], "Few-neuron soma diagnostic on $(tag) flatmap (column approximation)", fontsize=13)

    summary_lines = String[
        "Test goal: compare Soma_Region hierarchy with projected soma position",
        "Projection uses aligned gray-mid-white cortical columns, not raw nearest mid-surface",
        "Samples shown: $(nrow(df_sample)) neurons ($(per_group) per hierarchy when available)",
        "Hierarchy rule: IA*/FI -> Agranular; ID* -> Dysgranular; IG* -> Granular",
    ]
    Label(fig[2, 1], join(summary_lines, "\n"), tellwidth=false, fontsize=10, halign=:left)

    png_out = joinpath(TEST_OUTDIR, "diag_soma_hierarchy_column_samples_$(tag).png")
    svg_out = joinpath(TEST_OUTDIR, "diag_soma_hierarchy_column_samples_$(tag).svg")
    csv_out = joinpath(TEST_OUTDIR, "diag_soma_hierarchy_column_samples_$(tag).csv")
    md_out = joinpath(TEST_OUTDIR, "diag_soma_hierarchy_column_samples_$(tag).md")

    save(png_out, fig; px_per_unit=2, backend=CairoMakie)
    save(svg_out, fig; backend=CairoMakie)

    open(csv_out, "w") do io
        println(io, "NeuronID,source_region_norm,hierarchy,projdist_um,midsurf_dist_um,somaw,somauv_x,somauv_y")
        for r in eachrow(df_sample)
            println(io, "$(r.NeuronID),$(r.source_region_norm),$(r.hierarchy),$(r.projdist_um),$(r.midsurf_dist_um),$(r.somaw),$(r.somauv[1]),$(r.somauv[2])")
        end
    end

    counts = Dict(grp => nrow(filter(r -> r.hierarchy == grp, df_all)) for grp in ("Agranular", "Dysgranular", "Granular", "Unknown"))
    open(md_out, "w") do io
        println(io, "# Soma Hierarchy Diagnostic")
        println(io)
        println(io, "- Flatmap tag: `$(tag)`")
        println(io, "- Projected somata: `$(nrow(df_all))`")
        println(io, "- Sampling rule: lowest `projdist_um`, `$(per_group)` per hierarchy")
        println(io, "- Projection method: `column_approx` using aligned gray-mid-white surfaces")
        println(io, "- Known plotting issue found earlier: `plot_flatmap_soma!` resets axis limits, so wrapper enforces matching limits after plotting")
        println(io)
        println(io, "## Counts")
        println(io)
        for grp in ("Agranular", "Dysgranular", "Granular", "Unknown")
            println(io, "- `$(grp)`: `$(counts[grp])`")
        end
        println(io)
        println(io, "## Sampled neurons")
        println(io)
        for r in eachrow(df_sample)
            println(io, "- `$(r.NeuronID)`: `$(r.source_region_norm)` -> `$(r.hierarchy)`, `projdist_um=$(round(r.projdist_um, digits=1))`, `somaw=$(round(r.somaw, digits=3))`")
        end
    end

    println("Saved: ", png_out)
    println("Saved: ", svg_out)
    println("Saved: ", csv_out)
    println("Saved: ", md_out)
end

function render_base_only(; tag::Symbol=:leftfrontal, niter::Int=30000)
    objflat, objphy = load_or_build_flatmap(; tag, niter)
    highlights = zz1data.get_flatmap_highlights()

    fig = Figure(size=(650, 420), backgroundcolor=:white)
    ax = Axis(fig[1, 1])
    cd(MONKEYREC_ROOT) do
        zz2fig.plot_flatmap_base!(ax, highlights, objflat, objphy; label=(tag == :leftfrontal))
    end
    apply_consistent_limits!(ax; tag)
    Label(fig[0, 1], "Conservative Gou flatmap wrapper (base: $(tag))", fontsize=11)

    png_out = joinpath(BASE_OUTDIR, "flatmap_base_$(tag).png")
    svg_out = joinpath(BASE_OUTDIR, "flatmap_base_$(tag).svg")
    save(png_out, fig; px_per_unit=2, backend=CairoMakie)
    save(svg_out, fig; backend=CairoMakie)
    println("Saved: ", png_out)
    println("Saved: ", svg_out)
end

function render_real_soma_projection(; tag::Symbol=:lefths, niter::Int=30000, somafile::AbstractString=SOMA_TABLE, maxdist_um::Real=6000.0)
    bundle = load_or_build_flatmap_bundle(; tag, niter)
    objflat, objphy, objgray, objwhite = bundle.objflat, bundle.objphy, bundle.objgray, bundle.objwhite
    highlights = zz1data.get_flatmap_highlights()
    src = load_soma_table(; path=somafile)
    df = project_real_somas_to_flatmap_column(objflat, objphy, objgray, objwhite; somafile, maxdist_um=maxdist_um)
    nrow(df) == 0 && error("No somata passed projection filters for full rendering.")
    typcolor = zz2fig.typ2color_func(df)
    rowcolor = r -> typcolor(r.typ)

    fig = Figure(size=(650, 420), backgroundcolor=:white)
    ax = Axis(fig[1, 1])
    cd(MONKEYREC_ROOT) do
        zz2fig.plot_flatmap_base!(ax, highlights, objflat, objphy; label=false)
        zz2fig.plot_flatmap_soma!(ax, df; rowcolor)
    end
    apply_consistent_limits!(ax; tag)
    Label(fig[0, 1], "Conservative Gou flatmap wrapper (column-approx somata, mirrored to left)", fontsize=11)

    png_out = joinpath(SOMA_OUTDIR, "flatmap_real_somata_$(tag).png")
    svg_out = joinpath(SOMA_OUTDIR, "flatmap_real_somata_$(tag).svg")
    save(png_out, fig; px_per_unit=2, backend=CairoMakie)
    save(svg_out, fig; backend=CairoMakie)
    println("Projected somata: ", nrow(df), " / ", nrow(src))
    println("Saved: ", png_out)
    println("Saved: ", svg_out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    render_base_only(; tag=:leftfrontal)
    render_base_only(; tag=:lefths)
    render_real_soma_projection()
    render_hierarchy_diagnostic()
end
