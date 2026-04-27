# gou_analysis_minimal.jl
# Transferable analysis methods from Gou et al. 2025
#
# This file intentionally reuses Gou functions directly where possible.
# It depends on `gou_flatmap_minimal.jl`, which loads monkeyrec and patches atlas paths.

include(raw"D:\projectome_analysis\gou_julia_scripts\gou_flatmap_minimal.jl")

# -------------------------
# Re-export core flatmap APIs
# -------------------------
export load_somata, build_flatmap, run_pipeline, run_insula_pipeline
export get_insula_highlights, add_insula_labels!, insula_axis_limits

# -------------------------
# Transferable analysis wrappers (direct Gou reuse)
# -------------------------
export gou_prop_test
export gou_calc_twoside_projection!
export gou_calc_ipsicontra_len
export gou_group_into_inj_sites
export gou_calc_common_root_path
export gou_calc_dist_mat, gou_load_dist_mat
export gou_cluster_arbor_whist
export gou_loadhclust!
export gou_draw_neurons_rawxyz!
export gou_draw_pathways
export gou_neurite_length

"""
    gou_prop_test(x, n, p0)

Direct wrapper of Gou's `zz1data.prop_test`.
Use for proportion tests in laterality / projection asymmetry analysis.
"""
gou_prop_test(x, n, p0) = zz1data.prop_test(x, n, p0)

"""
    gou_calc_twoside_projection!(df)

Direct wrapper of Gou's `zz1data.calc_twoside_projection!`.
Adds merged two-side projection columns to DataFrame in-place.
"""
gou_calc_twoside_projection!(df) = zz1data.calc_twoside_projection!(df)

"""
    gou_calc_ipsicontra_len(dfx, arborx)

Direct wrapper of Gou's `zz1data.calc_ipsicontra_len`.
Requires SWC/arbor data (`Gapr.SwcTable`) and is SWC-dependent.
"""
gou_calc_ipsicontra_len(dfx, arborx) = zz1data.calc_ipsicontra_len(dfx, arborx)

"""
    gou_group_into_inj_sites(df)

Direct wrapper of Gou's `group_into_inj_sites` from `2-analysis.jl`.
Groups neurons into curated injection-site clusters.
"""
gou_group_into_inj_sites(df) = group_into_inj_sites(df)

"""
    gou_calc_common_root_path(paths)

Direct wrapper of Gou's `calc_common_root_path`.
Finds common root path for a list of SWC paths.
"""
gou_calc_common_root_path(paths) = calc_common_root_path(paths)

"""
    gou_calc_dist_mat(df_or_pair, dfdist, matdist; on=:path)

Direct wrapper of Gou's `calc_dist_mat` from `2-analysis.jl`.
Builds pairwise distance matrix by matching path keys.
"""
gou_calc_dist_mat(df_or_pair, dfdist, matdist; on=:path) = calc_dist_mat(df_or_pair, dfdist, matdist; on)

"""
    gou_load_dist_mat(dfx, path; on=:path)

Direct wrapper of Gou's `load_dist_mat` from `2-analysis.jl`.
Loads precomputed JLD2 distance matrix and maps it to `dfx`.
"""
gou_load_dist_mat(dfx, path; on=:path) = load_dist_mat(dfx, path; on)

"""
    gou_cluster_arbor_whist(arbors, df)

Direct wrapper of Gou's `cluster_arbor_whist`.
Performs depth-histogram-based clustering of arbor features.
"""
gou_cluster_arbor_whist(arbors, df) = cluster_arbor_whist(arbors, df)

"""
    gou_loadhclust!(df, path)

Direct wrapper of Gou's `loadhclust!`.
Loads hierarchical clustering labels into DataFrame.
"""
gou_loadhclust!(df, path) = loadhclust!(df, path)

"""
    gou_draw_neurons_rawxyz!(axspec, df, rawinfo)

Direct wrapper of Gou's `draw_neurons_rawxyz!` from `zz2fig`.
Draws neurons in raw XYZ space; SWC-dependent.
"""
gou_draw_neurons_rawxyz!(axspec, df, rawinfo) = zz2fig.draw_neurons_rawxyz!(axspec, df, rawinfo)

"""
    gou_draw_pathways(df, axspecs...; kwargs...)

Direct wrapper of Gou's `draw_pathways` from `zz2fig`.
Draws pathway trajectories on requested axes.
"""
function gou_draw_pathways(df, axspecs...; kwargs...)
    zz2fig.draw_pathways(df, axspecs...; kwargs...)
end

"""
    gou_neurite_length(swc)

Direct wrapper of Gou's `Gapr.neurite_length` from `Gapr/src/swc.jl`.
Computes total neurite cable length from SWC.
"""
gou_neurite_length(swc) = Gapr.neurite_length(swc)

# -------------------------
# Lightweight utilities for local workflow (non-GPU)
# -------------------------
export is_projcol_ic
export laterality_index
export summarize_laterality

"""
    is_projcol_ic(col)

Returns true for ipsi/contra projection columns named like `I|...` or `C|...`.
"""
is_projcol_ic(col) = startswith(String(col), "I|") || startswith(String(col), "C|")

"""
    laterality_index(ipsi, contra)

Compute LI = (ipsi - contra) / (ipsi + contra). Returns NaN for 0/0.
"""
laterality_index(ipsi, contra) = (ipsi - contra) / (ipsi + contra)

"""
    summarize_laterality(df, ipsi_cols, contra_cols)

Aggregate ipsi/contra columns and summarize LI distribution.
Returns DataFrame with mean/std/min/max LI.
"""
function summarize_laterality(df, ipsi_cols, contra_cols)
    lis = map(eachrow(df)) do r
        ip = sum(Float64(r[c]) for c in ipsi_cols)
        co = sum(Float64(r[c]) for c in contra_cols)
        laterality_index(ip, co)
    end
    DataFrame(;
        mean_LI=mean(lis),
        std_LI=std(lis),
        min_LI=minimum(lis),
        max_LI=maximum(lis),
    )
end

# -------------------------
# Explicitly mark non-portable Gou methods
# -------------------------
# `classify_itptct` in 2-analysis.jl requires Flux GPU (`|> Flux.gpu`) and is NOT exported here.
# `rnn_factory` in 1-rnn.jl requires Flux training and is NOT exported here.
