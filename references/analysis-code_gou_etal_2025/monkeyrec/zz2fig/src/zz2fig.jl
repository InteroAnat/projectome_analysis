module zz2fig

using ProgressMeter

using FileIO

using DataFrames
import Gapr

import RPRMakie
import CairoMakie
import GLMakie
using Makie

using MakieTeX
using LaTeXStrings

using Random
using StaticArrays
using LinearAlgebra

using GeometryBasics
using ColorTypes

import Images

using HypothesisTests
using StatsBase
using LinearRegression
import RCall

using zz1vis
import zz0atlas: NmtDat, Allen
using zz1data

export with_figure
include("zz2fig-base.jl")
include("2-plot.jl")

export plot_fig_recon
export plot_fig_compare2
include("zz2fig-recon.jl")

export plot_fig_result
export plot_result_depth_g97_dist
include("zz2fig-subtypes.jl")

export plot_fig_longproj
include("zz2fig-longproj.jl")

export plot_fig_intra_pfc
export plot_fig_intra_pfc_supp
include("zz2fig-intra.jl")

export plot_fig_it_contra_str
export plot_fig_it_contra_str_supp
export plot_fig_arbor_patches
include("zz2fig-itcs.jl")

export plot_fig_ctpt
export plot_fig_ctpt_supp
export plot_ctpt_cmp
include("zz2fig-ptct.jl")

export plot_fig_compare
export plot_fig_compare_supp
export plot_fig_compare_supp2
include("zz2fig-compare.jl")

export plot_rec_progress
export plot_all_injs
export plot_all_subtypes
export plot_find_arbors
include("zz2fig-extra.jl")

end # module zz2fig
