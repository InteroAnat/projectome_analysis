using ProgressMeter
using InteractiveUtils

using StaticArrays
using GeometryBasics
using ColorTypes

import Images
import Interpolations
import GIFTI
using FileIO
import XLSX

import DirectQhull
import GraphNeuralNetworks

import MarchingCubes
import Meshing
import Gmsh: gmsh

import RPRMakie
import CairoMakie
import GLMakie
import Makie
import LaTeXStrings
import MakieTeX

using Dates
import DataStructures
import Distributions
import StatsBase
import LinearRegression

import NetworkLayout

import NonparametricRegression

using SharedArrays

import Caching

using LinearAlgebra
using Statistics
using Random

using DataFrames
using DelimitedFiles
import HDF5

using Distributed

#import CUDA
import Flux

import JSON
import SHA

import JLD2

import CategoricalArrays

import Clustering

import NearestNeighbors

import HypothesisTests

import Graphs
import GraphMakie

import UMAP

import Optim

import PromptingTools

import Revise
using Jmake

using Gapr

Revise.includet("0-make.jl")
Revise.includet("0-meshes.jl")

Revise.includet("1-rnn.jl")
Revise.includet("1-nmr.jl")

### checked [
##push!(LOAD_PATH, @__DIR__) |> unique!
import zz0atlas.Allen
import zz0atlas.NmtDat
using zz1vis
using zz1data
using zz2fig
#### checked ]

Revise.includet("1-unibrain.jl")

Revise.includet("2-draw.jl")

Revise.includet("2-meshes.jl")
Revise.includet("2-analysis.jl")

