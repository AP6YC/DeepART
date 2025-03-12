"""
    layers.jl
"""

@info """
\n####################################
###### LAYERS EXPERIMENT ######
####################################
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

@info "------- Loading dependencies -------"
using Revise

using DeepART
using DrWatson
using Plots
using Random

using StatsPlots
using DataFrames

@info "------- Loading definitions -------"
include("lib/lib.jl")

@info "------- Loading Hebb module -------"
import .Hebb

n_add = 128


outdir(args...) = DeepART.results_dir("layers", args...)


df = collect_results!(outdir())

perfs = groupby(df, :rng)

perfs2 = [copy(perfsi[!, :perf]) for perfsi in perfs]

# errorline(1:5, perfs2)
