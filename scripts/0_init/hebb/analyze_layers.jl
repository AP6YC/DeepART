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

perfs2 = [Vector{Float32}(perfsi[!, :perf]) for perfsi in perfs]

# n_perfs = length(perfs2)
perfs_plot = reduce(hcat, perfs2)'
n_perfs = size(perfs_plot)[2]
# p = errorline(1:n_perfs, perfs_plot)
p = errorline(
    1:n_perfs,
    perfs_plot,
    errorstyle=:stick,
    label="Stick",
    secondarycolor=:matched,
)

DeepART.saveplot(p, "layers", ["hebb_dist_analyze", "layers"], paper=true, extension=".png")
