"""
    analyze_dist.jl

# Description
This script takes the results of the Monte Carlo of shuffled simulations
and generates plots of their statistics.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using DrWatson
using Plots
using StatsBase
using Latexify

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Point to the local sweep data directory
sweep_dir = DeepART.results_dir(
    "1_baselines",
)

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

# Collect the results into a single dataframe
df = collect_results!(sweep_dir)


# perf_df = DataFrame( = n_F2, Total = n_categories)
perf_df = df[:, [:m, :dataset, :perf]]
table = latexify(perf_df, env=:table)