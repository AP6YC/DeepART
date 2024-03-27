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

n_datasets = length(unique(perf_df.dataset))
n_models = length(unique(perf_df.m))
# n_scenarios = 2
scenarios = ["task-incremental", "task-homogeneous"]

out_mat = zeros(n_models*2, n_datasets)
for ix = 1:n_datasets
    for jx = 1:n_models
        for kx in eachindex(scenarios)
            local_scenario = scenarios[kx]
            out_mat[jx, ix, kx] = mean(perf_df[(perf_df.dataset .== ix) .& (perf_df.m .== jx) .& (perf_df.scenario .== local_scenario), :perf])
            out_mat[jx+n_models, ix] = std(perf_df[(perf_df.dataset .== ix) .& (perf_df.m .== jx) .& (perf_df.scenario .== local_scenario), :perf])
        end
        # end
        # out_mat[jx, ix] = mean(perf_df[(perf_df.dataset .== ix) .& (perf_df.m .== jx), :perf])
        # out_mat[jx+n_models, ix] = std(perf_df[(perf_df.dataset .== ix) .& (perf_df.m .== jx), :perf])
    end
end

table = latexify(perf_df, env=:table)
