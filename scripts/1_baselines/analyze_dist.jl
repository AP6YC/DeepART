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

# Get the sizes of the relevant elements in the dataframe
n_datasets = length(unique(perf_df.dataset))
n_models = length(unique(perf_df.m))
scenarios = ["task-incremental", "task-homogeneous"]
n_scenarios = length(scenarios)

# Compute the means and standard deviations of the final testing performances
out_mat = zeros(n_models, n_datasets, n_scenarios)
for ix = 1:n_datasets
    for jx = 1:n_models
        for kx in eachindex(scenarios)
            local_scenario = scenarios[kx]
            out_mat[jx, ix, kx] = mean(df[(df.dataset .== ix) .& (df.m .== jx) .& (df.scenario .== local_scenario), :perf])
            out_mat[jx+n_models, ix] = std(df[(df.dataset .== ix) .& (df.m .== jx) .& (df.scenario .== local_scenario), :perf])
        end
    end
end

# table = latexify(perf_df, env=:table)

# Compute the means and standard deviations of the confusion matrix for each experiment
# confusion_means = zeros(n_models*2, n_datasets)
# confusion_stds = zeros(n_models*2, n_datasets)
confs = zeros(n_models, n_dataset, n_scenarios)
for ix = 1:n_datasets
    for jx = 1:n_models
        for kx in eachindex(scenarios)
            local_scenario = scenarios[kx]
            confusion_matrix = df[(df.dataset .== ix) .& (df.m .== jx) .& (df.scenario .== local_scenario), :confusion_matrix]
            confusion_means[jx, ix, kx] = mean(confusion_matrix)
            confusion_stds[jx, ix, kx] = std(confusion_matrix)
        end
    end
end
