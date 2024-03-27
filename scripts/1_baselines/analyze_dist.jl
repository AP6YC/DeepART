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
using Printf
using DataStructures

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Point to the local sweep data directory
sweep_dir = DeepART.results_dir(
    "1_baselines",
)


EXP_TOP = "1_analyze"

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

# Collect the results into a single dataframe
df = collect_results!(sweep_dir)

# perf_df = DataFrame( = n_F2, Total = n_categories)
# perf_df = df[:, [:m, :dataset, :perf]]

# Get the sizes of the relevant elements in the dataframe
# datasets = unique(df.dataset)
datasets = OrderedDict(
    "mnist" => "MNIST",
    "fashionmnist" => "Fashion MNIST",
    "usps" => "USPS",
    "cifar10" => "CIFAR-10",
    "cifar100_coarse" => "CIFAR-100 (Coarse)",
    "cifar100_fine" => "CIFAR-100 (Fine)",
)
datasets_list = collect(keys(datasets))
models = Dict(
    "SFAM" => "SFAM",
    "DeepARTDense" => "MLP \\method",
    "DeepARTConv" => "CNN \\method",
    # "DeepARTDense" => "\\method (MLP)",
    # "DeepARTConv" => "\\method (CNN)",
    # "DeepARTDense" => "DeepART (MLP)",
    # "DeepARTConv" => "DeepART (CNN)",
)
models_list = collect(keys(models))
# models = unique(df.m)
# n_models = length(unique(perf_df.m))
# scenarios = unique(df.scenario)
scenarios = Dict(
    "task-homogenous" => "(TH)",
    "task-incremental" => "(TI)",
    # "task-homogenous" => "Task Homogeneous",
    # "task-incremental" => "Task Incremental",
)
scenarios_list = collect(keys(scenarios))

# Compute the means and standard deviations of the final testing performances
out_mat = zeros(n_models, n_datasets, n_scenarios, 2)
for ix in eachindex(datasets_list)
    for jx in eachindex(models_list)
        for kx in eachindex(scenarios_list)
            local_dataset = datasets_list[ix]
            local_model = models_list[jx]
            local_scenario = scenarios_list[kx]
            out_mat[jx, ix, kx, 1] = mean(df[(df.dataset .== local_dataset) .& (df.m .== local_model) .& (df.scenario .== local_scenario), :perf])
            out_mat[jx, ix, kx, 2] = std(df[(df.dataset .== local_dataset) .& (df.m .== local_model) .& (df.scenario .== local_scenario), :perf])
        end
    end
end
# for ix in collect(keys(datasets))
#     for jx in eachindex(models)
#         for kx in eachindex(scenarios)
#             local_dataset = datasets[ix]
#             local_model = models[jx]
#             local_scenario = scenarios[kx]
#                 # for mx = 1:2
#             out_mat[jx, ix, kx, 1] = mean(df[(df.dataset .== local_dataset) .& (df.m .== local_model) .& (df.scenario .== local_scenario), :perf])
#             out_mat[jx, ix, kx, 2] = std(df[(df.dataset .== local_dataset) .& (df.m .== local_model) .& (df.scenario .== local_scenario), :perf])
#         end
#     end
# end

out_mat

out_str = ""
for ix in eachindex(datasets_list)
    out_str *= "& $(datasets[datasets_list[ix]])"
end
out_str *= "\\\\ \n \\midrule \\\\ \n"
# & MNIST & Fashion MNIST & CIFAR-10 & CIFAR-100 (Fine) & CIFAR-100 (Coarse) & USPS \\
        # \midrule

for jx in eachindex(models_list)
    for kx in eachindex(scenarios_list)
        local_model = models[models_list[jx]] * " " * scenarios[scenarios_list[kx]]
        out_str *= "$(local_model)"
        for ix in eachindex(datasets_list)
            # out_str *= "\$ $(out_mat[jx, ix, kx, 1]) \\pm $(out_mat[jx, ix, kx, 2]) \$ & "
            # local_dataset  = datasets[datasets_list[ix]]

            out_str *= @sprintf "& \$ %.4f \\pm %.4f \$" out_mat[jx, ix, kx, 1] out_mat[jx, ix, kx, 1]
        end
        out_str *= "\\\\ \n"
    end
    # out_str *= "\\\\ \n"
    out_str *= "\\addlinespace \n \\hline \n \\addlinespace \n"
end

@info out_str
paper_out_dir(args...) = DeepART.paper_results_dir(EXP_TOP, args...)
mkpath(paper_out_dir())
results_out_dir(args...) = DeepART.results_dir(EXP_TOP, args...)
mkpath(results_out_dir())

open(paper_out_dir("basic.tex"), "w") do file
    write(file, out_str)
end

open(results_out_dir("basic.tex"), "w") do file
    write(file, out_str)
end

# Compute the means and standard deviations of the confusion matrix for each experiment
# confs = zeros(n_models, n_dataset, n_scenarios)
# for ix = 1:n_datasets
#     for jx = 1:n_models
#         for kx in eachindex(scenarios)
#             local_scenario = scenarios[kx]
#             confusion_matrix = df[(df.dataset .== ix) .& (df.m .== jx) .& (df.scenario .== local_scenario), :confusion_matrix]
#             confusion_means[jx, ix, kx] = mean(confusion_matrix)
#             confusion_stds[jx, ix, kx] = std(confusion_matrix)
#         end
#     end
# end


