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

# perfs = groupby(df, [:rng, :data])

# perfs_vec = []
datasets = ["usps", "mnist", "fashionmnist"]
perfs_dict = Dict()

for ix in eachindex(datasets)
    dataset = datasets[ix]
    local_df = subset(df, :data => d -> d .== dataset)
    local_perfs = groupby(local_df, [:rng])

    local_perfs2 = [Vector{Float32}(perfsi[!, :perf]) for perfsi in local_perfs]

    # n_perfs = length(perfs2)
    local_perfs_plot = reduce(hcat, local_perfs2)'
    perfs_dict[dataset] = local_perfs_plot
end

# perfs2 = [Vector{Float32}(perfsi[!, :perf]) for perfsi in perfs]
# n_perfs = length(perfs2)
# perfs_plot = reduce(hcat, perfs2)'


n_perfs = size(perfs_plot)[2]


# p = errorline(1:n_perfs, perfs_plot)
begin
    p = plot()
    dataset_labels = Dict(
        "usps" => "USPS",
        "mnist" => "MNIST",
        "fashionmnist" => "FashionMNIST",
    )
    linestyles = [:solid, :dash, :dashdot]
    for ix in eachindex(datasets)
        ticks = 3:1:n_perfs+2
        dataset_name = datasets[ix]
        bigfontsize=13
        smallfontsize=8
        errorline!(
            p,
            ticks,
            # perfs_plot,
            perfs_dict[dataset_name],
            xticks=ticks,
            yticks=0.4:0.1:0.9,
            ylims=(0.35, 0.9),
            xlabel="MLP DeepART Network Depth",
            ylabel="Testing Accuracy",
            errorstyle=:stick,
            # label="Stick",
            palette = palette(:okabe_ito),
            linewidth=4,
            # fontsize=19,
            labelfontsize=bigfontsize,
            annotationfontsize=bigfontsize,
            plot_titlefontsize=bigfontsize,
            tickfontsize=smallfontsize,
            legendfontsize=smallfontsize,
            linestyle=linestyles[ix],
            # colorscheme=:okabe_ito,
            label = dataset_labels[dataset_name],
            secondarycolor=:matched,
            fontfamily="Computer Modern",
            dpi=350,
        )
    end
    display(p)
end

DeepART.saveplot(p, "layers", ["hebb_dist_analyze", "layers"], paper=true, extension=".svg")
