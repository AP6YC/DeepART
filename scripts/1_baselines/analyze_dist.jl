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

# Point to the destination directories
paper_out_dir(args...) = DeepART.paper_results_dir(EXP_TOP, args...)
results_out_dir(args...) = DeepART.results_dir(EXP_TOP, args...)

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

# Collect the results into a single dataframe
df = collect_results!(sweep_dir)

# -----------------------------------------------------------------------------
# TABLE NAMES
# -----------------------------------------------------------------------------

# Get the sizes of the relevant elements in the dataframe
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

scenarios = Dict(
    "task-homogenous" => "(ST)",
    "task-incremental" => "(MT)",
    # "task-homogenous" => "Task Homogeneous",
    # "task-incremental" => "Task Incremental",
)
scenarios_list = collect(keys(scenarios))

# -----------------------------------------------------------------------------
# PERFORMANCES
# -----------------------------------------------------------------------------

function gen_tex_table(symb, filename, acc)
    # Compute the means and standard deviations of the final testing performances
    # out_mat = zeros(n_models, n_datasets, n_scenarios, 2)
    out_mat = zeros(length(models_list), length(datasets_list), length(scenarios), 2)

    for ix in eachindex(datasets_list)
        for jx in eachindex(models_list)
            for kx in eachindex(scenarios_list)
                local_dataset = datasets_list[ix]
                local_model = models_list[jx]
                local_scenario = scenarios_list[kx]
                out_mat[jx, ix, kx, 1] = mean(df[(df.dataset .== local_dataset) .& (df.m .== local_model) .& (df.scenario .== local_scenario), symb])
                out_mat[jx, ix, kx, 2] = std(df[(df.dataset .== local_dataset) .& (df.m .== local_model) .& (df.scenario .== local_scenario), symb])
            end
        end
    end

    # Construct the table contents string
    out_str = ""
    for ix in eachindex(datasets_list)
        out_str *= "& $(datasets[datasets_list[ix]])"
    end
    out_str *= "\\\\\n\\midrule\n\\addlinespace\n"

    for jx in eachindex(models_list)
        for kx in eachindex(scenarios_list)
            local_model = models[models_list[jx]] * " " * scenarios[scenarios_list[kx]]
            out_str *= "$(local_model)"
            for ix in eachindex(datasets_list)
                f_str = Printf.Format("& \$ %.$(acc)f \\pm %.$(acc)f \$")
                out_str *= Printf.format(f_str, out_mat[jx, ix, kx, 1], out_mat[jx, ix, kx, 2])
                # out_str *= @sprintf "& \$ %.4f \\pm %.4f \$" out_mat[jx, ix, kx, 1] out_mat[jx, ix, kx, 2]
            end
            out_str *= "\\\\\n"
        end
        # out_str *= "\\\\ \n"
        out_str *= "\\addlinespace\n\\hline\n\\addlinespace\n"
    end

    @info out_str

    # Create those directories if they don't exist
    mkpath(paper_out_dir())
    mkpath(results_out_dir())

    # Write to both
    open(paper_out_dir(filename), "w") do file
        write(file, out_str)
    end

    open(results_out_dir(filename), "w") do file
        write(file, out_str)
    end

    return
end

gen_tex_table(:perf, "basic.tex", 4)
gen_tex_table(:n_cat, "cats.tex", 1)

# -----------------------------------------------------------------------------
# CONFUSION
# -----------------------------------------------------------------------------

# Normalized confusion heatmap
# norm_cm = get_normalized_confusion(n_classes, data.test_y, y_hat)
# for ix in eachindex(dataset)
#     for jx in eachindex(models_list)
#         for kx in eachindex(scenarios_list)
#             norm_cm_df = df[:, :conf]
#         end
#     end
# end

# names_range = collect(1:n_classes)
# if DATASET == "mnist"
#     names_range .-= 1
# end
# names = string.(names_range)

n_classes_dict = Dict(
    "mnist" => 10,
    "fashionmnist" => 10,
    "usps" => 10,
    "cifar10" => 10,
    "cifar100_coarse" => 20,
    "cifar100_fine" => 100,
)

vec_h = []
for ix in eachindex(datasets_list)
    class_labels = string.(collect(1:n_classes_dict[datasets_list[ix]]))
    for jx in eachindex(models_list)
        for kx in eachindex(scenarios_list)
            local_dataset = datasets_list[ix]
            local_model = models_list[jx]
            local_scenario = scenarios_list[kx]
            local_conf = df[(df.dataset .== local_dataset) .& (df.m .== local_model) .& (df.scenario .== local_scenario), :conf]
            norm_cm = mean(local_conf)
            h = DeepART.create_custom_confusion_heatmap(class_labels, norm_cm, 7)
            push!(vec_h, h)

            DeepART.saveplot(
                h,
                "$(local_dataset)_$(local_model)_$(local_scenario)_conf",
                [EXP_TOP, "confusions"],
                paper=true,
                extension=".png",
            )

            # # Create the confusion matrix from this experiment
            # DeepART.plot_confusion_matrix(
            #     data.test.y,
            #     results["y_hats"],
            #     names,
            #     "conv_ti_confusion",
            #     EXP_TOP,
            # )

        end
    end
end

vec_h[1]

# # Create the confusion matrix from this experiment
# DeepART.plot_confusion_matrix(
#     data.test.y,
#     results["y_hats"],
#     names,
#     "conv_ti_confusion",
#     EXP_TOP,
# )

# function check_parts(parts::Vector{String})
#     results_out_dir(args...) = DeepART.results_dir(parts..., args...)
#     @info results_out_dir()
# end

# check_parts(["asdf", "qwer"])

# norm_cm_df = df[:, :conf]
# norm_cm = mean(norm_cm_df)
# h = DeepART.create_custom_confusion_heatmap(class_labels, norm_cm)
# DeepART.handle_display(h, pargs)


# create_custom_confusion_heatmap
# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

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
