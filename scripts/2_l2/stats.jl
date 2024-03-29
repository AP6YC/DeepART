"""
    stats.jl

# Description
Generates the statistics for all permutations of the generated l2metrics.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using CSV
using DrWatson
using DataFrames
using Latexify
using Statistics
using StatsBase
using DataStructures
using Printf

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

EXP_TOP = "l2metrics"
ACC = 4
COMBINED_TABLES = true
# EXP_TOP = "2_l2"

metrics_dir = DeepART.results_dir(EXP_TOP, "processed")

# Point to the destination directories
paper_out_dir(args...) = DeepART.paper_results_dir(EXP_TOP, args...)
results_out_dir(args...) = DeepART.results_dir(EXP_TOP, args...)

# Create those directories if they don't exist
mkpath(paper_out_dir())
mkpath(results_out_dir())

# -----------------------------------------------------------------------------
# LOAD RESULTS
# -----------------------------------------------------------------------------

# Initialize container for the dataframes
dfs = Dict{String, DataFrame}()

# Iterate over every file name
for metric_file in readdir(savedir())
    # Point to the full file name
    savefile = savedir(metric_file)
    # Get the name of the metric by removing the extension
    metric, _ = splitext(metric_file)
    # Load the df and add to the dictionary of dfs
    dfs[metric] = DataFrame(CSV.File(savefile))
end

# Names for the rows of the latex table
pretty_rows = OrderedDict(
    "performance" => "Performance",
    "art_activation" => "Activation",
    "art_match" => "Match",
)

# Initialize the output statistics dataframe
out_df = DataFrame(
    Dataset=String[],
    Method=String[],
    PM=String[],
    FTR=String[],
    BTR=String[]
)

# -----------------------------------------------------------------------------
# COMPUTE STATISTICS
# -----------------------------------------------------------------------------

met_syms = [:pm, :ftr, :btr]
methods = unique(dfs["performance"][:, :method])
datasets = unique(dfs["performance"][:, :dataset])
l2metrics = keys(dfs)

stats_dfs = Dict{String, DataFrame}()

for l2metric in l2metrics
    new_df = DataFrame(
        Dataset=String[],
        Method=String[],
        pmm=Float64[],  # Performance Maintenance Mean
        ftrm=Float64[], # Forward Transfer Ratio Mean
        btrm=Float64[], # Backward Transfer Ratio Mean
        pms=Float64[],  # Performance Maintenance Std
        ftrs=Float64[], # Forward Transfer Ratio Std
        btrs=Float64[], # Backward Transfer Ratio Std
    )

    for dataset in datasets
        for method in methods
            # Point to current dataframe
            cdf = dfs[l2metric]
            # Compute the means and standard deviations for each dataset and method tested for each l2metric
            means = [mean(cdf[(cdf.dataset .== dataset) .& (cdf.method .== method), sym]) for sym in met_syms]
            stds = [std(cdf[(cdf.dataset .== dataset) .& (cdf.method .== method), sym]) for sym in met_syms]
            new_entry = (
                dataset,
                method,
                means...,
                stds...,
            )
            push!(new_df, new_entry)
        end
    end
    stats_dfs[l2metric] = new_df
end

# -----------------------------------------------------------------------------
# GENERATE TABLES
# -----------------------------------------------------------------------------

pretty_l2metrics = OrderedDict(
    # :pm => "Performance Maintenance",
    # :ftr => "Forward Transfer Ratio",
    # :btr => "Backward Transfer Ratio",
    :pm => "PM",
    :ftr => "FTR",
    :btr => "BTR",
)

pretty_datasets = OrderedDict(
    "mnist" => "MNIST",
    "fashionmnist" => "Fashion MNIST",
    "usps" => "USPS",
    "cifar10" => "CIFAR-10",
    "cifar100_coarse" => "CIFAR-100 (Coarse)",
    "cifar100_fine" => "CIFAR-100 (Fine)",
)

pretty_methods = OrderedDict(
    "SFAM" => "FuzzyART",
    "DeepARTDense" => "MLP DeepART",
    "DeepARTConv" => "CNN DeepART",
)

l2metrics_names = Dict(
    :pm => Dict(
        "mean" => :ftrm,
        "std" => :ftrs,
    ),
    :ftr => Dict(
        "mean" => :ftrm,
        "std" => :ftrs,
    ),
    :btr => Dict(
        "mean" => :btrm,
        "std" => :btrs,
    ),
)

# -----------------------------------------------------------------------------
# COMBINED TABLES
# -----------------------------------------------------------------------------

for (metric, df) in stats_dfs
    # Point to the output file
    filename = "$(metric)-full.tex"

    # \\toprule
    table_str = """
    \\multirow{3}{*}{L2 Metric} & \\multirow{3}{*}{Method} & \\multicolumn{6}{c@{}}{Dataset} \\\\
    \\cmidrule(l){3-8} &
    """
    for (_, dataset) in pretty_datasets
        table_str *= " & $(dataset)"
    end
    table_str *= " \\\\\n"

    for (l2metric, title) in pretty_l2metrics
        table_str *= """
        \\midrule
        \\addlinespace
        \\multirow{3}{*}{$(title)} """

        # Construct the table contents string
        out_str = ""
        # for (method_key, method) in pretty_methods
        n_methods = length(pretty_methods)
        for ix = 1:n_methods
            method_key = collect(keys(pretty_methods))[ix]
            method = pretty_methods[method_key]
            # local_model = pretty_methods[methods[jx]]
            out_str *= "& $(method) "
            # for (data_key, dataset) in pretty_datasets
            n_datasets = length(pretty_datasets)
            for jx = 1:n_datasets
                data_key = collect(keys(pretty_datasets))[jx]
                dataset = pretty_datasets[data_key]
                f_str = Printf.Format("& \$ %.$(ACC)f \\pm %.$(ACC)f \$ ")
                local_mean = df[(df.Dataset .== data_key) .& (df.Method .== method_key), l2metrics_names[l2metric]["mean"]][1]
                local_std = df[(df.Dataset .== data_key) .& (df.Method .== method_key), l2metrics_names[l2metric]["std"]][1]
                out_str *= Printf.format(f_str, local_mean, local_std)
            end
            out_str *= "\\\\\n"
        end
        out_str *= "\\addlinespace\n"
        table_str *= out_str
    end
    table_str *= "\\bottomrule"

    # Write to both
    open(paper_out_dir(filename), "w") do file
        write(file, table_str)
    end

    open(results_out_dir(filename), "w") do file
        write(file, table_str)
    end

    @info table_str
end

# -----------------------------------------------------------------------------
# ONE BIG COMBINED TABLE
# -----------------------------------------------------------------------------

metric_order = ["performance", "art_activation", "art_match"]

begin
    function add_line_spacing!(table_str)
        return table_str *= "\\addlinespace\n\\midrule\n\\addlinespace\n"
    end

    filename = "l2metrics-big.tex"
    # \\toprule
    table_str = """
    \\multirow{3}{*}{L2 Metric} & \\multirow{3}{*}{Method} & \\multicolumn{6}{c@{}}{Dataset} \\\\
    \\cmidrule(l){3-8} &
    """

    for (_, dataset) in pretty_datasets
        table_str *= " & $(dataset)"
    end
    table_str *= " \\\\\n"
    # table_str *= " \\\\\n\\addlinespace\n"
    # add_line_spacing!(table_str)

    # for (metric, df) in stats_dfs
    for metric in metric_order
        df = stats_dfs[metric]

        table_str = add_line_spacing!(table_str)
        table_str *= """\\multicolumn{8}{c@{}}{$(pretty_rows[metric])} \\\\\n"""

        for (l2metric, title) in pretty_l2metrics
            table_str = add_line_spacing!(table_str)
            table_str *= """\\multirow{3}{*}{$(title)} """
            # table_str *= """
            # \\midrule
            # \\addlinespace
            # \\multirow{3}{*}{$(title)} """

            # Construct the table contents string
            out_str = ""
            # for (method_key, method) in pretty_methods
            n_methods = length(pretty_methods)
            for ix = 1:n_methods
                method_key = collect(keys(pretty_methods))[ix]
                method = pretty_methods[method_key]
                # local_model = pretty_methods[methods[jx]]
                out_str *= "& $(method) "
                # for (data_key, dataset) in pretty_datasets
                n_datasets = length(pretty_datasets)
                for jx = 1:n_datasets
                    data_key = collect(keys(pretty_datasets))[jx]
                    dataset = pretty_datasets[data_key]
                    f_str = Printf.Format("& \$ %.$(ACC)f \\pm %.$(ACC)f \$ ")
                    local_mean = df[(df.Dataset .== data_key) .& (df.Method .== method_key), l2metrics_names[l2metric]["mean"]][1]
                    local_std = df[(df.Dataset .== data_key) .& (df.Method .== method_key), l2metrics_names[l2metric]["std"]][1]
                    out_str *= Printf.format(f_str, local_mean, local_std)
                end
                out_str *= "\\\\\n"
            end
            # out_str *= "\\addlinespace\n"
            table_str *= out_str
        end
    end
    table_str *= "\\addlinespace\n\\bottomrule"
    @info table_str
    # Write to both
    open(paper_out_dir(filename), "w") do file
        write(file, table_str)
    end

    open(results_out_dir(filename), "w") do file
        write(file, table_str)
    end
end


# for (metric, df) in stats_dfs
#     for (l2metric, title) in pretty_l2metrics
#         # Construct the table contents string
#         out_str = ""
#         filename = "$(metric)-$(l2metric).tex"
#         if !COMBINED_TABLES
#             # for datatset in datasets
#             for (key, dataset) in pretty_datasets
#                 out_str *= "& $(dataset) "
#             end
#             out_str *= "\\\\\n\\midrule\n\\addlinespace\n"
#         end

#         # for (method_key, method) in pretty_methods
#         n_methods = length(pretty_methods)
#         for ix = 1:n_methods
#             method_key = collect(keys(pretty_methods))[ix]
#             method = pretty_methods[method_key]
#             # local_model = pretty_methods[methods[jx]]
#             out_str *= "$(method) "
#             # for ix in eachindex(datasets)
#             for (data_key, dataset) in pretty_datasets
#                 f_str = Printf.Format("& \$ %.$(ACC)f \\pm %.$(ACC)f \$ ")
#                 local_mean = df[(df.Dataset .== data_key) .& (df.Method .== method_key), l2metrics_names[l2metric]["mean"]][1]
#                 local_std = df[(df.Dataset .== data_key) .& (df.Method .== method_key), l2metrics_names[l2metric]["std"]][1]
#                 out_str *= Printf.format(f_str, local_mean, local_std)
#             end
#             if !COMBINED_TABLES
#                 out_str *= "\\\\\n"
#             else
#                 if ix != n_stats
#                     out_str *= "\\\\\n"
#                 end
#             end
#             # if ix != n_stats && !COMBINED_TABLES
#             #     out_str *= "\\\\\n"
#             # end
#             # out_str *= "\\addlinespace\n\\hline\n\\addlinespace\n"
#         end
#         # out_str *= "\\addlinespace\n\\hline\n\\addlinespace\n"
#         # out_str *= "\\addlinespace\n\\hline"
#         # out_str *= "\\addlinespace\n"
#         # out_str *= "\n"
#         if !COMBINED_TABLES
#             out_str *= "\\addlinespace\n\\bottomrule"
#         end

#         @info out_str
#         # Write to both
#         open(paper_out_dir(filename), "w") do file
#             write(file, out_str)
#         end

#         open(results_out_dir(filename), "w") do file
#             write(file, out_str)
#         end
#     end
# end

# syms = [:pm, :ftr, :btr]
# # Point to each l2metric symbol that we want to use here
# # Iterate over every metric
# for (metric, df) in dfs
#     # Create an empty new entry for the stats df
#     new_entry = String[]
#     # First entry is the pretty name of the metric
#     push!(new_entry, pretty_rows[metric])
#     # Iterate over every l2metric symbol
#     for sym in syms
#         # push!(new_entry, "$(mean(df[:, sym])) ± $(var(df[:, sym]))")
#         push!(new_entry, "$(mean(df[:, sym])) ± $(var(df[:, sym]))")
#     end
#     # for ix = 1:length(syms)
#     #     push!(new_entry, "$(mean(df[:, syms[ix]])) ± $(var(df[:, syms[ix]]))")
#     # end
#     # Add the entry to the output stats df
#     push!(out_df, new_entry)
# end

# # Make a latex version of the stats dataframe and save
# new_df_tex = latexify(out_df, env=:table, fmt="%.3f")
# table_file = paper_results_dir("perm_stats.tex")
# open(table_file, "w") do f
#     write(f, new_df_tex)
# end




# # Point to each l2metric symbol that we want to use here
# syms = [:pm, :ftr, :btr]
# # Iterate over every metric
# for (metric, df) in dfs
#     # Create an empty new entry for the stats df
#     new_entry = String[]
#     # First entry is the pretty name of the metric
#     push!(new_entry, pretty_rows[metric])
#     # Iterate over every l2metric symbol
#     for sym in syms
#         # push!(new_entry, "$(mean(df[:, sym])) ± $(var(df[:, sym]))")
#         push!(new_entry, "$(mean(df[:, sym])) ± $(var(df[:, sym]))")
#     end
#     # for ix = 1:length(syms)
#     #     push!(new_entry, "$(mean(df[:, syms[ix]])) ± $(var(df[:, syms[ix]]))")
#     # end
#     # Add the entry to the output stats df
#     push!(out_df, new_entry)
# end

# # Make a latex version of the stats dataframe and save
# new_df_tex = latexify(out_df, env=:table, fmt="%.3f")
# table_file = paper_results_dir("perm_stats.tex")
# open(table_file, "w") do f
#     write(f, new_df_tex)
# end