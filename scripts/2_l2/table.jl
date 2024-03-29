"""
    table.jl

# Description
Collects the l2metrics into a table with statistics.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

using
    CSV,
    DrWatson,
    DataFrames,
    DeepART,
    Revise,
    JSON,
    Latexify

experiment_top = "l2metrics"

# DCCR project files
# include(projectdir("src", "setup.jl"))

# Select the metrics that we are interested in consolidating into a new dict
desired_metrics = [
    "perf_maintenance_mrlep",
    "forward_transfer_ratio",
    "backward_transfer_ratio",
]
pretty_rows = [
    "Activation",
    "Match",
    "Performance",
]

# Point to the results directory containing all of the permutations
top_dir = DeepART.results_dir("l2metrics", "metrics")

# Create an empty destination for each dataframe
mets = Dict{String, DataFrame}()

# Iterate over each permutation
# for perm in readdir(perms_dir)
for dataset in readdir(top_dir)
    # Point to the directory of the most recent metrics in this permutation
    # full_perm = joinpath(perms_dir, perm)
    full_dataset = joinpath(top_dir, dataset)
    # top_dir = readdir(full_perm, join=true)[end]
    for method in readdir(full_dataset)
        full_method = joinpath(full_dataset, method)
        # Iterate over every metric in this permutation
        for metric in readdir(full_method)
            full_metric = joinpath(full_method, metric)
            for perm in readdir(full_metric)
                full_perm = joinpath(full_metric, perm)
                # @info full_perm
                # Check that we have a dataframe for each metric
                if !haskey(mets, metric)
                    # If we are missing a dataframe, initialize it with the correct columns
                    mets[metric] = DataFrame(
                        dataset=String[],
                        perm=String[],
                        method=String[],
                        pm=Float64[],
                        ftr=Float64[],
                        btr=Float64[],
                    )
                end
                # Load the metric file
                # metric_file = joinpath(top_dir, metric, metric * "_metrics.json")
                metric_file = joinpath(full_perm, metric * "_metrics.json")
                md = JSON.parsefile(metric_file)
                # Create a new dataframe entry manually from the l2metrics in the file
                new_entry = (
                    dataset,
                    method,
                    perm,
                    md["perf_maintenance_mrlep"],
                    md["forward_transfer_ratio"],
                    md["backward_transfer_ratio"],
                )
                push!(mets[metric], new_entry)
            end
        end
        # # Make a latex version of the dataframe and save
        # new_df_tex = latexify(new_df, env=:table, fmt="%.3f")
        # table_file = paper_results_dir("condensed.tex")
        # open(table_file, "w") do f
        #     write(f, new_df_tex)
        # end
    end
end

# Point to the save directory
savedir(args...) = DeepART.results_dir(experiment_top, "processed", args...)
mkpath(savedir())

# Save the raw metrics
for (metric, df) in mets
    savefile = savedir(metric * ".csv")
    CSV.write(savefile, df)
end
