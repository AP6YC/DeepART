"""
    dist_metrics.jl

# Description
Runs the l2metrics on the latest logs from within Julia.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Distributed

# N_SIMS = DEV ? 1 : 5
N_PROCS = 4

experiment_top = "l2metrics"

exp_dicts = Vector{Dict{String, Any}}()
log_top = DeepART.results_dir(experiment_top, "logs")
for data_dir in readdir(log_top)
    data_dir_full = joinpath(log_top, data_dir)
    for method_dir in readdir(data_dir_full)
        method_dir_full = joinpath(data_dir_full, method_dir)
        local_dict = Dict{String, Any}(
            "data" => data_dir,
            "method" => method_dir,
            "log_dir" => readdir(method_dir_full, join=true)[end],
            # "last_log" => joinpath(method_dir_full, readdir(method_dir_full)[end]),
        )
        push!(exp_dicts, local_dict)
    end
end

# -----------------------------------------------------------------------------
# DEPENDENT VARIABLES
# -----------------------------------------------------------------------------

# addprocs(N_PROCS, exeflags="--project=.")

@everywhere begin
    using Revise
    using DeepART

    function run_l2metrics(exp_dict::Dict{String, Any})
        experiment_top = "l2metrics"
        conda_env_name = "deepart-l2m"

        # Declare all of the metrics being calculated
        metrics = [
            "performance",
            "art_match",
            "art_activation",
        ]

        top_out_dir = DeepART.results_dir(experiment_top, "metrics", exp_dict["data"], exp_dict["method"])
        mkpath(top_out_dir)

        for log_dir in readdir(exp_dict["log_dir"])
            log_dir_full = joinpath(exp_dict["log_dir"], log_dir)
            # @info "Dir: $(isdir(log_dir_full)), $(log_dir_full)"
            # Iterate over every metric
            for metric in metrics
                # Point to the output directory for this metric
                # out_dir = results_dir(metrics_dir_name, text_order, last_log, metric)
                out_dir = joinpath(top_out_dir, log_dir, metric)
                mkpath(out_dir)
                # Set the common python l2metrics command
                l2metrics_command = `python -m l2metrics --no-plot -p $metric -o $metric -O $out_dir -l $log_dir_full`
                if Sys.iswindows()
                    run(`cmd /c activate $conda_env_name \&\& $l2metrics_command`)
                elseif Sys.isunix()
                    run(`$l2metrics_command`)
                end
            end
        end
    end
end

# Parallel map the sims
# pmap(run_l2metrics, exp_dicts)

println("--- Simulation complete ---")

# Close the workers after simulation
rmprocs(workers())

run_l2metrics(exp_dicts[1])


   # # Iterate over every one of the order folders
    # # for order in orders
    # # function local_sim(order::Vector{Int64})
    # function local_sim(datadir::String)
    #     # String of the permutation order
    #     # text_order = String(join(order))

    #     # Get the most recent log directory name
    #     # last_log = readdir(results_dir(log_dir_name, text_order))[end]

    #     # Set the full source directory
    #     src_dir = results_dir(log_dir_name, text_order, last_log)

    #     # Iterate over every metric
    #     for metric in metrics
    #         # Point to the output directory for this metric
    #         out_dir = results_dir(metrics_dir_name, text_order, last_log, metric)
    #         mkpath(out_dir)
    #         # Set the common python l2metrics command
    #         l2metrics_command = `python -m l2metrics --no-plot -p $metric -o $metric -O $out_dir -l $src_dir`
    #         if Sys.iswindows()
    #             run(`cmd /c activate $conda_env_name \&\& $l2metrics_command`)
    #         elseif Sys.isunix()
    #             run(`$l2metrics_command`)
    #         end
    #     end
    # end