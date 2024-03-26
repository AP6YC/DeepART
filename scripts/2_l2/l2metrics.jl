"""
    l2metrics.jl

# Description
Runs the l2metrics batch script from within Julia.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DeepART

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Experiment save directory name
experiment_top = "l2metrics"

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------

# Get the batch script location
# exp_dir(args...) = DeepART.projectdir(
#     "scripts",
#     experiment_top,
#     args...
# )

# Point to the batch script
full_l2m_script = DeepART.projectdir(
    "scripts",
    # experiment_top,
    "2_l2",
    "l2metrics.bat",
)

# Iterate over each group directory
# for group_dir in readdir(DeepART.results_dir(experiment_top, "logs"), join=true)
logs_dir = DeepART.results_dir(experiment_top, "logs")
for group_dir in readdir(logs_dir)
    full_group_dir = joinpath(logs_dir, group_dir)
    # Get the location of the last log
    last_log = readdir(full_group_dir, join=true)[end]
    out_dir = DeepART.results_dir(experiment_top, "metrics", group_dir)

    @info "Generating L2 metrics for dir: $last_log, $(isdir(last_log))"
    # @info out_dir
    # Run the command for the batch script
    run(`cmd /c activate deepart-l2m \&\& $full_l2m_script $last_log $out_dir`)
end
