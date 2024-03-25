"""
    dist_sfam.jl

# Description
This script runs the distributed task-homogenous train/test experiment.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""
# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DeepART

# -----------------------------------------------------------------------------
# ADDITIONAL DEPENDENCIES
# -----------------------------------------------------------------------------

using Distributed
using DrWatson

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

EXP_TOP = "1_baselines"
EXP_NAME = "dist"
N_PROCS = Sys.iswindows() ? 0 : 31

N_SIMS = Sys.iswindows() ? 1 : 1000
N_TRAIN = 1000
N_TEST = 1000

# Set the simulation parameters
sim_params = Dict{String, Any}(
    "m" => [
        "SFAM",
        "DeepARTDense",
        "DeepARTConv"
    ],
    "rho" => 0.6,
    "rng_seed" => collect(1:N_SIMS),
    "n_train" => N_TRAIN,
    "n_test" => N_TEST,
    "dataset" => [
        "mnist",
        "fashionmnist",
        "cifar10",
        "cifar100_fine",
        "cifar100_coarse",
        "usps",
    ],
)

# -----------------------------------------------------------------------------
# DEPENDENT VARIABLES
# -----------------------------------------------------------------------------

addprocs(N_PROCS, exeflags="--project=.")

# -----------------------------------------------------------------------------
# PARALLEL DEFINITIONS
# -----------------------------------------------------------------------------

@everywhere begin
    # Activate the project in case
    using Pkg
    Pkg.activate(".")
    using Revise
    using DeepART

    # Point to the sweep results
    sweep_results_dir(args...) = DeepART.results_dir(
        "1_dist",
        # "sfam",
        # "sweep",
        args...
    )

    # Make the path
    mkpath(sweep_results_dir())

    # Define the single-parameter function used for pmap
    local_sim(sim_dict) = DeepART.tt_dist(
        sim_dict,
        sweep_results_dir,
    )
end

# -----------------------------------------------------------------------------
# EXPERIMENT
# -----------------------------------------------------------------------------

# Log the simulation scale
@info "SFAM: $(dict_list_count(sim_params)) simulations across $(nprocs())."

# Turn the dictionary of lists into a list of dictionaries
dicts = dict_list(sim_params)

# Remove impermissible sim options
# filter!(d -> d["rho_ub"] > d["rho_lb"], dicts)
# @info "Testing permutations:" dicts

# Parallel map the sims
pmap(local_sim, dicts)

println("--- Simulation complete ---")

# -----------------------------------------------------------------------------
# CLEANUP
# -----------------------------------------------------------------------------

# Close the workers after simulation
rmprocs(workers())
