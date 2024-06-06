"""
    dist.jl

# Description
This script runs the distributed single-task and multi-task train/test experiments for each model and dataset combination.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Distributed
using DrWatson

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Accept data downloads
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Fix plotting on headless
ENV["GKSwstype"] = "100"

# Separate development and cluster settings
DEV = Sys.iswindows()

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

EXP_TOP = "1_baselines_new"
EXP_NAME = "dist"

N_PROCS = DEV ? 0 : 31
# N_SIMS = DEV ? 1 : 5
N_SIMS = DEV ? 1 : 25
N_TRAIN = DEV ? 1000 : 50000
N_TEST = DEV ? 500 : 10000
DISPLAY = DEV
# GPU = DEV
# GPU = true
GPU = false

# Set the simulation parameters
sim_params = Dict{String, Any}(
    "m" => [
        "DeepARTDense",
        "DeepARTConv",
        "SFAM",
    ],
    "rho" => [
        @onlyif("m" == "SFAM", 0.6),
        @onlyif("m" == "DeepARTDense", 0.3),
        @onlyif("m" == "DeepARTConv", 0.3),
    ],
    "beta_d" => 0.01,
    "beta_s" => 1.0,
    "rng_seed" => collect(1:N_SIMS),
    # "n_train" => N_TRAIN,
    "n_train" => [
        @onlyif("m" == "SFAM" && "dataset" in ["cifar10", "cifar100_fine", "cifar100_coarse"], 4000),
        @onlyif("m" == "SFAM" && !("dataset" in ["cifar10", "cifar100_fine", "cifar100_coarse"]), N_TRAIN),
        @onlyif("m" == "DeepARTDense", N_TRAIN),
        @onlyif("m" == "DeepARTConv", N_TRAIN),
    ],
    # "n_test" => N_TEST,
    "n_test" =>[
        @onlyif("m" == "SFAM" && "dataset" in ["cifar10", "cifar100_fine", "cifar100_coarse"], 2000),
        @onlyif("m" == "SFAM" && !("dataset" in ["cifar10", "cifar100_fine", "cifar100_coarse"]), N_TEST),
        @onlyif("m" == "DeepARTDense", N_TEST),
        @onlyif("m" == "DeepARTConv", N_TEST),
    ],
    # "head_dim" => 1024,
    "head_dim" => 784,
    "dataset" => [
        "mnist",
        "fashionmnist",
        "cifar10",
        # "cifar100_fine",
        # "cifar100_coarse",
        "usps",
    ],
    "display" => DISPLAY,
    "gpu" => GPU,
    "scenario" => [
        "task-incremental",
        "task-homogenous",
    ],
    "group_size" => [
        @onlyif("dataset" == "mnist", 2),
        @onlyif("dataset" == "fashionmnist", 2),
        @onlyif("dataset" == "cifar10", 2),
        @onlyif("dataset" == "cifar100_fine", 20),
        @onlyif("dataset" == "cifar100_coarse", 4),
        @onlyif("dataset" == "usps", 2),
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
        "1_baselines_gpu",
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
