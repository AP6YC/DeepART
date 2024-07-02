"""
    setup.jl

# Description
Setup script for the single model experiments.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using Logging
using DeepART
using Flux
using Random
# using CUDA
# using ProgressMeter
using AdaptiveResonance
using Plots

# theme(:dark)
# theme(:juno)
theme(:dracula)
# using StatsBase: norm

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

Random.seed!(1234)

# Accept data downloads
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
# Fix plotting on headless
ENV["GKSwstype"] = "100"

# Separate development and cluster settings
# DEV = Sys.iswindows()
# DEV = false
DEV = true
# Dataset selection
params = Dict{String, Any}(
    "dataset" => "mnist",
    # DATASET = "cifar10"
    # DATASET = "fashionmnist"
    # DATASET = "omniglot"
    "display" => true,
    "n_train" => DEV ? 1000 : 10000,
    "n_test" => DEV ? 500 : 1000,
    # N_TRAIN = DEV ? 1000 : 50000
    # N_TEST = DEV ? 1000 : 1000
    # GPU = !DEV
    "gpu" => true,
    # BETA_S = 0.5
    "beta_s" => 1.0,
    # BETA_D = 0.01
    "beta_d" => 1.0,
    "exp_top" => ["singles"],
)

# opts = Dict(
#     "beta_s" => 0.5,
#     "beta_d" => 0.01,
#     "rho" => 0.3,
# )

# head_dim = 256
# F2 layer size
# head_dim = 1024
head_dim = 784

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "----------------- LOADING DATA -----------------"

data = DeepART.load_one_dataset(
    params["dataset"],
    # "mnist",
    n_train=params["n_train"],
    n_test=params["n_test"],
)
fdata = DeepART.flatty(data)

n_classes = length(unique(data.train.y))

# Get the range of indices for the class names
names_range = collect(1:n_classes)

# Correction for digits being 0-9 while class labels are 1-10
if params["dataset"] == "mnist"
    names_range .-= 1
end

# Create class names directly from number of classes
names = string.(names_range)

# Get the number of input features from the flat dataset
n_input = size(fdata.train.x)[1]

@info "Loaded dataset: $(params["dataset"])"
