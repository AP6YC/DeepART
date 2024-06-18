"""
    setup.jl

# Description
Setup script for the single model experiments.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
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

# Accept data downloads
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Fix plotting on headless
ENV["GKSwstype"] = "100"

# Dataset selection
DATASET = "mnist"
# DATASET = "cifar10"
# DATASET = "fashionmnist"
# DATASET = "omniglot"
DISPLAY = true

# Separate development and cluster settings
DEV = Sys.iswindows()
# DEV = false
N_TRAIN = DEV ? 1000 : 10000
N_TEST = DEV ? 500 : 1000
# N_TRAIN = DEV ? 1000 : 50000
# N_TEST = DEV ? 1000 : 1000
# GPU = !DEV
GPU = true

# BETA_S = 0.5
BETA_S = 1.0
# BETA_D = 0.01
BETA_D = 0.1

EXP_TOP = ["singles"]

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
    DATASET,
    n_train=N_TRAIN,
    n_test=N_TEST,
)
fdata = DeepART.flatty(data)

n_classes = length(unique(data.train.y))

names_range = collect(1:n_classes)
if DATASET == "mnist"
    names_range .-= 1
end
names = string.(names_range)

n_input = size(fdata.train.x)[1]

@info "Loaded dataset: " DATASET