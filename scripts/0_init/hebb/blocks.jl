"""
    blocks.jl

# Description
Deep Hebbian learning experiment drafting script using a blocknet.
"""

@info """
\n####################################
###### NEW HEBBIAN EXPERIMENT ######
####################################
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

@info "------- Loading dependencies -------"
using Revise
using DeepART
using Flux
using Random

@info "------- Loading definitions -------"
include("lib/lib.jl")

@info "------- Loading Hebb module -------"
import .Hebb

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

@info "------- Setting options -------"
opts = Hebb.load_opts("blockbase.yml")

@info "------- Options post-processing -------"
# Random.seed!(opts["sim_opts"]["rng_seed"])
Hebb.set_seed!(opts)

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "------- Loading dataset -------"
data = Hebb.get_data(opts)

# n_preview = 2
n_preview = 4

dev_x, dev_y = data.train[1]


# Get the shape of the dataset
# dev_x, _ = data.train[1]
# n_input = size(dev_x)[1]
# n_class = length(unique(data.train.y))

# a = Hebb.ChainBlock(Hebb.get_model_opts(opts, 1), n_inputs = n_input)
# @info a.chain

b = Hebb.BlockNet(data, opts["block_opts"])

Hebb.forward(b, dev_x)
