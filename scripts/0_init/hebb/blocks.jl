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
Random.seed!(opts["sim_opts"]["rng_seed"])


