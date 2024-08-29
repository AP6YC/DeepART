"""
    hebb.jl

# Description
Deep Hebbian learning experiment drafting script.
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

if Sys.isunix()
    using ImageInTerminal
end

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

@info "------- Setting options -------"
# opts = Hebb.load_opts("base.yml")
# opts = Hebb.load_opts("fuzzy.yml")
opts_hebb = Hebb.load_opts("dense-fuzzy.yml")
# opts = Hebb.load_opts("conv-fuzzy.yml")

@info "------- Options post-processing -------"
Random.seed!(opts_hebb["rng_seed"])

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "------- Loading dataset -------"
data_hebb = Hebb.get_data(opts_hebb)

# n_preview = 4
# preview_weights = true
preview_weights = false
# n_preview = 10
# n_preview_2 = 8
n_preview = 4
n_preview_2 = 4

dev_x, dev_y = data.train[1]

opts_block = Hebb.load_opts("blockbase.yml")

hebb = Hebb.HebbModel(data, opts_hebb["model_opts"])
blocks = Hebb.BlockNet(data, opts_block["block_opts"])


h_weights = Hebb.get_weights(hebb.model)
b_weights = Hebb.get_weights(blocks.layers[1])
b_weights[1] .= h_weights[1]

h_weights[1]
b_weights[1]

b_weights[2] .= h_weights[2]

h_ins, h_outs = Hebb.get_incremental_activations(hebb.model, dev_x)
b_ins, b_outs = Hebb.get_incremental_activations(blocks.layers[1], dev_x)

h_ins[2]
b_ins[2]

h_outs[2]
b_outs[2]
@info all(h_outs[2] .== b_outs[2])

@info hebb.model.chain[2]
@info blocks.layers[1].chain[2]

@info hebb.model.chain[3]
@info blocks.layers[2].chain[1]

Hebb.forward(blocks, dev_x)
# Hebb.classify(hebb, dev_x)