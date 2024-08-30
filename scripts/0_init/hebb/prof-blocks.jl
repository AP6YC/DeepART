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
# opts = Hebb.load_opts("blockbase.yml")
opts = Hebb.load_opts("block-res.yml")

@info "------- Options post-processing -------"
# Random.seed!(opts["sim_opts"]["rng_seed"])
Hebb.set_seed!(opts)

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "------- Loading dataset -------"
data = Hebb.get_data(opts)

n_preview = 4

dev_x, dev_y = data.train[1]

model = Hebb.BlockNet(data, opts["block_opts"])
Hebb.forward(model, dev_x)

old_perf = Hebb.test(model, data)
@info "OLD PERF: $old_perf"


# Hebb.train!(model, dev_x, dev_y)

# n_preview = 4
# display(Hebb.view_weight_grid(model, n_preview, layer=1))

# # @info "------- Training -------"
# vals = Hebb.train_loop(
#     model,
#     data,
#     n_epochs = opts["sim_opts"]["n_epochs"],
#     n_vals = opts["sim_opts"]["n_vals"],
#     val_epoch = opts["sim_opts"]["val_epoch"],
# )



function local_profile(
    model,
    data,
    opts,
    n_epochs,
)
    _ = Hebb.train_loop(
        model,
        data,
        # n_epochs = opts["sim_opts"]["n_epochs"],
        n_epochs = n_epochs,
        n_vals = opts["sim_opts"]["n_vals"],
        val_epoch = opts["sim_opts"]["val_epoch"],
    )
    return
end

@static if Sys.iswindows()
    # compilation
    @profview local_profile(model, data, opts, 1)
    # pure runtime
    @profview local_profile(model, data, opts, 2)
end

display(Hebb.view_weight_grid(model, n_preview, layer=1))