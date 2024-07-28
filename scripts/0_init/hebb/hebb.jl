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

# perf = 0.9310344827586207
# perf = 0.9655172413793104

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

@info "------- Setting options -------"
# opts = Hebb.load_opts("base.yml")
# opts = Hebb.load_opts("fuzzy.yml")
opts = Hebb.load_opts("dense-fuzzy.yml")

@info "------- Options post-processing -------"
Random.seed!(opts["rng_seed"])

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "------- Loading dataset -------"
data = Hebb.get_data(opts)

# n_preview = 2
n_preview = 4

dev_x, dev_y = data.train[1]
# n_input = size(dev_x)[1]
# n_class = length(unique(data.train.y))

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

@info "------- Constructing model -------"
model = Hebb.HebbModel(data, opts["model_opts"])
# display(Hebb.view_weight_grid(model, n_preview))
# display(Hebb.view_weight_grid(model, 8, layer=2))
# display(Hebb.view_weight_grid(model, 4, layer=3))

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

@info "------- TESTING BEFORE TRAINING -------"
if model.opts["gpu"]
    model.model = model.model |> gpu
end
old_perf = Hebb.test(model, data)
@info "OLD PERF: $old_perf"

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# CUDA.@time train_loop(
# CUDA.@profile train_loop(
if opts["profile"]
    @info "------- Profiling -------"
    @static if Sys.iswindows()
        # compilation
        @profview Hebb.profile_test(3)
        # pure runtime
        @profview Hebb.profile_test(10)
    end
else
    if opts["dataset"] in Hebb.DATASETS["high_dimensional"]
        @info "weights before:"
        old_weights = deepcopy(model.model.chain[1][2].weight)
        # Hebb.view_weight(model, 1)
    # else
        # @info model[2].weight
        # @info sum(model[2].weight)
    end

    @info "------- Training -------"
    vals = Hebb.train_loop(
        model,
        data,
        n_epochs = opts["n_epochs"],
        n_vals = opts["n_vals"],
        val_epoch = opts["val_epoch"],
    )

    # local_plot = lineplot(
    #     vals,
    # )
    # show(local_plot)

    # Only visualize the weights if we are working with a computer vision dataset
    if opts["dataset"] in Hebb.DATASETS["high_dimensional"]
        @info "Weights after:"
        new_weights = deepcopy(model.model.chain[1][2].weight)
        @info "Weights difference:" sum(new_weights .- old_weights)
        Hebb.view_weight(model, 1)
    # else
        # @info model[2].weight
        # @info sum(model[2].weight)
    end
end

Hebb.view_weight_grid(model, n_preview, layer=1)
# Hebb.view_weight_grid(model, 8, layer=2)
