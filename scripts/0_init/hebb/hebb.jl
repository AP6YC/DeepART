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

# perf = 0.9310344827586207
# perf = 0.9655172413793104

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

@info "------- Setting options -------"
# opts = Hebb.load_opts("base.yml")
# opts = Hebb.load_opts("fuzzy.yml")
opts = Hebb.load_opts("dense-fuzzy.yml")
# opts = Hebb.load_opts("conv-fuzzy.yml")

@info "------- Options post-processing -------"
Random.seed!(opts["rng_seed"])

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "------- Loading dataset -------"
data = Hebb.get_data(opts)

# n_preview = 4
# preview_weights = true
preview_weights = false
# n_preview = 10
# n_preview_2 = 8
n_preview = 4
n_preview_2 = 4

dev_x, dev_y = data.train[1]
# n_input = size(dev_x)[1]
# n_class = length(unique(data.train.y))

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

@info "------- Constructing model -------"
model = Hebb.HebbModel(data, opts["model_opts"])
if preview_weights
    display(Hebb.view_weight_grid(model, n_preview))
    display(Hebb.view_weight_grid(model, n_preview_2, layer=2))
end
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
    # @static if Sys.iswindows()
    #     # compilation
    #     @profview Hebb.profile_test(3)
    #     # pure runtime
    #     @profview Hebb.profile_test(10)
    # end
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

if preview_weights
    display(Hebb.view_weight_grid(model, n_preview, layer=1))
    display(Hebb.view_weight_grid(model, n_preview_2, layer=2))
    # display(Hebb.view_weight_grid(model, 3, layer=2))
end

display(Hebb.view_weight_grid(model, n_preview))

a = Hebb.view_weight_grid(model, n_preview)

# using Images

# save(DeepART.paper_results_dir("fuzzy_weights.png"), a)
# save(DeepART.results_dir("fuzzy_weights.png"), a)



# Hebb.view_weight_grid(model, 8, layer=2)


# using ProgressMeter
# using UnicodePlots

# n = 20
# xs = Float64[]
# p = Progress(n)
# for iter = 1:n
#     append!(xs, rand())
#     sleep(0.5)
#     plot = lineplot(xs)
#     str = "\n" * string(plot; color=true) # use ANSI color codes and prepend newline
#     ProgressMeter.next!(p; showvalues = [(:UnicodePlot, str)])
# end
# https://github.com/timholy/ProgressMeter.jl/issues/224


# LeNet:
# model = Chain(
#     Conv((5,5),1 => 6, relu),
#     MaxPool((2,2)),
#     Conv((5,5),6 => 16, relu),
#     MaxPool((2,2)),
#     Flux.flatten,
#     Dense(256=>120,relu),
#     Dense(120=>84, relu),
#     Dense(84=>10, sigmoid),
#     softmax
# )