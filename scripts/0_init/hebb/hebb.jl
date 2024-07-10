"""
    hebb.jl

# Description
Deep Hebbian learning experiment drafting script.
"""

# @info "####################################"
# @info "###### NEW HEBBIAN EXPERIMENT ######"
# @info "####################################"
@info """\n####################################
###### NEW HEBBIAN EXPERIMENT ######
####################################
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

@info "------- Loading dependencies -------"
using Flux
using Random
using UnicodePlots

@info "------- Loading definitions -------"
include("definitions.jl")

@info "------- Loading Hebb module -------"
import .Hebb

# perf = 0.9310344827586207
# perf = 0.9655172413793104

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

@info "------- Setting options -------"
opts = Dict{String, Any}(
    "n_epochs" => 1000,
    # "n_epochs" => 200,
    # "n_epochs" => 10,
    # "n_epochs" => 50,

    "model_opts" => Dict{String, Any}(
        # "immediate" => true,
        "immediate" => false,

        "bias" => false,
        "eta" => 0.05,
        # "beta_d" => 0.0,
        "beta_d" => 0.1,
        # "eta" => 0.2,
        # "beta_d" => 0.2,
        # "eta" => 0.5,
        # "beta_d" => 0.5,
        # "eta" => 1.0,
        # "beta_d" => 1.0,
        # "beta_d" => 0.001,

        "final_sigmoid" => false,
        # "final_sigmoid" => true,

        "gpu" => false,

        # "model" => "dense",
        # "model" => "small_dense",
        # "model" => "fuzzy",
        # "model" => "conv",
        # "model" => "fuzzy_new",
        "model" => "dense_new",
        # "model" => "conv_new",

        "init" => Flux.rand32,
        # "init" => Flux.glorot_uniform,

        # "positive_weights" => true,
        "positive_weights" => false,

        # "beta_normalize" => false,
        "beta_normalize" => true,
        # "beta_rule" => "wta",
        # "beta_rule" => "contrast",
        "beta_rule" => "softmax",
    ),

    "profile" => false,
    # "profile" => true,

    # "dataset" => "wine",
    "dataset" => "iris",
    # "dataset" => "wave",
    # "dataset" => "face",
    # "dataset" => "flag",
    # "dataset" => "halfring",
    # "dataset" => "moon",
    # "dataset" => "ring",
    # "dataset" => "spiral",
    # "dataset" => "mnist",
    # "dataset" => "usps",

    "n_train" => 50000,
    "n_test" => 10000,
    # "flatten" => true,
    "rng_seed" => 1235,
)

@info "------- Options post-processing -------"

# Correct for Float32 types
opts["model_opts"]["eta"] = Float32(opts["model_opts"]["eta"])
opts["model_opts"]["beta_d"] = Float32(opts["model_opts"]["beta_d"])
Random.seed!(opts["rng_seed"])

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

datasets = Dict(
    "high_dimensional" => [
        "mnist",
        "usps",
    ],
    "low_dimensional" => [
        "wine",
        "iris",
        "wave",
        "face",
        "flag",
        "halfring",
        "moon",
        "ring",
        "spiral",
    ]
)

@info "------- Loading dataset -------"
data = Hebb.get_data(opts)

dev_x, dev_y = data.train[1]
n_input = size(dev_x)[1]
n_class = length(unique(data.train.y))

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

@info "------- Constructing model -------"

# model = Hebb.construct_model(data, opts)
model = Hebb.HebbModel(data, opts["model_opts"])

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

@info "------- TESTING BEFORE TRAINING -------"
if model.opts["gpu"]
    model.model = model.model |> gpu
end
Hebb.test(model, data)

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
    @info "------- Training -------"
    vals = Hebb.train_loop(
        model,
        data,
        n_epochs=opts["n_epochs"],
        # eta=opts["eta"],
        # beta_d=opts["beta_d"],
    )

    local_plot = lineplot(
        vals,
    )
    show(local_plot)

    # Only visualize the weights if we are working with a computer vision dataset
    if opts["dataset"] in datasets["high_dimensional"]
        Hebb.view_weight(model, 1)
    else
        # @info model[2].weight
        # @info sum(model[2].weight)
    end
end
