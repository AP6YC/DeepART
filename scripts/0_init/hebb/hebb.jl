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
# using UnicodePlots

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
    # "n_epochs" => 2000,
    # "n_epochs" => 100,
    "n_epochs" => 5,
    # "n_epochs" => 10,
    # "n_epochs" => 50,
    # "n_vals" => 100,
    "n_vals" => 50,
    "val_epoch" => true,

    "model_opts" => Dict{String, Any}(
        # "immediate" => true,
        "immediate" => false,

        "bias" => false,
        # "eta" => 0.05,
        "eta" => 0.005,     # The good one
        # "eta" => 0.001,
        # "beta_d" => 0.0,
        # "beta_d" => 0.0001,    # The good one
        # "beta_d" => 0.001,    # The good one
        # "beta_d" => 0.004,
        "beta_d" => 0.005,
        # "beta_d" => 0.01,       # Divergence
        # "eta" => 0.2,
        # "beta_d" => 0.2,
        # "eta" => 0.5,
        # "beta_d" => 0.5,
        # "eta" => 1.0,
        # "beta_d" => 1.0,
        # "beta_d" => 0.001,

        # "final_sigmoid" => false,
        "final_sigmoid" => true,

        "gpu" => false,

        # "model" => "dense",
        # "model" => "small_dense",
        # "model" => "fuzzy",
        # "model" => "conv",
        # "model" => "fuzzy_new",
        # "model" => "dense_new",
        "model" => "dense_spec",
        # "model" => "conv_new",

        # "n_neurons" => [128, 64, 32],
        # "n_neurons" => [64, 128, 32, 64, 16],
        "n_neurons" => [64],


        # "learning_rule" => "hebb",
        "learning_rule" => "oja",
        # "learning_rule" => "instar",
        # "learning_rule" => "fuzzyart",

        # "init" => Flux.rand32,
        "init" => Flux.glorot_uniform,

        # "middle_activation" => sigmoid_fast,
        "middle_activation" => Flux.tanh_fast,
        # "middle_activation" => Flux.celu,

        "positive_weights" => true,
        # "positive_weights" => false,

        "beta_normalize" => false,
        # "beta_normalize" => true,

        # "beta_rule" => "wta",
        "beta_rule" => "contrast",
        # "beta_rule" => "softmax",
        # "beta_rule" => "wavelet",
        # "sigma" => 1.0f0,
        "sigma" => 0.2,

        # "cc" => true,
        "cc" => false,

        # "model_spec" => []
    ),

    "profile" => false,
    # "profile" => true,

    # "dataset" => "wine",
    # "dataset" => "iris",
    # "dataset" => "wave",
    # "dataset" => "face",
    # "dataset" => "flag",
    # "dataset" => "halfring",
    # "dataset" => "moon",
    # "dataset" => "ring",
    # "dataset" => "spiral",
    # "dataset" => "mnist",
    # "dataset" => "fashionmnist",
    "dataset" => "usps",

    "n_train" => 50000,
    "n_test" => 10000,
    # "flatten" => true,
    "rng_seed" => 1235,
)

@info "------- Options post-processing -------"

# Correct for Float32 types
opts["model_opts"]["eta"] = Float32(opts["model_opts"]["eta"])
opts["model_opts"]["beta_d"] = Float32(opts["model_opts"]["beta_d"])
opts["model_opts"]["sigma"] = Float32(opts["model_opts"]["sigma"])
Random.seed!(opts["rng_seed"])

if opts["model_opts"]["beta_rule"] == "wavelet"
    n_samples = 10000
    plot_range = -0.5

    x = range(-plot_range, plot_range, length=n_samples)
    y = ricker_wavelet.(x, opts["model_opts"]["sigma"])

    min_y = minimum(y)
    inds = findall(x -> x == min_y, y)
    # @info x[inds]
    opts["model_opts"]["wavelet_offset"] = Float32(abs(x[inds][1]))
end

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

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
    if opts["dataset"] in Hebb.datasets["high_dimensional"]
        Hebb.view_weight(model, 1)
    # else
        # @info model[2].weight
        # @info sum(model[2].weight)
    end
end
