include("definitions.jl")

import .Hebb

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

opts = Dict{String, Any}(
    "n_epochs" => 1000,
    "eta" => 0.1,
    "beta_d" => 0.0,
    "final_sigmoid" => false,
    # "immediate" => true,
    "immediate" => false,
    "gpu" => false,
    "profile" => false,
    # "profile" => true,

    "model" => "dense",
    # "model" => "small_dense",
    # "model" => "fuzzy",
    # "model" => "conv",

    # "init" => Flux.rand32,
    "init" => Flux.glorot_uniform,

    # "positive_weights" => true,
    "positive_weights" => false,

    "wta" => true,
    # "wta" => false,

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

    "n_train" => 10000,
    "n_test" => 10000,
    # "flatten" => true,
    "rng_seed" => 1235,
    "bias" => false,
)

# Correct for Float32 types
opts["eta"] = Float32(opts["eta"])
opts["beta_d"] = Float32(opts["beta_d"])

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

data = Hebb.get_data(opts)

dev_x, dev_y = data.train[1]
# n_input = size(dev_x)[1]
# n_class = length(unique(data.train.y))

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

@info "------- Constructing model -------"

model1 = Hebb.construct_model(data, opts)

opts["model"] = "dense_new"

model2 = Hebb.construct_model(data, opts)
