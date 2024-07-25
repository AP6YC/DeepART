"""
    chains.jl

# Description
Collection of chain model definitions for the Hebb module.
"""

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

"""
Alias for the model options dictionary.
"""
const ModelOpts = Dict{String, Any}

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Abstract type for containers of Flux.Chain containing CC and non-CC layers.
"""
abstract type CCChain end

"""
Chains that group alternate CC and non-CC chain layers.
"""
struct GroupedCCChain{T <: Flux.Chain} <: CCChain
    chain::T
end

"""
Chains that simply alternate between CC and non-CC layers.
"""
struct AlternatingCCChain{T <: Flux.Chain} <: CCChain
    chain::T
end

# -----------------------------------------------------------------------------
# LAYER AND CHAIN DEFINITIONS
# -----------------------------------------------------------------------------

include("layers.jl")
include("dense.jl")
include("fuzzy.jl")
include("conv.jl")

# -----------------------------------------------------------------------------
# HIGH-LEVEL CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Dispatcher for model construction.
"""
const MODEL_MAP = Dict(
    "fuzzy" => get_fuzzy_model,
    "conv" => get_conv_model,
    "dense" => get_dense_model,
    "dense_new" => get_dense_groupedccchain,
    "fuzzy_new" => get_fuzzy_groupedccchain,
    "conv_new" => get_inc_conv_model,
    "dense_spec" => get_spec_dense_groupedccchain,
    "fuzzy_spec" => get_spec_fuzzy_groupedccchain,
)

function construct_model(
    data::DeepART.DataSplit,
    opts::ModelOpts,
)
    # Sanitize model option
    if !(opts["model"] in keys(MODEL_MAP))
        error("Invalid model type")
    end

    # Get the shape of the dataset
    dev_x, _ = data.train[1]
    n_input = size(dev_x)[1]
    n_class = length(unique(data.train.y))

    # If the convolutional model is selected, create a convolution input tuple
    local_input_size = if opts["model"] in ["conv", "conv_new"]
        (size(data.train.x)[1:3]..., 1)
    else
        n_input
    end

    # Construct the model from the model function map
    model = MODEL_MAP[opts["model"]](
        local_input_size,
        n_class,
        opts,
    )

    # Enforce positive weights if necessary
    if opts["positive_weights"]
        ps = Flux.params(model)
        for p in ps
            p .= abs.(p)
            p .= p ./ maximum(p)
        end
    end

    return model
end

# -----------------------------------------------------------------------------
# CHAIN BEHAVIOR
# -----------------------------------------------------------------------------

function get_weights(model::CCChain)
    return Flux.params(model.chain)
end

function get_activations(model::CCChain, x)
    return Flux.activations(model.chain, x)
end

function get_incremental_activations(
    chain::GroupedCCChain,
    x,
)
    # params
    n_layers = length(chain.chain)

    ins = []
    outs = []

    for ix = 1:n_layers
        pre_input = (ix == 1) ? x : outs[end]
        local_acts = Flux.activations(chain.chain[ix], pre_input)
        push!(ins, local_acts[1])
        push!(outs, local_acts[2])
    end
    return ins, outs
end
