"""
    blocknet.jl

# Description
Definitions for the BlockNet model.
"""

# -----------------------------------------------------------------------------
# ABSTRACT TYPES
# -----------------------------------------------------------------------------

"""
Block abstract type.

This abstract type enforces the following contract:
- A block must have a `forward` method.
"""
abstract type Block end

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
BlockOpts are ModelOpts that also have an index and input indices.
"""
const BlockOpts = Dict{String, Any}

# const BlockSimOpts = Dict{String, Any}

# const BlockFileOpts = Dict{String, Union{BlockOpts, BlockSimOpts}}

# -----------------------------------------------------------------------------
# CHAINS
# -----------------------------------------------------------------------------

function get_dense_chain(
    # n_in::Tuple,
    n_in::Integer,
    n_out::Integer,
    opts::ModelOpts;
    first_layer::Bool = false,
)
    first_activation = if first_layer
        identity
    elseif opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    input_dim = opts["bias"] ? n_in + 1 : n_in

    preprocess = if opts["layer_norm"] && !first_layer
        # LayerNorm(n_in, affine=false)
        Flux.normalise
    else
        identity
    end

    return Flux.@autosize (input_dim,) Chain(
    # return Flux.@autosize n_in Chain(
        Chain(
            # RandomTransform(_, 16, opts),
            # first_layer ? identity : sigmoid_fast,
            # first_layer ? identity : opts["middle_activation"],
            # LayerNorm(_, affine=false),
            # opts["layer_norm"] ? LayerNorm(input_dim, affine=false) : identity,
            preprocess,
            first_activation,
            opts["cc"] ? DeepART.CC() : identity,
        ),
        Dense(
            _, n_out,
            opts["post_synaptic"] ? opts["middle_activation"] : identity,
            # bias = opts["bias"],
            bias = false,
            init = opts["init"],
        ),
    )
end

# struct RandomTransform
#     chain::Dense
# end

# function RandomTransform(
#     n_in::Integer,
#     n_out::Integer,
#     opts::ModelOpts,
# )
#     return RandomTransform(
#         Dense(
#             n_in,
#             n_out,
#             # sigmoid_fast,
#             init = opts["init"],
#             opts["middle_activation"],
#             bias = false
#         )
#     )
# end

# function (m::RandomTransform)(x)
#     return m.chain(x)
# end

# Flux.@layer RandomTransform

# Flux.trainable(m::RandomTransform) = ()

function get_fuzzy_chain(
    n_in::Integer,
    # n_in::Tuple,
    n_out::Integer,
    opts::ModelOpts;
    first_layer::Bool = false,
)
    first_activation = if first_layer
        identity
    elseif opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    preprocess = if opts["layer_norm"] && !first_layer
        # LayerNorm(n_in, affine=false)
        Flux.normalise
    else
        identity
    end

    return Flux.@autosize (n_in,) Chain(
    # return Flux.@autosize n_in Chain(
        Chain(
            # RandomTransform(_, 16),
            # first_layer ? identity : sigmoid_fast,
            # first_layer ? identity : opts["middle_activation"],
            # opts["layer_norm"] ? LayerNorm(n_in, affine=false) : identity,
            preprocess,
            first_activation,
            opts["cc"] ? DeepART.CC() : identity,
        ),
        DeepART.Fuzzy(
            _, n_out,
            opts["post_synaptic"] ? opts["middle_activation"] : identity,
            init = opts["init"],
        ),
    )
end

function get_widrow_hoff_chain(
    n_in::Integer,
    # n_in::Tuple,
    n_out::Integer,
    opts::ModelOpts,
)
    input_dim = opts["bias"] ? n_in + 1 : n_in

    first_activation = if opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    preprocess = if opts["layer_norm"]
        # LayerNorm(n_in, affine=false)
        Flux.normalise
    else
        identity
    end

    return Flux.@autosize (input_dim,) Chain(
    # return Flux.@autosize n_in Chain(
        Chain(
            # identity
            # sigmoid_fast,
            # opts["middle_activation"],
            # opts["layer_norm"] ? LayerNorm(input_dim, affine=false) : identity,
            preprocess,
            first_activation,
        ),
        Dense(
            _, n_out,
            # bias = opts["bias"],
            opts["final_sigmoid"] ? sigmoid_fast : identity,
            # init = opts["init"],
            init = Flux.glorot_uniform,
            bias = false,
        ),
    )
end

const CHAIN_BLOCK_FUNC_MAP = Dict(
    "dense" => get_dense_chain,
    "fuzzy" => get_fuzzy_chain,
    "widrow_hoff" => get_widrow_hoff_chain,
)

# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------

struct ChainBlock{T <: Flux.Chain} <: Block
    chain::T
    opts::BlockOpts
end

function ChainBlock(
    opts::BlockOpts;
    n_inputs::Integer=0,
    n_outputs::Integer=0,
    # n_inputs::Tuple=(0,),
    # n_outputs::Tuple=(0,),
)
    # Determine the actual number of neurons per layer that we will have
    n_neurons = opts["n_neurons"]
    if n_inputs != 0
        n_neurons = [n_inputs, n_neurons...]
    end
    if n_outputs != 0
        n_neurons = [n_neurons..., n_outputs]
    end

    # The number of layers is determined by the above
    # (i.e., if we are at the input or output of the model)
    n_layers = length(n_neurons)

    # Determine if this is the first layer
    first_layer = opts["index"] == 1

    # Get the layer function
    layer_func = CHAIN_BLOCK_FUNC_MAP[opts["model"]]

    # Create the model
    model = Chain(
        (
            layer_func(n_neurons[ix], n_neurons[ix + 1], opts, first_layer=first_layer)
            for ix = 1:n_layers - 1
        )...,
    )

    # Enforce positive weights if necessary
    if opts["positive_weights"]
        ps = Flux.params(model)
        for p in ps
            p .= abs.(p)
            p .= p ./ maximum(p)
        end
    end

    return ChainBlock(model, opts)
end

function forward(block::ChainBlock, x)
    return block.chain(x)
end

function train(block::ChainBlock, x, y)
    return train(block.chain, x, y)
end

struct ARTBlock{T <: ARTModule} <: Block
    model::T
    opts::BlockOpts
end

function forward(block::ARTBlock, x)
    return classify(block.model, x)
end

function train(block::ARTBlock, x, y)
    return train(block.model, x, y)
end

# -----------------------------------------------------------------------------
# BLOCKNET
# -----------------------------------------------------------------------------


# """
# Many-to-one map of what types of blocks resolve to what types of structs.
# """
# const BLOCK_TYPES = Dict(
#     "dense" => "chain",
#     "fuzzy" => "chain",
#     "conv" => "chain",
#     "widrow_hoff" => "chain",
#     "fuzzyartmap" => "art",
# )

# """
# Map of block types to the functions that create them.
# """
# const BLOCK_FUNC_MAP = Dict(
#     "chain" => ChainBlock,
#     "art" => ARTBlock,
# )

"""
Map of block types to the functions that create them.
"""
const BLOCK_FUNC_MAP = Dict(
    "dense" => ChainBlock,
    "fuzzy" => ChainBlock,
    "conv" => ChainBlock,
    "widrow_hoff" => ChainBlock,
    "fuzzyartmap" => ARTBlock,
)

struct BlockNet
    layers::Vector{Block}
    opts::SimOpts
end

function BlockNet(
    data::DeepART.DataSplit,
    opts::SimOpts,
)
    dev_x, _ = data.train[1]
    n_input = size(dev_x)[1]
    n_class = length(unique(data.train.y))

    blocks = Vector{Block}()
    previous_size = []

    for block_opts in opts["blocks"]
        local_n_inputs = if block_opts["index"] == 1
            # local_input_size
                # If the convolutional model is selected, create a convolution input tuple
            if block_opts["model"] in ["conv",]
                (size(data.train.x)[1:3]..., 1)
            else
                n_input
            end
        else
            previous_size[1]
        end

        # local_block = BLOCK_FUNC_MAP[BLOCK_TYPES[block_opts["model"]]](
        local_block = BLOCK_FUNC_MAP[block_opts["model"]](
            block_opts,
            n_inputs=local_n_inputs,
        )

        # local_output = local_block.opts["n_neurons"][end]
        previous_size = Flux.outputsize(
            local_block.chain,
            (local_n_inputs,)
        )
        @info previous_size
        push!(blocks, local_block)
    end

    return BlockNet(blocks, opts)
end

# function BlockNet(
#     blocks::Vector{BlockOpts}
# )
#     layers = Vector{Block}()
#     for block in blocks
#         push!(layers, BLOCK_TYPES[block["type"]](block))
#     end
#     return BlockNet(layers, blocks)
# end


function forward(net::BlockNet, x)
    for layer in net.layers
        x = forward(layer, x)
    end
    return x
end

function train!(net::BlockNet, x, y)
    for layer in net.layers
        train(layer, x, y)
    end
    return
end


# function gen_blocks(opts::BlockOpts)
#     blocks = Vector{Block}()
#     for block in opts["blocks"]
#         if block["model"] == "chain"
#             chain = GroupedCCChain(block["model_opts"])
#             push!(blocks, ChainBlock(chain))
#         elseif block["model"] == "art"
#             model = ARTModule(block["model_opts"])
#             push!(blocks, ARTBlock(model))
#         else
#             error("Invalid block type: $(block["type"])")
#         end
#     end
#     return blocks
# end