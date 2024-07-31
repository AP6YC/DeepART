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


# """
# Sanitize the options dictionary.
# """
# function sanitize_opts!(opts::SimOpts)

# end

# function load_block_opts(name::AbstractString)::BlockFileOpts
#     # Load the options
#     opts = YAML.load_file(
#         joinpath(@__FILE__, "..", "..", "opts", name);
#         dicttype=BlockFileOpts
#     )

#     # Santize the options and do post-processing
#     sanitize_block_opts!(opts)

#     # Return the options
#     return opts
# end

# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------

struct ChainBlock{T <: Flux.Chain} <: Block
    chain::T
    opts::BlockOpts
end

const model_func_map = Dict(
    "dense" => get_dense_deepart_layer,
    # "conv" => get_conv_deepart_layer,
    "fuzzy" => get_fuzzy_deepart_layer,
    "widrow_hoff" => get_widrow_hoff_layer,
)

function ChainBlock(
    opts::BlockOpts;
    n_inputs::Integer=0,
    # n_outputs::Integer=0
)
    # Determine the actual number of neurons per layer that we will have
    n_neurons = opts["n_neurons"]
    if n_inputs != 0
        n_neurons = [n_inputs, n_neurons...]
    end
    # if n_outputs != 0
    #     n_neurons = [n_neurons..., n_outputs]
    # end

    # The number of layers is determined by the above
    # (i.e., if we are at the input or output of the model)
    n_layers = length(n_neurons)

    # Determine if this is the first layer
    first_layer = opts["index"] == 1

    # Get the layer function
    layer_func = model_func_map[opts["model"]]

    # Create the model
    model = Chain(
        (
            layer_func(n_neurons[ix], n_neurons[ix + 1], opts, first_layer=first_layer)
            for ix = 1:n_layers - 1
        )...,
    )

    return ChainBlock(model, opts)
end

# function ChainBlock(chain::Flux.Chain, opts::BlockOpts)
#     return ChainBlock(chain, opts)
# end

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

struct BlockNet
    layers::Vector{Block}
    opts::BlockOpts
end

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