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

# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------

const CHAIN_FUNC_MAP = Dict(
    "dense" => get_dense_deepart_layer,
    # "conv" => get_conv_deepart_layer,
    "fuzzy" => get_fuzzy_deepart_layer,
    "widrow_hoff" => get_widrow_hoff_layer,
)

struct ChainBlock{T <: Flux.Chain} <: Block
    chain::T
    opts::BlockOpts
end

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
    layer_func = CHAIN_FUNC_MAP[opts["model"]]

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


"""
Many-to-one map of what types of blocks resolve to what types of structs.
"""
const BLOCK_TYPES = Dict(
    "dense" => "chain",
    "fuzzy" => "chain",
    "conv" => "chain",
    "widrow_hoff" => "chain",
    "fuzzyartmap" => "art",
    # "chain" => [
    #     "dense",
    #     "fuzzy",
    #     "conv",
    #     "widrow_hoff",
    # ],
    # "art" => [
    #     "fuzzyartmap",
    # ],
)

"""
Map of block types to the functions that create them.
"""
const BLOCK_FUNC_MAP = Dict(
    "chain" => ChainBlock,
    "art" => ARTBlock,
)


struct BlockNet
    layers::Vector{Block}
    opts::SimOpts
end

function BlockNet(
    opts::SimOpts,
)
    blocks = Vector{Block}()
    for block_opts in opts["blocks"]
        local_block = BLOCK_FUNC_MAP[BLOCK_TYPES[block_opts["model"]]](block_opts)
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