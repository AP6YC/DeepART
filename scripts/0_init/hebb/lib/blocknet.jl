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

const BlockOpts = Dict{String, Any}

# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------

struct ChainBlock <: Block
    chain::GroupedCCChain
end

function forward(block::ChainBlock, x)
    return block.chain(x)
end

function train(block::ChainBlock, x, y)
    return train(block.chain, x, y)
end

struct ARTBlock <: Block
    model::ARTModule
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
