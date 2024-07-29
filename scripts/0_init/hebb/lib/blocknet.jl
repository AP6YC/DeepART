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

# struct ChainBlock <: Block
#     chain::Flux.Chain
# end

struct ChainBlock <: Block
    chain::GroupedCCChain
end

function forward(block::ChainBlock, x)
    return block.chain(x)
end

struct ARTBlock <: Block
    model::ARTModule
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
