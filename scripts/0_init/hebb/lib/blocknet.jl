"""
    blocknet.jl

# Description
Definitions for the BlockNet model.
"""

const BlockOpts = Dict{String, Any}

abstract type Block end

# struct ChainBlock <: Block
#     chain::Flux.Chain
# end

struct FuzzyBlock <: Block
    chain::Flux.Chain
end

struct BlockNet
    layers::Vector{Block}
    opts::BlockOpts
end
