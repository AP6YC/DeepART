"""
    deep.jl

3. Both fields are deep.

# Description
TODO
"""

"""
"""
const DEFAULT_N_SHARED = Vector{Int}([10, 20])

"""
"""
const DEFAULT_N_HEADS = Vector{Int}([20. 10])

"""
"""
@with_kw struct opts_MultiHeadField
    """
    """
    n_shared::Vector{Int} = DEFAULT_N_SHARED

    """
    """
    n_headS::Vector{Int} = DEFAULT_N_HEADS
end

"""
Container for a multihead [`DeeperART`](@ref) neural network field.
"""
struct MultiHeadField{T <: Flux.Chain}
    """
    The shared layers.
    """
    shared::T

    """
    The heads of the network.
    """
    heads::Vector{T}
end

"""
"""
function add_head!(model::MultiHeadField)
    local_head = Chain(
		Dense(size_tuple[1]=>120, sigmoid),
		Dense(84=>10, sigmoid),
		# softmax
	)
    push!(model.heads, local_head)

    # Explicitly empty return
    return
end

function MultiHeadField()
    shared = Chain(
		Dense(size_tuple[1]=>120, sigmoid),
		Dense(120=>84, sigmoid),
		Dense(84=>10, sigmoid),
		# softmax
	)

    heads = []
    for ix = 1:5

    end

    return MultiHeadField(
        shared,
        heads,
    )
end

"""
Options container for a [`DeeperART`](@ref) module.
"""
@with_kw struct opts_DeeperART
    rho::Float = 0.6
end

"""
Stateful information of a DeeperART module.
"""
struct DeeperART{T <: Flux.Chain}
    """
    Feature presentation layer.
    """
    F1::T

    """
    Feedback expectancy layer
    """
    F2::T

    """
    Options container
    """
    opts::opts_DeeperART
end

"""
Keyword argument constructor for a [`DeeperART`](@ref) module passing the keyword arguments to the [`opts_DeeperART`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function DeeperART(;kwargs...)
    # Create the options from the keyword arguments
    opts = opts_DeeperART(;kwargs...)

    # Instantiate and return a constructed module
    return DeeperART(
        opts,
    )
end

