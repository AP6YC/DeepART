"""
    deep.jl

3. Both fields are deep.

# Description
TODO
"""

"""
A specifier for the number of nodes per layer in a dense feedforward network.
"""
const DenseSpecifier = Vector{Int}

"""
The default shared hidden layer as a list of a number of nodes per layer, including the inputs.
"""
const DEFAULT_N_SHARED = DenseSpecifier([10, 20])

"""
The default shared head layers as a list of a number of nodes per layer, including the outputs.
"""
const DEFAULT_N_HEADS = DenseSpecifier([20, 10])

"""
The options container for a [`MultiHeadField`](@ref) module.
"""
@with_kw struct opts_MultiHeadField
    """
    """
    n_shared::DenseSpecifier = DEFAULT_N_SHARED

    """
    """
    n_heads::DenseSpecifier = DEFAULT_N_HEADS
end

"""
Container for a multihead [`DeeperART`](@ref) neural network field.
"""
struct MultiHeadField{T <: Flux.Chain}
    """
    The single shared layers object.
    """
    shared::T

    """
    The heads of the network as a list of layers.
    """
    heads::Vector{T}
end

# function get_

"""
"""
function get_dense(
    n_hidden::DenseSpecifier
# function get_head(
    # opts::opts_MultiHeadField
)
    local_chain = Chain(
        # [
        (
            Dense(
                n_hidden[ix] => n_hidden[ix + 1],
                sigmoid,
            ) for ix in range(1, length(n_hidden) - 1)
        )...
                # ]
		# Dense(size_tuple[1]=>120, sigmoid),
		# Dense(84=>10, sigmoid),
		# softmax
	)
    return local_chain
end


# """

# """
# function add_head!(
#     model::MultiHeadField
# )

#     push!(model.heads, get_head(model.opts))

#     # Explicitly empty return
#     return
# end

function MultiHeadField(
    opts::opts_MultiHeadField
)
    # shared = Chain(
	# 	Dense(size_tuple[1]=>120, sigmoid),
	# 	Dense(120=>84, sigmoid),
	# 	Dense(84=>10, sigmoid),
	# 	# softmax
	# )

    shared = get_dense(opts.n_shared)

    heads = []
    # for ix = 1:5
    for _ = 1:5
        # add_head!()
        push!(heads, get_dense(opts.n_heads))
    end

    return MultiHeadField(
        shared,
        heads,
    )
end


"""
Keyword argument constructor for a [`MultiHeadField`](@ref) module passing the keyword arguments to the [`opts_MultiHeadField`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function MultiHeadField(;kwargs...)
    # Create the options from the keyword arguments
    opts = opts_MultiHeadField(;kwargs...)

    # Instantiate and return a constructed module
    return MultiHeadField(
        opts,
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
    Feedback expectancy layer.
    """
    F2::T

    """
    An [`opts_DeeperART`](@ref) options container.
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

