
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
"""
struct ChainContainer{T <: Flux.Chain}
    chain::T
    activations::Vector{Vector{FluxFloat}}
end

"""
Container for a multihead [`DeeperART`](@ref) neural network field.
"""
struct MultiHeadField{T <: Flux.Chain, J <: Flux.Chain}
    """
    The single shared layers object.
    """
    shared::T

    """
    The heads of the network as a list of layers.
    """
    heads::Vector{J}

    """
    Container of the [`opts_MultiHeadField`](@ref) that created this field.
    """
    opts::opts_MultiHeadField
end

"""
Overload of the show function for [`MultiHeadField`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::MultiHeadField`: the [`MultiHeadField`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    field::MultiHeadField,
)
    print(io, "MultiHeadField(shared: $(field.opts.n_shared), heads: $(length(field.heads)) x $(field.opts.n_heads))")
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

"""
Constructor for a [`MultiHeadField`](@ref) taking a [`opts_MultiHeadField`](@ref) for construction options.

# Arguments
- `opts::opts_MultiHeadField`: the [`opts_MultiHeadField`](@ref) that specifies the construction options.
"""
function MultiHeadField(
    opts::opts_MultiHeadField
)
    # Create the shared network base
    shared = get_dense(opts.n_shared)

    # Create the heads
    heads = [get_dense(opts.n_heads) for _ = 1:5]

    # Construct and return the field
    return MultiHeadField(
        shared,
        heads,
        opts,
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
Computes the forward pass for a [`MultiHeadField`](@ref).

# Arguments
- `field::MultiHeadField`: the [`MultiHeadField`](@ref) object to compute activations for.
"""
function forward(field::MultiHeadField, input::RealArray)
    outs_shared = field.shared(input)
    outs_heads = [
        field.heads[ix](outs_shared) for ix = 1:length(field.heads)
    ]
    return outs_heads
end
