"""
    MultiHeadField.jl

# Description
Implementation of a multi-headed Flux.jl neural network field.
This does not use Parallel, instead using a shared network with multiple heads to be able to grow the heads.
"""

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
The default shared hidden layer as a list of a number of nodes per layer, including the inputs and outputs.
"""
const DEFAULT_SHARED_SPEC = DenseSpecifier([10, 20])

"""
The default shared head layers as a list of a number of nodes per layer, including the inputs and outputs.
"""
const DEFAULT_HEAD_SPEC = DenseSpecifier([20, 10])

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
The options container for a [`MultiHeadField`](@ref) module.
"""
@with_kw struct opts_MultiHeadField
    """
    The shared hidden layer as a list of a number of nodes per layer, including the inputs and outputs.
    """
    shared_spec::DenseSpecifier = DEFAULT_SHARED_SPEC

    """
    The head layers specifier as a list of a number of nodes per layer, including the inputs and outputs.
    """
    head_spec::DenseSpecifier = DEFAULT_HEAD_SPEC
end

# """
# """
# struct ChainContainer{T <: Flux.Chain}
#     chain::T
#     activations::Vector{Vector{FluxFloat}}
# end

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

# """

# """
# function add_head!(
#     model::MultiHeadField
# )

#     push!(model.heads, get_head(model.opts))

#     # Explicitly empty return
#     return
# end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Constructor for a [`MultiHeadField`](@ref) taking a [`opts_MultiHeadField`](@ref) for construction options.

# Arguments
- `opts::opts_MultiHeadField`: the [`opts_MultiHeadField`](@ref) that specifies the construction options.
"""
function MultiHeadField(
    opts::opts_MultiHeadField
)
    # Create the shared network base
    shared = get_dense(opts.shared_spec)

    # Create the heads
    heads = [get_dense(opts.head_spec) for _ = 1:5]

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

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Computes the forward pass for a [`MultiHeadField`](@ref).

# Arguments
- `field::MultiHeadField`: the [`MultiHeadField`](@ref) object to compute activations for.
- `x::AbstractArray`: the input data.
"""
function forward(
    field::MultiHeadField,
    x::RealArray,
)
    outs_shared = field.shared(x)
    outs_heads = [
        field.heads[ix](outs_shared) for ix = 1:length(field.heads)
    ]
    return outs_heads
end

"""
Computes the forward pass for a [`MultiHeadField`](@ref) and returns the activations of the shared and head layers.

# Arguments
- `field::MultiHeadField`: the [`MultiHeadField`](@ref) object to compute activations for.
- `x::AbstractArray`: the input data.
"""
function multi_activations(
    field::MultiHeadField,
    x::RealArray,
)
    outs_shared = Flux.activations(field.shared, x)

    outs_heads = [
        Flux.activations(field.heads[ix], outs_shared[end]) for ix = 1:length(field.heads)
    ]

    return outs_shared, outs_heads
end

function add_node!(
    field::MultiHeadField,
    x::AbstractArray,
)
    # Get the current head
    # head = field.heads[head_ix]
    push!(field.heads, get_dense(field.opts.head_spec))

    return
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

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
    print(io, "MultiHeadField(shared: $(field.opts.shared_spec), heads: $(length(field.heads)) x $(field.opts.head_spec))")
end
