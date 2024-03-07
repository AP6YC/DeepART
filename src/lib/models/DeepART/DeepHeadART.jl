"""
    DeepHeadART.jl

# Description
TODO
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Options container for a [`DeepHeadART`](@ref) module.
"""
@with_kw struct opts_DeepHeadART
    """
    The vigilance parameter of the [`DeepHeadART`](@ref) module, rho ∈ (0.0, 1.0].
    """
    rho::Float = 0.6; @assert rho > 0.0 && rho <= 1.0

    """
    Simple dense specifier for the F1 layer.
    """
    F1_spec::DenseSpecifier = [2, 5, 3]

    """
    Shared dense specifier for the F2 layer.
    """
    F2_shared::DenseSpecifier = [3, 6, 3]

    """
    Shared dense specifier for the F2 layer.
    """
    F2_heads::DenseSpecifier = [3, 5, 3]

    """
    Instar learning rate.
    """
    eta::Float = 0.1
end

"""
Stateful information of a DeepHeadART module.
"""
struct DeepHeadART{T <: Flux.Chain, U <: Flux.Chain, V <: Flux.Chain}
    """
    Feature presentation layer.
    """
    F1::T

    """
    Feedback expectancy layer.
    """
    F2::MultiHeadField{U, V}

    """
    An [`opts_DeepHeadART`](@ref) options container.
    """
    opts::opts_DeepHeadART
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Keyword argument constructor for a [`DeepHeadART`](@ref) module passing the keyword arguments to the [`opts_DeepHeadART`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function DeepHeadART(;kwargs...)
    # Create the options from the keyword arguments
    opts = opts_DeepHeadART(;kwargs...)

    # Instantiate and return a constructed module
    return DeepHeadART(
        opts,
    )
end

"""
Constructor for a [`DeepHeadART`](@ref) taking a [`opts_DeepHeadART`](@ref) for construction options.

# Arguments
- `opts::opts_DeepHeadART`: the [`opts_DeepHeadART`](@ref) that specifies the construction options.
"""
function DeepHeadART(
    opts::opts_DeepHeadART
)
    # # Create the shared network base
    # shared = get_dense(opts.shared_spec)

    # # Create the heads
    # heads = [get_dense(opts.head_spec) for _ = 1:5]

    F1 = get_dense(opts.F1_spec)
    # F2 = get_dense(opts.F2_spec)
    F2 = MultiHeadField(
        shared_spec=opts.F2_shared,
        head_spec=opts.F2_heads,
    )

    # Construct and return the field
    return DeepHeadART(
        F1,
        F2,
        opts,
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Forward pass for a [`DeepHeadART`](@ref) module.

# Arguments
$ARG_DEEPHEADART
$ARG_X
"""
function forward(
    art::DeepHeadART,
    x::RealArray,
)
    f1 = art.F1(x)
    f2 = forward(art.F2, f1)

    return f1, f2
end

"""
Forward pass for a [`DeepHeadART`](@ref) module with activations.

# Arguments
$ARG_DEEPHEADART
$ARG_X
"""
function multi_activations(
    art::DeepHeadART,
    x::RealArray,
)
    f1 = Flux.activations(art.F1, x)
    f2 = multi_activations(art.F2, f1[end])
    # f1 = Flux.activations(field.F1, x)
    # f2 = Flux.activations(field.F2, f1[end])

    return f1, f2
end

"""
Adds a node to the F2 layer of the [`DeepHeadART`](@ref) module.

# Arguments
$ARG_DEEPHEADART
$ARG_X
"""
function add_node!(
    art::DeepHeadART,
    x::RealArray,
)

    add_node!(art.F2, x)

    return
end

function learn!(
    art::DeepHeadART,
    # x::RealArray,
    activations::Tuple,
    index::Int,
)



    return
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`DeepHeadART`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::DeepHeadART`: the [`DeepHeadART`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    field::DeepHeadART,
)
    print(io, "DeepHeadART(F1: $(field.opts.F1_spec), F2: $(field.F2))")
end
