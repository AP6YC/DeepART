"""
    DeeperART.jl

# Description
TODO
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Options container for a [`DeeperART`](@ref) module.
"""
@with_kw struct opts_DeeperART
    """
    The vigilance parameter of the [`DeeperART`](@ref) module, rho ∈ (0.0, 1.0].
    """
    rho::Float = 0.6; @assert rho > 0.0 && rho <= 1.0

    """
    Simple dense specifier for the F1 layer.
    """
    F1_spec::DenseSpecifier = [2, 5, 3]

    """
    Simple dense specifier for the F2 layer.
    """
    F2_spec::DenseSpecifier = [3, 5, 3]
end

"""
Stateful information of a DeeperART module.
"""
struct DeeperART{T <: Flux.Chain, U <: Flux.Chain}
    """
    Feature presentation layer.
    """
    F1::T

    """
    Feedback expectancy layer.
    """
    F2::U

    """
    An [`opts_DeeperART`](@ref) options container.
    """
    opts::opts_DeeperART
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

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

"""
Constructor for a [`DeeperART`](@ref) taking a [`opts_DeeperART`](@ref) for construction options.

# Arguments
- `opts::opts_DeeperART`: the [`opts_DeeperART`](@ref) that specifies the construction options.
"""
function DeeperART(
    opts::opts_DeeperART
)
    # # Create the shared network base
    # shared = get_dense(opts.n_shared)

    # # Create the heads
    # heads = [get_dense(opts.n_heads) for _ = 1:5]

    F1 = get_dense(opts.F1_spec)
    F2 = get_dense(opts.F2_spec)

    # Construct and return the field
    return DeeperART(
        F1,
        F2,
        opts,
    )
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`DeeperART`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::DeeperART`: the [`DeeperART`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    field::DeeperART,
)
    print(io, "DeeperART(F1: $(field.opts.F1_spec), F2: $(field.opts.F2_spec))")
end
