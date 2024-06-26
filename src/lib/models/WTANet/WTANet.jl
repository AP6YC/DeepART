"""
    WTANet.jl

# Description
An implementation of a simple deep feedforward network trained with winner-take-all loss.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Options for the construction and usage of a [`WTANet`](@ref) module.
"""
@with_kw struct opts_WTANet
    """
    The vigilance parameter of the [`WTANet`](@ref) module, rho ∈ (0.0, 1.0].
    """
    rho::Float = 0.6; @assert rho > 0.0 && rho <= 1.0

    """
    Name of the optimiser to use.
    """
    optimiser::Symbol = :Descent

    """
    Simple dense specifier for the model.
    """
    model_spec::DenseSpecifier = [2, 10, 10]
end

"""
Container for the stateful information of a WTANet module.
"""
struct WTANet{
    T <: Flux.Chain,
    U <: NamedTuple,
    # U <: Flux.Optimise.AbstractOptimiser,
}
    """
    The feedforward network.
    """
    model::T

    """
    Container for the optimiser.
    """
    optim::U

    """
    The options for construction and usage.
    """
    opts::opts_WTANet
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Keyword argument constructor for a [`WTANet`](@ref) module passing the keyword arguments to the [`opts_WTANet`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function WTANet(;kwargs...)
    # Create the options from the keyword arguments
    opts = opts_WTANet(;kwargs...)

    # Instantiate and return a constructed module
    return WTANet(
        opts,
    )
end

"""
Constructor for a [`WTANet`](@ref) taking a [`opts_WTANet`](@ref) for construction options.

# Arguments
- `opts::opts_WTANet`: the [`opts_WTANet`](@ref) that specifies the construction options.
"""
function WTANet(
    opts::opts_WTANet
)
    # Create the model as a simple dense network
    model = get_dense(opts.model_spec)

    local_optim = eval(opts.optimiser)()

    optim = Flux.setup(local_optim, model)

    # Construct and return the field
    return WTANet(
        model,
        optim,
        opts,
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# function train!(model::WTANet, )
# end

# function wta_loss()
# end
