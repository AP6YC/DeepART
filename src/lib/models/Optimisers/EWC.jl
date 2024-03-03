"""
    EWC.jl

# Description
Elastic Weight Consolidation (EWC) optimiser for Flux.jl via an Optimisers.jl interface.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
The parameters if an EWCIncremental optimiser.
"""
@with_kw mutable struct EWC <: Flux.Optimisers.AbstractRule
    # Flux.Optimisers.@def struct EWC <: Flux.Optimisers.AbstractRule
    eta::Float = 0.01      # learning rate
    lambda::Float = 0.1    # regularization strength
    decay::Float = 0.9     # decay rate
    alpha::Float = 0.1
    new_task::Bool = true
    # enabled::Bool = false
end

"""
Custom state for the [`EWCState`](@ref) optimiser.
"""
struct EWCState
    """
    The Fisher Information Matrix (FIM) approximation.
    """
    FIM

    """
    The 'old parameters' that the FIM are computed on.
    """
    old_params
end

# -----------------------------------------------------------------------------
# LOSS DEFINITION
# -----------------------------------------------------------------------------

@with_kw mutable struct EWCLossOpts
    # Flux.Optimisers.@def struct EWC <: Flux.Optimisers.AbstractRule
    eta::Float = 0.01      # learning rate
    # lambda::Float = 0.1    # regularization strength
    lambda::Float = 100000.0    # regularization strength
    decay::Float = 0.9     # decay rate
    alpha::Float = 0.1
    first_task::Bool = false
    # enabled::Bool = false
end

"""
Custom state for the [`EWCState`](@ref) optimiser.
"""
struct EWCLossState
    """
    The Fisher Information Matrix (FIM) approximation.
    """
    FIM

    """
    The 'old parameters' that the FIM are computed on.
    """
    old_params
end

function EWCLossState()
    return EWCLossState(nothing, nothing)
end

function EWCLossState(state::EWCLossState, o::EWCLossOpts, x, dx)
    if isnothing(state.FIM)
        # @info "dx: $(size(dx)), $(typeof(dx))"
        new_FIM = dx .^ 2
        # new_FIM = dx.* dx
    else
        new_FIM = (1 - o.alpha) .* state.FIM + o.alpha .* dx.^2
        # new_FIM = dx .^ 2
        # new_FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx.^2
        # new_FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .* dx
    end

    return EWCLossState(new_FIM, copy(x))
end

function get_EWC_loss(state::EWCLossState, o::EWCLossOpts, x)
    # return ((o.lambda / 2) * (x .- state.old_params) .* state.FIM) .* o.eta

    # return ((o.lambda / 2) * sum((x .- state.old_params) .* state.FIM) .* o.eta)
    # return o.lambda * sum((state.FIM .* (x .- state.old_params) .^ 2) .* o.eta)
    return o.lambda * sum((state.FIM .* (x .- state.old_params) .^ 2)) # No eta
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

# Init overload for the EWC optimiser
function Flux.Optimisers.init(o::EWC, x::AbstractArray)
    return EWCState(nothing, x)
end

# Apply overload for the EWC optimiser
function Flux.Optimisers.apply!(o::EWC, state, x, dx)
    # If the signal has arrived for a new task, compute the new FIM and do not update weights
    if o.new_task
        # Because the FIM is a function of the gradients, initialize it here
        if isnothing(state.FIM)
            # state.FIM = dx .* dx
            new_FIM = dx .^ 2
        else
            # state.FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .* dx
            new_FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .^ 2
        end

        new_state = EWCState(new_FIM, copy(x))
        o.new_task = false

        return_dx = zero(x)
        # return_dx = dx
        # return new_state, zero(x)
    # Otherwise, copy the state and return the EWC penalty
    else
        new_state = state
        return_dx = ((o.lambda / 2) * (x .- state.old_params) .* state.FIM) .* o.eta
        # return_dx = zero(x)
        # return_dx = dx
        # return state, return_dx
    end

    return new_state, return_dx

    # eta = convert(float(eltype(x)), o.eta)
    # Flux.params(model)[ix] .-= (lambda / 2) * (Flux.params(model)[ix] .- mu[ix]) .* local_FIM
    # return state, Flux.Optimisers.@lazy dx * eta  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

"""
Overload of the show function for [`EWCState`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::EWCState`: the [`EWCState`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    state::EWCState,
)
    s1 = if isnothing(state.FIM)
        nothing
    else
        size(state.FIM)
    end
    s2 = size(state.old_params)

    print(io, "EWC(FIM: $(s1), old_params: $(s2))")
end
