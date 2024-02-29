"""
    IEWC.jl

# Description
Incremental Elastic Weight Consolidation (IEWC) optimiser for Flux.jl via an Optimisers.jl interface.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
The parameters if an IEWC optimiser.
"""
@with_kw mutable struct IEWC <: Flux.Optimisers.AbstractRule
    # Flux.Optimisers.@def struct EWC <: Flux.Optimisers.AbstractRule
    eta::Float = 0.01      # learning rate
    lambda::Float = 0.1    # regularization strength
    decay::Float = 0.9     # decay rate
    alpha::Float = 0.1
    new_task::Bool = true
end

"""
Custom state for the [`IEWC`](@ref) optimiser.
"""
mutable struct IEWCState{T <: AbstractArray, U <: AbstractArray}
    """
    The Fisher Information Matrix (FIM) approximation.
    """
    FIM::Vector{T}

    """
    The 'old parameters'.
    """
    old_params::Vector{U}
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

# Init overload for the IEWC optimiser
function Flux.Optimisers.init(o::IEWC, x::AbstractArray)
    return IEWCState(nothing, x)
end

# Apply overload for the IEWC optimiser
function Flux.Optimisers.apply!(o::IEWC, state, x, dx)    # If the signal has arrived for a new task, compute the new FIM and do not update weights
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
        # return new_state, zero(x)
    # Otherwise, copy the state and return the EWC penalty
    else
        new_state = state
        return_dx = (o.lambda / 2) * (x .- state.old_params) .* state.FIM
        # return state, return_dx
    end
    # # Because the FIM is a function of the gradients, initialize it here
    # if isnothing(state.FIM)
    #     # state.FIM = dx .* dx
    #     state.FIM = dx .^ 2
    # else
    #     # state.FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .* dx
    #     state.FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .^ 2
    # end
    # # eta = convert(float(eltype(x)), o.eta)
    # # Flux.params(model)[ix] .-= (lambda / 2) * (Flux.params(model)[ix] .- mu[ix]) .* local_FIM
    # return state, (o.lambda / 2) * (x .- state.old_params) .* state.FIM
    # # return state, Flux.Optimisers.@lazy dx * eta  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

"""
Overload of the show function for [`IEWCState`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::IEWCState`: the [`IEWCState`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    state::IEWCState,
)
    s1 = size(state.FIM)
    s2 = size(state.old_params)
    print(io, "IEWCState(FIM: $(s1), old_params: $(s2))")
end
