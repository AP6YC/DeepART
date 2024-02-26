"""
    deep.jl

3. Both fields are deep.

# Description
TODO
"""

# -----------------------------------------------------------------------------
# EWC updating each loop
# -----------------------------------------------------------------------------

# Flux.Optimisers.@def struct EWC <: Flux.Optimisers.AbstractRule
@with_kw mutable struct EWCIncremental <: Flux.Optimisers.AbstractRule
    eta::Float = 0.01      # learning rate
    lambda::Float = 0.1    # regularization strength
    decay::Float = 0.9     # decay rate
    alpha::Float = 0.1
    # enabled::Bool = true
end

mutable struct EWCIncrementalState
    FIM
    old_params
end

function Flux.Optimisers.apply!(o::EWCIncremental, state, x, dx)
    # Because the FIM is a function of the gradients, initialize it here
    if isnothing(state.FIM)
        # state.FIM = dx .* dx
        state.FIM = dx .^ 2
    else
        # state.FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .* dx
        state.FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .^ 2
    end
    # eta = convert(float(eltype(x)), o.eta)
    # Flux.params(model)[ix] .-= (lambda / 2) * (Flux.params(model)[ix] .- mu[ix]) .* local_FIM
    return state, (o.lambda / 2) * (x .- state.old_params) .* state.FIM
    # return state, Flux.Optimisers.@lazy dx * eta  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

function Flux.Optimisers.init(o::EWCIncremental, x::AbstractArray)
    return EWCIncrementalState(nothing, x)
end

# -----------------------------------------------------------------------------
# EWC "Traditional"
# -----------------------------------------------------------------------------


# Flux.Optimisers.@def struct EWC <: Flux.Optimisers.AbstractRule
@with_kw mutable struct EWC <: Flux.Optimisers.AbstractRule
    eta::Float = 0.01      # learning rate
    lambda::Float = 0.1    # regularization strength
    decay::Float = 0.9     # decay rate
    alpha::Float = 0.1
    new_task::Bool = true
end

struct EWCState
    FIM
    old_params
end

function Flux.Optimisers.apply!(o::EWC, state, x, dx)
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
    else
        new_state = state
        return_dx = (o.lambda / 2) * (x .- state.old_params) .* state.FIM
        # return state, return_dx
    end

    return new_state, return_dx

    # eta = convert(float(eltype(x)), o.eta)
    # Flux.params(model)[ix] .-= (lambda / 2) * (Flux.params(model)[ix] .- mu[ix]) .* local_FIM
    # return state, Flux.Optimisers.@lazy dx * eta  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

function Flux.Optimisers.init(o::EWC, x::AbstractArray)
    return EWCState(nothing, x)
end
