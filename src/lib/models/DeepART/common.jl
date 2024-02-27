"""
Common code for DeepART modules.
"""

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
A specifier for the number of nodes per layer in a dense feedforward network.
"""
const DenseSpecifier = Vector{Int}

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Creates a Flux.Chain of Flux.Dense layers according to the hidden layers [`DenseSpecifier`](@ref).

# Arguments
- `n_neurons::DenseSpecifier`: the [`DenseSpecifier`](@ref) that specifies the number of neurons per layer, including the input and output layers.
"""
function get_dense(
    n_neurons::DenseSpecifier
)
    chain_list = [
        Dense(
            n_neurons[ix] => n_neurons[ix + 1],
            sigmoid,
        ) for ix in range(1, length(n_neurons) - 1)
    ]

    # Broadcast until the types are more stable
    # https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain
    local_chain = Chain(chain_list...)

    # Return the chain
    return local_chain
end
