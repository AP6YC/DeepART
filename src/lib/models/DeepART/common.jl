"""
Common code for DeepART modules.
"""

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Returns the element-wise minimum between sample x and weight W.

# Arguments
- `x::RealVector`: the input sample.
- `W::RealVector`: the weight vector to compare the sample against.
"""
function element_min(x::RealVector, W::RealVector)
    # Get the length of the sample
    n_el = length(x)
    # Create a destination in memory of zeros of type and size like the sample
    min_vec = zero(x)
    # Iterate over every element of the sample
    for ix = 1:n_el
        # Get and assign the minimum of the sample and weight at index ix
        @inbounds min_vec[ix] = min(x[ix], W[ix])
    end
    # Return the element-minimum vector
    return min_vec
    # return @inbounds vec(minimum([x W], dims = 2))
end

"""
Low-level common function for computing the 1-norm of the element minimum of a sample and weights.

# Arguments
$(X_ARG_DOCSTRING)
$(W_ARG_DOCSTING)
"""
function x_W_min_norm(x::RealVector, W::RealVector)
    # return @inbounds norm(element_min(x, get_sample(W, index)), 1)
    return norm(element_min(x, W), 1)
end

"""
Low-level common function for computing the 1-norm of just the weight vector.

# Arguments
$(W_ARG_DOCSTING)
"""
function W_norm(W::RealVector)
    return norm(W, 1)
end

"""
Basic match function.

$(ART_X_W_ARGS)
"""
function basic_match(art::ARTModule, x::RealVector, W::RealVector)
    # return norm(element_min(x, get_sample(W, index)), 1) / art.config.dim
    return x_W_min_norm(x, W) / art.config.dim
end

"""
Simplified FuzzyARTMAP activation function.

$(ART_X_W_ARGS)
"""
function basic_activation(art::ARTModule, x::RealVector, W::RealVector)
    # return norm(element_min(x, get_sample(W, index)), 1) / (art.opts.alpha + norm(get_sample(W, index), 1))
    return x_W_min_norm(x, W) / (art.opts.alpha + W_norm(W))
end

"""
Instar learning rule.

# Arguments
$X_ARG_DOCSTRING
$W_ARG_DOCSTING,
- `eta::Float`: learning rate.
"""
function instar(
    x::AbstractArray,
    y::AbstractArray,
    W::AbstractArray,
    # x::RealVector,
    # W::RealVector,
    eta::Float,
)
    # return W .+ eta .* (x .- W)

    Wy, Wx = size(W)
    # @info size(W)
    local_x = repeat(x', Wy, 1)
    local_y = repeat(y, 1, Wx)
    # @info "W: $(size(W)), local_x: $(size(local_x)), local_y: $(size(local_y)), W: $(size(W))"
    return eta .* local_y .* (local_x .- W)
    # return eta .* local_x .* W

    # return eta .* y .* (x .- W)
    # return eta .* x .* W
end

"""
"""
function instar(
    # x::Tuple,
    x::RealArray,
    y::Tuple,
    W::Flux.Chain,
    eta::Float,
)
    # Iterate over the layers
    # for layer, actiativation in W, x

    # for ix in eachindex(W)
    # for ix in eachindex(Flux.params(W))
    for ix in eachindex(y)
        weights = Flux.params(W)[ix * 2 - 1]
        activation = y[ix]
        if ix == 1
            input = x
        else
            input = y[ix - 1]
        end

        # @info "weights: $(size(weights)), act: $(size(activation)), input: $(size(input))"

        # weights .= instar(activation, weights, 0.1)
        weights .+= instar(input, activation, weights, eta)
        # Update the weights
        # layer.weight = instar(activation, layer.weight, 0.1)
        # layer.bias = instar(activation, layer.bias, 0.1)
    end
    return
end
