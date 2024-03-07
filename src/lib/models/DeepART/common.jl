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
- `x::AbstractVector`: input vector.
- `w::AbstractVector`: weight vector.
- `η::Float`: learning rate.

# Returns
- `w::AbstractVector`: updated weight vector.
"""
function instar(x, w, η)
    return w .+ η .* (x .- w)
end
