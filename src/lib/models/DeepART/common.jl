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


const ARTStats = Dict{String, Any}

"""
Initializes an ARTStats dictionary with zero entries.
"""
function build_art_stats()
    # Create the stats dictionary
    stats = ARTStats()

    # Initialize zero entries for each element
    stats["M"] = 0.0
    stats["T"] = 0.0
    stats["bmu"] = 0
    stats["mismatch"] = false

    # Return the zero-initialized stats dictionary
    return stats
end

"""
Logs common statistics of an ART module after a training/classification iteration.

# Arguments
- `art::ARTModule`: the ART module that just underwent training/classification.
- `bmu::Integer`: the best-matching unit integer index.
- `mismatch::Bool`: flag of whether there was a mismatch in this iteration.
"""
function log_art_stats!(art::DeepARTModule, bmu::Integer, mismatch::Bool)
    # Overwrite the stats entries
    art.stats["M"] = art.M[bmu]
    art.stats["T"] = art.T[bmu]
    art.stats["bmu"] = bmu
    art.stats["mismatch"] = mismatch

    # Return empty
    return
end


"""
Gets the local learning parameter.
"""
function get_beta(art::DeepARTModule, outs::RealArray)
    local_beta = if art.opts.softwta == true
        # art.opts.beta .* (1 .- outs[ix])
        art.opts.beta .* Flux.softmax(
            outs,
        )
        # art.opts.beta .* (1 .- Flux.softmax(outs[ix]))
    else
        art.opts.beta
    end
    return local_beta
end

"""
Weight update rule for the deep model component of a [`DeepARTModule`](@ref).
"""
function learn_model(
    art::DeepARTModule,
    xf::RealArray;
    y::Integer=0,
)
    weights = Flux.params(art.model)
    acts = Flux.activations(art.model, xf)

    n_layers = length(weights)

    # trainables = weights
    ins = [acts[jx] for jx = 1:2:(n_layers*2)]
    outs = [acts[jx] for jx = 2:2:(n_layers*2)]

    # Leader neuron modification
    if !iszero(y) && art.opts.leader
        # outs[end][:] .= zero(eltype(xf))
        # outs[end][y] = one(eltype(xf))
        outs[end][:] .= -one(Float32)
        outs[end][y] = one(Float32)
        # @info outs[end]
    end

    for ix = 1:n_layers
        if art.opts.update == "art"
            # trainables[ix] .= DeepART.art_learn_cast(ins[ix], trainables[ix], art.opts.beta)
            # Get the local learning parameter beta
            # local_beta = get_beta(art, outs[ix])

            # If the layer is a convolution
            if ndims(weights[ix]) == 4
                full_size = size(weights[ix])
                n_kernels = full_size[4]
                kernel_shape = full_size[1:3]

                unfolded = Flux.NNlib.unfold(ins[ix], full_size)
                local_in = reshape(
                    mean(
                        reshape(unfolded, :, kernel_shape...),
                        dims=1,
                    ),
                    :
                )

                # Get the averaged and reshaped local output
                local_out = reshape(mean(outs[ix], dims=(1, 2)), n_kernels)
                # Reshape the weights to be (n_kernels, n_features)
                local_weight = reshape(weights[ix], :, n_kernels)'
                # Get the local learning parameter beta
                local_beta = get_beta(art, local_out)

                local_weight .= DeepART.art_learn_cast(
                    local_in,
                    local_weight,
                    local_beta,
                )
            else
                local_weight = weights[ix]
                local_in = ins[ix]
                local_out = outs[ix]
                local_beta = get_beta(art, local_out)

                local_weight .= DeepART.art_learn_cast(
                    local_in,
                    local_weight,
                    local_beta,
                )
            end

            # weights[ix] .= DeepART.art_learn_cast(
            #     ins[ix],
            #     weights[ix],
            #     local_beta,
            # )
        elseif art.opts.update == "instar"
            weights[ix] .+= DeepART.instar(
                ins[ix],
                outs[ix],
                weights[ix],
                art.opts.beta,
            )
        else
            error("Invalid update method: $(art.opts.update)")
        end
    end

    # for ix in eachindex(trainables)
        # weights[ix] .+= DeepART.instar(inputs[ix], acts[ix], weights[ix], eta)
    # end
    # DeepART.instar(xf, acts, model, 0.0001)

    return acts
end
