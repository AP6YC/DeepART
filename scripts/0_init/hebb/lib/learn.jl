"""
    learn.jl

# Description
Definitions for local learning rules.
"""

# -----------------------------------------------------------------------------
# LEARNING FUNCTIONS
# -----------------------------------------------------------------------------

function fuzzyart_learn(x, W, beta)
    return beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
end

function fuzzyart_learn_cast(x, W, beta)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    # _x = repeat(x, 1, Wx)
    _beta = repeat(beta, 1, Wx)

    # result = beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
    # result = beta .* minimum(cat(_x, W, dims=3), dims=3) + W .* (one(eltype(_beta)) .- _beta)
    result = beta .* min.(_x, W) + W .* (one(eltype(_beta)) .- _beta)
    return result
end

function instar_cast(x, W, y, beta)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    _beta = repeat(beta, 1, Wx)
    _y = repeat(y, 1, Wx)

    # dW = beta .* y .* (x .- W)
    result = _beta .* _y .* (_x .- W)
    return result
end

function oja_cast(x, W, y, beta)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    _beta = repeat(beta, 1, Wx)
    _y = repeat(y, 1, Wx)

    # dW = beta .* y .* (x .- y .* W)
    result = _beta .* _y .* (_x .- _y .* W)
    return result
end

function widrow_hoff_cast(weights, target, out, input, eta::Real)
    Wy, Wx = size(weights)
    _input = repeat(input', Wy, 1)
    # _target = repeat(target, 1, Wx)
    # _out = repeat(out, 1, Wx)
    middle = repeat(target .- out, 1, Wx)

    # weights[iw, :] .+= eta .* (target[iw] .- out[iw]) .* input
    # result = eta .* (_target .- _out) .* _input
    result = eta .* middle .* _input
    return result
end

function widrow_hoff_learn!(input, out, weights, target, opts::ModelOpts)
    weights .+= widrow_hoff_cast(weights, target, out, input, opts["eta"])
    return
end

function ricker_wavelet(t, sigma)
    # sigma = 1.0f0
    return 2.0f0 / (sqrt(3.0f0 * sigma) * pi^(1.0f0 / 4.0f0)) * (1.0f0 - (t / sigma)^2) * exp(-t^2 / (2.0f0 * sigma^2))
end

function gaussian(t, sigma)
    return exp(-t^2 / (2.0f0 * sigma^2))
end

function get_beta(out, opts::ModelOpts)
    if opts["beta_rule"] == "wta"
        # beta = zeros(Float32, size(out))
        # max_ind = argmax(out)
        # beta[max_ind] = opts["beta_d"] * one(Float32)
        beta = opts["beta_d"]

    elseif opts["beta_rule"] == "contrast"
        # Krotov / contrastive learning
        max_ind = argmax(out)
        local_soft = Flux.softmax(out)
        max_soft = opts["beta_normalize"] ? maximum(local_soft) : 1.0f0
        local_soft = -local_soft
        local_soft[max_ind] = -local_soft[max_ind]
        beta = opts["beta_d"] .* local_soft ./ max_soft
    elseif opts["beta_rule"] == "softmax"
        # Softmax
        local_soft = Flux.softmax(out)
        max_soft = opts["beta_normalize"] ? maximum(local_soft) : 1.0f0
        beta = opts["beta_d"] .* local_soft ./ max_soft
    elseif opts["beta_rule"] == "wavelet"
        # Ricker wavelet
        ind_max = argmax(out)
        wavelet_inputs = abs.(out .- out[ind_max])
        for ix in eachindex(wavelet_inputs)
            if ix != ind_max
                wavelet_inputs[ix] = wavelet_inputs[ix] + opts["wavelet_offset"]
            end
        end
        wavelet = ricker_wavelet.(wavelet_inputs, opts["sigma"])

        max_wave = opts["beta_normalize"] ? maximum(wavelet) : 1.0f0
        beta = opts["beta_d"] .* wavelet ./ max_wave
    elseif opts["beta_rule"] == "gaussian"
        # Gaussian
        ind_max = argmax(out)
        gaussian_inputs = abs.(out .- out[ind_max])
        gaussian_outs = gaussian.(gaussian_inputs, opts["sigma"])
        max_gaussian_outs = opts["beta_normalize"] ? maximum(gaussian_outs) : 1.0f0
        gaussian_outs = -gaussian_outs
        gaussian_outs[ind_max] = -gaussian_outs[ind_max]
        beta = opts["beta_d"] .* gaussian_outs ./ max_gaussian_outs
    # else
    #     error("Incorrect beta rule option ($(opts["beta_rule"])), must be in BETA_RULES")
    end

    return beta
end

function deepart_learn!(input, out, weights, opts::ModelOpts)
    # return beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
    if ndims(weights) == 4
        if opts["conv_strategy"] == "unfold"
            # full_size = size(weights[ix])
            full_size = size(weights)
            n_kernels = full_size[4]
            kernel_shape = full_size[1:3]

            unfolded = Flux.NNlib.unfold(input, full_size)
            local_in = reshape(mean(reshape(unfolded, :, kernel_shape...), dims=1), :)

            # Get the averaged and reshaped local output
            local_out = reshape(mean(out, dims=(1, 2)), n_kernels)

            # Reshape the weights to be (n_kernels, n_features)
            local_weight = reshape(weights, :, n_kernels)'
        elseif opts["conv_strategy"] == "patchwise"
            full_size = size(weights)
            n_kernels = full_size[4]
            kernel_shape = full_size[1:3]

            unfolded = Flux.NNlib.unfold(
                input,
                full_size,
                # pad=(2,2),
            )
            flat_in = unfolded[:, :, 1]
            n_windows = size(flat_in, 1)
            flat_out = reshape(out, n_windows, n_kernels)
            flat_weights = reshape(weights, :, n_kernels)'

            # Iterate over each unfolded window
            for ix in 1:n_windows

                local_in = flat_in[ix, :]
                local_out = flat_out[ix, :]
                local_weight = flat_weights

                # Get the local learning parameter beta
                beta = get_beta(local_out, opts)
                if opts["learning_rule"] == "fuzzyart"
                    if opts["beta_rule"] == "wta"
                        jx = argmax(local_out)
                        local_weight[jx, :] = fuzzyart_learn(local_in, local_weight[jx, :], beta)
                    else
                        local_weight .= fuzzyart_learn_cast(local_in, local_weight, beta)
                    end
                elseif opts["learning_rule"] == "instar"
                    # local_weight .+= opts["eta"] .* (beta .- local_out) .* local_in
                    # local_weight .+= beta .* local_out .* (local_in .- local_out .* local_weight)
                    local_weight .+= instar_cast(local_in, local_weight, local_out, beta)
                elseif opts["learning_rule"] == "oja"
                    local_weight .+= oja_cast(local_in, local_weight, local_out, beta)
                else
                    error("Incorrect learning rule option ($(opts["learning_rule"])), must be in LEARNING_RULES")
                end
            end

            # Break early because learning happened in each window of the convolution
            return
        end

    # Otherwise, we are in a dense layer
    else
        local_out = out
        local_weight = weights
        local_in = input
    end

    # Get the local learning parameter beta
    beta = get_beta(local_out, opts)
    if opts["learning_rule"] == "fuzzyart"
        # local_weight .= fuzzyart_learn_cast(local_in, local_weight, beta)
        if opts["beta_rule"] == "wta"
            jx = argmax(local_out)
            local_weight[jx, :] = fuzzyart_learn(local_in, local_weight[jx, :], beta)
        else
            local_weight .= fuzzyart_learn_cast(local_in, local_weight, beta)
        end
    elseif opts["learning_rule"] == "instar"
        # local_weight .+= opts["eta"] .* (beta .- local_out) .* local_in
        # local_weight .+= beta .* local_out .* (local_in .- local_out .* local_weight)
        local_weight .+= instar_cast(local_in, local_weight, local_out, beta)
    elseif opts["learning_rule"] == "oja"
        local_weight .+= oja_cast(local_in, local_weight, local_out, beta)
    else
        error("Incorrect learning rule option ($(opts["learning_rule"])), must be in LEARNING_RULES")
    end
    # weights .= fuzzyart_learn_cast_cache(input, weights, beta, cache)
    return
end