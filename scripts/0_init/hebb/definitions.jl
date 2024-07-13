"""
    definitions.jl

# Description
Definitions for the Hebbian learning module.
"""

module Hebb

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# @info "------- Loading dependencies -------"
using Revise
using DeepART
using Flux
using ProgressMeter
using Random
using CUDA
# using UnicodePlots
using StatsBase: mean
using NumericalTypeAliases

using UnicodePlots

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

include("chains.jl")

const SimOpts = Dict{String, Any}

datasets = Dict(
    "high_dimensional" => [
        "fashionmnist",
        "mnist",
        "usps",
    ],
    "low_dimensional" => [
        "wine",
        "iris",
        "wave",
        "face",
        "flag",
        "halfring",
        "moon",
        "ring",
        "spiral",
    ]
)

function get_data(opts::SimOpts)
    data = if opts["dataset"] in datasets["high_dimensional"]
        DeepART.load_one_dataset(
            opts["dataset"],
            n_train=opts["n_train"],
            n_test=opts["n_test"],
            # flatten=opts["flatten"],
            flatten = !(opts["model_opts"]["model"] in ["conv", "conv_new"]),
        )
    else
        DeepART.load_one_dataset(
            opts["dataset"],
        )
    end
    return data
end

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

# struct HebbModel{T <: Flux.Chain}
#     model::T
#     opts::ModelOpts
# end

struct HebbModel{T <: CCChain}
    model::T
    opts::ModelOpts
end

function HebbModel(
    data::DeepART.DataSplit,
    opts::ModelOpts,
)
    return HebbModel(
        construct_model(data, opts),
        opts,
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

function inspect_weights(model::HebbModel, layer::Integer)
    # weights = get_weights(model.model)
    # return weights[layer]
    return model.model.chain[layer][2].weight
end


function test(
    model::HebbModel,
    data::DeepART.DataSplit,
)
    n_test = length(data.test)

    y_hats = zeros(Int, n_test)
    test_loader = Flux.DataLoader(data.test, batchsize=-1)
    if model.opts["gpu"]
        y_hats = y_hats |> gpu
        test_loader = test_loader |> gpu
    end

    ix = 1
    for (x, _) in test_loader
        y_hats[ix] = argmax(model.model.chain(x))
        ix += 1
    end

    # y_hats = model(data.test.x |> gpu) |> cpu  # first row is prob. of true, second row p(false)
    # y_hats = argmaxmodel(data.test.x)  # first row is prob. of true, second row p(false)

    if model.opts["gpu"]
        y_hats = y_hats |> cpu
    end

    perf = DeepART.AdaptiveResonance.performance(y_hats, data.test.y)
    return perf
end

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

"""
FuzzyART learning rule with minor cachine. Probably deprecated.
"""
function fuzzyart_learn_cast_cache(x, W, beta::Real, cache)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    _beta = repeat(beta, 1, Wx)

    cache[:, :, 1] .= _x
    cache[:, :, 2] .= W

    # result = beta .* minimum(cat(_x, W, dims=3), dims=3) + W .* (one(eltype(_beta)) .- _beta)
    # result = beta .* minimum(cache, dims=3) + W .* (one(eltype(_beta)) .- _beta)
    result = beta .* min.(_x, W) + W .* (one(eltype(_beta)) .- _beta)
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

const BETA_RULES = [
    "wta",
    "contrast",
    "softmax",
]

function ricker_wavelet(t, sigma)
    # sigma = 1.0f0
    return 2.0f0 / (sqrt(3.0f0 * sigma) * pi^(1.0f0 / 4.0f0)) * (1.0f0 - (t / sigma)^2) * exp(-t^2 / (2.0f0 * sigma^2))
end

function get_beta(out, opts::ModelOpts)
    if opts["beta_rule"] == "wta"
        beta = zeros(Float32, size(out))
        max_ind = argmax(out)
        beta[max_ind] = one(Float32)
    elseif opts["beta_rule"] == "contrast"
        max_ind = argmax(out)
        local_soft = Flux.softmax(out)

        max_soft = opts["beta_normalize"] ? maximum(local_soft) : 1.0f0

        local_soft = -local_soft
        local_soft[max_ind] = -local_soft[max_ind]
        beta = opts["beta_d"] .* local_soft ./ max_soft
    elseif opts["beta_rule"] == "softmax"
        local_soft = Flux.softmax(out)
        max_soft = opts["beta_normalize"] ? maximum(local_soft) : 1.0f0
        beta = opts["beta_d"] .* local_soft ./ max_soft
    elseif opts["beta_rule"] == "wavelet"
        # Ricker wavelet
        # local_soft = Flux.softmax(out)
        # wavelet_inputs = out .- local_soft
        # wavelet = ricker_wavelet.(wavelet_inputs, opts["sigma"])
        # beta = opts["beta_d"] .* wavelet

        # local_soft = Flux.softmax(out)
        local_soft=out

        wavelet_inputs = abs.(out .- local_soft)
        ind_max = argmax(wavelet_inputs)
        for ix in eachindex(wavelet_inputs)
            if ix != ind_max
                wavelet_inputs[ix] = wavelet_inputs[ix] + opts["wavelet_offset"]
            end
        end
        wavelet = ricker_wavelet.(wavelet_inputs, opts["sigma"])

        max_wave = opts["beta_normalize"] ? maximum(wavelet) : 1.0f0
        beta = opts["beta_d"] .* wavelet ./ max_wave
    else
        error("Incorrect beta rule option ($(opts["beta_rule"])), must be in BETA_RULES")
    end

    return beta
end

function deepart_learn!(input, out, weights, opts::ModelOpts)
    # return beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
    if ndims(weights) == 4
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

    else
        local_out = out
        local_weight = weights
        local_in = input
    end

    # Get the local learning parameter beta
    beta = get_beta(local_out, opts)
    if opts["learning_rule"] == "fuzzyart"
        local_weight .= fuzzyart_learn_cast(local_in, local_weight, beta)
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

# @info "------- Defining train -------"
function train_hebb(
    model::HebbModel{T},
    x,
    y;
) where T <: AlternatingCCChain
    # chain = model.model.chain
    # params = Flux.params(chain)
    # acts = Flux.activations(chain, x)

    params = get_weights(model.model)
    acts = get_activations(model.model, x)
    n_layers = length(params)
    n_acts = length(acts)

    # Caches
    # caches = []
    # for p in params
    #     push!(caches, zeros(Float32, (size(p)..., 2)))
    # end

    # if bias
    #     n_layers = Int(length(params) / 2)
    # else
    #     n_layers = length(params)
    #     ins = [x, acts[1:end-1]...]
    #     outs = [acts...]
    # end

    ins = [acts[jx] for jx = 1:2:n_acts-1]
    outs = [acts[jx] for jx = 2:2:n_acts]

    target = zeros(Float32, size(outs[end]))
    # target = -ones(Float32, size(outs[end]))
    target[y] = 1.0
    if model.opts["gpu"]
        target = target |> gpu
    end

    for ix = 1:n_layers
        weights = params[ix]
        out = outs[ix]
        input = ins[ix]
        # cache = caches[ix]

        if ix == n_layers
            widrow_hoff_learn!(
                input,
                out,
                weights,
                target,
                model.opts,
            )
        else
            deepart_learn!(
                input,
                out,
                weights,
                model.opts,
            )
        end
    end

    return
end


function train_hebb(
    model::HebbModel{T},
    x,
    y;
) where T <: GroupedCCChain
    # Get the names for weights and iteration
    params = get_weights(model.model)
    n_layers = length(params)

    # Get the correct inputs and outputs for actuall learning
    ins, outs = get_incremental_activations(model.model, x)

    # Create the target vector
    target = zeros(Float32, size(outs[end]))
    # target = -ones(Float32, size(outs[end]))
    target[y] = 1.0

    if model.opts["gpu"]
        target = target |> gpu
    end

    for ix = 1:n_layers
        weights = params[ix]
        out = outs[ix]
        input = ins[ix]

        if ix == n_layers
            widrow_hoff_learn!(
                input,
                out,
                weights,
                target,
                model.opts,
            )
        else
            deepart_learn!(
                input,
                out,
                weights,
                model.opts,
            )
        end
    end

    return
end

function train_hebb_immediate(
    model::HebbModel{AlternatingCCChain},
    x,
    y;
)
    chain = model.model
    params = Flux.params(chain)
    n_layers = length(params)

    input = []
    out = []

    for ix = 1:n_layers
        weights = params[ix]

        # If the first layer, set up the recursion
        if ix == 1
            input = chain[1](x)
            out = chain[2](input)
        # If the last layer, set the supervised target
        elseif ix == n_layers
            input = chain[2*ix-1](out)
            out = chain[2*ix](input)
            target = zeros(Float32, size(out))
            target[y] = 1.0
        # Otherwise, recursion
        else
            input = chain[2*ix-1](out)
            out = chain[2*ix](input)
        end

        # If we are in the top supervised layer, use the supervised rule
        if ix == n_layers
            widrow_hoff_learn!(
                input,
                out,
                weights,
                target,
                model.opts,
            )
        # Otherwise, use the unsupervised rule(s)
        else
            deepart_learn!(
                input,
                out,
                weights,
                model.opts,
            )
        end
    end

    return
end

generate_showvalues(val) = () -> [(:val, val)]

function update_view_progress!(
    # ix_iter,
    # interval_vals,
    # ix_vals,
    # vals,
    p,
    loop_dict,
    model,
    data,
)
    if loop_dict["ix_iter"] % loop_dict["interval_vals"] == 0
        loop_dict["vals"][loop_dict["ix_vals"]] = test(model, data)
        loop_dict["ix_vals"] += 1
    end

    # Update progress bar
    report_value = if loop_dict["ix_vals"] > 1
        loop_dict["vals"][loop_dict["ix_vals"] - 1]
    else
        0.0
    end

    loop_dict["ix_iter"] += 1

    next!(p; showvalues=generate_showvalues(report_value))

    # next!(p; showvalues=generate_showvalues(report_value))
    # return report_value
    return
end

@info "------- Defining loop -------"
function train_loop(
    model::HebbModel,
    data;
    n_vals::Integer = 100,
    n_epochs::Integer = 10,
    val_epoch::Bool = false,
)
    loop_dict = Dict{String, Any}()

    # Set up the epochs progress bar
    n_iter = if val_epoch
        length(data.train)
    else
        n_epochs
    end

    # Set up the validation intervals
    local_n_vals = min(n_vals, n_iter)
    loop_dict["interval_vals"] = Int(floor(n_iter / local_n_vals))

    loop_dict["ix_vals"] = 1
    loop_dict["ix_iter"] = 1
    loop_dict["vals"] = zeros(Float32, local_n_vals)

    p = Progress(n_iter)

    # Iterate over each epoch
    for ie = 1:n_epochs
        # train_loader = Flux.DataLoader(data.train, batchsize=-1, shuffle=true)
        train_loader = Flux.DataLoader(data.train, batchsize=-1)
        if model.opts["gpu"]
            train_loader = train_loader |> gpu
        end

        # Iteratively train
        for (x, y) in train_loader
            if model.opts["immediate"]
                train_hebb_immediate(model, x, y)
            else
                train_hebb(model, x, y)
            end
            # if single_epoch
            if val_epoch
                update_view_progress!(
                    p,
                    loop_dict,
                    model,
                    data,
                )
            end
        end

        # Compute validation performance
        # if !single_epoch
        if !val_epoch
            update_view_progress!(
                p,
                loop_dict,
                model,
                data,
            )
            # next!(p; showvalues=generate_showvalues(report_value))
        else
            # Reset incrementers
            p = Progress(n_iter)
            loop_dict["ix_vals"] = 1
            loop_dict["ix_iter"] = 1
            local_plot = lineplot(
                loop_dict["vals"],
            )
            show(local_plot)
            println("\n")
        end
    end

    # @info "iter:" ix_iter
    perf = test(model, data)
    @info "perf = $perf"
    return loop_dict["vals"]
end

function view_weight(
    model::HebbModel,
    index::Integer,
)
    # weights = Flux.params(model.model.chain)
    weights = get_weights(model.model)
    @info size(weights[1])
    dim = Int(size(weights[1])[2])
    if model.opts["cc"]
        dim = Int(dim / 2)
    end
    dim = Int(sqrt(dim))
    local_weight = reshape(
        weights[1][index, :],
        dim,
        model.opts["cc"] ? dim*2 : dim,
    )
    lmax = maximum(local_weight)
    lmin = minimum(local_weight)
    DeepART.Gray.(local_weight .- lmin ./ (lmax - lmin))
end

function profile_test(n_epochs::Integer)
    _ = train_loop(
        model,
        data,
        n_epochs=n_epochs,
        eta=opts["eta"],
        beta_d=opts["beta_d"],
    )
end

end

