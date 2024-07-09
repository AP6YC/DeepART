"""
    hebb.jl

# Description
Deep Hebbian learning experiment drafting script.
"""

# @info "####################################"
# @info "###### NEW HEBBIAN EXPERIMENT ######"
# @info "####################################"
@info """\n####################################
###### NEW HEBBIAN EXPERIMENT ######
####################################
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

include("definitions.jl")

import .Hebb

# perf = 0.9310344827586207
# perf = 0.9655172413793104

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

opts = Dict{String, Any}(
    "n_epochs" => 1000,
    # "n_epochs" => 200,
    # "n_epochs" => 10,
    # "immediate" => true,
    "immediate" => false,

    "model_opts" => Dict{String, Any}(
        "bias" => false,
        "eta" => 0.1,
        "beta_d" => 0.0,
        # "beta_d" => 0.1,
        # "eta" => 0.2,
        # "beta_d" => 0.2,
        # "eta" => 0.5,
        # "beta_d" => 0.5,
        # "eta" => 1.0,
        # "beta_d" => 1.0,
        # "beta_d" => 0.001,

        "final_sigmoid" => false,
        # "final_sigmoid" => true,

        "gpu" => false,

        "model" => "dense",
        # "model" => "small_dense",
        # "model" => "fuzzy",
        # "model" => "conv",

        # "init" => Flux.rand32,
        "init" => Flux.glorot_uniform,

        # "positive_weights" => true,
        "positive_weights" => false,

        "wta" => true,
        # "wta" => false,
    ),

    "profile" => false,
    # "profile" => true,

    # "dataset" => "wine",
    "dataset" => "iris",
    # "dataset" => "wave",
    # "dataset" => "face",
    # "dataset" => "flag",
    # "dataset" => "halfring",
    # "dataset" => "moon",
    # "dataset" => "ring",
    # "dataset" => "spiral",
    # "dataset" => "mnist",
    # "dataset" => "usps",

    "n_train" => 10000,
    "n_test" => 10000,
    # "flatten" => true,
    "rng_seed" => 1235,
)

# Correct for Float32 types
opts["model_opts"]["eta"] = Float32(opts["model_opts"]["eta"])
opts["model_opts"]["beta_d"] = Float32(opts["model_opts"]["beta_d"])


Random.seed!(opts["rng_seed"])

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

datasets = Dict(
    "high_dimensional" => [
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


data = Hebb.get_data(opts)

dev_x, dev_y = data.train[1]
# n_input = size(dev_x)[1]
# n_class = length(unique(data.train.y))

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

@info "------- Constructing model -------"

# model = Hebb.construct_model(data, opts)
model = Hebb.HebbModel(data, opts["model_opts"])

# -----------------------------------------------------------------------------
# DEFINITIONS
# -----------------------------------------------------------------------------

@info "------- Defining test -------"
function test(model, data)
    n_test = length(data.test)

    y_hats = zeros(Int, n_test)
    test_loader = Flux.DataLoader(data.test, batchsize=-1)
    if model.opts["gpu"]
        y_hats = y_hats |> gpu
        test_loader = test_loader |> gpu
    end

    ix = 1
    for (x, _) in test_loader
        y_hats[ix] = argmax(model.model(x))
        ix += 1
    end
    # y_hats = model(data.test.x |> gpu) |> cpu  # first row is prob. of true, second row p(false)
    # y_hats = argmaxmodel(data.test.x)  # first row is prob. of true, second row p(false)

    if model.opts["gpu"]
        y_hats = y_hats |> cpu
    end

    # @info y_hats
    # @info "unique y_hats:" unique(y_hats)
    perf = DeepART.AdaptiveResonance.performance(y_hats, data.test.y)
    # @info "perf = $perf"
    return perf
end

@info "------- TESTING BEFORE TRAINING -------"
if model.opts["gpu"]
    model.model = model.model |> gpu
end
test(model, data)

@info "------- Defining fuzzyart_learn -------"
function fuzzyart_learn(x, W, beta)
    return beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
end

function fuzzyart_learn_cast(x, W, beta)
    Wy, Wx = size(W)
    # @info "sizes:" size(x) size(W) size(beta)
    _x = repeat(x', Wy, 1)
    # @info _x
    # _x = repeat(x, 1, Wx)
    _beta = repeat(beta, 1, Wx)

    # result = beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
    # result = beta .* minimum(cat(_x, W, dims=3), dims=3) + W .* (one(eltype(_beta)) .- _beta)
    # @info "sizes" size(W) size(_x) size(_beta)
    result = beta .* min.(_x, W) + W .* (one(eltype(_beta)) .- _beta)
    # @info "result is size" size(result)
    # @info sum(result - W)
    return result
end

function fuzzyart_learn_cast_cache(x, W, beta, cache)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    _beta = repeat(beta, 1, Wx)

    cache[:, :, 1] .= _x
    cache[:, :, 2] .= W

    # result = beta .* minimum(cat(_x, W, dims=3), dims=3) + W .* (one(eltype(_beta)) .- _beta)
    # result = beta .* minimum(cache, dims=3) + W .* (one(eltype(_beta)) .- _beta)
    result = beta .* min.(_x, W) + W .* (one(eltype(_beta)) .- _beta)
    # @info sum(beta .* minimum(cache, dims=3) - W .* (one(eltype(_beta)) .- _beta))
    # @info sum(result - W)
    return result
end


function widrow_hoff_cast(weights, target, out, input, eta)
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

function widrow_hoff_learn!(input, out, weights, target, opts)
    weights .+= widrow_hoff_cast(weights, target, out, input, opts["eta"])
    return
end

function deepart_learn!(input, out, weights, opts)
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

        # Get the local learning parameter beta
        if opts["wta"]
            beta = zeros(Float32, size(local_out))
            max_ind = argmax(local_out)
            beta[max_ind] = one(Float32)
        else
            local_soft = Flux.softmax(local_out)
            beta = opts["beta_d"] .* local_soft ./ maximum(local_soft)
            # beta = beta_d .* local_soft
        end

        local_weight .= fuzzyart_learn_cast(local_in, local_weight, beta)
    else
        if opts["wta"]
            beta = zeros(Float32, size(out))
            max_ind = argmax(out)
            beta[max_ind] = one(Float32)
        else
            local_soft = Flux.softmax(out)
            beta = opts["beta_d"] .* local_soft ./ maximum(local_soft)
            # beta = beta_d .* local_soft
        end
        weights .= fuzzyart_learn_cast(input, weights, beta)
        # weights .= fuzzyart_learn_cast_cache(input, weights, beta, cache)
    end
    return
end

@info "------- Defining train -------"
function train_hebb(
    model,
    x,
    y;
)
    chain = model.model
    params = Flux.params(chain)
    acts = Flux.activations(chain, x)
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

function train_hebb_immediate(
    model,
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
                model.opts["eta"],
            )
        # Otherwise, use the unsupervised rule(s)
        else
            deepart_learn!(
                input,
                out,
                weights,
                model.opts["beta_d"],
            )
        end
    end

    return
end

@info "------- Defining loop -------"
function train_loop(
    model,
    data;
    n_vals = 100,
    n_epochs = 10,
    kwargs...
)
    # Set up the validation intervals
    local_n_vals = min(n_vals, n_epochs)
    interval_vals = Int(floor(n_epochs / local_n_vals))
    ix_vals = 1
    vals = zeros(Float32, local_n_vals)

    # Set up the epochs progress bar
    p = Progress(n_epochs)
    generate_showvalues(val) = () -> [(:val, val)]

    # Iterate over each epoch
    for ie = 1:n_epochs
        # train_loader = Flux.DataLoader(data.train, batchsize=-1, shuffle=true)
        train_loader = Flux.DataLoader(data.train, batchsize=-1)
        if model.opts["gpu"]
            train_loader = train_loader |> gpu
        end

        # Iteratively train
        for (x, y) in train_loader
            if opts["immediate"]
                train_hebb_immediate(
                    model, x, y;
                    # kwargs...
                )
            else
                train_hebb(
                    model, x, y;
                    # kwargs...
                )
            end
        end

        # Compute validation performance
        if ie % interval_vals == 0
            vals[ix_vals] = test(model, data)
            ix_vals += 1
        end

        # Update progress bar
        report_value = if ix_vals > 1
            vals[ix_vals - 1]
        else
            0.0
        end
        next!(p; showvalues=generate_showvalues(report_value))
    end

    perf = test(model, data)
    @info "perf = $perf"
    return vals
end

function view_weight(model, index)
    weights = Flux.params(model.model)
    dim = Int(sqrt(size(weights[1])[2] / 2))
    local_weight = reshape(weights[1][index, :], dim, dim*2)
    lmax = maximum(local_weight)
    lmin = minimum(local_weight)
    DeepART.Gray.(local_weight .- lmin ./ (lmax - lmin))
end

function profile_test(n_epochs)
    vals = train_loop(
        model,
        data,
        n_epochs=n_epochs,
        eta=opts["eta"],
        beta_d=opts["beta_d"],
    )
end

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# CUDA.@time train_loop(
# CUDA.@profile train_loop(
if opts["profile"]
    @info "------- Profiling -------"
    @static if Sys.iswindows()
        # compilation
        @profview profile_test(3)
        # pure runtime
        @profview profile_test(10)
    end
else
    @info "------- Training -------"
    vals = train_loop(
        model,
        data,
        n_epochs=opts["n_epochs"],
        # eta=opts["eta"],
        # beta_d=opts["beta_d"],
    )

    local_plot = lineplot(
        vals,
    )
    show(local_plot)

    # Only visualize the weights if we are working with a computer vision dataset
    if opts["dataset"] in datasets["high_dimensional"]
        view_weight(model, 1)
    else
        # @info model[2].weight
        # @info sum(model[2].weight)
    end
end
