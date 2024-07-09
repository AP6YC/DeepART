
# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

module Hebb

@info "------- Loading dependencies -------"
using Revise
using DeepART
using Flux
using ProgressMeter
using Random
using CUDA
using UnicodePlots
using StatsBase: mean


function get_data(opts)
    @info "------- Loading dataset -------"
    data = if opts["dataset"] in ["mnist", "usps"]
        DeepART.load_one_dataset(
            opts["dataset"],
            n_train=opts["n_train"],
            n_test=opts["n_test"],
            # flatten=opts["flatten"],
            flatten = opts["model"] != "conv",
        )
    else
        DeepART.load_one_dataset(
            opts["dataset"],
        )
    end
end


# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------



function get_dense_deepart_layer(
    n_in::Integer,
    n_out::Integer,
    opts;
    first_layer::Bool = false,
)
    return Flux.@autosize (n_in,) Chain(
        Chain(
            first_layer ? identity : sigmoid_fast,
            DeepART.CC(),
        ),
        Dense(
            _, n_out,
            # bias=bias,
            bias = opts["bias"],
            # init=Flux.identity_init
            # init=rand,
            init=opts["init"],
        ),
    )
end

function get_widrow_hoff_layer(
    n_in::Integer,
    n_out::Integer,
    opts;
    # bias::Bool = false,
)
    return Flux.@autosize (n_in,) Chain(
        Chain(
            # identity
            sigmoid_fast,
        ),
        Dense(
            _, n_out,
            bias=opts["bias"],
            # init=Flux.identity_init
            # init=rand,
            init=opts["init"],
        ),
    )
end

function get_new_dense(
    n_input,
    n_class,
    opts,
)
    return Chain(
        get_dense_deepart_layer(n_input, 64, opts, first_layer=true),
        get_dense_deepart_layer(64, 32, opts),
        get_widrow_hoff_layer(32, n_class, opts)
    )
end


function get_incremental_activations(chain, x)
    # params
    n_layers = length(chain)

    ins = []
    outs = []

    for ix = 1:n_layers
        pre_input = (ix == 1) ? x : outs[end]
        local_acts = Flux.activations(chain[ix], pre_input)
        push!(ins, local_acts[1])
        push!(outs, local_acts[2])
    end
    return ins, outs
end

function train_new_hebb(
    model,
    x,
    y;
    # eta::Float32 = 0.1f0,
    # beta_d::Float32 = 0.1f0,
)
    chain = model.model
    params = Flux.params(chain)

    # # acts = Flux.activations(chain, x)
    n_layers = length(chain)
    # n_acts = length(acts)

    # ins = [acts[jx] for jx = 1:2:n_acts-1]
    # outs = [acts[jx] for jx = 2:2:n_acts]

    ins, outs = get_incremental_activations(chain, x)

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

        # if ix == n_layers
        #     widrow_hoff_learn!(input, out, weights, model.opts)
        # else
        #     deepart_learn!(input, out, weights, model.opts)
        # end

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

function get_conv_model(
    size_tuple::Tuple,
    head_dim::Integer,
    opts;
    # bias::Bool = false,
    # final_sigmoid::Bool = false,
)
    conv_model = Flux.@autosize (size_tuple,) Chain(
        # CC layer
        Chain(DeepART.CCConv()),

        # Conv layer
        Chain(
            Conv(
                (3, 3), _ => 8,
                # sigmoid_fast,
                # bias=false,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),

        # CC layer
        Chain(
            MaxPool((2,2)),
            sigmoid_fast,
            DeepART.CCConv(),
        ),

        # Conv layer
        Chain(
            Conv(
                (5,5), _ => 16,
                # sigmoid_fast,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),

        # CC layer
        Chain(
            Flux.AdaptiveMaxPool((4, 4)),
            Flux.flatten,
            sigmoid_fast,
            DeepART.CC(),
        ),

        # Dense layer
        Dense(_, 32,
            bias=opts["bias"],
            init=opts["init"],
        ),

        # Last layers
        Chain(identity),
        Chain(
            Dense(
                _, head_dim,
                # sigmoid_fast,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=["bias"],
                # init=opts["init"],
            ),
            vec,
        ),
        # DeepART.CC(),
    )
    return conv_model
end

function get_fuzzy_model(
    n_input,
    n_class,
    opts;
    # bias=false,
    # final_sigmoid=false,
)
    model = Flux.@autosize (n_input,) Chain(
        DeepART.CC(),
        DeepART.Fuzzy(_, 40,),
        Chain(sigmoid_fast, DeepART.CC()),
        DeepART.Fuzzy(_, 20,
            init=opts["init"],
        ),
        Chain(sigmoid_fast, DeepART.CC()),
        DeepART.Fuzzy(_, 20),
        Chain(identity),

        # LAST LAYER
        Chain(
            sigmoid_fast,
            Dense(
                _, n_class,
                # sigmoid_fast,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=opts["bias"],
            ),
        ),
    )
    return model
end

function get_model(
    n_input,
    n_class,
    # bias::Bool = false,
    # final_sigmoid = false,
    opts,
)
    model = Flux.@autosize (n_input,) Chain(
        Chain(DeepART.CC()),
        Dense(_, 64,
            bias=opts["bias"],
            # init=Flux.identity_init
            # init=abs(Flux.identity_init)
            # init=rand,
            init=opts["init"],
        ),

        Chain(sigmoid_fast, DeepART.CC()),
        Dense(_, 32,
            bias=opts["bias"],
            # init=Flux.identity_init
            # init=rand,
            init=opts["init"],
        ),

        # Chain(sigmoid_fast, DeepART.CC()),
        # Dense(_, 32, bias=bias),

        # LAST LAYER
        Chain(
            # identity
            sigmoid_fast,
        ),
        Chain(
            # sigmoid_fast,
            Dense(
                _, n_class,
                # sigmoid_fast,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=opts["bias"],
                # init=Flux.identity_init,
                # init=opts["init"],
            ),
        ),
    )
    return model
end

const ModelOpts = Dict{String, Any}

struct HebbModel{T <: Flux.Chain}
    model::T
    opts::ModelOpts
end


const MODEL_MAP = Dict(
    "fuzzy" => get_fuzzy_model,
    "conv" => get_conv_model,
    "dense" => get_model,
    "dense_new" => get_new_dense,
)

function construct_model(
    data,
    opts::ModelOpts,
)
    # Sanitize model option
    if !(opts["model"] in keys(MODEL_MAP))
        error("Invalid model type")
    end

    # Get the shape of the dataset
    dev_x, _ = data.train[1]
    n_input = size(dev_x)[1]
    n_class = length(unique(data.train.y))

    # If the convolutional model is selected, create a convolution input tuple
    local_input_size = if opts["model"] == "conv"
        (size(data.train.x)[1:3]..., 1)
    else
        n_input
    end

    # Construct the model
    model = MODEL_MAP[opts["model"]](
        local_input_size,
        n_class,
        opts,
    )


    # Enforce positive weights if necessary
    if opts["positive_weights"]
        ps = Flux.params(model)
        for p in ps
            p .= abs.(p)
            p .= p ./ maximum(p)
        end
    end


    return model
end

function HebbModel(
    data,
    opts::ModelOpts,
)
    return HebbModel(
        construct_model(data, opts),
        opts,
    )
end



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


# @info "------- Defining fuzzyart_learn -------"
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

const BETA_RULES = [
    "wta",
    "contrast",
    "softmax",
]

function get_beta(out, opts)
    if opts["beta_rule"] == "wta"
        beta = zeros(Float32, size(out))
        max_ind = argmax(out)
        beta[max_ind] = one(Float32)
    elseif opts["beta_rule"] == "contrast"
        max_ind = argmax(out)
        local_soft = Flux.softmax(out)
        max_soft = maximum(local_soft)
        local_soft = -local_soft
        local_soft[max_ind] = -local_soft[max_ind]
        beta = opts["beta_d"] .* local_soft ./ max_soft
    elseif opts["beta_rule"] == "softmax"
        local_soft = Flux.softmax(out)
        beta = opts["beta_d"] .* local_soft ./ maximum(local_soft)
        # beta = beta_d .* local_soft
    else
        error("Incorrect beta rule option ($(opts["beta_rule"])), must be in BETA_RULES")
    end

    return beta
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
        beta = get_beta(local_out, opts)
        local_weight .= fuzzyart_learn_cast(local_in, local_weight, beta)
    else
        # Get the local learning parameter beta
        beta = get_beta(out, opts)
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
            if model.opts["model"] == "dense_new"
                Hebb.train_new_hebb(model, x, y)
            elseif model.opts["immediate"]
                train_hebb_immediate(model, x, y)
            else
                train_hebb(model, x, y)
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

end
