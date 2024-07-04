"""
    hebb.jl

# Description
Deep Hebbian learning experiment drafting script.
"""

@info "####################################"
@info "###### NEW HEBBIAN EXPERIMENT ######"
@info "####################################"

@info "------- Loading dependencies -------"
using Revise
using DeepART
using Flux
using ProgressMeter
using Random
using CUDA
using UnicodePlots
# Random.seed!(1234)

# GPU = true
GPU = false
# PROFILE = true
PROFILE = false


# "face",
# "flag",
# "halfring",
# "iris",
# "moon",
# "ring",
# "spiral",
# "wave",
# "wine",
@info "------- Loading dataset -------"
data = DeepART.load_one_dataset(
    # "iris",
    "wine",
    # "wave",
    # "face",
    # "mnist",
    # "usps",
    # n_train=1000,
    # n_test=100,
    # flatten=true,
)

dev_x, dev_y = data.train[1]
n_input = size(dev_x)[1]
n_class = length(unique(data.train.y))

# perf = 0.9310344827586207
# perf = 0.9655172413793104
@info "------- Defining model definigion -------"
function get_model(
    n_input,
    n_class,
    bias=false,
)
    model = Flux.@autosize (n_input,) Chain(
        DeepART.CC(),
        Dense(
            # _, 20,
            _, 40,
            sigmoid_fast,
            bias=bias,
        ),

        # DeepART.CC(),
        # Dense(
        #     _, 128,
        #     sigmoid_fast,
        #     bias=bias,
        # ),

        # DeepART.CC(),
        # Dense(
        #     _, 64,
        #     sigmoid_fast,
        #     bias=bias,
        # ),

        DeepART.CC(),
        Dense(
            _, 20,
            # _, 20,
            sigmoid_fast,
            bias=bias,
        ),

        DeepART.CC(),
        Dense(
            _, 10,
            # _, 20,
            sigmoid_fast,
            bias=bias,
        ),

        # DeepART.CC(),
        # Dense(
        #     _, 10,
        #     # _, 20,
        #     sigmoid_fast,
        #     bias=bias,
        # ),

        # LAST LAYER
        Dense(
            _, n_class,
            sigmoid_fast,
            bias=bias,
        )
    )
    return model
end

function get_conv_model(size_tuple::Tuple, head_dim::Integer)
    conv_model = Flux.@autosize (size_tuple,) Chain(
        DeepART.CCConv(),
        Chain(
            Conv((3, 3), _ => 8, sigmoid_fast, bias=false),
        ),
        Chain(
            MaxPool((2,2)),
            DeepART.CCConv(),
        ),
        Chain(
            Conv((5,5), _ => 16, sigmoid_fast, bias=false),
        ),
        Chain(
            Flux.AdaptiveMaxPool((4, 4)),
            Flux.flatten,
            # DeepART.CC(),
        ),
        Chain(
            Dense(_, head_dim, sigmoid_fast, bias=false),
            vec,
        ),
        # DeepART.CC(),
    )
    return conv_model
end


@info "------- Constructing model -------"
model = get_model(n_input, n_class)

# size_tuple = (size(data.train.x)[1:3]..., 1)
# model = DeepART.get_rep_conv(size_tuple, n_class)

@info "------- Defining test -------"
function test(model, data)
    n_test = length(data.test)

    y_hats = zeros(Int, n_test)
    test_loader = Flux.DataLoader(data.test, batchsize=-1)
    if GPU
        y_hats = y_hats |> gpu
        test_loader = test_loader |> gpu
    end

    ix = 1
    for (x, _) in test_loader
        y_hats[ix] = argmax(model(x))
        ix += 1
    end

    if GPU
        y_hats = y_hats |> cpu
    end

    # @info y_hats
    # @info "unique y_hats:" unique(y_hats)
    perf = DeepART.AdaptiveResonance.performance(y_hats, data.test.y)
    # @info "perf = $perf"
    return perf
end

@info "------- TESTING BEFORE TRAINING -------"
test(model, data)
if GPU
    model = model |> gpu
end

@info "------- Defining fuzzyart_learn -------"
function fuzzyart_learn(x, W, beta)
    return beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
end

function fuzzyart_learn_cast(x, W, beta)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    _beta = repeat(beta, 1, Wx)

    # result = beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
    result = beta .* minimum(cat(_x, W, dims=3), dims=3) + W .* (one(eltype(_beta)) .- _beta)
    # @info sum(result - W)
    return result
end

function fuzzyart_learn_cast_cache(x, W, beta, cache)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    _beta = repeat(beta, 1, Wx)

    cache[:,:,1] .= _x
    cache[:,:,2] .= W

    # result = beta .* minimum(cat(_x, W, dims=3), dims=3) + W .* (one(eltype(_beta)) .- _beta)
    result = beta .* minimum(cache, dims=3) + W .* (one(eltype(_beta)) .- _beta)
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

@info "------- Defining train -------"
function train_hebb(
    chain,
    x,
    y;
    bias=false,
    eta = 0.1,
    beta_d = 0.1,
    # log_epoch = 0,
    # log_ix = 0,
    # decay = 0.01,
)
    params = Flux.params(chain)
    acts = Flux.activations(chain, x)
    n_layers = length(params)
    n_acts = length(acts)

    # Caches
    caches = []
    for p in params
        push!(caches, zeros(Float32, (size(p)..., 2)))
    end

    # if bias
    #     n_layers = Int(length(params) / 2)
    # else
    #     n_layers = length(params)
    #     ins = [x, acts[1:end-1]...]
    #     outs = [acts...]
    # end

    ins = push!([acts[jx] for jx = 1:2:((n_acts-1))], acts[n_acts-1])
    outs = push!([acts[jx] for jx = 2:2:((n_acts-1))], acts[n_acts])

    target = zeros(Float32, size(outs[end]))
    # target = -ones(Float32, size(outs[end]))
    target[y] = 1.0
    if GPU
        target = target |> gpu
    end

    for ix = 1:n_layers
        weights = params[ix]
        out = outs[ix]
        input = ins[ix]
        cache = caches[ix]

        if ix == n_layers
            weights .+= widrow_hoff_cast(weights, target, out, input, eta)
        else
            if ndims(weights[ix]) == 4
                full_size = size(weights[ix])
                n_kernels = full_size[4]
                kernel_shape = full_size[1:3]

                unfolded = Flux.NNlib.unfold(ins[ix], full_size)
                local_in = reshape(mean(reshape(unfolded, :, kernel_shape...), dims=1), :)
                # Get the averaged and reshaped local output
                local_out = reshape(mean(outs[ix], dims=(1, 2)), n_kernels)
                # Reshape the weights to be (n_kernels, n_features)
                local_weight = reshape(weights[ix], :, n_kernels)'
                # Get the local learning parameter beta
                # beta = Flux.softmax(local_out)
                local_soft = Flux.softmax(local_out)
                beta = beta_d .* local_soft ./ maximum(local_soft)
                weights .= fuzzyart_learn_cast(local_in, local_weight, beta)
            else
                # beta = Flux.softmax(out)
                # beta = ones(Float32, size(out))
                # beta = zeros(Float32, size(out))
                local_soft = Flux.softmax(out)
                beta = beta_d .* local_soft ./ maximum(local_soft)
                # weights .= fuzzyart_learn_cast(input, weights, beta)
                weights .= fuzzyart_learn_cast_cache(input, weights, beta, cache)
            end
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
    # eta=0.1,
    # beta_d=0.1,
)
    # n_train = length(data.train)
    # n_vals = 100

    interval_vals = Int(floor(n_epochs / n_vals))
    ix_vals = 1
    vals = zeros(Float32, n_vals)

    p = Progress(n_epochs)
    generate_showvalues(val) = () -> [(:val, val)]

    # @showprogress for ie = 1:n_epochs
    for ie = 1:n_epochs
        # train_loader = Flux.DataLoader(data.train, batchsize=-1, shuffle=true)
        train_loader = Flux.DataLoader(data.train, batchsize=-1)
        if GPU
            train_loader = train_loader |> gpu
        end

        # Iteratively train
        # ix = 1
        for (x, y) in train_loader
            train_hebb(
                model, x, y;
                kwargs...
                # eta=eta,
                # beta_d=beta_d,
            )
            # ix += 1
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
    weights = Flux.params(model)
    dim = Int(sqrt(size(weights[1])[2] / 2))
    local_weight = reshape(weights[1][index, :], dim, dim*2)
    lmax = maximum(local_weight)
    lmin = minimum(local_weight)
    DeepART.Gray.(local_weight .- lmin ./ (lmax - lmin))
end

@info "------- Training -------"
# CUDA.@time train_loop(
# CUDA.@profile train_loop(

function profile_test(n_epochs)
    vals = train_loop(
        model,
        data,
        n_epochs=n_epochs,
        eta=0.5,
        beta_d=0.5,
    )
end

if PROFILE
    # compilation
    @profview profile_test(100)
    # pure runtime
    @profview profile_test(1000)
else

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

    vals = train_loop(
        model,
        data,
        # n_epochs=5000,
        # n_epochs=300,
        # n_epochs=1000,
        n_epochs=10000,
        eta=1.0,
        # eta=0.5,
        # eta=0.1,
        beta_d=1.0,
        # beta_d=0.5,
        # beta_d=0.1,
    )

    local_plot = lineplot(
        vals,
    )
    show(local_plot)
end

# -----------------------------------------------------------------------------
# SCRATCH SPACE
# -----------------------------------------------------------------------------

# OLD TRAIN LOOP:
# n_weight = size(weights)[1]
# if ix == n_layers
#     # for iw = 1:n_weight
#     #     # (target - out)*input
#     #     # weights[iw, :] .+= eta .* (target[iw] .- out[iw]) .* (input - weights[iw, :])
#     #     weights[iw, :] .+= eta .* (target[iw] .- out[iw]) .* input
#     #     # @info (target[iw] .- out[iw])
#     #     # weights[iw, :] .+= eta .* target[iw] .* (input - weights[iw, :])
#     # end
#     weights .+= widrow_hoff_cast(weights, target, out, input, eta)
# else
#     beta = Flux.softmax(out)
#     # for iw = 1:n_weight
#     #     weights[iw, :] = fuzzyart_learn(input, weights[iw, :], beta[iw])
#     # end
#     weights .= fuzzyart_learn_cast(input, weights, beta)
# end



# @info "------- Post-training analysis -------"
# dev_x, dev_y = data.test[22]
# p = Flux.params(model)
# acts = Flux.activations(model, dev_x)
# n_layers = length(p)
# @info "acts" acts
# @info "inference:" dev_y model(dev_x)
# @info "params:" p


# n_layers = length(p)
# n_acts = length(acts)
# ins = push!([acts[jx] for jx = 1:2:((n_acts-1))], acts[n_acts-1])
# outs = push!([acts[jx] for jx = 2:2:((n_acts-1))], acts[n_acts])

# using DeepART

# head_dim = 10
# model = DeepART.get_rep_fia_dense(n_input, head_dim)
# model = Flux.@autosize (n_input,) Chain(
#     DeepART.CC(),
#     # Dense(_, 512, sigmoid_fast, bias=false),
#     # DeepART.CC(),
#     Dense(_, 256, sigmoid_fast, bias=false),
#     DeepART.CC(),
#     Dense(_, head_dim, sigmoid_fast, bias=false),
# )







# @info "------- Defining train -------"
# function train_hebb(
#     chain,
#     x,
#     y;
#     bias=false,
#     eta = 0.001,
#     log_epoch = 0,
#     log_ix = 0,
#     # decay = 0.01,
# )
#     params = Flux.params(chain)
#     # n_layers = length(params)
#     acts = Flux.activations(chain, x)

#     if any(isnan.(acts[1])) && !(GOTNAN)
#         @warn "NAN"
#         @warn acts
#         @warn params[1]
#         @warn "epoch:" log_epoch
#         @warn "ix:" log_ix
#         # @warn "weights" weights
#         # @warn "input" input
#         # @warn "out" outs
#         # @warn "target" target
#         global GOTNAN = true
#     end

#     if bias
#         n_layers = Int(length(params) / 2)
#     else
#         n_layers = length(params)
#         ins = [x, acts[1:end-1]...]
#         outs = [acts...]
#     end

#     # target = zeros(Float32, size(outs[end]))
#     target = -ones(Float32, size(outs[end]))
#     target[y] = 1.0

#     # target = ones(Float32, size(outs[end])).*0.25
#     # target[y] = 0.75
#     # @info target

#     # @info "sizes" length(params) size(params[1]) size(params[2]) size(ins[1]) size(ins[2]) size(outs[1]) size(outs[2])
#     for ix = 1:n_layers
#         weights = params[ix]
#         out = outs[ix]
#         input = ins[ix]
#         # @info "sizes:" size(weights) size(out) size(input)
#         n_weight = size(weights)[1]
#         # beta = Flux.softmax(out)
#         if ix == n_layers
#             # (target - out)*input
#             for iw = 1:n_weight
#                 # weights[iw, :] .+= input .* (target[iw] .- out[iw]) .* eta .- decay .* (target[iw] .- out[iw])

#                 # Instar
#                 # weights[ix, :] .+= eta .* (out[iw] .- target[iw]) .* (input .- weights[ix, :])

#                 # @info sum(update)
#                 # update = eta .* (target[iw] .- out[iw]) .* (input - weights[ix, :])
#                 # weights[iw, :] .*= update
#                 # weights[iw, :] .+= eta .* (target[iw] .- out[iw]) .* (input - weights[iw, :])
#                 weights[iw, :] .+= input .* out[iw] .* eta
#                 # weights[iw, :] .+= eta .* out[iw] .* (input .- weights[iw, :])

#                 # @info sum(eta .* out[iw] .* (input .- weights[ix, :]))
#                 # weights[ix, :] .+= eta .* out[iw] .* (input .- weights[ix, :])
#                 # weights[ix, :] .+= eta .* (out[iw] .- target[iw]) .* (input - weights[ix, :])
#                 # weights[ix, :] .+= eta .* target[iw] .* (input .- weights[ix, :])
#             end
#         else
#             for iw = 1:n_weight
#                 # Hebb
#                 # weights[iw, :] .+= input .* out[iw] .* eta .- decay .* weights[iw, :]
#                 # weights[iw, :] .+= input .* out[iw] .* eta .- decay .* out[iw]
#                 weights[iw, :] .+= input .* out[iw] .* eta

#                 # Instar
#                 # weights[ix, :] .+= eta .* out[iw] .* (input .- weights[ix, :])
#                 # weights[iw, :] .+= eta .* out[iw] .* (input .- weights[iw, :])

#                 # ART
#                 # weights[iw, :] = beta[iw] .* min.(input, weights[iw, :]) + weights[iw, :] .* (one(eltype(beta[iw])) .- beta[iw])
#             end
#         end
#     end
# end
