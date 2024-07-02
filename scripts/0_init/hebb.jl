@info "------- Loading dependencies -------"
using Revise
using DeepART
using Flux
using ProgressMeter
using Random
Random.seed!(1234)

@info "------- Loading dataset -------"
data = DeepART.load_one_dataset(
    # "iris",
    # "mnist",
    "usps",
    n_train=1000,
    n_test=100,
    flatten=true,
)
dev_x, dev_y = data.train[1]
n_input = size(dev_x)[1]
n_class = length(unique(data.train.y))

@info "------- Defining model definigion -------"
function get_model(
    n_input,
    n_class,
    bias=false,
)
    model = Flux.@autosize (n_input,) Chain(
        DeepART.CC(),
        # Dense(
        #     _, 20,
        #     sigmoid_fast,
        #     bias=bias,
        # ),
        Dense(
            _, 128,
            sigmoid_fast,
            bias=bias,
        ),
        # DeepART.CC(),
        # Dense(
        #     _ => 10,
        #     sigmoid_fast,
        #     bias=bias,
        # ),
        # DeepART.CC(),
        Dense(
            _, n_class,
            # sigmoid_fast,
            bias=bias,
        )
    )
    return model
end

@info "------- Constructing model -------"
model = get_model(n_input, n_class)

@info "------- Defining test -------"
function test(model, data)
    n_test = length(data.test)
    y_hats = zeros(Int, n_test)
    for ix = 1:n_test
        x, _ = data.test[ix]
        y_hats[ix] = argmax(model(x))
    end
    @info y_hats
    @info "uniques:" unique(y_hats)
    perf = DeepART.AdaptiveResonance.performance(y_hats, data.test.y)
    @info "perf = $perf"
end

# test(model, data)
# acts = Flux.activations(model, dev_x)
# @info "acts" acts
# @info "inference:" dev_y model(dev_x)

GOTNAN = false

function fuzzyart_learn(x, W, beta)
    return beta .* min.(x, W) + W .* (one(eltype(beta)) .- beta)
end

@info "------- Defining train -------"
function train_hebb(
    chain,
    x,
    y;
    bias=false,
    eta = 0.1,
    log_epoch = 0,
    log_ix = 0,
    # decay = 0.01,
)
    params = Flux.params(chain)
    # n_layers = length(params)
    acts = Flux.activations(chain, x)

    if any(isnan.(acts[1])) && !(GOTNAN)
        @warn "NAN"
        @warn acts
        @warn params[1]
        @warn "epoch:" log_epoch
        @warn "ix:" log_ix
        # @warn "weights" weights
        # @warn "input" input
        # @warn "out" outs
        # @warn "target" target
        global GOTNAN = true
    end

    # if bias
    #     n_layers = Int(length(params) / 2)
    # else
    #     n_layers = length(params)
    #     ins = [x, acts[1:end-1]...]
    #     outs = [acts...]
    # end
    n_layers = length(params)

    n_acts = length(acts)
    # # ins = push!([acts[jx] for jx = 1:2:(n_layers*2)-1], acts[n_layers-1])
    # ins = [acts[jx] for jx = 1:2:(n_acts*2-1)]
    # outs = push!([acts[jx] for jx = 2:2:(n_layers*2-1)], acts[n_layers])
    ins = push!([acts[jx] for jx = 1:2:((n_acts-1))], acts[n_acts-1])
    outs = push!([acts[jx] for jx = 2:2:((n_acts-1))], acts[n_acts])

    # @info "sizes" size(ins) size(outs) length(params) length(acts) n_layers
    # @info "sizes 2" size(ins[end]) size(outs[end]) size(params[n_layers])


    target = zeros(Float32, size(outs[end]))
    # target = -ones(Float32, size(outs[end]))
    target[y] = 1.0

    # target = ones(Float32, size(outs[end])).*0.25
    # target[y] = 0.75
    # @info target

    # @info "sizes" length(params) size(params[1]) size(params[2]) size(ins[1]) size(ins[2]) size(outs[1]) size(outs[2])
    for ix = 1:n_layers
        weights = params[ix]
        out = outs[ix]
        input = ins[ix]
        n_weight = size(weights)[1]


        if ix == n_layers
            # (target - out)*input
            for iw = 1:n_weight
                weights[iw, :] .+= eta .* (target[iw] .- out[iw]) .* (input - weights[iw, :])
                # weights[iw, :] .+= eta .* target[iw] .* (input - weights[iw, :])
            end
        else
            beta = Flux.softmax(out)
            for iw = 1:n_weight
                # Instar
                # weights[ix, :] .+= eta .* out[iw] .* (input .- weights[ix, :])
                # weights[iw, :] .+= eta .* out[iw] .* (input .- weights[iw, :])

                # ART
                # weights[iw, :] = beta[iw] .* min.(input, weights[iw, :]) + weights[iw, :] .* (one(eltype(beta[iw])) .- beta[iw])
                # weights[iw, :] = beta[iw] .* min.(input, weights[iw, :]) + weights[iw, :] .* (one(eltype(beta[iw])) .- beta[iw])
                weights[iw, :] = fuzzyart_learn(input, weights[iw, :], beta[iw])
            end
        end
    end

    # # Last layer
    # weights = params[n_layers]
    # # out = acts[n_layers]
    # out = outs[n_layers]
    # input = ins[n_layers]
    # n_weight = size(weights)[1]

    # for iw = 1:n_weight
    #     # weights[iw, :] .+= eta .* (target[iw] .- out[iw]) .* (input - weights[iw, :])
    #     @info "sizes" size(weights[iw, :]) target[iw] size(input)
    #     weights[iw, :] .+= eta .* target[iw] .* (input - weights[iw, :])
    # end
end

@info "------- Defining loop -------"
function train_loop(
    model,
    data;
    n_epochs = 10
)
    n_train = length(data.train)
    # n_layers = length(Flux.params(chain))
    @showprogress for ie = 1:n_epochs
        # old_weights = deepcopy(Flux.params(model))
        for ix = 1:n_train
            x, y = data.train[ix]
            train_hebb(model, x, y, log_epoch=ie, log_ix=ix)
            if GOTNAN
                @warn "LEAVING TRAIN"
                break
            end
        end
        if GOTNAN
            @warn "LEAVING EPOCH"
            break
        end
        # new_weights = deepcopy(Flux.params(model))
        # @info sum(new_weights[n_layers] - old_weights[n_layers])
    end

    test(model, data)
end

train_loop(
    model,
    data,
    n_epochs=100,
)

dev_x, dev_y = data.test[22]
p = Flux.params(model)
acts = Flux.activations(model, dev_x)
n_layers = length(p)
@info "acts" acts
@info "inference:" dev_y model(dev_x)
@info "params:" p


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
