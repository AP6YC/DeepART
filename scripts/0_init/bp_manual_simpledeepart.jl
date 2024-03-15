"""
Development script for a WTANet module.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
using StatsBase: norm

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

N_TRAIN = 1000
N_TEST = 1000
N_BATCH = 128
# N_BATCH = 1
N_EPOCH = 1
ACC_ITER = 10

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# all_data = DeepART.load_all_datasets()
# data = all_data["moon"]
data = DeepART.get_mnist()
fdata = DeepART.flatty_hotty(data)

n_classes = length(unique(data.train.y))
n_train = min(N_TRAIN, length(data.train.y))
n_test = min(N_TEST, length(data.test.y))

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

ix = 1
x = fdata.train.x[:, ix]

n_input = size(fdata.train.x)[1]

shared = Chain(
    Dense(n_input, 128, tanh),
    Dense(128, 64, tanh),
    # Dense(64, n_classes),
    # sigmoid,
    # softmax,
)


function get_head()
    Chain(
        # Dense(64, n_classes),
        # Dense(64, 32, tanh),
        Dense(64, 32, sigmoid),
        # DeepART.Fuzzy(64, 1),
        # sigmoid,
    )
end
# heads = Vector{Chain}()
# push!(heads, get_head())

heads = [get_head() for _ in 1:n_classes]
# forward(shared, heads, x)
# model = Chain(shared, Parallel(vcat, heads...))

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------

# Compute gradients from the forward pass
ix = 1
x = fdata.train.x[:, ix]
y = data.train.y[ix]
# shared(x)
# heads[1](shared(x))
# result = model(x)
# argmax(result)

# val, grads = Flux.withgradient(model) do m
#     result = model(x)
#     winner = argmax(result)
#     println(winner)
#     loss = sum(1.0 .- result[winner])
#     println(loss)
#     loss
# end

function forward(shared, heads, x)
    r_shared = shared(x)
    r_heads = [head(r_shared) for head in heads]
    r_heads
    # vcat(r_heads...)
end

function art_loss(r_heads, x)
    # r_heads[x]
end

# batch_y = 0
for (lx, ly) in dataloader
    # @info lx |> size
    result = forward(shared, heads, lx)
    # global batch_y = ly
    # @info ly
    new_w = DeepART.art_learn()
    break
end

function update(shared, heads, optim_shared, optim_heads, grads)
    # Winner index
    winner = findall(x -> !isnothing(x), grads[2])
    # Update the shared
    Flux.update!(optim_shared, shared, grads[1])
    # Update the correct head
    Flux.update!(optim_heads[winner], heads[winner], grads[2][winner])
end

function inference(shared, heads, data)
    # Flux.onecold(vcat(forward(shared, heads, data)...))
    Flux.onecold(forward(shared, heads, data))
end

function flux_accuracy(y_hat, y_truth, n_class::Int=0)
    # If the number of classes is specified, use that, otherwise infer from the training labels
    n_classes = DeepART.n_classor(y_truth, n_class)
    classes = collect(1:n_classes)
    Flux.mean(Flux.onecold(y_hat, classes) .== y_truth)
end

function flux_accuracy_cold(y_hat, y_truth)
    Flux.mean(y_hat .== y_truth)
end

optim_shared = Flux.setup(
    # Flux.Descent(0.01),
    Flux.Adam(),
    shared,
)
optim_heads = [
    Flux.setup(
        # Flux.Descent(0.01),
        Flux.Adam(),
        head,
    ) for head in heads
]

# val, grads = Flux.withgradient(shared, heads) do s, h
#     r_heads = forward(s, h, x)
#     loss = wta_loss(r_heads)
# end

# dataloader = Flux.DataLoader((fdata.train.x, fdata.train.y), batchsize=N_BATCH)
dataloader = Flux.DataLoader((fdata.train.x, data.train.y), batchsize=N_BATCH)

ix_acc = 0
acc_log = []

# @showprogress
for ep = 1:N_EPOCH

    for (lx, ly) in dataloader
        val, grads = Flux.withgradient(shared, heads) do s, h
            r_heads = forward(s, h, x)
            loss = wta_loss(r_heads, ly)
        end

        update(shared, heads, optim_shared, optim_heads, grads)

        if ix_acc % ACC_ITER == 0
            # local_y_hat = vcat(forward(shared, heads, fdata.test.x)...)

            acc = flux_accuracy_cold(
                inference(shared, heads, fdata.test.x),
                data.test.y,
            )
            push!(acc_log, acc)
            @info "Epoch $ep: $acc"
        end
        global ix_acc += 1
    end
end

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------

y_hats = Vector{Int}()
for ix = 1:n_test
    x = fdata.test.x[:, ix]
    y = fdata.test.y[ix]
    DeepART.test(m, x, y)
    push!(y_hats, y_hat)
end

perf = DeepART.ART.performance(y_hats, data.test.y[1:N_TEST])
@info perf unique(y_hats) model.art.n_categories
