"""
Development script for deep instar learning.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
using ProgressMeter
# using StatsBase: norm

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# N_TRAIN = 2000
N_TRAIN = 4000
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
# OLD MODELS
# -----------------------------------------------------------------------------

# model = @autosize (28, 28, 1, 1) Chain(
#     Conv((5,5),1=>6,relu),
#     Flux.flatten,
#     Dense(_=>15,relu),
#     Dense(15=>10,sigmoid),
#     softmax
# )

# size_tuple = (28, 28, 1, 1)

# Create a LeNet model
# model = @Flux.autosize (size_tuple,) Chain(
#     Conv((5,5),1 => 6, relu),
#     MaxPool((2,2)),
#     Conv((5,5),6 => 16, relu),
#     MaxPool((2,2)),
#     Flux.flatten,
#     Dense(256=>120,relu),
#     Dense(120=>84, relu),
#     Dense(84=>10, sigmoid),
#     softmax
# )

# for ix = 1:1000
#     x = reshape(data.train.x[:, :, ix], size_tuple)
#     acts = Flux.activations(model, x)
#     inputs = (xf, acts[1:end-1]...)
#     DeepART.instar(xf, acts, model, 0.0001)
# end

# ix = 1
# # x = fdata.train.x[:, ix]
# x = reshape(data.train.x[:, :, ix], size_tuple)
# y = data.train.y[ix]

# acts = Flux.activations(model, x)

# model = Chain(
#     Dense(n_input, 128, tanh),
#     Dense(128, 64, tanh),
#     Dense(64, n_classes, sigmoid),
#     # sigmoid,
#     # softmax,
# )

# model = Chain(
#     Dense(n_input*2, 128, tanh),
#     Dense(128, 64, tanh),
#     Dense(64, n_classes, sigmoid),
#     # sigmoid,
#     # softmax,
# )

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

n_input = size(fdata.train.x)[1]
head_dim = 64

model = Flux.@autosize (n_input,) Chain(
    # DeepART.CC(),
    # Dense(_, 256, sigmoid),
    DeepART.CC(),
    Dense(_, 128, sigmoid),
    DeepART.CC(),
    Dense(_, 64, sigmoid),
    DeepART.CC(),
    Dense(_, head_dim, sigmoid),
    # Dense(_, n_classes, sigmoid),
    # sigmoid,
    # softmax,
)

art = DeepART.INSTART(
    model,
    head_dim=head_dim,
    beta=0.01,
    # beta=1.0,
    rho=0.2,
)

xf = fdata.train.x[:, ix]
acts = Flux.activations(model, xf)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# create_category!(art, xf, y_hat)
@showprogress for ix = 1:n_train
    xf = fdata.train.x[:, ix]
    label = data.train.y[ix]
    DeepART.train!(art, xf, y=label)
end

y_hats = Vector{Int}()
@showprogress for ix = 1:n_test
    xf = fdata.test.x[:, ix]
    y_hat = DeepART.classify(art, xf, get_bmu=true)
    push!(y_hats, y_hat)
end

@info unique(y_hats)
perf = DeepART.ART.performance(y_hats, data.test.y[1:n_test])
@info "Perf: $perf, n_cats: $(art.n_categories)"

DeepART.create_confusion_heatmap(
    string.(collect(1:10)),
    data.test.y[1:n_test],
    y_hats,
)

# trainables = [weights[jx] for jx in [1, 3, 5]]
# ins = [acts[jx] for jx in [1, 3, 5]]
# outs = [acts[jx] for jx in [2, 4, 6]]
# for ix in eachindex(ins)
#     # weights[ix] .+= DeepART.instar(inputs[ix], acts[ix], weights[ix], eta)
#     trainables[ix] .+= DeepART.instar(ins[ix], outs[ix], trainables[ix], eta)
# end