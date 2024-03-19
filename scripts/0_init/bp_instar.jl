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

# model = Chain(
# )

# model = @autosize (28, 28, 1, 1) Chain(
#     Conv((5,5),1=>6,relu),
#     Flux.flatten,
#     Dense(_=>15,relu),
#     Dense(15=>10,sigmoid),
#     softmax
# )

size_tuple = (28, 28, 1, 1)

# Create a LeNet model
model = @Flux.autosize (size_tuple,) Chain(
    Conv((5,5),1 => 6, relu),
    MaxPool((2,2)),
    Conv((5,5),6 => 16, relu),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(256=>120,relu),
    Dense(120=>84, relu),
    Dense(84=>10, sigmoid),
    softmax
)

# for ix = 1:1000
#     x = reshape(data.train.x[:, :, ix], size_tuple)
#     acts = Flux.activations(model, x)
#     inputs = (xf, acts[1:end-1]...)
#     DeepART.instar(xf, acts, model, 0.0001)
# end

ix = 1
# x = fdata.train.x[:, ix]
x = reshape(data.train.x[:, :, ix], size_tuple)
y = data.train.y[ix]

acts = Flux.activations(model, x)

n_input = size(fdata.train.x)[1]

model = Chain(
    Dense(n_input, 128, tanh),
    Dense(128, 64, tanh),
    Dense(64, n_classes, sigmoid),
    # sigmoid,
    # softmax,
)

# model = Chain(
#     Dense(n_input*2, 128, tanh),
#     Dense(128, 64, tanh),
#     Dense(64, n_classes, sigmoid),
#     # sigmoid,
#     # softmax,
# )

head_dim = 128

model = Flux.@autosize (784,) Chain(
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

# heads = [get_head() for ix = 1:n_classes]

a = Flux.params(model)

art = DeepART.INSTART(model)

# create_category!(art, xf, y_hat)

@showprogress for ix = 1:1000
    xf = fdata.train.x[:, ix]
    label = data.train.y[ix]
    DeepART.train!(art, xf, y=label)
end

xf = fdata.train.x[:, ix]
# xf = DeepART.complement_code(xf)
acts = Flux.activations(model, xf)

y_hats = Vector{Int}()
@showprogress for ix in eachindex(data.test.y[1:N_TEST])
    y_hat = argmax(model(fdata.test.x[:, ix]))
    push!(y_hats, y_hat)
end

@info unique(y_hats)

model(xf)
argmax(model(xf))
data.train.y[ix]
# weights = Flux.params(model)

# trainables = [weights[jx] for jx in [1, 3, 5]]
# ins = [acts[jx] for jx in [1, 3, 5]]
# outs = [acts[jx] for jx in [2, 4, 6]]
# for ix in eachindex(ins)
#     # weights[ix] .+= DeepART.instar(inputs[ix], acts[ix], weights[ix], eta)
#     trainables[ix] .+= DeepART.instar(ins[ix], outs[ix], trainables[ix], eta)
# end
