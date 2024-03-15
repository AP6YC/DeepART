"""
Development script for a WTANet module.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

N_TRAIN = 1000
N_TEST = 1000
N_BATCH = 128
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

# m = DeepART.WTANet(
#     model_spec=[2, 10, n_classes],
# )

ix = 1
x = fdata.train.x[:, ix]
# f1 = m.model(x)
# argmax(f1)

n_input = size(fdata.train.x)[1]

# Make a simple multilayer perceptron
model = Chain(
    Dense(n_input, 128, relu),
    Dense(128, 64, relu),
    Dense(64, n_classes),
    sigmoid,
    # softmax,
)

model(x)

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------


# Compute gradients from the forward pass
ix = 1
x = fdata.train.x[:, ix]
y = data.train.y[ix]

result = model(x)
winner = result[y]
sum(1.0 .- winner)

val, grads = Flux.withgradient(model) do m
    result = m(x)
    winner = result[y]
    sum(1.0 .- winner)
end


# for ix = 1:n_train
#     x = fdata.train.x[:, ix]
#     y = fdata.train.y[ix]
#     DeepART.train!(m, x, y)
# end

# dataloader = Flux.DataLoader((fdata.train.x, fdata.train.y), batchsize=N_BATCH)
dataloader = Flux.DataLoader((fdata.train.x, data.train.y), batchsize=N_BATCH)
# optim = Flux.setup(Flux.Adam(), model)
optim = Flux.setup(Flux.Descent(), model)

ix_acc = 0
acc_log = []

function flux_accuracy(y_hat, y_truth, n_class::Int=0)
    # If the number of classes is specified, use that, otherwise infer from the training labels
    n_classes = DeepART.n_classor(y_truth, n_class)
    classes = collect(1:n_classes)
    Flux.mean(Flux.onecold(y_hat, classes) .== y_truth)
end

# @showprogress
for ep = 1:N_EPOCH

    for (lx, ly) in dataloader
        # Compute gradients from the forward pass
        val, grads = Flux.withgradient(model) do m
            result = m(lx)
            # loss(result, ly)
            # Flux.logitcrossentropy(result, ly)
            # winner = result[argmax(result)]
            winner = result[ly]
            sum(1.0 .- winner)
        end

        Flux.update!(optim, model, grads[1])

        if ix_acc % ACC_ITER == 0
            acc = flux_accuracy(model(fdata.test.x), data.test.y, n_classes)
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
