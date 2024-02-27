"""
    mnist.jl

# Description
Boilerplate MNIST training and testing with no modifications.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
using CUDA
using ProgressMeter
using UnicodePlots

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# N_BATCH = 128
N_BATCH = 12
N_EPOCH = 2

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# Load the dataset
# data = DeepART.get_mnist()
all_data = DeepART.load_all_datasets()
data = all_data["moon"]

# Get the number of classes
n_classes = length(unique(data.train.y))

# Get the flat features and one-hot labels
x, y, xt, yt = DeepART.flatty_hotty(data)

n_input = size(x)[1]

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

# Make a simple multilayer perceptron
model = Chain(
    Dense(n_input, 128, relu),
    # Dense(n_input, 128, sigmoid),
    Dense(128, 64, relu),
    # Dense(128, 64, sigmoid),
    Dense(64, n_classes),
    sigmoid,
)
# model |> gpu

# model = DeepART.get_dense([n_input, 128, n_classes])

# optim = Flux.setup(DeepART.EWC(), model)  # will store optimiser momentum, etc.
optim = Flux.setup(Flux.Adam(), model)

# -----------------------------------------------------------------------------
# Normal training loop
# -----------------------------------------------------------------------------

# Identify the loss function
loss(x, y) = Flux.crossentropy(x, y)

# dataloader = Flux.DataLoader((x, y_hot), batchsize=32)
dataloader = Flux.DataLoader((x, y), batchsize=N_BATCH)

Flux.Optimisers.adjust!(optim, enabled = true)

# flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), classes) .== y);
function flux_accuracy(y_hat, y_truth, n_class::Int=0)
    # If the number of classes is specified, use that, otherwise infer from the training labels
    n_classes = DeepART.n_classor(y_truth, n_class)
    classes = collect(1:n_classes)
    Flux.mean(Flux.onecold(y_hat, classes) .== y_truth)
end

# @showprogress
ix_acc = 0
acc_iter = 10
acc_log = []

for ep = 1:N_EPOCH
    # # @showprogress
    # ix_acc = 0
    # acc_iter = 10
    # acc_log = []

    for (lx, ly) in dataloader
        # Compute gradients from the forward pass
        val, grads = Flux.withgradient(model) do m
            result = m(lx)
            # loss(result, ly)
            Flux.logitcrossentropy(result, ly)
        end

        Flux.update!(optim, model, grads[1])

        if ix_acc % acc_iter == 0
            acc = flux_accuracy(model(xt), data.test.y, n_classes)
            push!(acc_log, acc)
            @info acc
        end
        global ix_acc += 1
    end
end

lineplot(
    acc_log,
    title="Accuracy Trend",
    xlabel="Iteration",
    ylabel="Test Accuracy",
)

# Flux.Optimisers.adjust!(optim, enabled = false)
# end
