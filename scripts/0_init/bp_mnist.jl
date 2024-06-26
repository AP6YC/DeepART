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
using Plots

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

N_BATCH = 128
# N_BATCH = 12
N_EPOCH = 1
ACC_ITER = 10

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# Load the dataset
data = DeepART.get_mnist()
# all_data = DeepART.load_all_datasets()
# data = all_data["spiral"]
# data = all_data["moon"]
# data = all_data["ring"]

# Get the number of classes
n_classes = length(unique(data.train.y))

# Get the flat features and one-hot labels
fdata = DeepART.flatty_hotty(data)
n_input = size(fdata.train.x)[1]

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

# Make a simple multilayer perceptron
model = Chain(
    Dense(n_input, 128, relu),
    Dense(128, 64, relu),
    Dense(64, n_classes),
    sigmoid,
    # softmax,
)

function CC()
    Parallel(vcat,
        identity,
        DeepART.complement_code,
    )
end

model = Flux.@autosize (n_input,) Chain(
    CC(),
    Dense(_, 128, sigmoid),
    CC(),
    Dense(_, 64, sigmoid),
    CC(),
    Dense(_, n_classes, sigmoid),
    # Dense(_, n_classes, sigmoid),
    # sigmoid,
    # softmax,
)


# model = Chain(
#     # Dense(n_input, 64),
#     Dense(n_input, 128),
#     relu,
#     Dense(128, 64),
#     # sigmoid,
#     relu,
#     Dense(64, n_classes),
#     # sigmoid,
#     # relu,
#     softmax,
# )

# model = DeepART.get_dense([n_input, 200, 100, 10, n_classes])
# model = DeepART.get_dense([n_input, 128, n_classes])

# model |> gpu

# optim = Flux.setup(DeepART.EWC(), model)  # will store optimiser momentum, etc.
optim = Flux.setup(Flux.Adam(), model)
# Flux.Optimisers.adjust!(optim, enabled = false)

# -----------------------------------------------------------------------------
# Normal training loop
# -----------------------------------------------------------------------------

# Identify the loss function
loss(x, y) = Flux.crossentropy(x, y)

# dataloader = Flux.DataLoader((x, y_hot), batchsize=32)
dataloader = Flux.DataLoader((fdata.train.x, fdata.train.y), batchsize=N_BATCH)

# Flux.Optimisers.adjust!(optim, enabled = true)

# flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), classes) .== y);
function flux_accuracy(y_hat, y_truth, n_class::Int=0)
    # If the number of classes is specified, use that, otherwise infer from the training labels
    n_classes = DeepART.n_classor(y_truth, n_class)
    classes = collect(1:n_classes)
    Flux.mean(Flux.onecold(y_hat, classes) .== y_truth)
end

ix_acc = 0
acc_log = []

# @showprogress
for ep = 1:N_EPOCH

    for (lx, ly) in dataloader
        # Compute gradients from the forward pass
        val, grads = Flux.withgradient(model) do m
            result = m(lx)
            # loss(result, ly)
            Flux.logitcrossentropy(result, ly)
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

lineplot(
    acc_log,
    title="Accuracy Trend",
    xlabel="Iteration",
    ylabel="Test Accuracy",
)

# plot(
#     acc_log,
#     title="Accuracy Trend",
#     xlabel="Iteration",
#     ylabel="Test Accuracy",
# )

# Flux.Optimisers.adjust!(optim, enabled = false)

# function plot_f(x, y)
#     classes = collect(1:n_classes)
#     y_hat = model([x, y])
#     return Flux.onecold(y_hat, classes)
# end

# p = plot()

# ccol = cgrad([RGB(1,.3,.3), RGB(.4,1,.4), RGB(.3,.3,1), RGB(.3,.6,.1)])
# r = 0:.05:1

# contour!(
#     p,
#     r,
#     r,
#     plot_f,
#     f=true,
#     nlev=4,
#     c=ccol,
#     # c=DeepART.COLORSCHEME,
#     leg=:none
# )

# p = scatter!(
#     p,
#     data.train.x[1, :],
#     data.train.x[2, :],
#     group=data.train.y,
# )