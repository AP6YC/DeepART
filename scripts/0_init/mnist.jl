
using Revise
using DeepART
using Flux
using CUDA
using ProgressMeter

mnist = DeepART.get_mnist()

# -----------------------------------------------------------------------------
# Normal training loop
# -----------------------------------------------------------------------------

loss(x, y) = Flux.crossentropy(x, y)

# Make a simple multilayer perceptron
model = Chain(
    Dense(784, 128, relu),
    Dense(128, 64, relu),
    Dense(64, 10),
    sigmoid,
)
# model |> gpu

# optim = Flux.setup(DeepART.EWC(), model)  # will store optimiser momentum, etc.
optim = Flux.setup(
    Flux.Adam(),
    model,
)

n_classes = 10

n_samples = length(mnist.train.y)
x = reshape(
    mnist.train.x[:,:,1:n_samples],
    784, n_samples
)
# x |> gpu

n_test_samples = length(mnist.test.y)
x_test = reshape(
    mnist.test.x[:,:,1:n_test_samples],
    784, n_test_samples
)

y_cold = mnist.train.y[1:n_samples]
y_hot = zeros(Int, n_classes, n_samples)
for jx = 1:n_samples
    y_hot[y_cold[jx], jx] = 1
end

# dataloader = Flux.DataLoader((x, y_hot), batchsize=32)
dataloader = Flux.DataLoader((x, y_hot), batchsize=128)

Flux.Optimisers.adjust!(optim, enabled = true)

# flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), classes) .== y);
classes = collect(1:10)
flux_accuracy(y_hat, y) = Flux.mean(Flux.onecold(y_hat, classes) .== y);

# @showprogress
ix_acc = 1
acc_iter = 10
acc_log = []
for (lx, ly) in dataloader
    # y_hot |> gpu

    val, grads = Flux.withgradient(model) do m
        result = m(lx)
        # loss(result, ly)
        Flux.logitcrossentropy(result, ly)
    end

    if ix_acc == acc_iter
        acc = flux_accuracy(model(x_test), mnist.test.y)
        push!(acc_log, acc)
        @info acc
        ix_acc = 1
    else
        ix_acc += 1
    end
    # @info "$val"

    Flux.update!(optim, model, grads[1])
end
# Flux.Optimisers.adjust!(optim, enabled = false)
# end
