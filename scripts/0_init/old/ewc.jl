"""
    wta.jl

# Description
Implementation of a feedforward network trained with a simple winner-take-all loss.
"""

using Revise
using DeepART
using Flux
using CUDA
using ProgressMeter

all_data = DeepART.load_all_datasets()
data = all_data["moon"]

# n_classes = unique(data.train.y)

# small_model = Chain(
#     Dense(2 => 3, tanh),   # activation function inside layer
#     BatchNorm(3),
#     Dense(3 => 2),
#     sigmoid,
#     # softmax,
# ) |> gpu

# opt = Flux.setup(Flux.Momentum(), small_model)
# Momentum |> fieldnames
# Flux.Optimisers.adjust!(opt, rho=0.1)

# optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.
# Flux.train!(model, train_set, optim) do m, x, y
#     loss(m(x), y)
# end

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
    Flux.Optimisers.OptimiserChain(
        DeepART.EWC(),
        Flux.Descent(),
        # Flux.Optimisers.Adam(0.1)
    ),
    model,
)

Flux.Optimisers.adjust!(optim, enabled = false)
Flux.Optimisers.adjust!(optim, enabled = true)

loss(x, y) = Flux.crossentropy(x, y)

mnist = DeepART.get_mnist()
cimnist = DeepART.TaskIncrementalDataSplit(mnist)

# -----------------------------------------------------------------------------
# Class incremental
# -----------------------------------------------------------------------------

for ix = 1:n_classes
    # x = reshape(mnist.train.x[:, :, ix], 784)
    # y = mnist.train.y[ix]

    # n_samples = n_train
    n_samples = length(cimnist.train[ix].y)
    x = reshape(
        cimnist.train[ix].x[:,:,1:n_samples],
        784, n_samples
    )
    # x |> gpu

    y_cold = cimnist.train[ix].y[1:n_samples]
    y_hot = zeros(Int, n_classes, n_samples)
    for jx = 1:n_samples
        y_hot[y_cold[jx], jx] = 1
    end

    dataloader = Flux.DataLoader((x, y_hot), batchsize=32)

    Flux.Optimisers.adjust!(optim, enabled = true)

    for (lx, ly) in dataloader
        # y_hot |> gpu

        val, grads = Flux.withgradient(model) do m
            result = m(lx)
            loss(result, ly)
        end
        @info "$ix: $val"

        Flux.update!(optim, model, grads[1])
    end
    # Flux.Optimisers.adjust!(optim, enabled = false)
end

# Optimisers.adjust!(opt, rho = 0.95)  # change ρ for the whole model
# train_set = (x, y)
# Flux.train!(model, train_set, optim) do m, x, y
#     loss(m(x), y)
# end

# opt_state = Flux.setup(Adam(), model)

# my_log = []
# # for epoch in 1:100
#   losses = Float32[]
# #   for (i, data) in enumerate(train_set)
#     input, label = data

#     val, grads = Flux.withgradient(model) do m
#       # Any code inside here is differentiated.
#       # Evaluation of the model and loss must be inside!
#       result = m(input)
#       my_loss(result, label)
#     end

#     # Save the loss from the forward pass. (Done outside of gradient.)
#     push!(losses, val)

#     # Detect loss of Inf or NaN. Print a warning, and then skip update!
#     if !isfinite(val)
#       @warn "loss is $val on item $i" epoch
#       continue
#     end

#     Flux.update!(opt_state, model, grads[1])
#   end

#   # Compute some accuracy, and save details as a NamedTuple
#   acc = my_accuracy(model, train_set)
#   push!(my_log, (; acc, losses))

#   # Stop training when some criterion is reached
#   if  acc > 0.95
#     println("stopping after $epoch epochs")
#     break
#   end
# end


# using ForwardDiff  # an example of a package which only likes one array

# model = Chain(  # much smaller model example, as ForwardDiff is a slow algorithm here
#           Conv((3, 3), 3 => 5, pad=1, bias=false),
#           BatchNorm(5, relu),
#           Conv((3, 3), 5 => 3, stride=16),
#         )
# image = rand(Float32, 224, 224, 3, 1);
# @show sum(model(image));

# flat, re = destructure(model)
# st = Optimisers.setup(rule, flat)  # state is just one Leaf now

# ∇flat = ForwardDiff.gradient(flat) do v
#   m = re(v)      # rebuild a new object like model
#   sum(m(image))  # call that as before
# end

# st, flat = Optimisers.update(st, flat, ∇flat)
# @show sum(re(flat)(image));



# # Training loop, using the whole data set 1000 times:
# losses = []
# @showprogress for epoch in 1:1_000
#     # for (x, y) in loader
#     # for
#         loss, grads = Flux.withgradient(model) do m
#             # Evaluate model and loss inside gradient context:
#             y_hat = m(x)
#             Flux.crossentropy(y_hat, y)
#         end
#         Flux.update!(optim, model, grads[1])
#         push!(losses, loss)  # logging, outside gradient context
#     end
# end

# optim # parameters, momenta and output have all changed
# out2 = model(noisy |> gpu) |> cpu  # first row is prob. of true, second row p(false)

# mean((out2[1,:] .> 0.5) .== truth)  # accuracy 94% so far!


# https://pub.towardsai.net/overcoming-catastrophic-forgetting-a-simple-guide-to-elastic-weight-consolidation-122d7ac54328
# def get_fisher_diag(model, dataset, params, empirical=True):
#     fisher = {}
#     for n, p in deepcopy(params).items():
#         p.data.zero_()
#         fisher[n] = Variable(p.data)

#     model.eval()
#     for input, gt_label in dataset:
#         model.zero_grad()
#         output = model(input).view(1, -1)
#         if empirical:
#             label = gt_label
#         else:
#             label = output.max(1)[1].view(-1)
#         negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
#         negloglikelihood.backward()

#         for n, p in model.named_parameters():
#             fisher[n].data += p.grad.data ** 2 / len(dataset)

#     fisher = {n: p for n, p in fisher.items()}
#     return fisher

# def get_ewc_loss(model, fisher, p_old):
#     loss = 0
#     for n, p in model.named_parameters():
#         _loss = fisher[n] * (p - p_old[n]) ** 2
#         loss += _loss.sum()
#     return loss

# model = model_trained_on_task_A
# dataset = a_small_sample_from_dataset_A
# params = {n: p for n, p in model.named_parameters() if p.requires_grad}
# p_old = {}

# for n, p in deepcopy(params).items():
#     p_old[n] = Variable(p.data)

# fisher_matrix = get_fisher_diag(model, dataset, params)
# ewc_loss = get_ewc_loss(model, fisher_matrix, p_old)
