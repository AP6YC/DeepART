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

# all_data = DeepART.load_all_datasets()
# data = all_data["moon"]

# n_classes = unique(data.train.y)

# model = Chain(
#     Dense(2 => 3, tanh),   # activation function inside layer
#     BatchNorm(3),
#     Dense(3 => 2),
#     sigmoid,
#     # softmax,
# ) |> gpu

# opt = Flux.setup(Flux.Momentum(), model)

Flux.Optimisers.@def struct EWC <: Flux.Optimisers.AbstractRule
	# eta = 0.01
    eta = 0.01      # learning rate
    lambda = 0.1    # regularization strength
    decay = 0.9     # decay rate
    alpha = 0.1
end

mutable struct EWCState
    FIM
    old_params
end

function Flux.Optimisers.apply!(o::EWC, state, x, dx)
    # Because the FIM is a function of the gradients, initialize it here
    if isnothing(state.FIM)
        # new_state = dx .* dx
        state.FIM = dx .* dx
    else
        # new_state = (1 - o.alpha) .* state + o.alpha .* dx .* dx
        state.FIM = (1 - o.alpha) .* state.old_params + o.alpha .* dx .* dx
    end
	# eta = convert(float(eltype(x)), o.eta)
    # Flux.params(model)[ix] .-= (lambda / 2) * (Flux.params(model)[ix] .- mu[ix]) .* local_FIM
    # (o.lambda / 2) * (x .- state.old_params) .* local_FIM
    return state, (o.lambda / 2) * (x .- state.old_params) .* state.FIM
	# return new_state, (o.lambda / 2) *
    # return state, Flux.Optimisers.@lazy dx * eta  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
end

function Flux.Optimisers.init(o::EWC, x::AbstractArray)
    # State is the FIM
    # new_FIM = [(1 - alpha) * FIM[ix] + alpha * (gradients[ix] .* gradients[ix]) for ix in eachindex(gradients)]
    # new_FIM = [(1 - alpha) * FIM[ix] + alpha * (gradients[ix] .* gradients[ix]) for ix in eachindex(gradients)]
    return EWCState(nothing, x)
    # return nothing
end

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

optim = Flux.setup(EWC(), model)  # will store optimiser momentum, etc.

loss(x, y) = Flux.crossentropy(x, y)

mnist = DeepART.get_mnist()
ix = 1
x = reshape(mnist.train.x[:, :, ix], 784)
y = mnist.train.y[ix]
# gradient = Flux.gradient(() -> loss(x, y), Flux.params(model))
grads = Flux.gradient(model) do m
    result = m(x)
    loss(result, y)
end

Flux.update!(optim, model, grads[1])

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
