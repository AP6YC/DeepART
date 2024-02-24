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

n_classes = unique(data.train.y)

model = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2),
    sigmoid,
    # softmax,
) |> gpu

# model = Flux.Chain(
#     Dense
# )

optim = Flux.setup(Flux.Adam(0.01), model)  # will store optimiser momentum, etc.

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