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
using Printf

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

N_BATCH = 128
# N_BATCH = 4
N_EPOCH = 2
ACC_ITER = 10
GPU = false

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

# Turn these into a class-incremental dataset
cidata = DeepART.ClassIncrementalDataSplit(fdata)

# Combine and shuffle to multiple classes per task
# groupings = [[1,2], [3,4]]
# groupings = [[1,2],[3,4],[5,6],[7,8],[9,10]]
groupings = [collect(1:5), collect(6:10)]
# groupings = [collect(1:10)]

tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)

GPU && tidata |> gpu

# Infer the input size
n_input = size(fdata.train.x)[1]

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------

# Make a simple multilayer perceptron
model = Chain(
    Dense(n_input, 128, relu),
    Dense(128, 64, relu),
    Dense(64, n_classes),
    # sigmoid,
    softmax,
)

# model = Chain(
#     Conv((5,5),1=>6,relu),
#     Flux.flatten,
#     Dense(4704=>15,relu),
#     Dense(15=>10,sigmoid),
#     softmax
# )

ix_acc = 0
acc_log = []

flat, re = Flux.destructure(model)

EWC_opts = DeepART.EWCLossOpts(
    lambda = 100000000.0,
)
EWC_state = DeepART.EWCLossState()

function ewc_loss(flat)
    return DeepART.get_EWC_loss(EWC_state, EWC_opts, flat)
end

n_tasks = length(tidata.train)

# (Re)initialize the optimiser for this task
# optim = Flux.setup(Flux.Adam(), flat)
# optim = Flux.setup(Flux.Descent(), flat)
stps = []

# function loss(result, ly)
#     # result = re(m)(lx)
#     # ewc_loss = DeepART.get_EWC_loss(EWC_state, EWC_opts, flat)
#     return Flux.logitcrossentropy(result, ly) + ewc_loss
# end

for ix = 1:n_tasks
    # (Re)initialize the optimiser for this task
    optim = Flux.setup(Flux.Adam(), flat)

    for ep = 1:N_EPOCH
        # Create a dataloader for this task
        task_x = tidata.train[ix].x
        task_y = tidata.train[ix].y

        task_xt = tidata.test[ix].x
        task_yt = tidata.test[ix].y

        dataloader = Flux.DataLoader((task_x, task_y), batchsize=N_BATCH)

        for (lx, ly) in dataloader
            # Flux.Optimisers.adjust!(optim, new_task = true)

            # Compute gradients from the forward pass
            # val, grads = Flux.withgradient(model) do m

            # val, grads = Flux.withgradient(flat) do m
            #     # result = m(lx)
            #     result = re(m)(lx)
            #     # loss(result, ly)
            #     if first_task
            #         Flux.logitcrossentropy(result, ly)
            #     else
            #         ewc_loss = DeepART.get_EWC_loss(EWC_state, EWC_opts, flat)
            #         Flux.logitcrossentropy(result, ly) + ewc_loss
            #         # Flux.logitcrossentropy(result, ly) + DeepART.get_EWC_loss(EWC_state, EWC_opts, flat)
            #     end
            # end

            ewc_loss = 0.0

            val, grads = Flux.withgradient(flat) do m
                result = re(m)(lx)
                # ewc_loss = DeepART.get_EWC_loss(EWC_state, EWC_opts, flat)
                # return Flux.logitcrossentropy(result, ly) + ewc_loss
                return Flux.logitcrossentropy(result, ly) + ewc_loss(flat)
                # loss(m(lx), ly)
                # Flux.logitcrossentropy(result, ly) + DeepART.get_EWC_loss(EWC_state, EWC_opts, flat)
            end

            Flux.update!(optim, flat, grads[1])

            if ix_acc % ACC_ITER == 0
                # acc = DeepART.flux_accuracy(model(fdata.test.x), fdata.test.y, n_classes)
                # acc = DeepART.flux_accuracy(re(flat)(fdata.test.x), fdata.test.y, n_classes)
                acc = DeepART.flux_accuracy(re(flat)(task_xt), task_yt, n_classes)
                push!(acc_log, acc)
                # @info "Epoch: $(ep)\t acc: $(acc)\t loss: $(val)\t task: $(ix)\t classes: $(groupings[ix])"
                @info @sprintf "Epoch: %i\t acc: %.4f\t loss: %.4f\t ewc: %.8f\t task: %i\t classes: %i %i" ep acc val ewc_loss ix groupings[ix][1] groupings[ix][2]
            end
            global ix_acc += 1
        end
    end

    # Compute single task performances
    local_stp = []
    for jx = 1:ix
        local_perf = DeepART.flux_accuracy(re(flat)(tidata.test[jx].x), tidata.test[jx].y, n_classes)
        push!(local_stp, local_perf)
    end
    push!(stps, local_stp)

    # Compute the new FIM
    local_mem_data = DeepART.group_datasets(tidata.train, collect(1:ix), true)
    _, full_grads = Flux.withgradient(flat) do m
        # result = re(m)(task_x)
        # Flux.logitcrossentropy(result, task_y)
        result = re(m)(local_mem_data.x)
        Flux.logitcrossentropy(result, local_mem_data.y)
    end

    global EWC_state = DeepART.EWCLossState(EWC_state, EWC_opts, flat, full_grads[1])
    global EWC_opts.first_task = false
end

# _, full_grads = Flux.withgradient(flat) do m
#     result = re(m)(fdata.test.x)
#     Flux.logitcrossentropy(result, fdata.test.y)
# end

@info stps
@info "Accuracy: $(DeepART.flux_accuracy(re(flat)(fdata.test.x), fdata.test.y, n_classes))"

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

# model |> gpu

# model = DeepART.get_dense([n_input, 128, n_classes])

# optim = Flux.setup(DeepART.EWC(), model)  # will store optimiser momentum, etc.
# optim = Flux.setup(Flux.Adam(), model)
# optim = Flux.setup(
#     Flux.Optimisers.OptimiserChain(
#         DeepART.EWC(),
#         Flux.Adam(),
#         # Flux.Descent(),
#         # Flux.Optimisers.Adam(0.1)
#     ),
#     model,
# )

# Flux.Optimisers.adjust!(optim, new_task = false)
# Flux.Optimisers.adjust!(optim, new_task = true)

# Identify the loss function
# loss(x, y) = Flux.crossentropy(x, y)

# dataloader = Flux.DataLoader((x, y_hot), batchsize=32)
# dataloader = Flux.DataLoader((fdata.train.x, fdata.train.y), batchsize=N_BATCH)

# Flux.Optimisers.adjust!(optim, enabled = true)

# flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), classes) .== y);
# function flux_accuracy(y_hat, y_truth, n_class::Int=0)
#     # If the number of classes is specified, use that, otherwise infer from the training labels
#     n_classes = DeepART.n_classor(y_truth, n_class)
#     classes = collect(1:n_classes)
#     Flux.mean(Flux.onecold(y_hat, classes) .== y_truth)
# end




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
