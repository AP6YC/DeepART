"""
    train-test.jl

# Description
Implements the variety of training/testing start-to-finish experiments.
"""

# -----------------------------------------------------------------------------
# TRAIN/TEST FUNCTIONS
# -----------------------------------------------------------------------------

"""
Task-homogenous training loop for a DeepART model.

# Arguments
$ARG_COMMONARTMODULE
$ARG_DATASPLIT
$ARG_N_TRAIN
"""
function basic_train!(
    art::CommonARTModule,
    data::DataSplit,
    n_train::Integer=IInf,
)
    # Get the number of training samples
    l_n_train = get_n(n_train, data.train)

    # Iterate over the training samples
    pr = Progress(n_train; desc="Task-Homogenous Training")
    for ix = 1:l_n_train
        xf = data.train.x[:, ix]
        label = data.train.y[ix]
        incremental_supervised_train!(art, xf, label)
        next!(pr)
    end
end

"""
Task-homogenous testing loop for a [`DeepARTModule`](@ref) model.

# Arguments
$ARG_COMMONARTMODULE
$ARG_DATASPLIT
$ARG_N_TEST
"""
function basic_test(
    art::CommonARTModule,
    data::DataSplit,
    n_test::Integer=IInf,
)
    # Get the number of testing samples
    l_n_test = get_n(n_test, data.test)

    # Get the estimates on the test data
    y_hats = Vector{Int}()
    pr = Progress(n_test; desc="Task-Homogenous Testing")
    for ix = 1:l_n_test
        xf = data.test.x[:, ix]
        # y_hat = DeepART.classify(art, xf, get_bmu=true)
        y_hat = incremental_classify(art, xf)
        push!(y_hats, y_hat)
        next!(pr)
    end

    # # Calculate the performance and log
    # perf = DeepART.ART.performance(y_hats, data.test.y[1:l_n_test])
    # @info "Perf: $perf, n_cats: $(art.n_categories), uniques: $(unique(y_hats))"

    # Return the estimates
    return y_hats
end

"""
Task-incremental training/testing loop.

# Arguments
$ARG_DEEPARTMODULE
$ARG_TIDATA
$ARG_N_TRAIN
"""
function train_inc!(
    # art::DeepARTModule,
    art::CommonARTModule,
    tidata::ClassIncrementalDataSplit,
    n_train::Integer=IInf,
)
    # Infer the number of tasks to train over
    n_tasks = length(tidata.train)

    # Iterate over the tasks
    for ix = 1:n_tasks
        # Get the local batch of training data
        task_x = tidata.train[ix].x
        task_y = tidata.train[ix].y

        # l_n_train = min(n_train, length(task_y))
        l_n_train = get_n(n_train, tidata.train[ix])

        # Incrementally train over the current task's training data
        pr = Progress(n_train; desc="Task-Incremental Training: Task $(ix)")
        for jx = 1:l_n_train
            # Get the current sample and label
            xf = task_x[:, jx]
            label = task_y[jx]

            # Train the module
            # DeepART.train!(art, xf, y=label)
            incremental_supervised_train!(art, xf, label)

            # Update the progress bar
            next!(pr)
        end
    end
end

"""
Computes the performance of the ART module given some estimates.
"""
function get_perf(
    art::CommonARTModule,
    data::DataSplit,
    y_hats::Vector{Int},
    n_test::Integer=IInf,
)
    # Get the number of testing samples
    l_n_test = min(n_test, length(data.test.y))

    perf = ART.performance(y_hats, data.test.y[1:l_n_test])
    @info "Perf: $perf, n_cats: $(art.n_categories), uniques: $(unique(y_hats))"

    return perf
end

# -----------------------------------------------------------------------------
# FULL EXPERIMENTS
# -----------------------------------------------------------------------------

"""
Task-homogenous training/testing loop.

# Arguments
$ARG_COMMONARTMODULE
$ARG_DATASPLIT
$ARG_N_TRAIN
$ARG_N_TEST
"""
function tt_basic!(
    # art::DeepARTModule,
    art::CommonARTModule,
    data::DataSplit,
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    # Train
    basic_train!(art, data, n_train)

    # Test
    y_hats = basic_test(art, data, n_test)

    # Compute the performance fromt the test results
    perf = get_perf(art, data, y_hats, n_test)

    # Compile the experiment results
    out_dict = Dict(
        "y_hats" => y_hats,
        "perf" => perf,
    )

    return out_dict

    # # Confusion matrix
    # p = DeepART.create_confusion_heatmap(
    #     string.(collect(0:9)),
    #     data.test.y[1:n_test],
    #     y_hats,
    # )

    # # Return the plot
    # return p
end

"""
Task-incremental training/testing loop for [`DeepARTModule`](@ref)s.

# Arguments
$ARG_DEEPARTMODULE
$ARG_TIDATA
$ARG_DATASPLIT
$ARG_N_TRAIN
$ARG_N_TEST
"""
function tt_inc!(
    # art::DeepARTModule,
    art::CommonARTModule,
    tidata::ClassIncrementalDataSplit,
    data::DataSplit,
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    # Task-incremental training
    train_inc!(art, tidata, n_train)

    # Get the number of testing samples
    l_n_test = min(n_test, length(data.test.y))

    # Test
    y_hats = basic_test(art, data, l_n_test)

    # Compute the performance fromt the test results
    perf = get_perf(art, data, y_hats, n_test)

    # Compile the experiment results
    out_dict = Dict(
        "y_hats" => y_hats,
        "perf" => perf,
    )

    return out_dict
end

# -----------------------------------------------------------------------------
# OLD EXPERIMENTS
# -----------------------------------------------------------------------------

"""
One-cold vector encoding of a one-hot encoded array.
"""
function one_coldify(y_hat::AbstractArray)
    return vec([x[1] for x in argmax(y_hat, dims=1)])
end

"""
Definition of testing accuracy for Flux.jl training loop logs.

# Arguments
- `y_hat::AbstractMatrix`: the predicted labels as a matrix.
- `y_truth::AbstractMatrix`: the true labels as a matrix.
- `n_class::Int`: the number of classes in the dataset.
"""
function flux_accuracy(
    y_hat::AbstractMatrix,
    y_truth::AbstractMatrix,
    n_class::Int,
    # n_class::Int=0,
)
    # flux_accuracy(x, y) = mean(Flux.onecold(flux_model(x), classes) .== y);
    # If the number of classes is specified, use that, otherwise infer from the training labels
    # n_classes = DeepART.n_classor(y_truth, n_class)
    classes = collect(1:n_class)

    # real_y_hat = vec([x[1] for x in argmax(y_hat, dims=1)])
    real_y_hat = one_coldify(y_hat)

    return Flux.mean(
        real_y_hat .== Flux.onecold(y_truth, classes)
        # Flux.onecold(y_hat, classes) .== Flux.onecold(y_truth, classes)
    )
    # return Flux.mean(Flux.onecold(y_hat, classes) .== y_truth)
end

"""
Simple train/test split experiment.
"""
function train_test!(
    # model::T,
    data::DataSplit,
    opts::Dict,
)
# ) where T <: Flux.Chain
    # Get the number of classes
    n_classes = length(unique(data.train.y))

    # Get the flat features and one-hot labels
    fdata = DeepART.flatty_hotty(data)

    # Get the resulting feature size
    n_input = size(fdata.train.x)[1]

    # Make a simple multilayer perceptron
    model = Chain(
        Dense(n_input, 128, relu),
        Dense(128, 64, relu),
        Dense(64, n_classes),
        sigmoid,
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

    # model |> gpu

    # model = DeepART.get_dense([n_input, 128, n_classes])

    # optim = Flux.setup(DeepART.EWC(), model)  # will store optimiser momentum, etc.
    optim = Flux.setup(Flux.Adam(), model)

    # Identify the loss function
    loss(x, y) = Flux.crossentropy(x, y)

    # dataloader = Flux.DataLoader((x, y_hot), batchsize=32)
    dataloader = Flux.DataLoader((fdata.train.x, fdata.train.y), batchsize=opts["N_BATCH"])

    # Flux.Optimisers.adjust!(optim, enabled = true)
    acc_log = []

    # @showprogress
    generate_showvalues(ep, i_train, acc) = () -> [(:Epoch, ep), (:Iter, i_train), (:Acc, acc)]

    for ep = 1:opts["N_EPOCH"]

        i_train = 0
        acc=0

        p = Progress(length(dataloader); showspeed=true)

        @info "Length: $(length(dataloader))"

        for (lx, ly) in dataloader
            # Compute gradients from the forward pass
            val, grads = Flux.withgradient(model) do m
                result = m(lx)
                # loss(result, ly)
                Flux.logitcrossentropy(result, ly)
            end

            # Update the model with the optimizer from these gradients
            Flux.update!(optim, model, grads[1])

            # Performance logging
            if i_train % opts["ACC_ITER"] == 0
                # acc = flux_accuracy(model(fdata.test.x), data.test.y, n_classes)
                acc = flux_accuracy(model(fdata.test.x), fdata.test.y, n_classes)
                push!(acc_log, acc)
                # @info "Epoch $ep: $acc"
            end

            i_train += 1

            next!(
                p;
                showvalues = generate_showvalues(ep, i_train, acc)
                # showvalues = [(:Epoch, ep), (:Iter, i_train), (:Acc, acc)],
            )
        end
    end

    lineplot(
        acc_log,
        title="Accuracy Trend",
        xlabel="Iteration",
        ylabel="Test Accuracy",
    )

    return acc_log

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
end
