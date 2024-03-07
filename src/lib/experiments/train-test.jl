"""
    train-test.jl

# Description
Implements the variety of training/testing start-to-finish experiments.
"""

"""
One-cold vector encoding of a one-hot encoded array.
"""
function one_coldify(y_hat::AbstractArray)
    return vec([x[1] for x in argmax(y_hat, dims=1)])
end

"""
Definition of testing accuracy for Flux.jl training loop logs.
"""
function flux_accuracy(
    y_hat::AbstractArray,
    y_truth::AbstractArray,
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
) where T <: Flux.Chain
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
