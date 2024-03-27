"""
    train-test.jl

# Description
Implements the variety of training/testing start-to-finish experiments.
"""

# -----------------------------------------------------------------------------
# TRAIN/TEST FUNCTIONS
# -----------------------------------------------------------------------------

"""
Overload for getting a UnitRange of a [`SupervisedDataset`](@ref).
"""
function Base.getindex(
    data::SupervisedDataset,
    ix::Integer
)
    return get_sample(data, ix), data.y[ix]
end

"""
Overload for getting a UnitRange of a [`SupervisedDataset`](@ref).
"""
function Base.getindex(
    data::SupervisedDataset,
    ix::UnitRange,
)
    dims_x = size(data.x)
    n_dims_x = ndims(data.x)

    samples = if n_dims_x == 4
        reshape(data.x[:, :, :, ix], dims_x[1:3]..., 1)
    else
        data.x[:, ix]
    end

    return samples, data.y[ix]
end

"""
Overload for the length of a [`SupervisedDataset`](@ref).
"""
function Base.length(
    data::SupervisedDataset,
)
    return length(data.y)
end

"""
Generates a data loader for a [`CommonARTModule`](@ref) training/testing loop.
"""
function get_loader(
    art::ART.ARTModule,
    data::SupervisedDataset,
)
    # Create the data loader
    loader = Flux.DataLoader(data, batchsize=1)

    return loader
end

"""
Generates a data loader for a [`CommonARTModule`](@ref) training/testing loop.
"""
function get_loader(
    # art::CommonARTModule,
    art::DeepARTModule,
    data::SupervisedDataset,
)
    # Create the data loader
    loader = Flux.DataLoader(data, batchsize=1)

    # If using the gpu, push the data onto it
    if art.opts.gpu
        loader = loader |> gpu
    end

    return loader
end

"""
Task-homogenous training loop for a DeepART model.

# Arguments
$ARG_COMMONARTMODULE
$ARG_DATASPLIT
$ARG_N_TRAIN
"""
function basic_train!(
    art::CommonARTModule,
    # data::DataSplit;
    data::SupervisedDataset;
    display::Bool=false,
    desc::AbstractString="Task-Homogenous Train",
)
    # Create the data loader
    train_loader = get_loader(art, data)

    # Iterate over the training samples
    pr = Progress(
        length(train_loader);
        desc=desc,
        enabled = display,
    )

    # Get the estimates on the training data
    y_hats = Vector{Int}()
    for (xf, label) in train_loader
        # Correction for ART modules
        if art isa ART.ARTModule
            xf = vec(xf)
        end
        # Train
        y_hat = incremental_supervised_train!(art, xf, cpu(label)[1])
        # Push the estimate
        push!(y_hats, y_hat)
        # Update the display iterator
        next!(pr; showvalues=[
            (:NCat, art.n_categories),
        ])
    end

    # Return the estimates
    return y_hats
end

"""
Task-homogenous testing loop for a [`DeepARTModule`](@ref) model.

# Arguments
$ARG_COMMONARTMODULE
$ARG_N_TEST
"""
function basic_test(
    art::CommonARTModule,
    data::SupervisedDataset;
    display::Bool=false,
    desc::AbstractString="Task-Homogenous Test",
)
    # Create the data loader
    test_loader = get_loader(art, data)

    # Create a display iterator
    pr = Progress(
        length(test_loader);
        desc=desc,
        enabled = display,
    )

    # Get the estimates on the test data
    y_hats = Vector{Int}()
    for (xf, _) in test_loader
        # Correction for ART modules
        if art isa ART.ARTModule
            xf = vec(xf)
        end
        # Classify the sample
        y_hat = incremental_classify(art, xf)
        # Push the estimate
        push!(y_hats, y_hat)
        # Update the display iterator
        next!(pr)
    end

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
    tidata::ClassIncrementalDataSplit;
    display::Bool=false,
    desc::AbstractString="Task-Incremental Train",
    # n_train::Integer=IInf,
)
    # Infer the number of tasks to train over
    n_tasks = length(tidata.train)

    # Iterate over the tasks
    y_hats = Vector{Vector{Int}}()
    for ix = 1:n_tasks
        # Run a single task-homogenous training loop
        local_desc = "$desc: Task $(ix)"
        local_y_hats = basic_train!(
            art,
            tidata.train[ix],
            display=display,
            desc=local_desc,
        )
        push!(y_hats, local_y_hats)
    end
    return y_hats
end

"""
Computes the performance of the ART module given some estimates.
"""
function get_perf(
    data::SupervisedDataset,
    y_hats::Vector{Int},
)
    # perf = ART.performance(y_hats, data.test.y[1:l_n_test])
    perf = ART.performance(y_hats, data.y)
    # @info "Perf: $perf, n_cats: $(art.n_categories), uniques: $(unique(y_hats))"

    return perf
end

"""
Wrapper for loading simulation results with arbitrarily many fields.

# Arguments
- `data_file::AbstractString`: the location of the datafile for loading.
- `args...`: the string names of the files to open.
"""
function load_sim_results(data_file::AbstractString, args...)
    # Load and return the tuple of entries from the data file
    return JLD2.load(data_file, args...)
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
    art::CommonARTModule,
    data::DataSplit;
    display::Bool=false,
)
    # Train
    # y_hats_train = basic_train!(art, data.train, display=display)
    _ = basic_train!(art, data.train, display=display)

    # Test
    y_hats = basic_test(art, data.test, display=display)

    # Compute the performance fromt the test results
    perf = get_perf(data.test, y_hats)

    # Compile the experiment results
    out_dict = Dict(
        "n_cat" => art.n_categories,
        "y_hats" => y_hats,
        "perf" => perf,
    )

    # Return the experiment results
    return out_dict
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
    data::DataSplit;
    display::Bool=false,
)
    # Task-incremental training
    _ = train_inc!(art, tidata, display=display)

    # Test
    y_hats = basic_test(art, data.test, display=display)

    # Compute the performance fromt the test results
    perf = get_perf(data.test, y_hats)

    # Compile the experiment results
    out_dict = Dict(
        "n_cat" => art.n_categories,
        "y_hats" => y_hats,
        "perf" => perf,
    )

    return out_dict
end

"""
Get a list of the percentage accuracies.

# Arguments
- `y::IntegerVector`: the target values.
- `y_hat::IntegerVector`: the agent's estimates.
- `n_classes::Integer`: the number of total classes in the test set.
"""
function get_accuracies(y::IntegerVector, y_hat::IntegerVector, n_classes::Integer)
    cm = get_confusion(y, y_hat, n_classes)
    correct = [cm[i,i] for i = 1:n_classes]
    # total = sum(cm, dims=1)
    total = sum(cm, dims=2)'
    accuracies = correct'./total

    return accuracies
end

# -----------------------------------------------------------------------------
# MULTI EXPERIMENT WRAPPERS
# -----------------------------------------------------------------------------

# function train_test_plot(
#     data::DataSplit,
#     opts::Dict,
#     conv::Bool=false,
# )
#     # Load the dataset with the provided options
#     # isconv = !(d["m"] == "DeepARTConv")
#     # data = load_one_dataset(
#     #     d["dataset"],
#     #     flatten=isconv,
#     #     gray=true,
#     #     n_train=d["n_train"],
#     #     n_test=d["n_test"],
#     # )

#     head_dim = 1024
#     size_tuple = (size(data.train.x)[1:3]..., 1)
#     conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

#     # Construct the module from the options
#     art = get_module_from_options(d, data)


#     art = DeepART.ARTINSTART(
#         conv_model,
#         head_dim=head_dim,
#         beta=BETA_D,
#         beta_s=BETA_S,
#         # rho=0.6,
#         rho=0.3,
#         update="art",
#         softwta=true,
#         gpu=GPU,
#     )

#     results = DeepART.tt_basic!(
#         art,
#         data,
#         display=DISPLAY
#     )
#     @info "Results: " results["perf"] results["n_cat"]

#     # Create the confusion matrix from this experiment
#     DeepART.plot_confusion_matrix(
#         data.test.y,
#         results["y_hats"],
#         string.(collect(0:9)),
#         "conv_basic_confusion",
#         EXP_TOP,
#     )

#     return results

# end

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
Simple train/test split experiment using an MLP with gradient descent on the datset.
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


# """
# Task-homogenous testing loop for a [`DeepARTModule`](@ref) model.

# # Arguments
# $ARG_COMMONARTMODULE
# $ARG_DATASPLIT
# $ARG_N_TEST
# """
# function basic_test(
#     art::CommonARTModule,
#     data::DataSplit,
#     # n_test::Integer=IInf,
# )
#     # Get the number of testing samples
#     # l_n_test = get_n(n_test, data.test)

#     # Get the estimates on the test data
#     y_hats = Vector{Int}()


#     test_loader = Flux.DataLoader(data.test, batchsize=1)

#     if art.opts.gpu
#         test_loader = test_loader |> gpu
#     end

#     pr = Progress(
#         # l_n_test;
#         length(test_loader);
#         desc="Task-Homogenous Testing",
#         enabled = Sys.iswindows(),
#     )

#     # for ix = 1:l_n_test
#     for (xf, _) in test_loader
#         # xf = data.test.x[:, ix]
#         # xf = get_sample(data.test, ix)
#         # y_hat = DeepART.classify(art, xf, get_bmu=true)
#         y_hat = incremental_classify(art, xf)
#         push!(y_hats, y_hat)
#         next!(pr)
#     end

#     # # Calculate the performance and log
#     # perf = DeepART.ART.performance(y_hats, data.test.y[1:l_n_test])
#     # @info "Perf: $perf, n_cats: $(art.n_categories), uniques: $(unique(y_hats))"

#     # Return the estimates
#     return y_hats
# end
