"""
    data.jl

# Description
A collection of types and utilities for loading and handling datasets for the project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
A struct containing a supervised set of features in a matrix `x` mapping to integer labels `y`.
"""
struct SupervisedDataset{T <: AbstractFeatures, U <: AbstractLabels}
    """
    A set of features.
    """
    x::T

    """
    The labels corresponding to each feature.
    """
    y::U
end

"""
A train/test split of supervised datasets.
"""
struct DataSplit
    """
    The training portion of the dataset.
    """
    train::SupervisedDataset

    """
    The test portion of the dataset.
    """
    test::SupervisedDataset
end

"""
A class-incremental variant of a [`DataSplit`](@ref) containing instead vectors of [`SupervisedDataset`](@ref)s.
"""
struct ClassIncrementalDataSplit
    """
    The vector of training class datasets.
    """
    train::Vector{SupervisedDataset}

    """
    The vector of testing class datasets.
    """
    test::Vector{SupervisedDataset}
end

"""
Turns a normal [`SupervisedDataset`](@ref) into a class-incremental vector of [`SupervisedDataset`](@ref)s.

# Arguments
$ARG_SUPERVISEDDATASET
"""
function class_incrementalize(data::SupervisedDataset)
    # Initialize the new class incremental vector
    new_data = Vector{SupervisedDataset}()

    local_y = if ndims(data.y) == 2
        Flux.onecold(data.y)
    else
        data.y
    end

    n_classes = length(unique(local_y))
    n_dim = ndims(data.x)

    # Iterate over all integer class labels
    for ix = 1:n_classes
        # Get all of the indices corresponding to the integer class label

        class_indices = findall(x->x==ix, local_y)

        # Fragile, but it works for now
        local_features = if n_dim == 4
            data.x[:, :, :, class_indices]
        elseif n_dim == 3
            data.x[:, :, class_indices]
        elseif n_dim == 2
            data.x[:, class_indices]
        end

        local_labels = if ndims(data.y) == 2
            data.y[:, class_indices]
        else
            data.y[class_indices]
        end

        # Create a new dataset from just these features and labels
        local_dataset = SupervisedDataset(
            local_features,
            local_labels,
        )
        # Add the local dataset to the vector of datasets to return
        push!(new_data, local_dataset)
    end

    return new_data
end

"""
Constructor for a [`ClassIncrementalDataSplit`](@ref) taking a normal [`DataSplit`](@ref).

# Arguments
$ARG_DATASPLIT
"""
function ClassIncrementalDataSplit(datasplit::DataSplit)
    return ClassIncrementalDataSplit(
        class_incrementalize(datasplit.train),
        class_incrementalize(datasplit.test),
    )
end

# """
# A task-incremental variant of a [`DataSplit`](@ref) containing multiple classes per task.
# """
# struct TaskIncrementalDataSplit
#     """
#     The vector of training class datasets.
#     """
#     train::Vector{SupervisedDataset}

#     """
#     The vector of testing class datasets.
#     """
#     test::Vector{SupervisedDataset}
# end

function group_datasets(
    data::Vector{SupervisedDataset},
    group::Vector{Int},
    shuffle::Bool=true,
)
    # Cat the features
    local_features = hcat([data[ix].x for ix in group]...)

    # If we have one-hot encoded labels, we need to stack them differently
    if ndims(data[1].y) == 2
        local_labels = hcat([data[ix].y for ix in group]...)
    else
        local_labels = vcat([data[ix].y for ix in group]...)
    end

    if shuffle
        local_features, local_labels = shuffle_pairs(local_features, local_labels)
        local_features = copy(local_features)
        local_labels = copy(local_labels)
    end

    return SupervisedDataset(local_features, local_labels)
end

function task_incrementalize(
    data::Vector{SupervisedDataset},
    groupings::Vector{Vector{Int}},
    shuffle::Bool=true,
)
    new_data = Vector{SupervisedDataset}()

    for group in groupings
        # push!(new_data, SupervisedDataset(local_features, local_labels))
        push!(new_data, group_datasets(data, group, shuffle))
    end

    return new_data
end

function TaskIncrementalDataSplit(
    datasplit::ClassIncrementalDataSplit,
    groupings::Vector{Vector{Int}};
    shuffle::Bool=true,
)
    return ClassIncrementalDataSplit(
        task_incrementalize(datasplit.train, groupings, shuffle),
        task_incrementalize(datasplit.test, groupings, shuffle),
    )
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Convenience constructor for a supervised [`DataSplit`](@ref) that takes each set of features `x` and labels `y`separately.

# Arguments
- `X_train::AbstractFeatures`: the training features.
- `y_train::AbstractLabels`: the training integer labels.
- `X_test::AbstractFeatures`: the testing features.
- `y_test::AbstractLabels`: the testing integer labels.
"""
function DataSplit(
    X_train::AbstractFeatures,
    y_train::AbstractLabels,
    X_test::AbstractFeatures,
    y_test::AbstractLabels,
)
    return DataSplit(
        SupervisedDataset(
            X_train,
            y_train,
        ),
        SupervisedDataset(
            X_test,
            y_test,
        ),
    )
end

"""
Constructor for a [`DataSplit`](@ref) taking a set of features and options for the split ratio and shuffle flag.

# Arguments
- `features::AbstractFeatures`: the input features as an array of samples.
- `labels::AbstractLabels`: the supervised labels.
$ARG_P
$ARG_SHUFFLE
"""
function DataSplit(
    features::AbstractFeatures,
    labels::AbstractLabels;
    p::Float=DEFAULT_P,
    shuffle::Bool=DEFAULT_SHUFFLE,
)
    # Get the features and labels
    ls, ll = if shuffle
        # ls, ll = shuffleobs((features, labels))
        shuffle_pairs(features, labels)
    else
        (features, labels)
    end

    # Create a train/test split
    (X_train, y_train), (X_test, y_test) = splitobs((ls, ll); at=p)

    # Create and return a single container for this train/test split
    return DataSplit(
        copy(X_train),
        copy(y_train),
        copy(X_test),
        copy(y_test),
    )
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`DataSplit`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::DataSplit`: the [`DataSplit`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    ds::DataSplit,
)
    # Compute all of the dimensions of the dataset
    s_train = size(ds.train.x)
    s_test = size(ds.test.x)

    # Get the number of features, training samples, and testing samples
    n_dims = length(s_train)
    n_features = s_train[1:n_dims - 1]
    n_train = s_train[end]
    n_test = s_test[end]

    # print(io, "DataSplit(features: $(size(ds.train.x)), test: $(size(ds.test.x)))")
    print(io, "DataSplit(features: $(n_features), train: $(n_train), test: $(n_test))")
end

"""
Overload of the show function for [`ClassIncrementalDataSplit`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::ClassIncrementalDataSplit`: the [`ClassIncrementalDataSplit`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    ds::ClassIncrementalDataSplit,
)
    # Compute all of the dimensions of the dataset
    s_train = size(ds.train[1].x)

    # Get the number of features, training samples, and testing samples
    n_dims = length(s_train)
    n_features = s_train[1:n_dims - 1]
    n_classes = length(ds.train)

    # print(io, "DataSplit(features: $(size(ds.train.x)), test: $(size(ds.test.x)))")
    print(io, "ClassIncrementalDataSplit(features: $(n_features), n_classes: $(n_classes))")
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Loads the MNIST dataset using MLDatasets.
"""
function get_mnist()
    trainset = MLDatasets.MNIST(:train)
    testset = MLDatasets.MNIST(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)

    # dataset = tensorize_datasplit(dataset)

    return dataset
end

"""
Loads the MNIST dataset using MLDatasets.
"""
function get_cifar10()
    trainset = MLDatasets.CIFAR10(:train)
    testset = MLDatasets.CIFAR10(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)

    # dataset = tensorize_datasplit(dataset)

    return dataset
end

"""
Turns the features of a dataset into a tensor.

# Arguments
$ARG_SUPERVISEDDATASET
"""
function tensorize_dataset(data::SupervisedDataset)
    dims = size(data.x)
    new_dataset = SupervisedDataset(
        reshape(data.x, dims[1:end-1]..., 1, :),
        data.y,
    )
    return new_dataset
end

"""
Tensorizes both the training and testing components of a [`DataSplit`](@ref).

# Arguments
$ARG_DATASPLIT
"""
function tensorize_datasplit(data::DataSplit)
    new_dataset = DataSplit(
        tensorize_dataset(data.train),
        tensorize_dataset(data.test),
    )
    return new_dataset
end

"""
Returns the number of classes given a vector of labels.

If the number of classes is provided, that is used; otherwise, the number of classes is inferred from the labels.

# Arguments
- `y::IntegerVector`: the vector of integer labels.
$ARG_N_CLASS
"""
function n_classor(y::IntegerVector, n_class::Int=0)
    # If the number of classes is specified, use that, otherwise infer from the training labels
    n_classes = if n_class == 0
        length(unique(y))
    else
        n_class
    end

    return n_classes
end

"""
Flattens a set of features to a 2D matrix.

Every dimension except the last is reshaped into the first dimension.

# Arguments
- `x::AbstractFeatures`: the array of features to flatten.
"""
function flatten(x::AbstractFeatures)
    dims = size(x)
    n_dims = length(dims)

    # If the array is already 2D, return it
    x_new = if n_dims == 2
        x
    # Otherwise, reshape into the product of the first (n_dims-1) dimensions
    else
        flat_dim = prod([dims[ix] for ix = 1:n_dims-1])
        reshape(x, flat_dim, :)
    end

    return x_new
end

"""
One-hot encodes the vector of labels into a matrix of ones.

# Arguments
- `y::IntegerVector`: the vector of integer labels.
$ARG_N_CLASS
"""
function one_hot(y::IntegerVector, n_class::Int=0)
    # Get the number of samples and classes for iteration
    # n_samples = length(y)
    n_classes = n_classor(y, n_class)

    if FLUXONEHOT
        y_hot = Flux.onehotbatch(y, collect(1:n_classes))
    else
        # Initialize the one-hot matrix
        y_hot = zeros(Int, n_classes, n_samples)

        # For each sample, set a one at the index of the value of the integer label
        for jx = 1:n_samples
            y_hot[y[jx], jx] = 1
        end
    end

    return y_hot
end

"""
Flattens the feature dimensions of a [`SupervisedDataset`](@ref) and one-hot encodes the labels.

# Arguments
$ARG_SUPERVISEDDATASET
$ARG_N_CLASS
"""
function flatty_hotty(data::SupervisedDataset, n_class::Int=0)
    # Flatten the features
    x_flat = flatten(data.x)
    # x |> gpu

    # One-hot encode the labels
    y_hot = one_hot(data.y, n_class)
    # y_hot |> gpu

    # return x_flat, y_hot
    return SupervisedDataset(
        x_flat,
        y_hot,
    )
end

"""
Flattens and one-hot encodes a [`DataSplit`](@ref).

# Arguments
$ARG_DATASPLIT
$ARG_N_CLASS
"""
function flatty_hotty(data::DataSplit, n_class::Int=0)
    new_train = flatty_hotty(data.train, n_class)
    new_test = flatty_hotty(data.test, n_class)

    # Construct and return the new DataSplit
    return DataSplit(
        new_train,
        new_test,
    )
end

# function data_shape

"""
Loads a dataset from a local file.

# Arguments
- `filename::AbstractString`: the location of the file to load with a default value.
"""
function load_dataset(
    filename::AbstractString,
)
    # Load the data
    data = readdlm(filename, ',', header=false)

    return data
end

"""
Loades the datasets from the data package experiment.

# Arguments
- `topdir::AbstractString`: default `data_dir("data-package")`, the directory containing the CSV data package files.
$ARG_SHUFFLE
$ARG_P
"""
function load_all_datasets(
    topdir::AbstractString=data_dir("data-package"),
    shuffle::Bool=true,
    p::Float=0.8,
)
    # Walk the directory
    data = Dict{String, Any}()
    for (root, _, files) in walkdir(topdir)
        # Iterate over all of the files
        for file in files
            # Get the full filename for the current data file
            filename = joinpath(root, file)
            data_name = splitext(file)[1]
            data[data_name] = load_dataset(filename)
            # @info typeof(data[data_name])
        end
    end

    # Turn each dataset into a SupervisedDataset
    data_splits = Dict{String, DataSplit}()
    for (name, dataset) in data
        # Assume that the last column is the labels, all others are features
        n_features = size(dataset)[2] - 1

        # Get the features and labels (Float32 precision for Flux dense networks)
        features = Matrix{FluxFloat}(dataset[:, 1:n_features]')
        labels = Vector{Int}(dataset[:, end])
        # @info typeof(features)
        # @info typeof(labels)
        # Create a DataSplit
        data_splits[name] = DataSplit(
            features,
            labels,
            shuffle=shuffle,
            p=p,
        )
    end

    # return data
    return data_splits
end
