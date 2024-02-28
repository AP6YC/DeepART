"""
    data.jl

# Description
A collection of types and utilities for loading and handling datasets for the project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# ABTRACT TYPES
# -----------------------------------------------------------------------------

# abstract type  end

const AbstractFeatures = RealArray

const AbstractLabels = IntegerArray

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
A struct containing a supervised set of features in a matrix `x` mapping to integer labels `y`.
"""
struct SupervisedDataset{T <: RealArray, U <: IntegerArray}
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

struct ClassIncrementalDataSplit
    train::Vector{SupervisedDataset}
    test::Vector{SupervisedDataset}
end

function class_incrementalize(data::SupervisedDataset)
    # Initialize the new class incremental vector
    new_data = Vector{SupervisedDataset}()
    n_classes = length(unique(data.y))
    n_dim = length(size(data.x))

    # Iterate over all integer class labels
    for ix = 1:n_classes
        # Get all of the indices corresponding to the integer class label
        class_indices = findall(x->x==ix, data.y)
        # Fragile, but it works for now
        local_features = if n_dim == 3
            data.x[:, :, class_indices]
        elseif n_dim == 2
            data.x[:, class_indices]
        end
        # Create a new dataset from just these features and labels
        local_dataset = SupervisedDataset(
            local_features,
            data.y[class_indices],
        )
        # Add the local dataset to the vector of datasets to return
        push!(new_data, local_dataset)
    end

    return new_data
end

function ClassIncrementalDataSplit(datasplit::DataSplit)
    return ClassIncrementalDataSplit(
        class_incrementalize(datasplit.train),
        class_incrementalize(datasplit.test),
    )
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Convenience constructor for a supervised [`DataSplit`](@ref) that takes each set of features `x` and labels `y`separately.

# Arguments
- `X_train::RealArray`: the training features.
- `y_train::IntegerArray`: the training integer labels.
- `X_test::RealArray`: the testing features.
- `y_test::IntegerArray`: the testing integer labels.
"""
function DataSplit(
    X_train::RealArray,
    y_train::IntegerArray,
    X_test::RealArray,
    y_test::IntegerArray,
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
Wrapper for shuffling features and their labels.

# Arguments
- `features::AbstractArray`: the set of data features.
- `labels::AbstractArray`: the set of labels corresponding to the features.
"""
function shuffle_pairs(features::AbstractArray, labels::AbstractArray)
    # Use the MLUtils function for shuffling
    ls, ll = shuffleobs((features, labels))

    # Return the pairs
    return ls, ll
end

"""
Constructor for a [`DataSplit`](@ref) taking a set of features and options for the split ratio and shuffle flag.

# Arguments
- `features::RealArray`: the input features as an array of samples.
- `labels::IntegerVector`: the supervised labels as a vector of integers.
$ARG_P
$ARG_SHUFFLE
"""
function DataSplit(
    features::RealArray,
    labels::IntegerArray;
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
Turns the features of a dataset into a tensor.

# Arguments
- `data::SupervisedDataset`: the
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

function n_classor(y::Vector{Int}, n_class::Int=0)
    # If the number of classes is specified, use that, otherwise infer from the training labels
    n_classes = if n_class == 0
        length(unique(y))
    else
        n_class
    end

    return n_classes
end

function flatten(x::RealArray)
    dims = size(x)
    n_dims = length(dims)

    # Fragile, but it works for now
    x_new = if n_dims == 2
        x
    else
        flat_dim = prod([dims[ix] for ix = 1:n_dims-1])
        reshape(x, flat_dim, :)
    end

    return x_new
end

function one_hot(y::Vector{Int}, n_class::Int=0)
    n_samples = length(y)
    n_classes = n_classor(y, n_class)

    # y_cold = y[1:n_samples]
    y_hot = zeros(Int, n_classes, n_samples)
    for jx = 1:n_samples
        y_hot[y[jx], jx] = 1
    end
    return y_hot
end

"""
Specifically for the MNIST dataset as an example, flattens the feature dimensions and one-hot encodes the labels.
"""
function flatty_hotty(data::SupervisedDataset, n_class::Int=0)
    x_flat = flatten(data.x)
    # x |> gpu

    y_hot = one_hot(data.y, n_class)
    # y_hot |> gpu

    return x_flat, y_hot
end

function flatty_hotty(data::DataSplit, n_class::Int=0)
    x, y = flatty_hotty(data.train, n_class)
    xt, yt = flatty_hotty(data.test, n_class)
    return x, y, xt, yt
end

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
