"""
    data.jl

# Description
A collection of types and utilities for loading and handling datasets for the project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Converts a vector of string targets to a vector of integer targets using a target map.

# Arguments
- `targets::Vector{String}`: the vector of string targets to convert.
- `target_map::Dict{String, Int}`: the mapping of string targets to integer targets.
"""
function text_targets_to_ints(
    targets::Vector{String},
    # target_map::Dict{String, Int},
)
    unique_strings = unique(targets)
    # target_map = Dict{String, Int}()
    target_map = Dict{String, Int}(
        unique_strings[i] => i for i in eachindex(unique_strings)
    )

    return [target_map[t] for t in targets]
end

"""
Loads the MNIST dataset using MLDatasets.
"""
function get_mnist(;
    flatten::Bool=false,
)
    trainset = MLDatasets.MNIST(:train)
    testset = MLDatasets.MNIST(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)
    # dataset = tensorize_datasplit(dataset)

    if flatten
        dataset = flatty(dataset)
    end

    return dataset
end

"""
Loads the CIFAR10 dataset using MLDatasets.
"""
function get_cifar10(;
    flatten::Bool=false,
    gray::Bool=false,
)
    trainset = MLDatasets.CIFAR10(:train)
    testset = MLDatasets.CIFAR10(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    if gray
        X_train = mean(X_train, dims=3)
        X_test = mean(X_test, dims=3)
    end

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)
    # dataset = tensorize_datasplit(dataset)

    if flatten
        dataset = flatty(dataset)
    end

    return dataset
end

"""
Loads the fine CIFAR100 dataset using MLDatasets.
"""
function get_cifar100_fine(;
    flatten::Bool=false,
    gray::Bool=false,
)
    trainset = MLDatasets.CIFAR100(:train)
    testset = MLDatasets.CIFAR100(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    y_train = y_train.fine
    y_test = y_test.fine

    if gray
        X_train = mean(X_train, dims=3)
        X_test = mean(X_test, dims=3)
    end

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)
    # dataset = tensorize_datasplit(dataset)

    if flatten
        dataset = flatty(dataset)
    end

    return dataset
end

"""
Loads the coarse CIFAR100 dataset using MLDatasets.
"""
function get_cifar100_coarse(;
    flatten::Bool=false,
    gray::Bool=false,
)
    trainset = MLDatasets.CIFAR100(:train)
    testset = MLDatasets.CIFAR100(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    y_train = y_train.coarse
    y_test = y_test.coarse

    if gray
        X_train = mean(X_train, dims=3)
        X_test = mean(X_test, dims=3)
    end

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)
    # dataset = tensorize_datasplit(dataset)

    if flatten
        dataset = flatty(dataset)
    end

    return dataset
end

"""
Loads the FashionMNIST dataset using MLDatasets.
"""
function get_fashionmnist(;
    flatten::Bool=false,
)
    trainset = MLDatasets.FashionMNIST(:train)
    testset = MLDatasets.FashionMNIST(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)
    # dataset = tensorize_datasplit(dataset)

    if flatten
        dataset = flatty(dataset)
    end

    return dataset
end

"""
Loads the Omniglot dataset using MLDatasets.
"""
function get_omniglot(;
    flatten::Bool=false,
)
    trainset = MLDatasets.Omniglot(:train)
    testset = MLDatasets.Omniglot(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    y_train = text_targets_to_ints(y_train)
    y_test = text_targets_to_ints(y_test)

    dataset = DataSplit(X_train, y_train, X_test, y_test)
    # dataset = tensorize_datasplit(dataset)

    if flatten
        dataset = flatty(dataset)
    else
        X_train = reshape(dataset.train.x, 105, 105, 1, :)
        X_test = reshape(dataset.test.x, 105, 105, 1, :)
        dataset = DataSplit(X_train, y_train, X_test, y_test)
    end

    return dataset
end

function get_usps(;
    flatten::Bool=false,
)
    # Load the train and test datasets locally
    train = transpose(load_dataset(data_dir("usps/train.csv")))
    test = transpose(load_dataset(data_dir("usps/test.csv")))

    X_train = train[1:end-1, 2:end]
    y_train = Int.(train[end, 2:end])

    X_test = test[1:end-1, 2:end]
    y_test = Int.(test[end, 2:end])

    # Opposite of flatten operation since the dataset is already flat
    if !flatten
        X_train = reshape(X_train, 16, 16, 1, :)
        X_test = reshape(X_test, 16, 16, 1, :)
    end

    # Create a DataSplit
    dataset = DataSplit(X_train, y_train, X_test, y_test)

    return dataset
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
