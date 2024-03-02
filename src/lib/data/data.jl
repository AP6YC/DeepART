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
