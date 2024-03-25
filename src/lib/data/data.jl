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

function get_x_subset(
    x::AbstractArray,
    n_samples::Integer=IInf,
)
    # Fragile, but it works for now
    l_n_dim=ndims(x)
    local_features = if l_n_dim == 4
        x[:, :, :, 1:n_samples]
    elseif l_n_dim == 3
        x[:, :, 1:n_samples]
    elseif l_n_dim == 2
        x[:, 1:n_samples]
    end

    return local_features
end

function get_y_subset(
    y::AbstractArray,
    n_samples::Integer=IInf,
)
    l_n_dim = ndims(y)
    local_labels = if l_n_dim == 2
        y[:, 1:n_samples]
    else
        y[1:n_samples]
    end

    return local_labels
end

function get_data_subset(
    data::DataSplit;
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    return DataSplit(
        get_x_subset(data.train.x, n_train),
        get_y_subset(data.train.y, n_train),
        get_x_subset(data.test.x, n_test),
        get_y_subset(data.test.y, n_test),
    )
end

"""
Loads the MNIST dataset using MLDatasets.
"""
function get_mnist(;
    flatten::Bool=false,
    gray::Bool=false,       # MNIST is already grayscale
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    trainset = MLDatasets.MNIST(:train)
    testset = MLDatasets.MNIST(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)
    # dataset = tensorize_datasplit(dataset)

    # Get a subset of the data if necessary
    if (n_train != IInf) || (n_test != IInf)
        l_n_train = min(n_train, length(y_train))
        l_n_test = min(n_test, length(y_test))
        dataset = get_data_subset(
            dataset,
            n_train=l_n_train,
            n_test=l_n_test,
        )
    end

    if flatten
        dataset = flatty(dataset)
    else
        dataset = DataSplit(
            reshape(dataset.train.x, 28, 28, 1, :),
            dataset.train.y,
            reshape(dataset.test.x, 28, 28, 1, :),
            dataset.test.y,
        )
    end

    return dataset
end

"""
Loads the CIFAR10 dataset using MLDatasets.
"""
function get_cifar10(;
    flatten::Bool=false,
    gray::Bool=false,
    n_train::Integer=IInf,
    n_test::Integer=IInf,
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

    # Get a subset of the data if necessary
    if (n_train != IInf) || (n_test != IInf)
        l_n_train = min(n_train, length(y_train))
        l_n_test = min(n_test, length(y_test))
        dataset = get_data_subset(
            dataset,
            n_train=l_n_train,
            n_test=l_n_test,
        )
    end

    if flatten
        dataset = flatty(dataset)
    else
        dataset = DataSplit(
            reshape(dataset.train.x, 32, 32, 1, :),
            dataset.train.y,
            reshape(dataset.test.x, 32, 32, 1, :),
            dataset.test.y,
        )
    end

    return dataset
end

"""
Loads the fine CIFAR100 dataset using MLDatasets.
"""
function get_cifar100_fine(;
    flatten::Bool=false,
    gray::Bool=false,
    n_train::Integer=IInf,
    n_test::Integer=IInf,
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

    # Get a subset of the data if necessary
    if (n_train != IInf) || (n_test != IInf)
        l_n_train = min(n_train, length(y_train))
        l_n_test = min(n_test, length(y_test))
        dataset = get_data_subset(
            dataset,
            n_train=l_n_train,
            n_test=l_n_test,
        )
    end

    if flatten
        dataset = flatty(dataset)
    else
        dataset = DataSplit(
            reshape(dataset.train.x, 32, 32, 1, :),
            dataset.train.y,
            reshape(dataset.test.x, 32, 32, 1, :),
            dataset.test.y,
        )
    end

    return dataset
end

"""
Loads the coarse CIFAR100 dataset using MLDatasets.
"""
function get_cifar100_coarse(;
    flatten::Bool=false,
    gray::Bool=false,
    n_train::Integer=IInf,
    n_test::Integer=IInf,
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

    # Get a subset of the data if necessary
    if (n_train != IInf) || (n_test != IInf)
        l_n_train = min(n_train, length(y_train))
        l_n_test = min(n_test, length(y_test))
        dataset = get_data_subset(
            dataset,
            n_train=l_n_train,
            n_test=l_n_test,
        )
    end

    if flatten
        dataset = flatty(dataset)
    else
        dataset = DataSplit(
            reshape(dataset.train.x, 32, 32, 1, :),
            dataset.train.y,
            reshape(dataset.test.x, 32, 32, 1, :),
            dataset.test.y,
        )
    end

    return dataset
end

"""
Loads the FashionMNIST dataset using MLDatasets.
"""
function get_fashionmnist(;
    flatten::Bool=false,
    gray::Bool=false,       # FashionMNIST is already grayscale
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    trainset = MLDatasets.FashionMNIST(:train)
    testset = MLDatasets.FashionMNIST(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)
    # dataset = tensorize_datasplit(dataset)

    # Get a subset of the data if necessary
    if (n_train != IInf) || (n_test != IInf)
        l_n_train = min(n_train, length(y_train))
        l_n_test = min(n_test, length(y_test))
        dataset = get_data_subset(
            dataset,
            n_train=l_n_train,
            n_test=l_n_test,
        )
    end

    if flatten
        dataset = flatty(dataset)
    else
        dataset = DataSplit(
            reshape(dataset.train.x, 28, 28, 1, :),
            dataset.train.y,
            reshape(dataset.test.x, 28, 28, 1, :),
            dataset.test.y,
        )
    end

    return dataset
end

"""
Loads the Omniglot dataset using MLDatasets.
"""
function get_omniglot(;
    flatten::Bool=false,
    gray::Bool=false,       # Omniglot is already grayscale
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    trainset = MLDatasets.Omniglot(:train)
    testset = MLDatasets.Omniglot(:test)

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    y_train = text_targets_to_ints(y_train)
    y_test = text_targets_to_ints(y_test)

    dataset = DataSplit(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # dataset = tensorize_datasplit(dataset)

    if flatten
        dataset = flatty(dataset)
    else
        X_train = reshape(dataset.train.x, 105, 105, 1, :)
        X_test = reshape(dataset.test.x, 105, 105, 1, :)
        dataset = DataSplit(X_train, y_train, X_test, y_test)
    end

    # Get a subset of the data if necessary
    if (n_train != IInf) || (n_test != IInf)
        l_n_train = min(n_train, length(y_train))
        l_n_test = min(n_test, length(y_test))
        dataset = get_data_subset(
            dataset,
            n_train=l_n_train,
            n_test=l_n_test,
        )
    end

    return dataset
end

function get_usps(;
    flatten::Bool=false,
    gray::Bool=false,       # USPS is already grayscale
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    # Load the train and test datasets locally
    train = transpose(load_dataset_file(data_dir("usps/train.csv")))
    test = transpose(load_dataset_file(data_dir("usps/test.csv")))

    X_train = train[1:end-1, 2:end]
    y_train = Int.(train[end, 2:end]) .+ 1

    X_test = test[1:end-1, 2:end]
    y_test = Int.(test[end, 2:end]) .+ 1

    # Opposite of flatten operation since the dataset is already flat
    if !flatten
        X_train = reshape(X_train, 16, 16, 1, :)
        X_test = reshape(X_test, 16, 16, 1, :)
    end

    # Create a DataSplit
    dataset = DataSplit(X_train, y_train, X_test, y_test)

    # Get a subset of the data if necessary
    if (n_train != IInf) || (n_test != IInf)
        l_n_train = min(n_train, length(y_train))
        l_n_test = min(n_test, length(y_test))
        dataset = get_data_subset(
            dataset,
            n_train=l_n_train,
            n_test=l_n_test,
        )
    end

    return dataset
end

# function get_sample(
#     data::SupervisedDataset,
#     index::Integer,
# )
#     sample = if ndims(data.x) == 4
#         data.x[:, :, :, index]
#     else
#         data.x[:, index]
#     end
#     return sample
# end

"""
Loads a dataset from a local file.

# Arguments
- `filename::AbstractString`: the location of the file to load with a default value.
"""
function load_dataset_file(
    filename::AbstractString,
)
    # Load the data
    data = readdlm(filename, ',', header=false)

    return data
end

"""
Constructs a [`DataSplit`](@ref) from an existing dataset.

This assumes that the last column is the labels and all others are features.

# Arguments
- `dataset::AbstractMatrix`: the dataset to split.
$ARG_SHUFFLE
$ARG_P
"""
function DataSplit(
    dataset::AbstractMatrix;
    shuffle::Bool=true,
    p::Float=0.8,
)
    # Assume that the last column is the labels, all others are features
    n_features = size(dataset)[2] - 1

    # Get the features and labels (Float32 precision for Flux dense networks)
    features = Matrix{FluxFloat}(dataset[:, 1:n_features]')
    labels = Vector{Int}(dataset[:, end])

    # Create and return a DataSplit
    DataSplit(
        features,
        labels,
        shuffle=shuffle,
        p=p,
    )
end

const DATA_DISPATCH = Dict(
    "mnist" => get_mnist,
    "fashionmnist" => get_fashionmnist,
    "cifar10" => get_cifar10,
    "cifar100_fine" => get_cifar100_fine,
    "cifar100_coarse" => get_cifar100_coarse,
    "omniglot" => get_omniglot,
    "usps" => get_usps,
    # "CBB-R15" => get_data_package_dataset,
)

const DATA_PACKAGE_NAMES = [
    "CBB-Aggregation",
    "CBB-Aggregation",
    "CBB-Compound",
    "CBB-flame",
    "CBB-jain",
    "CBB-pathbased",
    "CBB-R15",
    "CBB-spiral",
    "face",
    "flag",
    "halfring",
    "iris",
    "moon",
    "ring",
    "spiral",
    "wave",
    "wine",
]

function load_data_package_dataset(
    name::AbstractString;
    shuffle::Bool=true,
    p::Float=0.8,
)
    # Load the dataset from file
    local_data = load_dataset_file(
        data_dir("data-package", "$(name).csv")
    )

    # Construct and return a DataSplit
    return DataSplit(
        local_data,
        shuffle=shuffle,
        p=p,
    )
end

"""
Loads a single dataset by name, dispatching accordingly.

# Arguments
- `name::AbstractString`: the name of the dataset to load.
- `args...`: additional arguments to pass to the dataset loading function.
"""
function load_one_dataset(name::AbstractString; kwargs...)
    # If the name is in the datasets function dispatch map
    if name in keys(DATA_DISPATCH)
        return DATA_DISPATCH[name](;kwargs...)
    elseif name in DATA_PACKAGE_NAMES
        return load_data_package_dataset(name; kwargs...)
    else
        error("The dataset name $(name) is not set up.")
    end
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
    # Initialize the output data splits dictionary
    data_splits = Dict{String, DataSplit}()

    # Iterate over all of the files
    for file in readdir(topdir)
        # Get the filename for the current data file
        data_name = splitext(file)[1]
        data_splits[data_name] = load_data_package_dataset(
            data_name,
            shuffle=shuffle,
            p=p,
        )
    end

    return data_splits
end
