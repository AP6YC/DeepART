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
Gets a subset of the dataset samples from the first index up to the number requested.
"""
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

"""
Gets a subset of the dataset labels from the first index up to the number requested.
"""
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

function assign_sample!(
    data::SupervisedDataset,
    new_x::AbstractArray,
    new_y::Integer,
    index::Integer,
)
    l_n_dim = ndims(new_x)
    if l_n_dim == 3
        data.x[:, :, :, index] .= new_x
    elseif l_n_dim == 2
        data.x[:, :, index] .= new_x
    elseif l_n_dim == 1
        data.x[:, index] .= new_x
    end

    data.y[index] = new_y

    return
end

function get_supervised_subset(
    data::SupervisedDataset,
    n_samples::Integer=IInf,
    # n_train::Integer=IInf,
    # n_test::Integer=IInf,
)
    original_classes = unique(data.y)
    n_original_classes = length(original_classes)

    # Create a new dataset with some
    new_data = SupervisedDataset(
        get_x_subset(data.x, n_samples),
        get_y_subset(data.y, n_samples),
    )

    new_classes = unique(new_data.y)
    n_new_classes = length(new_classes)
    # If there is a missing class
    if n_new_classes != n_original_classes
        # Get the ids that are in the original but not new
        problem_ids = setdiff(original_classes, new_classes)
        # For each id
        for pid in problem_ids
            # Get all of the counts of the labels
            dest_ids = MLUtils.group_counts(new_data.y)
            # Get a destination id that is represented at least 2 times otherwise
            open_spot = findfirst(x -> dest_ids[new_data.y[x]] > 1, 1:length(new_data.y))
            # Get the first sample belonging to the problem ID from the old data
            old_data_id = findfirst(x -> data.y[x] == pid, 1:length(data.y))
            new_x = get_sample(data, old_data_id)
            new_y = data.y[old_data_id]

            # Assign the sample in place to the new data
            assign_sample!(new_data, new_x, new_y, open_spot)
        end
    end

    return new_data
end

"""
Gets a training and testing subset of the data from a [`DataSplit`](@ref) object.
"""
function get_data_subset(
    data::DataSplit;
    n_train::Integer=IInf,
    n_test::Integer=IInf,
)
    return DataSplit(
        get_supervised_subset(data.train, n_train),
        get_supervised_subset(data.test, n_test),
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
    # else
    #     dataset = DataSplit(
    #         reshape(dataset.train.x, 32, 32, 1, :),
    #         dataset.train.y,
    #         reshape(dataset.test.x, 32, 32, 1, :),
    #         dataset.test.y,
    #     )
    end

    return dataset
end

function fix_missing_els(new_dataset::SupervisedDataset, old_dataset::SupervisedDataset, n_classes::Integer)
    if length(unique(new_dataset.train.y)) < n_classes
        @warn "CIFAR100 fine dataset has less than 100 classes."
        missing_els = setdiff(1:100, unique(new_dataset.train.y))
        for el in missing_els
            new_dataset.train.y[new_dataset.train.y .> el] .-= 1
            new_dataset.test.y[new_dataset.test.y .> el] .-= 1
        end
        dataset = new_dataset
    end

    return new_dataset
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

    # original_size = size(X_train)[1:end-1]

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
        # dataset = DataSplit(
        #     fix_missing_els(new_dataset.train, dataset.train, 100),
        #     fix_missing_els(new_dataset.test, dataset.test, 100),
        # )
    end

    if flatten
        dataset = flatty(dataset)
    # else
    #     dataset = DataSplit(
    #         reshape(dataset.train.x, 32, 32, 1, :),
    #         # reshape(dataset)
    #         dataset.train.y,
    #         reshape(dataset.test.x, 32, 32, 1, :),
    #         dataset.test.y,
    #     )
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
    # else
    #     dataset = DataSplit(
    #         reshape(dataset.train.x, 32, 32, 1, :),
    #         dataset.train.y,
    #         reshape(dataset.test.x, 32, 32, 1, :),
    #         dataset.test.y,
    #     )
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
        shuffle=true,
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

function double_tensor(data::AbstractArray)
    nx, ny, ns = size(data)
    tnx, tny = nx*2, ny*2
    dest = zeros(eltype(data), (tnx, tny, ns))
    for ix = 1:ns
        dest[:, :, ix] = imresize(data[:, :, ix], (tnx, tny))
    end
    return dest
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
        # Make arrays
        X_train = double_tensor(reshape(X_train, 16, 16, :))
        X_test = double_tensor(reshape(X_test, 16, 16, :))
        # Interpolate

        # X_train = reshape(X_train, 16, 16, 1, :)
        # X_test = reshape(X_test, 16, 16, 1, :)
        X_train = reshape(X_train, 32, 32, 1, :)
        X_test = reshape(X_test, 32, 32, 1, :)
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

# const HOSTNAME_MAP = Dict(
#     "SASHA-XPS" => "laptop",
#     "Sasha-PC" => "pc",
#     "linux" => "cluster",
# )

# function get_comp_context()
#     # Get the current machine's name for dispatch
#     HOSTNAME = gethostname()

#     context = if Sys.islinux()
#         HOSTNAME_MAP["linux"]
#     else
#         HOSTNAME_MAP[HOSTNAME]
#     end

#     return context
# end

"""
Points to the directory containing the Indoor Scene Recognition dataset depending on the host machine.
"""
function get_isr_dir()
    # Get the current machine's name for dispatch
    HOSTNAME = gethostname()
    # context = get_comp_context()

    # Laptop
    data_dir = if HOSTNAME == "SASHA-XPS"
    # data_dir = if context == "SASHA-XPS"
        joinpath(
            "C:",
            "Users",
            "sap62",
            "Repos",
            "github",
            "DeepART",
            "work",
            "data",
            "indoorCVPR_09",
        )
    # PC
    elseif HOSTNAME == "Sasha-PC"
        joinpath(
            "E:",
            "dev",
            "data",
            "indoorCVPR_09",
        )
    # Cluster
    elseif Sys.islinux()
        joinpath(
            "lustre",
            "scratch",
            "sap625",
            "data",
            "indoorCVPR_09",
        )
    else
        error("Unknown hostname: $HOSTNAME")
    end

    return data_dir
end

"""
Loads the Indoor Scene Recognition dataset from a local directory.
"""
function get_isr(;
    shuffle::Bool=true,
    p::Float=0.8,
    dir::AbstractString=get_isr_dir(),
)
    images_dir = joinpath(dir, "Images")
    labels_dir = joinpath(dir, "Labels")

    # Load the dataset from file
    # local_data = load_dataset_file(
    #     data_dir("indoorcpr_09.csv")
    # )

    # Construct and return a DataSplit
    # return DataSplit(
    #     local_data,
    #     shuffle=shuffle,
    #     p=p,
    # )

    local_image = nothing
    for class in readdir(images_dir)
        # println("$image")
        for image in readdir(joinpath(images_dir, class), join=true)
            # println("$image")
            local_image = channelview(load(image))
            original_size = size(local_image)
            local_image = reshape(local_image, original_size[2], original_size[3], :, 1)
            break
        end
        break
    end

    return images_dir, labels_dir, local_image
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
    data = readdlm(filename, ',', Float32, header=false)

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

"""
A map of dataset names to their loading functions.
"""
const DATA_DISPATCH = Dict(
    "mnist" => get_mnist,
    "fashionmnist" => get_fashionmnist,
    "cifar10" => get_cifar10,
    "cifar100_fine" => get_cifar100_fine,
    "cifar100_coarse" => get_cifar100_coarse,
    "omniglot" => get_omniglot,
    "usps" => get_usps,
    "isr" => get_isr,
    # "CBB-R15" => get_data_package_dataset,
)

"""
A list of the data package names, mainly used as clustering benchmarks.
"""
const DATA_PACKAGE_NAMES = [
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

"""
Loader function for the data package datasets.
"""
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
    # If the name is in the data package names
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
