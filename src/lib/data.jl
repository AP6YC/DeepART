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

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
A struct containing a supervised set of features in a matrix `x` mapping to integer labels `y`.
"""
struct SupervisedDataset
    """
    A set of features.
    """
    x::RealArray

    """
    The labels corresponding to each feature.
    """
    y::IntegerVector
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

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Convenience constructor for a supervised [`DataSplit`](@ref) that takes each set of features `x` and labels `y`separately.

# Arguments
- `X_train::RealArray`: the training features.
- `y_train::IntegerVector`: the training integer labels.
- `X_test::RealArray`: the testing features.
- `y_test::IntegerVecto`: the testing integer labels.
"""
function DataSplit(
    X_train::RealArray,
    y_train::IntegerVector,
    X_test::RealArray,
    y_test::IntegerVector
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

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Loads the MNIST dataset using MLDatasets.
"""
function get_mnist()
    trainset = MLDatasets.MNIST(:train)
    testset = MLDatasets.MNIST(:test)

    # d = MNIST

    X_train, y_train = trainset[:]
    X_test, y_test = testset[:]

    dataset = DataSplit(X_train, y_train.+1, X_test, y_test.+1)

    return dataset
end

function tensorize_dataset(data::SupervisedDataset)
    dims = size(data.x)
    new_dataset = SupervisedDataset(
        reshape(data.x, dims[1:end-1]..., 1, :),
        data.y,
    )
    return new_dataset
end

function tensorize_datasplit(data::DataSplit)
    # for
    new_dataset = DataSplit(
        tensorize_dataset(data.train),
        tensorize_dataset(data.test),
    )
    return new_dataset
end
