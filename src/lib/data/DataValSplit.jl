"""
    DataValSplit.jl

# Description
Definitions for a train/test split of supervised datasets.
"""

"""
A train/test split of supervised datasets.
"""
struct DataValSplit
    """
    The training portion of the dataset.
    """
    train::SupervisedDataset

    """
    The test portion of the dataset.
    """
    test::SupervisedDataset

    """
    The validation portion of the dataset.
    """
    val::SupervisedDataset
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Convenience constructor for a supervised [`DataValSplit`](@ref) that takes each set of features `x` and labels `y`separately.

# Arguments
- `X_train::AbstractFeatures`: the training features.
- `y_train::AbstractLabels`: the training integer labels.
- `X_test::AbstractFeatures`: the testing features.
- `y_test::AbstractLabels`: the testing integer labels.
- `X_val::AbstractFeatures`: the validation features.
- `y_val::AbstractLabels`: the validation integer labels.
$ARG_SHUFFLE
"""
function DataValSplit(
    X_train::AbstractFeatures,
    y_train::AbstractLabels,
    X_test::AbstractFeatures,
    y_test::AbstractLabels;
    X_val::AbstractFeatures,
    y_val::AbstractLabels,
    shuffle::Bool=DEFAULT_SHUFFLE,
)
    return DataValSplit(
        SupervisedDataset(
            X_train,
            y_train,
            shuffle,
        ),
        SupervisedDataset(
            X_test,
            y_test,
            shuffle,
        ),
        SupervisedDataset(
            X_val,
            y_val,
            shuffle,
        ),
    )
end

"""
Constructor for a [`DataValSplit`](@ref) taking a set of features and options for the split ratio and shuffle flag.

# Arguments
- `features::AbstractFeatures`: the input features as an array of samples.
- `labels::AbstractLabels`: the supervised labels.
$ARG_P
$ARG_SHUFFLE
"""
function DataValSplit(
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
    splits = (p, 0.1)
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = MLUtils.splitobs(
        (ls, ll);
        at=splits
    )

    # Create and return a single container for this train/test split
    return DataValSplit(
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
Tensorizes both the training and testing components of a [`DataValSplit`](@ref).

# Arguments
$ARG_DataValSplit
"""
function tensorize_DataValSplit(data::DataValSplit)
    new_dataset = DataValSplit(
        tensorize_dataset(data.train),
        tensorize_dataset(data.test),
    )
    return new_dataset
end

"""
Flattens a [`DataValSplit`](@ref).

# Arguments
$ARG_DataValSplit
$ARG_N_CLASS
"""
function flatty(data::DataValSplit)
    new_train = flatty(data.train)
    new_test = flatty(data.test)

    # Construct and return the new DataValSplit
    return DataValSplit(
        new_train,
        new_test,
    )
end

"""
Flattens and one-hot encodes a [`DataValSplit`](@ref).

# Arguments
$ARG_DataValSplit
$ARG_N_CLASS
"""
function flatty_hotty(data::DataValSplit, n_class::Int=0)
    new_train = flatty_hotty(data.train, n_class)
    new_test = flatty_hotty(data.test, n_class)

    # Construct and return the new DataValSplit
    return DataValSplit(
        new_train,
        new_test,
    )
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`DataValSplit`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::DataValSplit`: the [`DataValSplit`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    ds::DataValSplit,
)
    # Compute all of the dimensions of the dataset
    s_train = size(ds.train.x)
    s_test = size(ds.test.x)

    # Get the number of features, training samples, and testing samples
    n_dims = length(s_train)
    n_features = s_train[1:n_dims - 1]
    n_train = s_train[end]
    n_test = s_test[end]

    # print(io, "DataValSplit(features: $(size(ds.train.x)), test: $(size(ds.test.x)))")
    print(io, "DataValSplit(features: $(n_features), train: $(n_train), test: $(n_test))")
end
