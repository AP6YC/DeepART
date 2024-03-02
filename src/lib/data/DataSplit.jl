"""
    DataSplit.jl

# Description
Definitions for a train/test split of supervised datasets.
"""

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
# FUNCTIONS
# -----------------------------------------------------------------------------

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
