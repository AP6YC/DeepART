"""
    common.jl

# Description
Common dataset types and routines.
"""

# -----------------------------------------------------------------------------
# ABTRACT TYPES
# -----------------------------------------------------------------------------

# abstract type  end

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

"""
Abstract type alias for features.
"""
const AbstractFeatures = RealArray

"""
Abstract type alias for labels.
"""
const AbstractLabels = IntegerArray

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Wrapper for shuffling features and their labels.

# Arguments
- `features::AbstractArray`: the set of data features.
- `labels::AbstractArray`: the set of labels corresponding to the features.
"""
function shuffle_pairs(
    features::AbstractArray,
    labels::AbstractArray,
)
    # Use the MLUtils function for shuffling
    ls, ll = shuffleobs((features, labels))

    # Return the pairs
    return ls, ll
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
