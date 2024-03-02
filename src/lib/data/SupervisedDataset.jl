"""
    SupervisedDataset.jl

# Description
Type and function definitions for the `SupervisedDataset` struct.
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

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`SupervisedDataset`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::SupervisedDataset`: the [`SupervisedDataset`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    ds::SupervisedDataset,
)
    # Compute all of the dimensions of the dataset
    s_data = size(ds.x)

    # Get the number of features, training samples, and testing samples
    n_dims = ndims(ds.x)
    n_features = s_data[1:n_dims - 1]
    n_samples = s_data[end]

    # Show the dataset dimensions
    print(io, "SupervisedDataset(features: $(n_features), samples: $(n_samples))")
end
