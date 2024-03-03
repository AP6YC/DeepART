"""
    ClassIncrementalDataSplit.jl

# Description
A class-incremental variant of a [`DataSplit`](@ref) containing vectors of [`SupervisedDataset`](@ref)s.
"""

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

"""
Type alias for a a class-incremental dataset as a vector of [`SupervisedDataset`](@ref)s.
"""
const ClassIncrementalSupervisedDataset = Vector{SupervisedDataset}

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
A class-incremental variant of a [`DataSplit`](@ref) containing instead vectors of [`SupervisedDataset`](@ref)s.
"""
struct ClassIncrementalDataSplit
    """
    The vector of training class datasets.
    """
    train::ClassIncrementalSupervisedDataset

    """
    The vector of testing class datasets.
    """
    test::ClassIncrementalSupervisedDataset
end

# """
# A task-incremental variant of a [`DataSplit`](@ref) containing multiple classes per task.
# """
# struct TaskIncrementalDataSplit
#     """
#     The vector of training class datasets.
#     """
#     train::Vector{SupervisedDataset}

#     """
#     The vector of testing class datasets.
#     """
#     test::Vector{SupervisedDataset}
# end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Turns a normal [`SupervisedDataset`](@ref) into a class-incremental vector of [`SupervisedDataset`](@ref)s.

# Arguments
$ARG_SUPERVISEDDATASET
"""
function class_incrementalize(data::SupervisedDataset)
    # Initialize the new class incremental vector
    new_data = ClassIncrementalSupervisedDataset()

    # If the labels are one-hot encoded, one-cold them back
    local_y = if ndims(data.y) == 2
        Flux.onecold(data.y)
    else
        data.y
    end

    # Get the number of classes and dimensions
    n_classes = length(unique(local_y))
    n_dim = ndims(data.x)

    # Iterate over all integer class labels
    for ix = 1:n_classes
        # Get all of the indices corresponding to the integer class label

        class_indices = findall(x->x==ix, local_y)

        # Fragile, but it works for now
        local_features = if n_dim == 4
            data.x[:, :, :, class_indices]
        elseif n_dim == 3
            data.x[:, :, class_indices]
        elseif n_dim == 2
            data.x[:, class_indices]
        end

        local_labels = if ndims(data.y) == 2
            data.y[:, class_indices]
        else
            data.y[class_indices]
        end

        # Create a new dataset from just these features and labels
        local_dataset = SupervisedDataset(
            local_features,
            local_labels,
        )
        # Add the local dataset to the vector of datasets to return
        push!(new_data, local_dataset)
    end

    return new_data
end

"""
Constructor for a [`ClassIncrementalDataSplit`](@ref) taking a normal [`DataSplit`](@ref).

# Arguments
$ARG_DATASPLIT
"""
function ClassIncrementalDataSplit(datasplit::DataSplit)
    return ClassIncrementalDataSplit(
        class_incrementalize(datasplit.train),
        class_incrementalize(datasplit.test),
    )
end

"""
Returns a [`SupervisedDataset`](@ref)s that combines the datasets in a [`ClassIncrementalSupervisedDataset`](@ref) at the indices given by `group`.

# Arguments
- `data::ClassIncrementalSupervisedDataset`: the vector of [`SupervisedDataset`](@ref)s to select and combine from.
- `group::Vector{Int}`: the indices to select from for combining.
- `shuffle::Bool`: flag for pairwise shuffling the dataset after it has been combined, default true.
"""
function group_datasets(
    data::ClassIncrementalSupervisedDataset,
    group::Vector{Int},
    shuffle::Bool=true,
)
    # Cat the features
    local_features = hcat([data[ix].x for ix in group]...)

    # If we have one-hot encoded labels, we need to stack them differently
    if ndims(data[1].y) == 2
        local_labels = hcat([data[ix].y for ix in group]...)
    else
        local_labels = vcat([data[ix].y for ix in group]...)
    end

    # Shuffle if necessary
    if shuffle
        local_features, local_labels = shuffle_pairs(local_features, local_labels)
        local_features = copy(local_features)
        local_labels = copy(local_labels)
    end

    # Construct and return the new supervised dataset
    return SupervisedDataset(local_features, local_labels)
end

"""
Groups multiple datasets within a [`ClassIncrementalSupervisedDataset`](@ref) according to a vector of groupings.

# Arguments
- `data:ClassIncrementalSupervisedDataset`: the vector of datasets to group.
- `groupings::Vector{Vector{Int}}`: the set of groupings to perform.
$ARG_SHUFFLE
"""
function task_incrementalize(
    data::ClassIncrementalSupervisedDataset,
    groupings::Vector{Vector{Int}},
    shuffle::Bool=true,
)
    new_data = ClassIncrementalSupervisedDataset()

    for group in groupings
        # push!(new_data, SupervisedDataset(local_features, local_labels))
        push!(new_data, group_datasets(data, group, shuffle))
    end

    return new_data
end

"""
Combines classes in the training and testing datasets of a [`ClassIncrementalDataSplit`](@ref) according to the provided `groupings`.

# Arguments
- `datasplit::ClassIncrementalDataSplit`: a [`ClassIncrementalDataSplit`](@ref) to combine elements of according to the groupings
- `groupings::Vector{Vector{Int}}`: the set of groupings to perform.
$ARG_SHUFFLE
"""
function TaskIncrementalDataSplit(
    datasplit::ClassIncrementalDataSplit,
    groupings::Vector{Vector{Int}};
    shuffle::Bool=true,
)
    return ClassIncrementalDataSplit(
        task_incrementalize(datasplit.train, groupings, shuffle),
        task_incrementalize(datasplit.test, groupings, shuffle),
    )
end

# -----------------------------------------------------------------------------
# OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`ClassIncrementalDataSplit`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `ds::ClassIncrementalDataSplit`: the [`ClassIncrementalDataSplit`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    ds::ClassIncrementalDataSplit,
)
    # Compute all of the dimensions of the dataset
    s_train = size(ds.train[1].x)

    # Get the number of features, training samples, and testing samples
    n_dims = length(s_train)
    n_features = s_train[1:n_dims - 1]
    n_classes = length(ds.train)

    # print(io, "DataSplit(features: $(size(ds.train.x)), test: $(size(ds.test.x)))")
    print(io, "ClassIncrementalDataSplit(features: $(n_features), n_classes: $(n_classes))")
end
