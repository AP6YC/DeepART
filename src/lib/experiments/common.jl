"""
    common.jl

# Description
Common functions and types for experiment drivers.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Infinity for integers, used for getting the minimum of training/testing values.
"""
const IInf = typemax(Int)

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Helper function for making sure that the selected number of samples is within the bounds of the current dataset.

# Arguments
- `n::Integer`: the selected number of samples to train/test on.
- `data::SupervisedDataset`: the dataset to check against.
"""
function get_n(n::Integer, data::SupervisedDataset)
    return min(n, length(data.y))
end
