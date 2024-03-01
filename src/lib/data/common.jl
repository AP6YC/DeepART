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
