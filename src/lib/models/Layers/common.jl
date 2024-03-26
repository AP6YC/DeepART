"""
Common code for custom Flux.jl layer definitions.
"""

# -----------------------------------------------------------------------------
# ABSTRACT TYPES
# -----------------------------------------------------------------------------

"""
Abstract type for custom Flux.jl layers.
"""
abstract type CustomLayer end

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

const ALPHA = 1e-3

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Returns the complement code of the input
"""
function complement_code(
    x
    # x::AbstractVecOrMat{Float32}
)
    # vcat(x, 1.0 .- x)
    # vcat(x, one(Float32) .- x)
    vcat(x, one(eltype(x)) .- x)
end