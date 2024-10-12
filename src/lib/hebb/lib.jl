"""
    lib.jl

# Description
Definitions for the Hebbian learning module.
"""

module Hebb

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
# using DeepART
using ..DeepART

using AdaptiveResonance
using Flux
using ProgressMeter
using Random
using CUDA
using StatsBase: mean
using NumericalTypeAliases
using YAML

using UnicodePlots

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

include("opts.jl")

include("data.jl")

include("chains/lib.jl")

include("learn.jl")

include("hebbmodel.jl")

include("loop.jl")

include("blocknet.jl")

# include("overloads.jl")

end
