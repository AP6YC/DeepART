"""
Aggregates custom Flux layer definitions.
"""

# Common layer definitions
include("common.jl")

# Fuzzy intersection layer
include("FuzzyLayer.jl")

# Hypersphere layer
include("HypersphereLayer.jl")

# Complement code layer
include("CCLayer.jl")
