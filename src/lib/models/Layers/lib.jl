"""
Aggregates custom Flux layer definitions.
"""

# Common layer definitions
include("common.jl")

# Fuzzy intersection layer
include("FuzzyLayer.jl")

# Complement code layer
include("CCLayer.jl")
