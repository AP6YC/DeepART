"""
    lib.jl

# Description
Aggregates all models files.
Machine learning models and their functions for training and testing.
"""

# Common model code
include("common.jl")

# Custom 'optimisers'
include("Optimisers/lib.jl")

# Simple WTA network
include("WTANet/lib.jl")

# SimpleDeepART
include("SimpleDeepART/lib.jl")

# Fully deep
include("DeepART/lib.jl")

# Custom Flux layers
include("Layers/lib.jl")
