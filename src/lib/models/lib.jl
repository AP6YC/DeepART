"""
    lib.jl

# Description
Aggregates all models files.
Machine learning models and their functions for training and testing.
"""

# Common model code
include("common.jl")

# Simple WTA network
include("WTANet.jl")

# Vanilla
include("FeatureART.jl")

# EWC
include("EWC.jl")

# Fully deep
include("DeepART.jl")
