"""
    lib.jl

# Description
Aggregates all models files.
Machine learning models and their functions for training and testing.
"""

# Common model code
include("common.jl")

include("0_wta_deep.jl")

# Vanilla
include("1_vanilla.jl")

# EWC
include("2_ewc.jl")

# Fully deep
include("3_deep.jl")
