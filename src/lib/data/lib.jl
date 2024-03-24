"""
    lib.jl

# Description
Aggregates the data set files.
"""

# Common dataset code
include("common.jl")

# x/y dataset container
include("SupervisedDataset.jl")

# Train/test dataset container
include("DataSplit.jl")

# Task incremental train/test dataset container
include("ClassIncrementalDataSplit.jl")

# Dataset loaders
include("data.jl")

# Common functions for post-processing
include("post-common.jl")
