"""
Aggregates the experiment files.
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# # Common types and functions for all experiments
# include("common.jl")

# Incremental train/test functions and overloads
include("incremental.jl")

# Train-test experiments
include("train-test.jl")

# Experiment drivers for baselines
include("baselines.jl")

# Distributed experiment drivers
include("dist.jl")
