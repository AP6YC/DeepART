"""
Aggregates the experiment files.
"""

# Train-test experiments
include("train-test.jl")

# Experiment drivers for baselines
include("baselines.jl")
