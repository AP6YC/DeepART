"""
    lib.jl

# Description
This file aggregates the library code from other files for the `DeepART` project.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Color definitions
include("colors.jl")

# Constant values
include("constants.jl")

# Docstring variables and templates
include("docstrings.jl")

# DrWatson added functionality
include("drwatson.jl")

# # Common types and functions for all experiment code
# include("common.jl")

# Datasets and their utilities
include("data/lib.jl")

# Machine learning models and functions
include("models/lib.jl")

# Utilities
include("utils.jl")

# Plotting functions
include("plots.jl")

# Pluto utils
# include("pluto.jl")

# Experiment driver files that aggregate these tools
include("experiments/lib.jl")

# L2 definitions
include("l2/lib.jl")
