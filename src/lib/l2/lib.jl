"""
    lib.jl

# Description
Aggregates all of the lib_l2 source files.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# Common definitions and variables
include("common.jl")

# Experience definitions
include("experience.jl")

# Scenario and experience definitions
include("scenario.jl")

# # Agent definitions that depend on scenario definitions
# include("agents.jl")
