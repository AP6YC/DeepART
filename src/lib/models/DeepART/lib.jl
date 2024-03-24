"""
Aggregates all of the deep models for the project.
"""

# Common deep ART types and functions
include("common.jl")

# Definition of a multi-head field
include("MultiHeadField.jl")

# Definition of a DeeperART module
include("DeeperART.jl")

# Definition of a DeepHeadART module
include("DeepHeadART.jl")

# Deep instar learning
include("INSTART.jl")

# Deep instar learning with existing ART module as head
include("ARTINSTART.jl")
