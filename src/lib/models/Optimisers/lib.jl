"""
Aggregates all of the custom 'optimisers' for the library.

Yes, I know the American spelling is 'optimizer' and British is 'optimiser,' but that's how Optimisers.jl is spelled, so I am following that convention...begrudgingly.
"""

# EWC optimiser
include("EWC.jl")

# Incremental EWC
include("IEWC.jl")

# EWC as a loss
include("EWCLoss.jl")
