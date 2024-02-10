"""
    DeepART.jl

# Description
Definition of the `DeepART` module, which encapsulates experiment driver code.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

"""
A module encapsulating the experimental driver code for the DeepART project.

# Imports

The following names are imported by the package as dependencies:
$(IMPORTS)

# Exports

The following names are exported and available when `using` the package:
$(EXPORTS)
"""
module DeepART

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

# Full usings (which supports comma-separated import notation)
using
    AdaptiveResonance,
    DocStringExtensions,
    DrWatson,
    Flux

end
