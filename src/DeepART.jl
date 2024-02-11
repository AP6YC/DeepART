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
    ColorSchemes,
    Colors,
    DataFrames,
    DocStringExtensions,
    Flux,
    HypertextLiteral,
    ImageShow,
    Markdown,
    MLDatasets,
    MLUtils,
    MLBase,                 # confusmat
    NumericalTypeAliases,
    Plots,
    Pluto,
    PlutoUI

# Precompile concrete type methods
import PrecompileSignatures: @precompile_signatures

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

# Authorize downloads to prevent interactive download blocking
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

include("lib/lib.jl")

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------

export
    # Configs
    NumConfig,
    OptConfig,

    # Pluto utils
    correct,
    almost,
    hint,
    keep_working

# -----------------------------------------------------------------------------
# PRECOMPILE
# -----------------------------------------------------------------------------

# Precompile any concrete-type function signatures
@precompile_signatures(DeepART)

end
