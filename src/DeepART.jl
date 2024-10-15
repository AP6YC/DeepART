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
    Combinatorics,
    CUDA,
    DataFrames,
    DataStructures,
    DrWatson,
    DelimitedFiles,
    DocStringExtensions,
    Flux,
    # HypertextLiteral,
    ImageShow,
    Images,
    JLD2,
    JSON,
    # Markdown,
    MLDatasets,
    MLUtils,
    MLBase,                 # confusmat
    NumericalTypeAliases,
    Parameters,
    Plots,
    # Pluto,
    # PlutoUI,
    Printf,
    ProgressMeter,
    # PythonCall,
    Random,
    # SHA,
    UnicodePlots

using LinearAlgebra: norm   # Trace and norms

# Precompile concrete type methods
import PrecompileSignatures: @precompile_signatures

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------

"""
Internal alias for the AdaptiveResonance.jl package.
"""
const ART = AdaptiveResonance

"""
Flag for using Flux.onehotbatch or an internal implementation.
"""
const FLUXONEHOT = true

# Authorize downloads to prevent interactive download blocking
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Fix plotting on headless
ENV["GKSwstype"] = "100"

# -----------------------------------------------------------------------------
# INCLUDES
# -----------------------------------------------------------------------------

# The full library
include("lib/lib.jl")

# Exported constant for the version of the package
include("version.jl")

# # New work import
# include(projectdir("scripts/0_init/hebb/lib/lib.jl"))

# import .Hebb

# -----------------------------------------------------------------------------
# EXPORTS
# -----------------------------------------------------------------------------

export
    # Constants
    DEEPART_VERSION

    # Configs
    # NumConfig,
    # OptConfig,

    # Pluto utils
    # correct,
    # almost,
    # hint,
    # keep_working

# -----------------------------------------------------------------------------
# PRECOMPILE
# -----------------------------------------------------------------------------

# Precompile any concrete-type function signatures
# @precompile_signatures(DeepART)

end
