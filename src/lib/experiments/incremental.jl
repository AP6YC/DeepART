"""
    incrementa.jl

# Description
Incremental training/classification functions, defining how a single sample is used by each type of module.
"""

# -----------------------------------------------------------------------------
# INCREMENTAL TRAINING
# -----------------------------------------------------------------------------

"""
Dispatch overload for incremental supervised training for an ART.ART module.

# Arguments
- `art::ART.ART`: the supervised ART.ART module.
$X_ARG_DOCSTRING
$ARG_Y
"""
function incremental_supervised_train!(
    art::ART.ART,
    x::RealVector,
    y::Integer,
)
    return ART.train!(art, x, y=y)
end

"""
Dispatch overload for incremental supervised training for an ART.ARTMAP module.

# Arguments
- `art::ART.ARTMAP`: the supervised ART.ARTMAP module.
$X_ARG_DOCSTRING
$ARG_Y
"""
function incremental_supervised_train!(
    art::ART.ARTMAP,
    x::RealVector,
    y::Integer,
)
    return ART.train!(art, x, y)
end

"""
Overload for incremental supervised training for a [`DeepARTModule`](@ref) model.

# Arguments
$ARG_DEEPARTMODULE
$X_ARG_DOCSTRING
$ARG_Y
"""
function incremental_supervised_train!(
    art::DeepARTModule,
    x::RealVector,
    y::Integer,
)
    return DeepART.train!(art, x, y=y)
end

# -----------------------------------------------------------------------------
# INCREMENTAL CLASSIFICATION
# -----------------------------------------------------------------------------

"""
Dispatch overload for the incremental classification with an ART.ARTModule.

# Arguments
- `art::ART.ARTModule`: the ART.ARTModule to use for classification.
$X_ARG_DOCSTRING
"""
function incremental_classify(
    art::ART.ARTModule,
    x::RealVector,
)
    return ART.classify(art, x, get_bmu=true)
end

"""
Dispatch overload for the incremental classification with a [`DeepARTModule`](@ref).

# Arguments
$ARG_DEEPARTMODULE
$X_ARG_DOCSTRING
"""
function incremental_classify(
    art::DeepARTModule,
    x::RealVector,
)
    return DeepART.classify(art, x, get_bmu=true)
end
