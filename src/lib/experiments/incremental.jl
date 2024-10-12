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
    x::RealArray,
    y::Integer,
)
    # return ART.train!(art, x, y=y)
    y_hat = ART.train!(art, x, y=y)
    # bmu = art.labels[art.stats["bmu"]]
    # bmu = isempty(art.labels) ? 0 : art.labels[art.stats["bmu"]]
    # return (bmu == 0) ? y_hat : bmu
    bmu = art.stats["bmu"]
    return iszero(bmu) ? y_hat : art.labels[bmu]
    # return isempty(art.labels) ? y_hat : art.labels[art.stats["bmu"]]
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
    x::RealArray,
    y::Integer,
)
    # return ART.train!(art, x, y)

    y_hat = ART.train!(art, x, y)
    # bmu = isempty(art.labels) ? 0 : art.labels[art.stats["bmu"]]
    # return (bmu == 0) ? y_hat : bmu
    bmu = art.stats["bmu"]
    return iszero(bmu) ? y_hat : art.labels[bmu]
    # return isempty(art.labels) ? y_hat : art.labels[art.stats["bmu"]]
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
    x::RealArray,
    y::Integer,
)
    # return DeepART.train!(art, x, y=y)

    y_hat = DeepART.train!(art, x, y=y)
    # return art.labels[art.stats["bmu"]]
    # bmu = art.labels[art.stats["bmu"]]
    # bmu = isempty(art.labels) ? 0 : art.labels[art.stats["bmu"]]
    bmu = art.head.stats["bmu"]
    return iszero(bmu) ? y_hat : art.head.labels[bmu]
    # return isempty(art.labels) ? y_hat : art.labels[art.stats["bmu"]]
end

function incremental_supervised_train!(
    art::FIA,
    x::RealArray,
    y::Integer,
)
    y_hat = DeepART.train!(art, x, y=y)
    # return iszero(bmu) ? y_hat : art.head.labels[bmu]
    return y_hat
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
    x::RealArray,
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
    x::RealArray,
)
    return DeepART.classify(art, x, get_bmu=true)
end

function incremental_supervised_train!(
    art::Hebb.HebbModel,
    x::RealArray,
    y::Integer,
)
    # @info art x y
    # @info art.model.chain[1]

    # @info argmax(vec(art.model.chain(x)))
    # y_hat = DeepART.train_hebb(art, x, y)
    y_hat = Hebb.train_hebb(art, vec(x), y)
    # y_hat = Hebb.train_hebb(art, vec(x), y)
    # return iszero(bmu) ? y_hat : art.head.labels[bmu]
    return y
end

# Hebb

function incremental_supervised_train!(
    art::Hebb.BlockNet,
    x::RealArray,
    y::Integer,
)
    # y_hat = DeepART.train_hebb(art, x, y)
    y_hat = Hebb.train!(art, x, y)
    # return iszero(bmu) ? y_hat : art.head.labels[bmu]
    return y_hat
end


function incremental_classify(
    art::Hebb.HebbModel,
    x::RealArray,
)
    # return DeepART.classify(art, x, get_bmu=true)
    return argmax(vec(art.model.chain(x)))
end

function incremental_classify(
    art::Hebb.BlockNet,
    x::RealArray,
)
    # return DeepART.classify(art, x, get_bmu=true)
    return Hebb.forward(art, x)
end
