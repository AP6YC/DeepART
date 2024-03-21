"""
    HypersphereLayer.jl

# Description
Definition of a FuzzyART-like layer for a Flux.jl model.
"""

# -----------------------------------------------------------------------------
# Single Fuzzy
# -----------------------------------------------------------------------------

# const R_bar = 0.5 * length(_weight) # 1/2 * max

struct HypersphereLayer{M<:AbstractVector} <: CustomLayer
    weight::M
    radius::Float32
    R_bar::Float32
end

function HypersphereLayer(
    weight,
)
    R_bar = Float32(0.5 * norm(length(weight))) # 1/2 * max_{p,q} ||w_p - w_q||_2
    HypersphereLayer(
        weight,
        norm(weight, 2),
        R_bar,
    )
end

function HypersphereLayer(
    in::Integer;
    init = Flux.rand32
)
    _weight = init(in)
    HypersphereLayer(
        _weight,
    )
end

Flux.@layer HypersphereLayer

function (a::HypersphereLayer)(x::AbstractVecOrMat)
    Flux._size_check(a, x, 1 => length(a.weight))
#   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
    xT = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
#   return σ.(a.weight * xT .+ a.bias)
    # _weight = complement_code(a.weight')
    # _x = complement_code(xT)
    _x = xT

    xw_norm_max = max(a.radius, norm(min.(_x, a.weight), 2))

    M = 1 - xw_norm_max / a.R_bar
    T = (a.R_bar - xw_norm_max) / (a.R_bar - a.radius + ALPHA)

    return [M, T]
    # return norm(min.(_x, _weight), 1) / (ALPHA + norm(_weight, 1))
end

# function (a::HypersphereLayer)(x::AbstractArray)
#     Flux._size_check(a, x, 1 => size(a.weight, 2))
#     reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
# end

function Base.show(io::IO, l::HypersphereLayer)
    print(io, "HypersphereLayer(", length(l.weight))
    # l.σ == identity || print(io, ", ", l.σ)
    # l.bias == false && print(io, "; bias=false")
    print(io, ")")
end
