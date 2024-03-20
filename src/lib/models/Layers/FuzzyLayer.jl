"""
    FuzzyLayer.jl

# Description
Definition of a FuzzyART-like layer for a Flux.jl model.
"""

# -----------------------------------------------------------------------------
# Single Fuzzy
# -----------------------------------------------------------------------------

struct SingleFuzzy{M<:AbstractVector} <: CustomLayer
    weight::M
end

function SingleFuzzy(
    in::Integer;
    init = Flux.rand32
)
    SingleFuzzy(init(in))
end

Flux.@layer SingleFuzzy

# function linear_normalization(x, W)
#     return norm(min.(x, W), 1) / (1e-3 + norm(W, 1))
# end

function (a::SingleFuzzy)(x::AbstractVecOrMat)
    Flux._size_check(a, x, 1 => length(a.weight))
#   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
    xT = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
#   return σ.(a.weight * xT .+ a.bias)
    # _weight = complement_code(a.weight')
    # _x = complement_code(xT)
    _weight = a.weight
    _x = xT
    xw_norm = norm(min.(_x, _weight), 1)
    M = xw_norm / (ALPHA + norm(_weight, 1))
    T = xw_norm / (length(a.weight) / 2)
    return [M, T]
    # return norm(min.(_x, _weight), 1) / (ALPHA + norm(_weight, 1))
end

# function (a::SingleFuzzy)(x::AbstractArray)
#     Flux._size_check(a, x, 1 => size(a.weight, 2))
#     reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
# end

function Base.show(io::IO, l::SingleFuzzy)
    print(io, "SingleFuzzy(", length(l.weight))
    # l.σ == identity || print(io, ", ", l.σ)
    # l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

# -----------------------------------------------------------------------------
# FuzzyMatrix
# -----------------------------------------------------------------------------

# struct Dense{F, M<:AbstractMatrix, B}
struct Fuzzy{M<:AbstractMatrix}
    weight::M
end

# function Fuzzy(
#     (in, out)::Pair{<:Integer, <:Integer};
#     init = Flux.glorot_uniform
# )
#     Fuzzy(init(out, in))
# end

function Fuzzy(
    in::Integer, out::Integer;
    # init = Flux.glorot_uniform
    init = Flux.rand32
)
    Fuzzy(init(out, in))
end

Flux.@layer Fuzzy

# function linear_normalization(x, W)
#     return norm(min.(x, W), 1) / (1e-3 + norm(W, 1))
# end

const ALPHA = 1e-3

function (a::Fuzzy)(x::AbstractVecOrMat)
    Flux._size_check(a, x, 1 => size(a.weight, 2))
#   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
    xT = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
#   return σ.(a.weight * xT .+ a.bias)
    # _weight = complement_code(a.weight')
    # _x = complement_code(xT)
    _weight = a.weight'
    _x = xT
    return norm(min.(_x, _weight), 1) / (ALPHA + norm(_weight, 1))
end

function (a::Fuzzy)(x::AbstractArray)
    Flux._size_check(a, x, 1 => size(a.weight, 2))
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::Fuzzy)
    print(io, "Fuzzy(", size(l.weight, 2), " => ", size(l.weight, 1))
    # l.σ == identity || print(io, ", ", l.σ)
    # l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

# const BETA = 1.0

# function art_learn(x, W)
#     # return eval(art.opts.update)(art, x, get_sample(art.W, index))
#     return BETA * min.(x, W) + W * (1.0 - art.opts.beta)
# end
