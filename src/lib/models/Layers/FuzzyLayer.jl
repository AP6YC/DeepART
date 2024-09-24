"""
    FuzzyLayer.jl

# Description
Definition of a FuzzyART-like layer for a Flux.jl model.
"""

# -----------------------------------------------------------------------------
# Single Fuzzy
# -----------------------------------------------------------------------------

"""
A single FuzzyART-like layer for a Flux.jl model, implemented for use in a vector container.
"""
struct SingleFuzzy{M<:AbstractVector} <: CustomLayer
    """
    The weight vector for the layer.
    """
    weight::M
end

"""
Constructor for a [`SingleFuzzy`](@ref) layer taking the inut dimension and an optional weight initialization function.
"""
function SingleFuzzy(
    in::Integer;
    init = Flux.rand32
)
    SingleFuzzy(init(in))
end

# Declares the layer as a Flux.jl layer
Flux.@layer SingleFuzzy

# function linear_normalization(x, W)
#     return norm(min.(x, W), 1) / (1e-3 + norm(W, 1))
# end

"""
Inference definition for a [`SingleFuzzy`](@ref) layer computing the activation and match values.
"""
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

"""
Pretty print definition for a [`SingleFuzzy`](@ref) layer.
"""
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
# struct Fuzzy{M<:AbstractMatrix}
struct Fuzzy{F, M<:AbstractMatrix}
    weight::M
    activation::F
    cc_dim::Integer
end

# function Fuzzy(
#     (in, out)::Pair{<:Integer, <:Integer};
#     init = Flux.glorot_uniform
# )
#     Fuzzy(init(out, in))
# end

function Fuzzy(
    in::Integer, out::Integer,
    # init = Flux.glorot_uniform
    activation=identity;
    init = Flux.rand32
)
    Fuzzy(
        init(out, in),
        # init(in, out),
        activation,
        in,
    )
end

Flux.@layer Fuzzy

# function linear_normalization(x, W)
#     return norm(min.(x, W), 1) / (1e-3 + norm(W, 1))
# end

function (a::Fuzzy)(x::AbstractVecOrMat)
    Flux._size_check(a, x, 1 => size(a.weight, 2))
    # Flux._size_check(a, x, 1 => size(a.weight, 1))
#   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
    # a.cache["xT"] = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
    # xT = x
#   return σ.(a.weight * xT .+ a.bias)
    _weight = a.weight'
    # _weight = a.weight

    # NOTE: removing the repeat here because casting works as intended in xw_norm
    # _x = repeat(xT, 1, size(_weight, 2))
    # _x = xT
    _x = x

    xw_norm = sum(abs.(min.(_x, _weight)), dims=1)

    # w_norm = sum(abs.(_weight), dims=1)
    # M = a.activation(vec(xw_norm ./ (ALPHA .+ w_norm)))
    # return M

    # T = a.activation(vec(xw_norm ./ (size(_weight, 1) ./ 2.0f0)))
    # T = a.activation(vec(xw_norm ./ (size(_weight, 2) ./ 2.0f0)))
    T = a.activation.(vec(xw_norm ./ (a.cc_dim ./ 2.0f0)))
    # T = a.activation.(vec(xw_norm ./ (a.cc_dim)))
    # T = a.activation.(vec(xw_norm ./ (1e-3 .+ w_norm)))
    return T
end

# Tell Flux that not every gosh darn cached parameter is trainable
Flux.trainable(a::Fuzzy) = (; W=a.weight)

# function (a::Fuzzy)(x::AbstractVecOrMat)
#     Flux._size_check(a, x, 1 => size(a.weight, 2))
# #   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
#     xT = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
# #   return σ.(a.weight * xT .+ a.bias)
#     # _weight = complement_code(a.weight')
#     # _x = complement_code(xT)
#     _weight = a.weight'
#     # _x = xT
#     _x = repeat(xT, 1, size(_weight, 2))

#     xw_norm = sum(abs.(min.(_x, _weight)), dims=1)
#     # @info "sizes: $(size(_x)) $(size(_weight))"

#     w_norm = sum(abs.(_weight), dims=1)
#     M = a.activation(vec(xw_norm ./ (ALPHA .+ w_norm)))
#     # T = a.activation(vec(xw_norm ./ size(_weight, 1) ./ 2))
#     # return T
#     return M
#     # return norm(min.(_x, _weight), 1) / (ALPHA + norm(_weight, 1))
# end

function (a::Fuzzy)(x::AbstractArray)
    Flux._size_check(a, x, 1 => size(a.weight, 2))
    # Flux._size_check(a, x, 1 => size(a.weight, 1))
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::Fuzzy)
    print(io, "Fuzzy(", size(l.weight, 2), " => ", size(l.weight, 1))
    # print(io, "Fuzzy(", size(l.weight, 1), " => ", size(l.weight, 2))
    # l.σ == identity || print(io, ", ", l.σ)
    # l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

# function art_learn(x, W)
#     # return eval(art.opts.update)(art, x, get_sample(art.W, index))
#     return BETA * min.(x, W) + W * (1.0 - art.opts.beta)
# end


# -----------------------------------------------------------------------------
# Random Fuzzy Matrix
# -----------------------------------------------------------------------------

# struct Dense{F, M<:AbstractMatrix, B}
# struct Fuzzy{M<:AbstractMatrix}
struct RandFuzzy{F, M<:AbstractMatrix}
    rand_weight::M
    weight::M
    activation::F
end

function RandFuzzy(
    in::Integer, out::Integer,
    activation=identity;
    init = Flux.rand32,
    rand_dim = 16,
)
    RandFuzzy(
        init(rand_dim, in),
        init(out, rand_dim),
        activation,
    )
end

Flux.@layer RandFuzzy

# Tell Flux that not every gosh darn cached parameter is trainable
Flux.trainable(a::RandFuzzy) = (; W=a.weight)

function (a::RandFuzzy)(x::AbstractVecOrMat)
    _weight = a.weight'
    # _x = repeat(a.cache["xT"], 1, size(_weight, 2))
    _x = a.rand_weight * x
    # Flux._size_check(a, _x, 1 => size(a.weight, 2))
    xw_norm = sum(abs.(min.(_x, _weight)), dims=1)
    w_norm = sum(abs.(_weight), dims=1)
    M = a.activation(vec(xw_norm ./ (ALPHA .+ w_norm)))
    # return M

    T = a.activation(vec(xw_norm ./ size(_weight, 1) ./ 2.0f0))
    # return T
    return [M, T]
end

function (a::RandFuzzy)(x::AbstractArray)
    Flux._size_check(a, x, 1 => size(a.weight, 2))
    reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
end

function Base.show(io::IO, l::RandFuzzy)
    print(io, "RandFuzzy(", size(l.weight, 2), " => ", size(l.weight, 1))
    # l.σ == identity || print(io, ", ", l.σ)
    # l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

function train_randfuzzy(a::RandFuzzy, x::AbstractVecOrMat)
    # _weight = a.weight'
    # _x = a.rand_weight * x
    # xw_norm = sum(abs.(min.(_x, _weight)), dims=1)
    # w_norm = sum(abs.(_weight), dims=1)
    # M = a.activation(vec(xw_norm ./ (ALPHA .+ w_norm)))

    _x = a.rand_weight * x
    return M
end


# -----------------------------------------------------------------------------
# CACHED VERSIONS
# -----------------------------------------------------------------------------

# # -----------------------------------------------------------------------------
# # FuzzyMatrix
# # -----------------------------------------------------------------------------

# # struct Dense{F, M<:AbstractMatrix, B}
# # struct Fuzzy{M<:AbstractMatrix}
# struct Fuzzy{F, M<:AbstractMatrix}
#     weight::M
#     activation::F
#     cache::Dict{String, Any}
# end

# # function Fuzzy(
# #     (in, out)::Pair{<:Integer, <:Integer};
# #     init = Flux.glorot_uniform
# # )
# #     Fuzzy(init(out, in))
# # end

# function Fuzzy(
#     in::Integer, out::Integer,
#     # init = Flux.glorot_uniform
#     activation=identity;
#     init = Flux.rand32
# )
#     Fuzzy(
#         init(out, in),
#         activation,
#         Dict{String, Any}(),
#     )
# end

# Flux.@layer Fuzzy

# # function linear_normalization(x, W)
# #     return norm(min.(x, W), 1) / (1e-3 + norm(W, 1))
# # end

# function (a::Fuzzy)(x::AbstractVecOrMat)
#     Flux._size_check(a, x, 1 => size(a.weight, 2))
# #   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
#     # a.cache["xT"] = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
#     # @info typeof(x)
#     a.cache["xT"] = x
# #   return σ.(a.weight * xT .+ a.bias)
#     a.cache["_weight"] = a.weight'
#     a.cache["_x"] = repeat(a.cache["xT"], 1, size(a.cache["_weight"], 2))

#     a.cache["xw_norm"] = sum(abs.(min.(a.cache["_x"], a.cache["_weight"])), dims=1)
#     # a.cache["xw_norm"] = sum(min.(a.cache["_x"], a.cache["_weight"]), dims=1)

#     a.cache["w_norm"] = sum(abs.(a.cache["_weight"]), dims=1)
#     # M = a.activation(vec(a.cache["xw_norm"] ./ (ALPHA .+ a.cache["w_norm"])))
#     # return M

#     # T = a.activation(vec(xw_norm ./ size(_weight, 1) ./ 2))
#     T = a.activation(vec(a.cache["xw_norm"] ./ (size(a.cache["_weight"], 1) ./ 2.0f0)))
#     return T
# end

# # Tell Flux that not every gosh darn cached parameter is trainable
# Flux.trainable(a::Fuzzy) = (; W=a.weight)

# # function (a::Fuzzy)(x::AbstractVecOrMat)
# #     Flux._size_check(a, x, 1 => size(a.weight, 2))
# # #   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
# #     xT = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
# # #   return σ.(a.weight * xT .+ a.bias)
# #     # _weight = complement_code(a.weight')
# #     # _x = complement_code(xT)
# #     _weight = a.weight'
# #     # _x = xT
# #     _x = repeat(xT, 1, size(_weight, 2))

# #     xw_norm = sum(abs.(min.(_x, _weight)), dims=1)
# #     # @info "sizes: $(size(_x)) $(size(_weight))"

# #     w_norm = sum(abs.(_weight), dims=1)
# #     M = a.activation(vec(xw_norm ./ (ALPHA .+ w_norm)))
# #     # T = a.activation(vec(xw_norm ./ size(_weight, 1) ./ 2))
# #     # return T
# #     return M
# #     # return norm(min.(_x, _weight), 1) / (ALPHA + norm(_weight, 1))
# # end

# function (a::Fuzzy)(x::AbstractArray)
#     Flux._size_check(a, x, 1 => size(a.weight, 2))
#     reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
# end

# function Base.show(io::IO, l::Fuzzy)
#     print(io, "Fuzzy(", size(l.weight, 2), " => ", size(l.weight, 1))
#     # l.σ == identity || print(io, ", ", l.σ)
#     # l.bias == false && print(io, "; bias=false")
#     print(io, ")")
# end

# # function art_learn(x, W)
# #     # return eval(art.opts.update)(art, x, get_sample(art.W, index))
# #     return BETA * min.(x, W) + W * (1.0 - art.opts.beta)
# # end


# # -----------------------------------------------------------------------------
# # Random Fuzzy Matrix
# # -----------------------------------------------------------------------------

# # struct Dense{F, M<:AbstractMatrix, B}
# # struct Fuzzy{M<:AbstractMatrix}
# struct RandFuzzy{F, M<:AbstractMatrix}
#     rand_weight::M
#     weight::M
#     activation::F
#     cache::Dict{String, Any}
# end

# function RandFuzzy(
#     in::Integer, out::Integer,
#     # init = Flux.glorot_uniform
#     activation=identity;
#     init = Flux.rand32,
#     rand_dim = 16,
# )
#     RandFuzzy(
#         # init(out, in),
#         init(rand_dim, in),
#         init(out, rand_dim),
#         activation,
#         Dict{String, Any}(),
#     )
# end

# Flux.@layer RandFuzzy

# # Tell Flux that not every gosh darn cached parameter is trainable
# Flux.trainable(a::RandFuzzy) = (; W=a.weight)

# function (a::RandFuzzy)(x::AbstractVecOrMat)
#     # Flux._size_check(a, x, 1 => size(a.weight, 2))
# #   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
#     # a.cache["xT"] = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
#     # @info typeof(x)
#     # a.cache["xT"] = x * a.rand_weight
# #   return σ.(a.weight * xT .+ a.bias)
#     a.cache["_weight"] = a.weight'
#     # a.cache["_x"] = repeat(a.cache["xT"], 1, size(a.cache["_weight"], 2))
#     a.cache["_x"] = a.rand_weight * x

#     # Flux._size_check(a, a.cache["_x"], 1 => size(a.weight, 2))

#     a.cache["xw_norm"] = sum(abs.(min.(a.cache["_x"], a.cache["_weight"])), dims=1)
#     # a.cache["xw_norm"] = sum(min.(a.cache["_x"], a.cache["_weight"]), dims=1)

#     a.cache["w_norm"] = sum(abs.(a.cache["_weight"]), dims=1)
#     M = a.activation(vec(a.cache["xw_norm"] ./ (ALPHA .+ a.cache["w_norm"])))
#     return M

#     # T = a.activation(vec(xw_norm ./ size(_weight, 1) ./ 2))
#     # T = a.activation(vec(a.cache["xw_norm"] ./ (size(a.cache["_weight"], 1) ./ 2.0f0)))
#     return T
# end

# function (a::RandFuzzy)(x::AbstractArray)
#     Flux._size_check(a, x, 1 => size(a.weight, 2))
#     reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
# end

# function Base.show(io::IO, l::Fuzzy)
#     print(io, "Fuzzy(", size(l.weight, 2), " => ", size(l.weight, 1))
#     # l.σ == identity || print(io, ", ", l.σ)
#     # l.bias == false && print(io, "; bias=false")
#     print(io, ")")
# end
