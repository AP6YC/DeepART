"""
    CCLayer.jl

# Description
Definition of a complement coding layer for a Flux.jl model.
"""

function CC()
    complement_code
    # Parallel(vcat,
    #     identity,
    #     complement_code,
    # )
end

# # struct Dense{F, M<:AbstractMatrix, B}
# struct CC{M<:AbstractMatrix}
#     weight::M
# end

# # function Fuzzy(
# #     (in, out)::Pair{<:Integer, <:Integer};
# #     init = Flux.glorot_uniform
# # )
# #     Fuzzy(init(out, in))
# # end

# function CC(
#     in::Integer, out::Integer;
#     init = Flux.glorot_uniform
# )
#     CC(init(out, in))
# end

# Flux.@layer CC

# function (a::CC)(x::AbstractVecOrMat)
#     Flux._size_check(a, x, 1 => size(a.weight, 2))
# #   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
#     xT = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
# #   return σ.(a.weight * xT .+ a.bias)
#     _weight = complement_code(a.weight')
#     _x = complement_code(xT)
#     return norm(min.(_x, _weight), 1) / (ALPHA + norm(_weight, 1))
# end