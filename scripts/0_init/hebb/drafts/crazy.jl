
module Crazy
    using Flux
    struct SingleFuzzy{M<:AbstractMatrix, V<:AbstractVector, F}
        rand_weight::M
        weight::V
        rand_activation::F
        activation::F
    end

    function SingleFuzzy(
        in::Integer,
        # out::Integer;
        activation = identity,
        rand_activation = identity,
        init = Flux.rand32,
        rand_dim = 16,
    )
        SingleFuzzy(
            init(rand_dim, in),
            init(rand_dim),
            rand_activation,
            activation,
        )
    end

    Flux.@layer SingleFuzzy
    const ALPHA = Float32(1e-3)
    function (a::SingleFuzzy)(x::AbstractVecOrMat)
        # Flux._size_check(a, x, 1 => length(a.weight))
    #   σ = NNlib.fast_act(a.σ, x)  # replaces tanh => tanh_fast, etc
        xT = Flux._match_eltype(a, x)  # fixes Float64 input, etc.
    #   return σ.(a.weight * xT .+ a.bias)
        # _weight = complement_code(a.weight')
        # _x = complement_code(xT)
        _weight = a.weight
        # _x = xT
        _x = a.rand_weight * xT
        # xw_norm = norm(min.(_x, _weight), 1)
        # M = xw_norm / (ALPHA + norm(_weight, 1))
        # T = xw_norm / (length(a.weight) / 2)

        xw_norm = sum(abs.(min.(_x, _weight)), dims=1)
        w_norm = sum(abs.(_weight), dims=1)
        M = a.activation(vec(xw_norm ./ (ALPHA .+ w_norm)))
        return M
        # T = a.activation(vec(xw_norm ./ size(_weight, 1) ./ 2.0f0))
        # return [M, T]
        # return norm(min.(_x, _weight), 1) / (ALPHA + norm(_weight, 1))
    end

    function Base.show(io::IO, l::SingleFuzzy)
        print(io, "SingleFuzzy(", length(l.weight))
        # l.σ == identity || print(io, ", ", l.σ)
        # l.bias == false && print(io, "; bias=false")
        print(io, ")")
    end

end

import .Crazy
using Flux
a = Crazy.SingleFuzzy(5)

a(rand(5))

b = [a, a]

new_model = Parallel(vcat, b)
push!(b, a)

@info new_model

new_model(rand(5))