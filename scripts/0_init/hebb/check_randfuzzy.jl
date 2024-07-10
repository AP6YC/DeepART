using DeepART
using Flux

model = Flux.@autosize (2,) Chain(
    DeepART.CC(),
    DeepART.RandFuzzy(_, 5),
)

x = rand(2)

M, T = model(x)


new_model = Parallel(vcat, model, model)


x = DeepART.CC()(x)
a  = model[2]
_weight = a.weight'
_x = a.rand_weight * x
xw_norm = sum(abs.(min.(_x, _weight)), dims=1)
w_norm = sum(abs.(_weight), dims=1)
M = a.activation(vec(xw_norm ./ (ALPHA .+ w_norm)))
# return M

T = a.activation(vec(xw_norm ./ size(_weight, 1) ./ 2.0f0))



sort!(T, rev=true)
for (m, T) in zip(M, T)
    @info "m: $m, t: $T"
end

Flux.softmax(T)
