using DeepART
using Flux

model = Flux.@autosize (2,) Chain(
    DeepART.RandFuzzy(_, 5),
)

x = rand(2)

model(x)