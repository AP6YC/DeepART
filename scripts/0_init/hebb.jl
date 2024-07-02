# using Flux

# model = Chain(
#     Dense(
#         1 => 1
#     )
# )

# weights = Flux.params(model)

# weights[1]

using Revise
using DeepART

head_dim = 10
# model = DeepART.get_rep_fia_dense(n_input, head_dim)
model = Flux.@autosize (n_input,) Chain(
    DeepART.CC(),
    Dense(_, 512, sigmoid_fast, bias=false),
    DeepART.CC(),
    Dense(_, 256, sigmoid_fast, bias=false),
    DeepART.CC(),
    Dense(_, head_dim, sigmoid_fast, bias=false),
)