"""
    representatives.jl

# Description
Representative models for the deep component of DeepART modules.
"""

function get_rep_dense(n_input::Integer, head_dim::Integer)
    model = Flux.@autosize (n_input,) Chain(
        DeepART.CC(),
        Dense(_, 512, sigmoid_fast, bias=false),
        DeepART.CC(),
        # Dense(_, 256, sigmoid_fast, bias=false),
        # DeepART.CC(),
        # Dense(_, 128, sigmoid, bias=false),
        # Dense(_, 64, sigmoid, bias=false),
        # DeepART.CC(),
        Dense(_, head_dim, sigmoid_fast, bias=false),
    )
    return model
end

function get_rep_conv(size_tuple, head_dim::Integer)
    # model = Flux.@autosize size_tuple Chain(
    #     DeepART.CCConv(),
    #     Conv((3, 3), 1=>32, relu),
    #     MaxPool((2, 2)),
    #     Conv((3, 3), 32=>64, relu),
    #     MaxPool((2, 2)),
    #     Conv((3, 3), 64=>128, relu),
    #     MaxPool((2, 2)),
    #     Conv((3, 3), 128=>256, relu),
    #     MaxPool((2, 2)),
    #     Dense(256, head_dim, sigmoid_fast),
    # )

    conv_model = Flux.@autosize (size_tuple,) Chain(
        DeepART.CCConv(),
        Chain(
            Conv((5,5), _ => 6, sigmoid_fast, bias=false),
        ),
        # Chain(
        #     MaxPool((2,2)),
        #     DeepART.CCConv(),
        # ),
        # Chain(
        #     Conv((5,5), _ => 6, sigmoid, bias=false),
        # ),
        # BatchNorm(_),
        Chain(
            # MaxPool((2,2)),
            MaxPool((2,2)),
            Flux.flatten,
            DeepART.CC(),
        ),
        Dense(_, 256, sigmoid_fast, bias=false),
        DeepART.CC(),
        Chain(
            Dense(_, head_dim, sigmoid_fast, bias=false),
            vec,
        ),
        # Dense(15=>10, sigmoid),
        # Flux.flatten,
        # Dense(_=>15,relu),
        # Dense(15=>10,sigmoid),
        # softmax
    )

    return conv_model
end

# conv_model = Flux.@autosize (size_tuple,) Chain(
#     DeepART.CCConv(),
#     Chain(
#         Conv((5,5), _ => 6, sigmoid, bias=false),
#     ),
#     # Chain(
#     #     MaxPool((2,2)),
#     #     DeepART.CCConv(),
#     # ),
#     # Chain(
#     #     Conv((5,5), _ => 6, sigmoid, bias=false),
#     # ),
#     # BatchNorm(_),
#     Chain(
#         # MaxPool((2,2)),
#         MaxPool((2,2)),
#         Flux.flatten,
#         DeepART.CC(),
#     ),
#     # Dense(_, 128, sigmoid, bias=false),
#     # DeepART.CC(),
#     Chain(
#         Dense(_, head_dim, sigmoid, bias=false),
#         vec,
#     ),
#     # Dense(15=>10, sigmoid),
#     # Flux.flatten,
#     # Dense(_=>15,relu),
#     # Dense(15=>10,sigmoid),
#     # softmax
# )

# # Model definition
# model = Flux.@autosize (n_input,) Chain(
#     # DeepART.CC(),
#     # Dense(_, 512, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, 256, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, 128, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, 64, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, head_dim, sigmoid, bias=false),
#     # Dense(_, head_dim, bias=false),
#     # softmax,
#     # Dense(_, n_classes, sigmoid),
#     # sigmoid,
#     # softmax,
# )

# model = Flux.@autosize (n_input,) Chain(
#     DeepART.CC(),
#     DeepART.Fuzzy(_, 256, sigmoid),
#     DeepART.CC(),
#     DeepART.Fuzzy(_, 128, sigmoid),
#     DeepART.CC(),
#     DeepART.Fuzzy(_, 64, sigmoid),
#     DeepART.CC(),
#     DeepART.Fuzzy(_, head_dim, sigmoid),
# )

# model = Flux.@autosize (n_input,) Chain(
#     DeepART.CC(),
#     DeepART.Fuzzy(_, 256, sigmoid),
#     DeepART.CC(),
#     DeepART.Fuzzy(_, 128, sigmoid),
#     DeepART.CC(),
#     DeepART.Fuzzy(_, 64, sigmoid),
#     DeepART.CC(),
#     DeepART.Fuzzy(_, head_dim, sigmoid),
# )

# model = Flux.@autosize (n_input,) Chain(
#     DeepART.CC(),
#     Dense(_, 512, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, 256, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, 128, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, 64, sigmoid, bias=false),
#     DeepART.CC(),
#     Dense(_, head_dim, sigmoid, bias=false),
# )