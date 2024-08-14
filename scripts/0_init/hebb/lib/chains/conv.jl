"""
    conv.jl

# Desription
Definitions for convolutional models.
"""

# function get_conv_layer(
#     n_in::Integer,
#     n_out::Integer,
#     kernel::Tuple,
#     opts::ModelOpts;
#     first_layer::Bool = false,
#     n_pool::Tuple = (),
# )
#     return Flux.@autosize (n_in,) Chain(
#         # CC layer
#         Chain(
#             first_layer ? identity :
#                 Chain(
#                     MaxPool(n_pool),
#                     sigmoid_fast,
#                 ),
#             DeepART.CCConv()
#         ),

#         # Conv layer
#         Chain(
#             Conv(
#                 kernel, _ => n_out,
#                 # sigmoid_fast,
#                 bias=opts["bias"],
#                 init=opts["init"],
#             ),
#         ),
#     )
# end

function get_inc_conv_model(
    size_tuple::Tuple,
    head_dim::Integer,
    opts::ModelOpts
)::GroupedCCChain

    first_activation = if opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    # preprocess = if opts["layer_norm"]
    #     LayerNorm(_, affine=false)
    # else
    #     identity
    # end

    conv_model = Flux.@autosize (size_tuple,) Chain(
        # get_conv_layer(, 8, (3, 3), opts, first_layer=true),
        Chain(
            Chain(
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                # (3, 3), _ => 8,
                (5, 5), _ => 16,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                # pad=(2,2),
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                MaxPool(
                    # (2,2),
                    (3, 3),
                    stride=(2,2),
                ),
                # sigmoid_fast,
                # opts["middle_activation"],
                # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
                LayerNorm(_, affine=false),
                opts["post_synaptic"] ? identity : opts["middle_activation"],
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                (5,5), _ => 16,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                # pad=(2,2),
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                Flux.AdaptiveMaxPool(
                    (4, 4)
                    # (8, 8),
                ),
                Flux.flatten,
                vec,
                # sigmoid_fast,
                # opts["middle_activation"],
                # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
                LayerNorm(_, affine=false),
                opts["post_synaptic"] ? identity : opts["middle_activation"],
                opts["cc"] ? DeepART.CC() : identity,
            ),
            Dense(_, 32,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        # Last layers
        get_widrow_hoff_layer(32, head_dim, opts),
        # vec,
        # Chain(
        #     Chain(
        #         # identity,
        #         # sigmoid_fast,
        #         opts["middle_activation"],
        #     ),
        #     Chain(
        #         Dense(
        #             _, head_dim,
        #             opts["final_sigmoid"] ? sigmoid_fast : identity,
        #             bias=opts["bias"],
        #         ),
        #         vec,
        #     ),
        # ),
    )

    return GroupedCCChain(conv_model)
end



# function get_inc_conv_model(
#     size_tuple::Tuple,
#     head_dim::Integer,
#     opts::ModelOpts
# )::GroupedCCChain

#     first_activation = if opts["post_synaptic"]
#         identity
#     else
#         opts["middle_activation"]
#     end

#     # preprocess = if opts["layer_norm"]
#     #     LayerNorm(_, affine=false)
#     # else
#     #     identity
#     # end

#     conv_model = Flux.@autosize (size_tuple,) Chain(
#         # get_conv_layer(, 8, (3, 3), opts, first_layer=true),
#         Chain(
#             Chain(
#                 opts["cc"] ? DeepART.CCConv() : identity,
#             ),
#             Conv(
#                 # (3, 3), _ => 8,
#                 (5, 5), _ => 16,
#                 opts["post_synaptic"] ? opts["middle_activation"] : identity,
#                 # pad=(2,2),
#                 bias=opts["bias"],
#                 init=opts["init"],
#             ),
#         ),
#         Chain(
#             Chain(
#                 MaxPool(
#                     # (2,2),
#                     (3, 3),
#                     stride=(2,2),
#                 ),
#                 # sigmoid_fast,
#                 # opts["middle_activation"],
#                 # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
#                 LayerNorm(_, affine=false),
#                 opts["post_synaptic"] ? identity : opts["middle_activation"],
#                 opts["cc"] ? DeepART.CCConv() : identity,
#             ),
#             Conv(
#                 (5,5), _ => 16,
#                 opts["post_synaptic"] ? opts["middle_activation"] : identity,
#                 # pad=(2,2),
#                 bias=opts["bias"],
#                 init=opts["init"],
#             ),
#         ),
#         Chain(
#             Chain(
#                 Flux.AdaptiveMaxPool(
#                     (4, 4)
#                     # (8, 8),
#                 ),
#                 Flux.flatten,
#                 vec,
#                 # sigmoid_fast,
#                 # opts["middle_activation"],
#                 # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
#                 LayerNorm(_, affine=false),
#                 opts["post_synaptic"] ? identity : opts["middle_activation"],
#                 opts["cc"] ? DeepART.CC() : identity,
#             ),
#             Dense(_, 32,
#                 opts["post_synaptic"] ? opts["middle_activation"] : identity,
#                 bias=opts["bias"],
#                 init=opts["init"],
#             ),
#         ),
#         # Last layers
#         get_widrow_hoff_layer(32, head_dim, opts),
#         # vec,
#         # Chain(
#         #     Chain(
#         #         # identity,
#         #         # sigmoid_fast,
#         #         opts["middle_activation"],
#         #     ),
#         #     Chain(
#         #         Dense(
#         #             _, head_dim,
#         #             opts["final_sigmoid"] ? sigmoid_fast : identity,
#         #             bias=opts["bias"],
#         #         ),
#         #         vec,
#         #     ),
#         # ),
#     )

#     return GroupedCCChain(conv_model)
# end



function get_conv_model(
    size_tuple::Tuple,
    head_dim::Integer,
    opts::ModelOpts
)::AlternatingCCChain
    conv_model = Flux.@autosize (size_tuple,) Chain(
        # CC layer
        Chain(
            opts["cc"] ? DeepART.CCConv() : identity,
        ),

        # Conv layer
        Chain(
            Conv(
                (3, 3), _ => 8,
                # (5, 5), _ => 6, pad=(2,2),
                # sigmoid_fast,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),

        # CC layer
        Chain(
            MaxPool(
                (2,2),
                # stride=(2,2),
            ),
            # sigmoid_fast,
            # opts["middle_activation"],
            opts["post_synaptic"] ? identity : opts["middle_activation"],
            opts["cc"] ? DeepART.CCConv() : identity,
        ),

        # Conv layer
        Chain(
            Conv(
                (5,5), _ => 16,
                # sigmoid_fast,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),

        # CC layer
        Chain(
            Flux.AdaptiveMaxPool((4, 4)),
            Flux.flatten,
            # sigmoid_fast,
            # opts["middle_activation"],
            opts["post_synaptic"] ? identity : opts["middle_activation"],
            opts["cc"] ? DeepART.CC() : identity,
        ),

        # Dense layer
        Dense(_, 32,
            opts["post_synaptic"] ? opts["middle_activation"] : identity,
            bias=opts["bias"],
            init=opts["init"],
        ),

        # Last layers
        Chain(
            # identity,
            # sigmoid_fast,
            # opts["middle_activation"],
            opts["post_synaptic"] ? identity : opts["middle_activation"],
        ),
        Chain(
            Dense(
                _, head_dim,
                # sigmoid_fast,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=["bias"],
                # init=opts["init"],
            ),
            vec,
        ),
    )
    return AlternatingCCChain(conv_model)
end
