"""
    blocknet.jl

# Description
Definitions for the BlockNet model.
"""

# -----------------------------------------------------------------------------
# ABSTRACT TYPES
# -----------------------------------------------------------------------------

"""
Block abstract type.

This abstract type enforces the following contract:
- A block must have a `forward` method.
"""
abstract type Block end

abstract type FluxBlock <: Block end


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
BlockOpts are ModelOpts that also have an index and input indices.
"""
const BlockOpts = Dict{String, Any}

# const BlockSimOpts = Dict{String, Any}

# const BlockFileOpts = Dict{String, Union{BlockOpts, BlockSimOpts}}

# -----------------------------------------------------------------------------
# CHAINS
# -----------------------------------------------------------------------------

function get_dense_chain(
    # n_in::Tuple,
    n_in::Integer,
    n_out::Integer,
    opts::ModelOpts;
    first_layer::Bool = false,
)
    first_activation = if first_layer
        identity
    elseif opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    second_activation = if opts["post_synaptic"]
        opts["middle_activation"]
    else
        identity
    end

    input_dim = opts["bias"] ? n_in + 1 : n_in

    preprocess = if opts["layer_norm"] && !first_layer
        # LayerNorm(n_in, affine=false)
        Flux.normalise
    else
        identity
    end

    return Flux.@autosize (input_dim,) Chain(
    # return Flux.@autosize n_in Chain(
        Chain(
            # RandomTransform(_, 16, opts),
            # first_layer ? identity : sigmoid_fast,
            # first_layer ? identity : opts["middle_activation"],
            # LayerNorm(_, affine=false),
            # opts["layer_norm"] ? LayerNorm(input_dim, affine=false) : identity,
            preprocess,
            first_activation,
            opts["cc"] ? DeepART.CC() : identity,
        ),
        Dense(
            _, n_out,
            # opts["post_synaptic"] ? opts["middle_activation"] : identity,
            second_activation,
            # bias = opts["bias"],
            bias = false,
            init = opts["init"],
        ),
    )
end

# struct RandomTransform
#     chain::Dense
# end

# function RandomTransform(
#     n_in::Integer,
#     n_out::Integer,
#     opts::ModelOpts,
# )
#     return RandomTransform(
#         Dense(
#             n_in,
#             n_out,
#             # sigmoid_fast,
#             init = opts["init"],
#             opts["middle_activation"],
#             bias = false
#         )
#     )
# end

# function (m::RandomTransform)(x)
#     return m.chain(x)
# end

# Flux.@layer RandomTransform

# Flux.trainable(m::RandomTransform) = ()

function get_fuzzy_chain(
    n_in::Integer,
    # n_in::Tuple,
    n_out::Integer,
    opts::ModelOpts;
    first_layer::Bool = false,
)
    first_activation = if first_layer
        identity
    elseif opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    preprocess = if opts["layer_norm"] && !first_layer
        # LayerNorm(n_in, affine=false)
        Flux.normalise
    else
        identity
    end

    return Flux.@autosize (n_in,) Chain(
    # return Flux.@autosize n_in Chain(
        Chain(
            # RandomTransform(_, 16),
            # first_layer ? identity : sigmoid_fast,
            # first_layer ? identity : opts["middle_activation"],
            # opts["layer_norm"] ? LayerNorm(n_in, affine=false) : identity,
            preprocess,
            first_activation,
            opts["cc"] ? DeepART.CC() : identity,
        ),
        DeepART.Fuzzy(
            _, n_out,
            opts["post_synaptic"] ? opts["middle_activation"] : identity,
            init = opts["init"],
        ),
    )
end

function get_widrow_hoff_chain(
    n_in::Integer,
    # n_in::Tuple,
    n_out::Integer,
    opts::ModelOpts;
    first_layer::Bool = false,
)
    input_dim = opts["bias"] ? n_in + 1 : n_in

    first_activation = if opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    preprocess = if opts["layer_norm"]
        # LayerNorm(n_in, affine=false)
        Flux.normalise
    else
        identity
    end

    return Flux.@autosize (input_dim,) Chain(
    # return Flux.@autosize n_in Chain(
        Chain(
            # identity
            # sigmoid_fast,
            # opts["middle_activation"],
            # opts["layer_norm"] ? LayerNorm(input_dim, affine=false) : identity,
            preprocess,
            first_activation,
        ),
        Dense(
            _, n_out,
            # bias = opts["bias"],
            opts["final_sigmoid"] ? sigmoid_fast : identity,
            # init = opts["init"],
            init = Flux.glorot_uniform,
            bias = false,
        ),
    )
end



function get_conv_chain(
    # size_tuple::Tuple,
    # head_dim::Integer,
    # opts::ModelOpts
    n_in::Tuple,
    n_out::Integer,
    # kernel::Tuple,
    opts::ModelOpts;
    first_layer::Bool = false,
    # n_pool::Tuple = (),
)
    # first_activation = if opts["post_synaptic"]
    #     identity
    # else
    #     opts["middle_activation"]
    # end

    # preprocess = if opts["layer_norm"]
    #     LayerNorm(_, affine=false)
    #     # Flux.normalise
    # else
    #     identity
    # end

    # @info n_in

    conv_model = Flux.@autosize (n_in,) Chain(
        # get_conv_layer(, 8, (3, 3), opts, first_layer=true),
        Chain(
            Chain(
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                (3, 3), _ => 8,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                MaxPool((2,2)),
                # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
                LayerNorm(_, affine=false),
                opts["post_synaptic"] ? identity : opts["middle_activation"],
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                (5,5), _ => 16,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                Flux.AdaptiveMaxPool((4, 4)),
                Flux.flatten,
                vec,
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
    )

    return conv_model
end


const CHAIN_BLOCK_FUNC_MAP = Dict(
    "dense" => get_dense_chain,
    "fuzzy" => get_fuzzy_chain,
    "widrow_hoff" => get_widrow_hoff_chain,
    "conv" => get_conv_chain,
)

const SUPERVISED_MODELS = [
    "widrow_hoff",
    "fuzzyartmap",
]

# -----------------------------------------------------------------------------
# BLOCKS
# -----------------------------------------------------------------------------

struct ChainBlock{T <: Flux.Chain} <: FluxBlock
    chain::T
    # activations::Vector{Vector{Float32}}
    opts::BlockOpts
end

struct ConvBlock{T <: Flux.Chain} <: FluxBlock
    chain::T
    opts::BlockOpts
end


function ChainBlock(
    opts::BlockOpts;
    n_inputs::Integer=0,
    n_outputs::Integer=0,
    # n_inputs::Tuple=(0,),
    # n_outputs::Tuple=(0,),
)
    # Determine the actual number of neurons per layer that we will have
    if opts["model"] == "widrow_hoff"
        # Special case for Widrow-Hoff
        n_neurons = [n_inputs, n_outputs]
    else
        # Otherwise, use the n_neurons option and handle the input layer
        n_neurons = opts["n_neurons"]
        if n_inputs != 0
            n_neurons = [n_inputs, n_neurons...]
        end
    end

    # The number of layers is determined by the above
    # (i.e., if we are at the input or output of the model)
    n_layers = length(n_neurons) - 1

    # Determine if this is the first layer
    first_layer = opts["index"] == 1

    # Get the layer function
    layer_func = CHAIN_BLOCK_FUNC_MAP[opts["model"]]

    # Create the model
    model = Chain(
        (
            layer_func(n_neurons[ix], n_neurons[ix + 1], opts, first_layer=first_layer)
            for ix = 1:n_layers
        )...,
    )

    # Enforce positive weights if necessary
    if opts["positive_weights"]
        ps = Flux.params(model)
        for p in ps
            p .= abs.(p)
            p .= p ./ maximum(p)
        end
    end

    return ChainBlock(
        model,
        opts,
    )
end



function ConvBlock(
    opts::BlockOpts;
    n_inputs::Tuple=(0,),
    n_outputs::Integer=0,
)
    # Determine if this is the first layer
    first_layer = opts["index"] == 1

    # Get the layer function
    layer_func = CHAIN_BLOCK_FUNC_MAP[opts["model"]]

    # Create the model
    model = Chain(
        (
            layer_func(n_inputs, 0, opts, first_layer=first_layer)
        )...,
    )

    # Enforce positive weights if necessary
    if opts["positive_weights"]
        ps = Flux.params(model)
        for p in ps
            p .= abs.(p)
            p .= p ./ maximum(p)
        end
    end

    return ConvBlock(
        model,
        opts,
    )
end

function forward(block::FluxBlock, x)
    return block.chain(x)
end

function train(block::FluxBlock, x, y)
    return train(block.chain, x, y)
end

struct ARTBlock{T <: ARTModule} <: Block
    model::T
    opts::BlockOpts
end

function ARTBlock(
    opts::BlockOpts;
    n_inputs::Integer=0,
    n_outputs::Integer=0,
)

    # Create the head
    model = AdaptiveResonance.SFAM(
        rho=opts["rho"],
        epsilon=1e-4,
        beta=opts["beta_s"],
    )
    model.config = AdaptiveResonance.DataConfig(0.0, 1.0, n_inputs)

    # # model = DeepART.FuzzyARTMap(
    # model = AdaptiveResonance.DDVFA(
    #     n_inputs,
    #     n_outputs,
    # )

    return ARTBlock(
        model,
        opts,
    )
end

function forward(block::ARTBlock, x)
    y_hat = if block.model.n_categories > 0
        AdaptiveResonance.classify(block.model, x, get_bmu=true)
    else
        0
    end

    return y_hat
end

function train(block::ARTBlock, x, y)
    return train(block.model, x, y)
end

# -----------------------------------------------------------------------------
# BLOCKNET
# -----------------------------------------------------------------------------

# """
# Many-to-one map of what types of blocks resolve to what types of structs.
# """
# const BLOCK_TYPES = Dict(
#     "dense" => "chain",
#     "fuzzy" => "chain",
#     "conv" => "chain",
#     "widrow_hoff" => "chain",
#     "fuzzyartmap" => "art",
# )

# """
# Map of block types to the functions that create them.
# """
# const BLOCK_FUNC_MAP = Dict(
#     "chain" => ChainBlock,
#     "art" => ARTBlock,
# )

"""
Map of block types to the functions that create them.
"""
const BLOCK_FUNC_MAP = Dict(
    "dense" => ChainBlock,
    "fuzzy" => ChainBlock,
    "widrow_hoff" => ChainBlock,
    "conv" => ConvBlock,
    "fuzzyartmap" => ARTBlock,
)

const OutsVector = Vector{Vector{Float32}}

struct BlockNet
    layers::Vector{Block}
    outs::OutsVector
    opts::SimOpts
end

function BlockNet(
    data::DeepART.DataSplit,
    opts::SimOpts,
)

    dev_x, _ = data.train[1]
    n_input = size(dev_x)[1]
    n_class = length(unique(data.train.y))

    blocks = Vector{Block}()
    out_sizes = []
    outs = OutsVector()

    for block_opts in opts["blocks"]

        is_conv = block_opts["model"] in ["conv",]

        # If this is the first layer
        if block_opts["index"] == 1
            # If the convolutional model is selected, create a convolution input tuple
            local_n_inputs = if is_conv
                (size(data.train.x)[1:3]..., 1)
            else
                n_input
            end
            # push!(outs, zeros(Float32, local_n_inputs))
        # Otherwise, get the size of the previous layer(s)
        else
            # Get the combined sizes of the previous layers
            local_n_inputs = if (length(block_opts["inputs"]) > 1)
                sum([out_sizes[ix] for ix in block_opts["inputs"]])
            # Otherwise, get the size of the single previous layer
            else
                # previous_size[1]
                out_sizes[block_opts["inputs"]]
            end
        end

        # If the last layer, set the number of outputs to the number of classes
        local_n_outputs = if (block_opts["index"] == length(opts["blocks"]))
            n_class
        else
            0
        end

        # Generate the block
        local_block = BLOCK_FUNC_MAP[block_opts["model"]](
            block_opts,
            n_inputs=local_n_inputs,
            n_outputs=local_n_outputs,
        )

        # local_output = local_block.opts["n_neurons"][end]
        test_size = is_conv ? local_n_inputs : (local_n_inputs,)

        # Correction for the ouptut shape for ART module blocks
        if local_block isa ARTBlock
            block_out_size = n_class
        # Otherwise, get the output size from the model
        else
            block_out_size = Flux.outputsize(
                local_block.chain,
                test_size
            )[1]
        end

        # Append the blocks and metadata
        push!(blocks, local_block)
        push!(out_sizes, block_out_size)
        push!(outs, zeros(Float32, block_out_size))
    end

    return BlockNet(
        blocks,
        outs,
        opts,
    )
end

# function get_inputs(net::BlockNet, index::Integer)
#     # If this is the first layer
#     local_n_inputs = if net.opts["index"] == 1
#         # If the convolutional model is selected, create a convolution input tuple
#         if block_opts["model"] in ["conv",]
#             (size(data.train.x)[1:3]..., 1)
#         else
#             n_input
#         end
#     # Otherwise, get the size of the previous layer(s)
#     else
#         # Get the combined sizes of the previous layers
#         if length(block_opts["inputs"]) > 1
#             sum([out_sizes[ix] for ix in block_opts["inputs"]])
#         # Otherwise, get the size of the single previous layer
#         else
#             # previous_size[1]
#             out_sizes[block_opts["inputs"]]
#         end
#     end

#     # inputs = Vector{Any}()
#     # for block in net.layers
#     #     push!(inputs, x)
#     #     x = forward(block, x)
#     # end
#     # return inputs
# end


function forward(net::BlockNet, x)

    for ix in eachindex(net.layers)
        layer = net.layers[ix]

        # If this is the first layer, use the input data
        if layer.opts["index"] == 1
            local_input = x
        # Otherwise, get the output from the previous layer(s)
        else
            local_input = if length(layer.opts["inputs"]) > 1
                vcat((net.outs[ix] for ix in layer.opts["inputs"])...)
            else
                net.outs[layer.opts["inputs"]]
            end
        end

        # Compute the forward pass for the layer
        y = forward(layer, local_input)

        # If the layer is an ART block, convert the output to a one-hot vector
        if layer isa ARTBlock
            y_out = zeros(Float32, length(net.outs[end]))
            if y > 0
                y_out[y] = 1.0
            end
        # Otherwise, use the output as-is
        else
            y_out = y
        end

        # Store the output to the cached outputs vector
        net.outs[ix] .= y_out
    end

    # Return the last layer output as the output of the block net
    return net.outs[end]
end

function train!(net::BlockNet, x, y)
    for layer in net.layers
        train(layer, x, y)
    end
    return
end


# function get_conv_chain(
#     n_in::Tuple,
#     n_out::Integer,
#     # kernel::Tuple,
#     opts::ModelOpts;
#     first_layer::Bool = false,
#     # n_pool::Tuple = (),
# )
#     # return Flux.@autosize n_in Chain(
#     #     # CC layer
#     #     Chain(
#     #         first_layer ? identity :
#     #             Chain(
#     #                 MaxPool(n_pool),
#     #                 sigmoid_fast,
#     #             ),
#     #         DeepART.CCConv()
#     #     ),

#     #     # Conv layer
#     #     Chain(
#     #         Conv(
#     #             kernel, _ => n_out,
#     #             # sigmoid_fast,
#     #             bias=opts["bias"],
#     #             init=opts["init"],
#     #         ),
#     #     ),
#     # )

#     @info n_in

#     conv_model = Flux.@autosize (n_in,) Chain(
#         Chain(
#             Chain(
#                 opts["cc"] ? DeepART.CCConv() : identity,
#             ),
#             Conv(
#                 (3, 3), _ => 8,
#                 bias=opts["bias"],
#                 init=opts["init"],
#             ),
#         ),
#         Chain(
#             Chain(
#                 MaxPool((2,2)),
#                 # sigmoid_fast,
#                 opts["middle_activation"],
#                 opts["cc"] ? DeepART.CCConv() : identity,
#             ),
#             Conv(
#                 (5,5), _ => 16,
#                 bias=opts["bias"],
#                 init=opts["init"],
#             ),
#             Chain(
#                 Flux.AdaptiveMaxPool((4, 4)),
#                 Flux.flatten,
#             ),
#         ),
#         # Chain(
#         #     Chain(
#         #         Flux.AdaptiveMaxPool((4, 4)),
#         #         Flux.flatten,
#         #         # sigmoid_fast,
#         #         opts["middle_activation"],
#         #         opts["cc"] ? DeepART.CC() : identity,
#         #     ),
#         #     Dense(_, 32,
#         #         bias=opts["bias"],
#         #         init=opts["init"],
#         #     ),
#         # ),
#     )

#     return conv_model
# end

