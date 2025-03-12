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

"""
FluxBlock abstract type, which encapsulates Flux chain-like models.
"""
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

"""
Dense chain layer constructor.
"""
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
        LayerNorm(input_dim, affine=false)
        # Flux.normalise
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

"""
Fuzzy chain layer constructor.
"""
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
        # Flux.normalise
        LayerNorm(n_in, affine=false)
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

"""
Widrow-Hoff chain layer constructor as a final (supervised) layer.
"""
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
        # Flux.normalise
        LayerNorm(input_dim, affine=false)
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


# LeNet:
# model = Chain(
#     Conv((5,5),1 => 6, relu),
#     MaxPool((2,2)),
#     Conv((5,5),6 => 16, relu),
#     MaxPool((2,2)),
#     Flux.flatten,
#     Dense(256=>120,relu),
#     Dense(120=>84, relu),
#     Dense(84=>10, sigmoid),
#     softmax
# )

"""
Convolutional chain layer constructor.

NOTE: this is not currently set up to be modular and should be used as the first "layer" in a model chain.
"""
function get_lenet_conv_chain(
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
                # (3, 3), _ => 8,
                (5, 5), _ => 16,
                # (5, 5), _ => 6,
                # pad=(2,2),
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                # MaxPool(
                MeanPool(
                    (2,2),
                    # (3, 3),
                    stride=(2,2),
                ),
                # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
                LayerNorm(_, affine=false),
                opts["post_synaptic"] ? identity : opts["middle_activation"],
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                (5,5), _ => 16,
                # (5,5), _ => 32,
                # (5,5), _ => 8,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        #     Chain(
        #         MeanPool(
        #             (2, 2),
        #             stride=(2,2),
        #         ),
        #         Flux.flatten,
        #         vec,
        #     )
        ),

        # # TEMP: ADDED BLOCK
        # Chain(
        #     Chain(
        #         # MaxPool(
        #         MeanPool(
        #             (2,2),
        #             stride=(2,2),
        #         ),
        #         # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
        #         LayerNorm(_, affine=false),
        #         opts["post_synaptic"] ? identity : opts["middle_activation"],
        #         opts["cc"] ? DeepART.CCConv() : identity,
        #     ),
        #     Conv(
        #         (3,3), _ => 16,
        #         opts["post_synaptic"] ? opts["middle_activation"] : identity,
        #         bias=opts["bias"],
        #         init=opts["init"],
        #     ),
        # ),
        Chain(
            Chain(
                # Flux.AdaptiveMaxPool(
                #     # (4, 4)
                #     (5, 5)
                # ),
                MeanPool(
                    (2, 2),
                    stride=(2,2),
                ),
                Flux.flatten,
                vec,
                # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
                LayerNorm(_, affine=false),
                opts["post_synaptic"] ? identity : opts["middle_activation"],
                opts["cc"] ? DeepART.CC() : identity,
            ),
            Dense(
                # _, 64,
                _, 120,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
    )

    return conv_model
end

"""
Convolutional chain layer constructor.

NOTE: this is not currently set up to be modular and should be used as the first "layer" in a model chain.
"""
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
                # (3, 3), _ => 8,
                # (5, 5), _ => 16,
                (5, 5), _ => 6,
                # pad=(2,2),
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                # MaxPool(
                MeanPool(
                    (2,2),
                    # (3, 3),
                    stride=(2,2),
                ),
                # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
                LayerNorm(_, affine=false),
                opts["post_synaptic"] ? identity : opts["middle_activation"],
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                (5,5), _ => 16,
                # (5,5), _ => 32,
                # (5,5), _ => 8,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        #     Chain(
        #         MeanPool(
        #             (2, 2),
        #             stride=(2,2),
        #         ),
        #         Flux.flatten,
        #         vec,
        #     )
        ),
        Chain(
            Chain(
                # Flux.AdaptiveMaxPool(
                #     # (4, 4)
                #     (5, 5)
                # ),
                MeanPool(
                    (2, 2),
                    stride=(2,2),
                ),
                Flux.flatten,
                vec,
                # opts["layer_norm"] ? LayerNorm(_, affine=false) : identity,
                LayerNorm(_, affine=false),
                opts["post_synaptic"] ? identity : opts["middle_activation"],
                opts["cc"] ? DeepART.CC() : identity,
            ),
            Dense(
                # _, 64,
                _, 120,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
    )

    return conv_model
end

"""
Map of block types to the functions that create them.
"""
const CHAIN_BLOCK_FUNC_MAP = Dict(
    "dense" => get_dense_chain,
    "fuzzy" => get_fuzzy_chain,
    "widrow_hoff" => get_widrow_hoff_chain,
    "conv" => get_conv_chain,
    "lenet" => get_lenet_conv_chain,
)

"""
List of supervised layers.
"""
const SUPERVISED_MODELS = [
    "widrow_hoff",
    "fuzzyartmap",
]

# -----------------------------------------------------------------------------
# CHAIN BLOCKS
# -----------------------------------------------------------------------------

"""
Block definition for a single dense chain.
"""
struct ChainBlock{T <: Flux.Chain} <: FluxBlock
    chain::T
    # activations::Vector{Vector{Float32}}
    opts::BlockOpts
end

"""
Block definition for a single convolutional chain.
"""
struct ConvBlock{T <: Flux.Chain} <: FluxBlock
    chain::T
    opts::BlockOpts
end

"""
Constructs a dense chain block.
"""
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
    # first_layer = opts["index"] == 1
    first_layer(x) = opts["index"] == 1 && x == 1

    # Get the layer function
    layer_func = CHAIN_BLOCK_FUNC_MAP[opts["model"]]

    # Create the model
    model = Chain(
        (
            layer_func(n_neurons[ix], n_neurons[ix + 1], opts, first_layer=first_layer(ix))
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

"""
Constructs a convolutional chain block.
"""
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

"""
Inference function for a chain block.
"""
function forward(block::FluxBlock, x)
    return block.chain(x)
end


function get_weights(block::FluxBlock)
    return Flux.params(block.chain)
end

function get_activations(block::FluxBlock, x)
    return Flux.activations(block.chain, x)
end

function get_incremental_activations(
    block::FluxBlock,
    x,
)
    n_layers = length(block.chain)
    ins = []
    outs = []
    for ix = 1:n_layers
        pre_input = (ix == 1) ? x : outs[end]
        local_acts = Flux.activations(block.chain[ix], pre_input)
        push!(ins, local_acts[1])
        push!(outs, local_acts[2])
        # push!(outs, local_acts[end])
    end
    return ins, outs
end


"""
Training function for a chain block.
"""
function train!(block::FluxBlock, x, y)
    # Get the names for weights and iteration
    # params = get_weights(block.chain)
    params = get_weights(block)
    n_layers = length(params)

    # Get the correct inputs and outputs for actuall learning
    # ins, outs = get_incremental_activations(block.chain, x)
    ins, outs = get_incremental_activations(block, x)

    # Create the target vector
    target = zeros(Float32, size(outs[end]))
    target[y] = 1.0

    # if block.opts["gpu"]
    #     target = target |> gpu
    # end

    for ix = 1:n_layers
        weights = params[ix]
        out = outs[ix]
        input = ins[ix]

        if block.opts["model"] == "widrow_hoff"
            # local_input = model.opts["final_bias"] ? [1.0; input] : input
            widrow_hoff_learn!(
                input,
                out,
                weights,
                target,
                block.opts,
            )
        else
            deepart_learn!(
                input,
                out,
                weights,
                block.opts,
            )
        end
    end

    return outs[end]
end

# -----------------------------------------------------------------------------
# ART BLOCKS
# -----------------------------------------------------------------------------

"""
ART block definition.
"""
struct ARTBlock{T <: ARTModule, U <: Flux.Chain} <: Block
    model::T
    chain::U
    opts::BlockOpts
end

"""
ART block constructor.
"""
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
    # model = AdaptiveResonance.DDVFA(
    #     # rho=opts["rho"],
    #     rho_lb=0.4,
    #     rho_ub=0.75,
    # )

    model.config = AdaptiveResonance.DataConfig(
        0.0,
        # -1.0,
        1.0,
        n_inputs,
    )

    chain = Chain(
        # LayerNorm(n_inputs, affine=false),
        # relu,
        LayerNorm(n_inputs, affine=false),
        sigmoid_fast,
        # tanh_fast,
        # relu,
    )

    # # model = DeepART.FuzzyARTMap(
    # model = AdaptiveResonance.DDVFA(
    #     n_inputs,
    #     n_outputs,
    # )

    return ARTBlock(
        model,
        chain,
        opts,
    )
end

"""
Inference function for an ART block.
"""
function forward(block::ARTBlock, x)
    y_hat = if block.model.n_categories > 0
        preprocess_x = block.chain(x)
        AdaptiveResonance.classify(block.model, preprocess_x, get_bmu=true)
    else
        0
    end

    return y_hat
end

"""
Training function for an ART block.
"""
function train!(block::ARTBlock, x, y)
    preprocess_x = block.chain(x)
    return AdaptiveResonance.train!(
        block.model,
        preprocess_x,
        y,
        # y=y,
    )
end

# -----------------------------------------------------------------------------
# BLOCKNET
# -----------------------------------------------------------------------------

"""
Map of block types to the functions that create them.
"""
const BLOCK_FUNC_MAP = Dict(
    "dense" => ChainBlock,
    "fuzzy" => ChainBlock,
    "widrow_hoff" => ChainBlock,
    "conv" => ConvBlock,
    "lenet" => ConvBlock,
    "fuzzyartmap" => ARTBlock,
)

"""
Definition of the block outputs type.
"""
const OutsVector = Vector{Vector{Float32}}

"""
BlockNet definition as a container of blocks.
"""
struct BlockNet
    layers::Vector{Block}
    outs::OutsVector
    opts::SimOpts
end

# """
# BlockNet constructor.
# """
# function BlockNet(
#     data::DeepART.DataSplit,
#     opts::SimOpts,
# )

#     dev_x, _ = data.train[1]
#     n_input = size(dev_x)[1]
#     n_class = length(unique(data.train.y))

#     blocks = Vector{Block}()
#     out_sizes = []
#     outs = OutsVector()

#     for block_opts in opts["blocks"]

#         is_conv = block_opts["model"] in ["conv", "lenet"]

#         # If this is the first layer
#         if block_opts["index"] == 1
#             # If the convolutional model is selected, create a convolution input tuple
#             local_n_inputs = if is_conv
#                 (size(data.train.x)[1:3]..., 1)
#             else
#                 n_input
#             end
#             # push!(outs, zeros(Float32, local_n_inputs))
#         # Otherwise, get the size of the previous layer(s)
#         else
#             # Get the combined sizes of the previous layers
#             local_n_inputs = if (length(block_opts["inputs"]) > 1)
#                 sum([out_sizes[ix] for ix in block_opts["inputs"]])
#             # Otherwise, get the size of the single previous layer
#             else
#                 # previous_size[1]
#                 out_sizes[block_opts["inputs"]]
#             end
#         end

#         # If the last layer, set the number of outputs to the number of classes
#         local_n_outputs = if (block_opts["index"] == length(opts["blocks"]))
#             n_class
#         else
#             0
#         end

#         # Generate the block
#         local_block = BLOCK_FUNC_MAP[block_opts["model"]](
#             block_opts,
#             n_inputs=local_n_inputs,
#             n_outputs=local_n_outputs,
#         )

#         # local_output = local_block.opts["n_neurons"][end]
#         test_size = is_conv ? local_n_inputs : (local_n_inputs,)

#         # Correction for the ouptut shape for ART module blocks
#         if local_block isa ARTBlock
#             block_out_size = n_class
#         # Otherwise, get the output size from the model
#         else
#             block_out_size = Flux.outputsize(
#                 local_block.chain,
#                 test_size
#             )[1]
#         end

#         # Append the blocks and metadata
#         push!(blocks, local_block)
#         push!(out_sizes, block_out_size)
#         push!(outs, zeros(Float32, block_out_size))
#     end

#     return BlockNet(
#         blocks,
#         outs,
#         opts,
#     )
# end


"""
BlockNet constructor.
"""
function BlockNet(
    data::DeepART.SupervisedDataset,
    opts::SimOpts,
)
    # dev_x, _ = data.train[1]
    dev_x = DeepART.get_sample(data, 1)
    n_input = size(dev_x)[1]
    # n_class = length(unique(data.train.y))
    n_class = length(unique(data.y))

    blocks = Vector{Block}()
    out_sizes = []
    outs = OutsVector()

    for block_opts in opts["blocks"]

        is_conv = block_opts["model"] in ["conv", "lenet"]

        # If this is the first layer
        if block_opts["index"] == 1
            # If the convolutional model is selected, create a convolution input tuple
            local_n_inputs = if is_conv
                # (size(data.train.x)[1:3]..., 1)
                (size(data.x)[1:3]..., 1)
                # (size(dev_x)..., 1)
                # size(dev_x)
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

"""
BlockNet constructor.
"""
function BlockNet(
    data::DeepART.DataSplit,
    opts::SimOpts,
)
    return BlockNet(
        data.train,
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

"""
Inference definition for a BlockNet.
"""
function forward(net::BlockNet, x)
    # Loop through the block layers
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

"""
Training definition for a BlockNet.
"""
function train!(net::BlockNet, x, y)
    # for layer in net.layers
    #     train(layer, x, y)
    # end
    # Loop through the block layers
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
        y_hat = train!(layer, local_input, y)

        # If the layer is an ART block, convert the output to a one-hot vector
        if layer isa ARTBlock
            y_out = zeros(Float32, length(net.outs[end]))
            if y_hat > 0
                y_out[y_hat] = 1.0
            end
        # Otherwise, use the output as-is
        else
            y_out = y_hat
        end

        # Store the output to the cached outputs vector
        net.outs[ix] .= y_out
    end

    # Return the last layer output as the output of the block net
    return net.outs[end]
end


function train_loop(
    model::BlockNet,
    data;
    n_vals::Integer = 100,
    n_epochs::Integer = 10,
    val_epoch::Bool = false,
    toshow::Bool = true
)
    loop_dict = LoopDict()

    # Set up the epochs progress bar
    loop_dict["n_iter"] = if val_epoch
        length(data.train)
    else
        n_epochs
    end

    # Set up the validation intervals
    local_n_vals = min(n_vals, loop_dict["n_iter"])
    loop_dict["interval_vals"] = Int(floor(loop_dict["n_iter"] / local_n_vals))
    loop_dict["vals"] = zeros(Float32, local_n_vals)

    # Init the progress bar and loop tracking variables
    p = init_progress(loop_dict)

    # Iterate over each epoch
    for ie = 1:n_epochs
        # train_loader = Flux.DataLoader(data.train, batchsize=-1, shuffle=true)
        train_loader = Flux.DataLoader(data.train, batchsize=-1)
        if model.opts["gpu"]
            train_loader = train_loader |> gpu
        end

        # Iteratively train
        for (x, y) in train_loader
            # if model.opts["immediate"]
            #     train_hebb_immediate(model, x, y)
            # else
            #     train_hebb(model, x, y)
            # end
            train!(model, x, y)

            if val_epoch
                update_view_progress!(
                    p,
                    loop_dict,
                    model,
                    data,
                )
            end
        end

        if toshow
            # Compute validation performance
            if !val_epoch
                update_view_progress!(
                    p,
                    loop_dict,
                    model,
                    data,
                )
            else
                # Reset incrementers
                p = init_progress(loop_dict)

                local_plot = lineplot(
                    loop_dict["vals"],
                )
                show(local_plot)
                println("\n")
            end
        end
    end

    perf = test(model, data)
    @info "perf = $perf"
    return loop_dict["vals"]
end

function test(
    model::BlockNet,
    data::DeepART.DataSplit,
)
    n_test = length(data.test)

    y_hats = zeros(Int, n_test)
    test_loader = Flux.DataLoader(data.test, batchsize=-1)
    if model.opts["gpu"]
        y_hats = y_hats |> gpu
        test_loader = test_loader |> gpu
    end

    ix = 1
    for (x, _) in test_loader
        # y_hats[ix] = argmax(model.model.chain(x))
        y_hats[ix] = argmax(forward(model, x))
        ix += 1
    end

    # y_hats = model(data.test.x |> gpu) |> cpu  # first row is prob. of true, second row p(false)
    # y_hats = argmaxmodel(data.test.x)  # first row is prob. of true, second row p(false)

    if model.opts["gpu"]
        y_hats = y_hats |> cpu
    end

    perf = DeepART.AdaptiveResonance.performance(y_hats, data.test.y)
    return perf
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


"""
Helper function: gets the weights of a neuron with a specified index at a specified layer.
"""
function get_weight_slice(
    model::BlockNet,
    layer::Integer,
    index::Integer,
)
    # weights = get_weights(model.model)
    # weights = Flux.params(model.model.chain)
    # weights = get_weights(model.model)
    weights = get_weights(model.layers[layer])
    dim = Int(size(weights[layer])[2])

    if model.layers[layer].opts["bias"]
        dim -= 1
        local_weights = weights[layer][index, 2:end]
    else
        local_weights = weights[layer][index, :]
    end
    # local_weights = weights[layer][index, :]

    if model.layers[layer].opts["cc"]
        dim = Int(dim / 2)
    end

    dim = Int(floor(sqrt(dim)))
    local_weight = reshape(
        # weights[layer][index, :],
        local_weights,
        dim,
        model.layers[layer].opts["cc"] ? dim*2 : dim,
    )

    return local_weight
end


"""
Helper function: visualizes the weight index at a specified layer.
"""
function view_weight(
    model::BlockNet,
    index::Integer;
    layer::Integer=1
)
    # if model.opts["bias"]
    #     dim_x -= 1
    # end

    if model.layers[layer].chain[1][2] isa Flux.Conv
        weights = model.layers[layer].chain[1][2].weight
        lmax = maximum(weights)
        lmin = minimum(weights)
        local_weights = weights[:, :, :, index] .- lmin ./ (lmax - lmin)
        img = DeepART.Gray.(
            vcat(
                (local_weights[:, :, jx] for jx = 1:size(local_weights)[3])...,
                # weights[:, :, 1, index] .- lmin ./ (lmax - lmin),
                # weights[:, :, 2, index] .- lmin ./ (lmax - lmin)
            ),
        )
    else
        # # weights = Flux.params(model.model.chain)
        # weights = get_weights(model.model)
        # dim = Int(size(weights[layer])[2])
        # if model.opts["cc"]
        #     dim = Int(dim / 2)
        # end

        # dim = Int(sqrt(dim))
        # local_weight = reshape(
        #     weights[layer][index, :],
        #     dim,
        #     model.opts["cc"] ? dim*2 : dim,
        # )
        local_weight = get_weight_slice(model, layer, index)

        lmax = maximum(local_weight)
        lmin = minimum(local_weight)
        img = DeepART.Gray.(local_weight .- lmin ./ (lmax - lmin))
    end

    return img
end

# """
# Helper function: visualizes a grid of weights at a specified layer.
# """
# function view_weight_grid(model::Hebb.BlockNet, n_grid::Int; layer=1)
#     # Infer the size of the weight matrix
#     a = Hebb.view_weight(model, 1, layer=layer)
#     (dim_x, dim_y) = size(a)

#     # if model.opts["bias"]
#     #     dim_x -= 1
#     # end

#     # Create the output grid
#     out_grid = zeros(DeepART.Gray{Float32}, dim_x * n_grid, dim_y * n_grid)

#     # Populate the grid iteratively
#     for ix = 1:n_grid
#         for jx = 1:n_grid
#             local_weight = Hebb.view_weight(
#                 model,
#                 n_grid * (ix - 1) + jx,
#                 layer=layer,
#             )
#             out_grid[(ix - 1) * dim_x + 1:ix * dim_x,
#                      (jx - 1) * dim_y + 1:jx * dim_y] = local_weight
#         end
#     end

#     # Return the tranpose for visualization
#     return out_grid'
# end
