"""
    chains.jl

# Description
Collection of chain model definitions for the Hebb module.
"""

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

const ModelOpts = Dict{String, Any}

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Abstract type for containers of Flux.Chain containing CC and non-CC layers.
"""
abstract type CCChain end

"""
Chains that group alternate CC and non-CC chain layers.
"""
struct GroupedCCChain{T <: Flux.Chain} <: CCChain
    chain::T
end

"""
Chains that simply alternate between CC and non-CC layers.
"""
struct AlternatingCCChain{T <: Flux.Chain} <: CCChain
    chain::T
end

# -----------------------------------------------------------------------------
# ALTERNATING CHAIN CONSTRUCTORS
# -----------------------------------------------------------------------------

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
                # sigmoid_fast,
                opts["post_synaptic"] ? opts["middle_activation"] : identity,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),

        # CC layer
        Chain(
            MaxPool((2,2)),
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

function get_fuzzy_model(
    n_input::Integer,
    n_class::Integer,
    opts::ModelOpts
)::AlternatingCCChain
    model = Flux.@autosize (n_input,) Chain(
        opts["cc"] ? DeepART.CC() : identity,
        DeepART.Fuzzy(
            _, 40,
            init=opts["init"],
        ),
        Chain(
            # sigmoid_fast,
            opts["middle_activation"],
            opts["cc"] ? DeepART.CC() : identity,
        ),
        DeepART.Fuzzy(
            _, 20,
            init=opts["init"],
        ),

        # LAST LAYER
        Chain(
            # identity,
            # sigmoid_fast,
            opts["middle_activation"],
        ),
        Chain(
            # sigmoid_fast,
            Dense(
                _, n_class,
                # sigmoid_fast,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=opts["bias"],
            ),
        ),
    )
    return AlternatingCCChain(model)
end

function get_dense_model(
    n_input::Integer,
    n_class::Integer,
    opts::ModelOpts,
)::AlternatingCCChain
    model = Flux.@autosize (n_input,) Chain(
        Chain(opts["cc"] ? DeepART.CC() : identity,),
        Dense(_, 64,
            bias=opts["bias"],
            init=opts["init"],
        ),

        Chain(
            # sigmoid_fast,
            opts["middle_activation"],
            opts["cc"] ? DeepART.CC() : identity,
        ),
        Dense(_, 32,
            bias=opts["bias"],
            init=opts["init"],
        ),

        # Chain(sigmoid_fast, DeepART.CC()),
        # Dense(_, 32, bias=bias),

        # LAST LAYER
        Chain(
            # identity
            # sigmoid_fast,
            opts["middle_activation"],
        ),
        Chain(
            Dense(
                _, n_class,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=opts["bias"],
                # init=Flux.identity_init,
                # init=opts["init"],
            ),
        ),
    )
    return AlternatingCCChain(model)
end

# -----------------------------------------------------------------------------
# GROUPED CHAIN LAYERS
# -----------------------------------------------------------------------------

function get_dense_deepart_layer(
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

    return Flux.@autosize (n_in,) Chain(
        Chain(
            # RandomTransform(_, 16, opts),
            # first_layer ? identity : sigmoid_fast,
            # first_layer ? identity : opts["middle_activation"],
            first_activation,
            opts["cc"] ? DeepART.CC() : identity,
        ),
        Dense(
            _, n_out,
            opts["post_synaptic"] ? opts["middle_activation"] : identity,
            bias = opts["bias"],
            init = opts["init"],
        ),
    )
end

struct RandomTransform
    chain::Dense
end

function RandomTransform(
    n_in::Integer,
    n_out::Integer,
    opts::ModelOpts,
)
    return RandomTransform(
        Dense(
            n_in,
            n_out,
            # sigmoid_fast,
            opts["middle_activation"],
            bias=false
        )
    )
end

function (m::RandomTransform)(x)
    return m.chain(x)
end

Flux.@layer RandomTransform

Flux.trainable(m::RandomTransform) = ()

function get_fuzzy_deepart_layer(
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

    return Flux.@autosize (n_in,) Chain(
        Chain(
            # RandomTransform(_, 8),
            # first_layer ? identity : sigmoid_fast,
            # first_layer ? identity : opts["middle_activation"],
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

function get_widrow_hoff_layer(
    n_in::Integer,
    n_out::Integer,
    opts::ModelOpts,
)
    return Flux.@autosize (n_in,) Chain(
        Chain(
            # identity
            # sigmoid_fast,
            opts["middle_activation"],
        ),
        Dense(
            _, n_out,
            bias = opts["bias"],
            init = opts["init"],
        ),
    )
end

# -----------------------------------------------------------------------------
# GROUPED CHAIN CONSTRUCTORS
# -----------------------------------------------------------------------------

function get_dense_groupedccchain(
    n_input::Integer,
    n_class::Integer,
    opts::ModelOpts,
)::GroupedCCChain
    model = Chain(
        get_dense_deepart_layer(n_input, 64, opts, first_layer=true),
        get_dense_deepart_layer(64, 32, opts),
        get_widrow_hoff_layer(32, n_class, opts)
    )
    return GroupedCCChain(model)
end

function get_fuzzy_groupedccchain(
    n_input::Integer,
    n_class::Integer,
    opts::ModelOpts,
)::GroupedCCChain
    model = Chain(
        get_fuzzy_deepart_layer(n_input, 64, opts, first_layer=true),
        get_fuzzy_deepart_layer(64, 32, opts),
        get_widrow_hoff_layer(32, n_class, opts)
    )
    return GroupedCCChain(model)
end

function get_spec_dense_groupedccchain(
    # n_neurons::Vector{Int},
    n_input::Integer,
    n_class::Integer,
    opts::ModelOpts,
)::GroupedCCChain

    n_neurons = [n_input, opts["n_neurons"]..., n_class]

    n_layers = length(n_neurons)

    model = Chain(
        # Input layer
        get_dense_deepart_layer(n_neurons[1], n_neurons[2], opts, first_layer=true),
        # Hidden layers
        (get_dense_deepart_layer(n_neurons[ix], n_neurons[ix + 1], opts)
        for ix = 2:n_layers - 2)...,
        # Output layer
        get_widrow_hoff_layer(n_neurons[n_layers-1], n_neurons[n_layers], opts)
    )
    return GroupedCCChain(model)
end

function get_spec_fuzzy_groupedccchain(
    n_neurons::Vector{Int},
    opts::ModelOpts,
)::GroupedCCChain

    n_layers = length(n_neurons)

    model = Chain(
        get_fuzzy_deepart_layer(n_neurons[1], n_neurons[2], opts, first_layer=true),
        (get_fuzzy_deepart_layer(n_neurons[ix], n_neurons[ix + 1], opts)
        for ix = 2:n_layers - 2)...,
        get_widrow_hoff_layer(n_neurons[n_layers-1], n_neurons[n_layers], opts)
    )
    return GroupedCCChain(model)
end

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
    conv_model = Flux.@autosize (size_tuple,) Chain(
        # get_conv_layer(, 8, (3, 3), opts, first_layer=true),
        Chain(
            Chain(
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                (3, 3), _ => 8,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                MaxPool((2,2)),
                # sigmoid_fast,
                opts["middle_activation"],
                opts["cc"] ? DeepART.CCConv() : identity,
            ),
            Conv(
                (5,5), _ => 16,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        Chain(
            Chain(
                Flux.AdaptiveMaxPool((4, 4)),
                Flux.flatten,
                # sigmoid_fast,
                opts["middle_activation"],
                opts["cc"] ? DeepART.CC() : identity,
            ),
            Dense(_, 32,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
        # Last layers
        Chain(
            Chain(
                # identity,
                # sigmoid_fast,
                opts["middle_activation"],
            ),
            Chain(
                Dense(
                    _, head_dim,
                    opts["final_sigmoid"] ? sigmoid_fast : identity,
                    bias=opts["bias"],
                ),
                vec,
            ),
        ),
    )

    return GroupedCCChain(conv_model)
end

# -----------------------------------------------------------------------------
# HIGH-LEVEL CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Dispatcher for model construction.
"""
const MODEL_MAP = Dict(
    "fuzzy" => get_fuzzy_model,
    "conv" => get_conv_model,
    "dense" => get_dense_model,
    "dense_new" => get_dense_groupedccchain,
    "fuzzy_new" => get_fuzzy_groupedccchain,
    "conv_new" => get_inc_conv_model,
    "dense_spec" => get_spec_dense_groupedccchain,
)

function construct_model(
    data::DeepART.DataSplit,
    opts::ModelOpts,
)
    # Sanitize model option
    if !(opts["model"] in keys(MODEL_MAP))
        error("Invalid model type")
    end

    # Get the shape of the dataset
    dev_x, _ = data.train[1]
    n_input = size(dev_x)[1]
    n_class = length(unique(data.train.y))

    # If the convolutional model is selected, create a convolution input tuple
    local_input_size = if opts["model"] in ["conv", "conv_new"]
        (size(data.train.x)[1:3]..., 1)
    else
        n_input
    end

    # Construct the model
    # model = if opts["model_spec"] == "dense_spec"
    #     get_spec_dense_groupedccchain(
    #         [local_input_size, opts["n_neurons"]..., n_class],
    #         opts,
    #     )
    # else
    #     MODEL_MAP[opts["model"]](
    #         local_input_size,
    #         n_class,
    #         opts,
    #     )
    # end

    model = MODEL_MAP[opts["model"]](
        local_input_size,
        n_class,
        opts,
    )

    # Enforce positive weights if necessary
    if opts["positive_weights"]
        ps = Flux.params(model)
        for p in ps
            p .= abs.(p)
            p .= p ./ maximum(p)
        end
    end

    return model
end

# -----------------------------------------------------------------------------
# CHAIN BEHAVIOR
# -----------------------------------------------------------------------------

function get_weights(model::CCChain)
    return Flux.params(model.chain)
end

function get_activations(model::CCChain, x)
    return Flux.activations(model.chain, x)
end

function get_incremental_activations(
    chain::GroupedCCChain,
    x,
)
    # params
    n_layers = length(chain.chain)

    ins = []
    outs = []

    for ix = 1:n_layers
        pre_input = (ix == 1) ? x : outs[end]
        local_acts = Flux.activations(chain.chain[ix], pre_input)
        push!(ins, local_acts[1])
        push!(outs, local_acts[2])
    end
    return ins, outs
end
