"""
    models.jl

# Description
Collection of model definitions for the Hebb module.
"""

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

const ModelOpts = Dict{String, Any}

# -----------------------------------------------------------------------------
# MODEL CONSTRUCTORS
# -----------------------------------------------------------------------------

function get_conv_model(
    size_tuple::Tuple,
    head_dim::Integer,
    opts::ModelOpts
)
    conv_model = Flux.@autosize (size_tuple,) Chain(
        # CC layer
        Chain(
            DeepART.CCConv()
        ),

        # Conv layer
        Chain(
            Conv(
                (3, 3), _ => 8,
                # sigmoid_fast,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),

        # CC layer
        Chain(
            MaxPool((2,2)),
            sigmoid_fast,
            DeepART.CCConv(),
        ),

        # Conv layer
        Chain(
            Conv(
                (5,5), _ => 16,
                # sigmoid_fast,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),

        # CC layer
        Chain(
            Flux.AdaptiveMaxPool((4, 4)),
            Flux.flatten,
            sigmoid_fast,
            DeepART.CC(),
        ),

        # Dense layer
        Dense(_, 32,
            bias=opts["bias"],
            init=opts["init"],
        ),

        # Last layers
        Chain(
            # identity,
            sigmoid_fast,
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
)
    model = Flux.@autosize (n_input,) Chain(
        DeepART.CC(),
        DeepART.Fuzzy(
            _, 40,
            init=opts["init"],
        ),
        Chain(
            sigmoid_fast,
            DeepART.CC()
        ),
        DeepART.Fuzzy(
            _, 20,
            init=opts["init"],
        ),

        # LAST LAYER
        Chain(
            # identity,
            sigmoid_fast,
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
)
    model = Flux.@autosize (n_input,) Chain(
        Chain(DeepART.CC()),
        Dense(_, 64,
            bias=opts["bias"],
            init=opts["init"],
        ),

        Chain(sigmoid_fast, DeepART.CC()),
        Dense(_, 32,
            bias=opts["bias"],
            init=opts["init"],
        ),

        # Chain(sigmoid_fast, DeepART.CC()),
        # Dense(_, 32, bias=bias),

        # LAST LAYER
        Chain(
            # identity
            sigmoid_fast,
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


function get_dense_deepart_layer(
    n_in::Integer,
    n_out::Integer,
    opts::ModelOpts;
    first_layer::Bool = false,
)
    return Flux.@autosize (n_in,) Chain(
        Chain(
            first_layer ? identity : sigmoid_fast,
            DeepART.CC(),
        ),
        Dense(
            _, n_out,
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
)
    return RandomTransform(Dense(n_in, n_out, bias=false))
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
    return Flux.@autosize (n_in,) Chain(
        Chain(
            # RandomTransform(_, 16),
            first_layer ? identity : sigmoid_fast,
            DeepART.CC(),
        ),
        DeepART.Fuzzy(
            _, n_out,
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
            sigmoid_fast,
        ),
        Dense(
            _, n_out,
            bias = opts["bias"],
            init = opts["init"],
        ),
    )
end

function get_new_dense_model(
    n_input::Integer,
    n_class::Integer,
    opts::ModelOpts,
)
    model = Chain(
        get_dense_deepart_layer(n_input, 64, opts, first_layer=true),
        get_dense_deepart_layer(64, 32, opts),
        get_widrow_hoff_layer(32, n_class, opts)
    )
    return GroupedCCChain(model)
end

function get_new_fuzzy_model(
    n_input::Integer,
    n_class::Integer,
    opts::ModelOpts,
)
    model = Chain(
        get_fuzzy_deepart_layer(n_input, 64, opts, first_layer=true),
        get_fuzzy_deepart_layer(64, 32, opts),
        get_widrow_hoff_layer(32, n_class, opts)
    )
    return GroupedCCChain(model)
end

function get_conv_layer(
    n_in::Integer,
    n_out::Integer,
    kernel::Tuple,
    opts::ModelOpts;
    first_layer::Bool = false,
    n_pool::Tuple = (),
)
    return Flux.@autosize (n_in,) Chain(
        # CC layer
        Chain(
            first_layer ? identity :
                Chain(
                    MaxPool(n_pool),
                    sigmoid_fast,
                ),
            DeepART.CCConv()
        ),

        # Conv layer
        Chain(
            Conv(
                kernel, _ => n_out,
                # sigmoid_fast,
                bias=opts["bias"],
                init=opts["init"],
            ),
        ),
    )
end

function get_inc_conv_model(
    size_tuple::Tuple,
    head_dim::Integer,
    opts::ModelOpts
)
    conv_model = Flux.@autosize (size_tuple,) Chain(
        # get_conv_layer(, 8, (3, 3), opts, first_layer=true),
        Chain(
            Chain(
                DeepART.CCConv()
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
                sigmoid_fast,
                DeepART.CCConv(),
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
                sigmoid_fast,
                DeepART.CC(),
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
                sigmoid_fast,
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

const MODEL_MAP = Dict(
    "fuzzy" => get_fuzzy_model,
    "conv" => get_conv_model,
    "dense" => get_dense_model,
    "dense_new" => get_new_dense_model,
    "fuzzy_new" => get_new_fuzzy_model,
    "conv_new" => get_inc_conv_model,
)


# # "model" => "dense",
# # "model" => "small_dense",
# # "model" => "fuzzy",
# # "model" => "conv",
# # "model" => "fuzzy_new",
# # "model" => "dense_new",
# "model" => "conv_new",

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
