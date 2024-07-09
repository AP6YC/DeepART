
const ModelOpts = Dict{String, Any}

function get_conv_model(
    size_tuple::Tuple,
    head_dim::Integer,
    opts;
    # bias::Bool = false,
    # final_sigmoid::Bool = false,
)
    conv_model = Flux.@autosize (size_tuple,) Chain(
        # CC layer
        Chain(DeepART.CCConv()),

        # Conv layer
        Chain(
            Conv(
                (3, 3), _ => 8,
                # sigmoid_fast,
                # bias=false,
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
        Chain(identity),
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
        # DeepART.CC(),
    )
    return conv_model
end

function get_fuzzy_model(
    n_input,
    n_class,
    opts;
    # bias=false,
    # final_sigmoid=false,
)
    model = Flux.@autosize (n_input,) Chain(
        DeepART.CC(),
        DeepART.Fuzzy(_, 40,),
        Chain(sigmoid_fast, DeepART.CC()),
        DeepART.Fuzzy(_, 20,
            init=opts["init"],
        ),
        Chain(sigmoid_fast, DeepART.CC()),
        DeepART.Fuzzy(_, 20),
        Chain(identity),

        # LAST LAYER
        Chain(
            sigmoid_fast,
            Dense(
                _, n_class,
                # sigmoid_fast,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=opts["bias"],
            ),
        ),
    )
    return model
end

function get_model(
    n_input,
    n_class,
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
            # sigmoid_fast,
            Dense(
                _, n_class,
                # sigmoid_fast,
                opts["final_sigmoid"] ? sigmoid_fast : identity,
                bias=opts["bias"],
                # init=Flux.identity_init,
                # init=opts["init"],
            ),
        ),
    )
    return model
end



function get_dense_deepart_layer(
    n_in::Integer,
    n_out::Integer,
    opts;
    first_layer::Bool = false,
)
    return Flux.@autosize (n_in,) Chain(
        Chain(
            first_layer ? identity : sigmoid_fast,
            DeepART.CC(),
        ),
        Dense(
            _, n_out,
            # bias=bias,
            bias = opts["bias"],
            # init=Flux.identity_init
            # init=rand,
            init=opts["init"],
        ),
    )
end

function get_widrow_hoff_layer(
    n_in::Integer,
    n_out::Integer,
    opts;
    # bias::Bool = false,
)
    return Flux.@autosize (n_in,) Chain(
        Chain(
            # identity
            sigmoid_fast,
        ),
        Dense(
            _, n_out,
            bias=opts["bias"],
            # init=Flux.identity_init
            # init=rand,
            init=opts["init"],
        ),
    )
end

function get_new_dense(
    n_input,
    n_class,
    opts,
)
    return Chain(
        get_dense_deepart_layer(n_input, 64, opts, first_layer=true),
        get_dense_deepart_layer(64, 32, opts),
        get_widrow_hoff_layer(32, n_class, opts)
    )
end



const MODEL_MAP = Dict(
    "fuzzy" => get_fuzzy_model,
    "conv" => get_conv_model,
    "dense" => get_model,
    "dense_new" => get_new_dense,
)

function construct_model(
    data,
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
    local_input_size = if opts["model"] == "conv"
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