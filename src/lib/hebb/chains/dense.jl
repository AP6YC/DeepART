"""
    dense.jl

# Desription
Definitions for dense models.
"""

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
