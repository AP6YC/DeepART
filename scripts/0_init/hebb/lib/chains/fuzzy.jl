"""
    fuzzy.jl

# Desription
Definitions for fuzzy-logic-based models.
"""

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
