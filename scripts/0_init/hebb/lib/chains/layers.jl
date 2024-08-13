"""
    layers.jl

# Desription
Definitions for layers for each type of model.
"""

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

    input_dim = opts["bias"] ? n_in + 1 : n_in

    preprocess = if opts["layer_norm"] && !first_layer
        LayerNorm(input_dim, affine=false)
        # Flux.normalise
    else
        identity
    end

    return Flux.@autosize (input_dim,) Chain(
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
            opts["post_synaptic"] ? opts["middle_activation"] : identity,
            # bias = opts["bias"],
            bias = false,
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
            init = opts["init"],
            opts["middle_activation"],
            bias = false
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

    preprocess = if opts["layer_norm"] && !first_layer
        LayerNorm(n_in, affine=false)
        # Flux.normalise
    else
        identity
    end

    return Flux.@autosize (n_in,) Chain(
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

function get_widrow_hoff_layer(
    n_in::Integer,
    n_out::Integer,
    opts::ModelOpts,
)
    # input_dim = opts["bias"] ? n_in + 1 : n_in
    input_dim = opts["final_bias"] ? n_in + 1 : n_in

    first_activation = if opts["post_synaptic"]
        identity
    else
        opts["middle_activation"]
    end

    preprocess = if opts["layer_norm"]
        LayerNorm(input_dim, affine=false)
        # Flux.normalise
    else
        identity
    end

    return Flux.@autosize (input_dim,) Chain(
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
