
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
