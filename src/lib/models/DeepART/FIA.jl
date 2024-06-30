"""
    FIA.jl

# Description
An implementation of a fully deep instar learning network (Fullly Instar ART).
This is not a good name for the model, but I couldn't think of a better one.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Options container for a [`FIA`](@ref) module.
"""
@with_kw struct opts_FIA @deftype Float32
# @with_kw struct opts_FIA{R<:Real} @deftype R
    """
    The vigilance parameter of the [`FIA`](@ref) module, rho ∈ (0.0, 1.0].
    """
    rho = 0.6; @assert rho >= 0.0 && rho <= 1.0

    """
    Instar learning rate.
    """
    eta = 0.1

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-3; @assert alpha > 0.0

    """
    Deep model learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0

    """
    Flag to use an uncommitted node when learning.

    If true, new weights are created with ones(dim) and learn on the complement-coded sample.
    If false, fast-committing is used where the new weight is simply the complement-coded sample.
    """
    uncommitted::Bool = false

    """
    Flag for pushing the models to the GPU.
    """
    gpu::Bool = false

    """
    Update method ∈ ["art", "instar"].
    """
    update::String = "art"

    """
    Soft WTA update rule flag.
    """
    softwta::Bool = false

    """
    Flag for the use of a leader neuron, which negates the use of the SFAM head.
    """
    leader::Bool=false
end

"""
Stateful information of an FIA model.
"""
mutable struct FIA{T <: Flux.Chain} <: DeepARTModule
    """
    The shared model.
    """
    model::T

    """
    An [`opts_FIA`](@ref) options container.
    """
    opts::opts_FIA

    """
    The statistics dictionary for logging.
    """
    stats::ARTStats
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Constructor for a [`FIA`](@ref) taking a [`opts_FIA`](@ref) for construction options.

# Arguments
- `opts::opts_FIA`: the [`opts_FIA`](@ref) that specifies the construction options.
"""
function FIA(
    model,
    opts::opts_FIA
)
    if opts.gpu
        model = gpu(model)
    end

    # Construct and return the field
    return FIA(
        model,
        opts,
        build_art_stats(),
    )
end

"""
Keyword argument constructor for a [`FIA`](@ref) module passing the keyword arguments to the [`opts_FIA`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function FIA(
    model;
    kwargs...
)
    # Create the options from the keyword arguments
    opts = opts_FIA(;kwargs...)

    # Instantiate and return a constructed module
    return FIA(
        model,
        opts,
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

function train!(
    art::FIA,
    x;
    y::Integer=0,
)
    # Compute the activations
    # acts = Flux.activations(art.model, x)
    acts = learn_model(art, x, y=y)
    out = vec(cpu(acts[end]))

    n_out = length(out)

    # y_hat = Int(argmax(out))

    y_hat = Int(argmax(out[1 : Int(n_out / 2)]))

    return y_hat
end

# COMMON DOC: FuzzyART incremental classification method
function classify(
    art::FIA,
    x::RealArray;
    preprocessed::Bool=false,
    get_bmu::Bool=true,
)

    # acts = Flux.activations(art.model, x)
    out = vec(cpu(art.model(x)))

    n_out = length(out)

    # y_hat = Int(argmax(out))
    y_hat = Int(argmax(out[1: Int(n_out / 2)]))

    # Return the inferred label
    return y_hat
end

# -----------------------------------------------------------------------------
# EXPERIMENTAL
# -----------------------------------------------------------------------------

"""
Specific weight update rule for the deep model component of a [`FIA`](@ref).
"""
function learn_model(
    art::FIA,
    xf::RealArray;
    y::Integer=0,
)
    # Extract the weights and compute the activations
    weights = Flux.params(art.model)
    acts = Flux.activations(art.model, xf)
    n_layers = length(weights)

    # Index the input and output activations
    ins = [acts[jx] for jx = 1:2:(n_layers*2)]
    outs = [acts[jx] for jx = 2:2:(n_layers*2)]

    # FIA modification

    # Get the length
    # n_out = length(outs[end])
    # half_out = Int(n_out / 2)

    # Clear the output
    # outs[end][:] .= zero(Float32)
    # outs[end][:] .= -one(Float32)
    # Set the target index high
    # outs[end][y] = one(Float32)
    # Set the complement section high
    # outs[end][half_out + 1 : end] .= one(Float32)
    # @info outs[end]
    for ix = 1:n_layers
        if art.opts.update == "art"
            # If the layer is a convolution
            if ndims(weights[ix]) == 4
                full_size = size(weights[ix])
                n_kernels = full_size[4]
                kernel_shape = full_size[1:3]

                unfolded = Flux.NNlib.unfold(ins[ix], full_size)
                local_in = reshape(
                    mean(
                        reshape(unfolded, :, kernel_shape...),
                        dims=1,
                    ),
                    :
                )

                # Get the averaged and reshaped local output
                local_out = reshape(mean(outs[ix], dims=(1, 2)), n_kernels)

                # Reshape the weights to be (n_kernels, n_features)
                local_weight = reshape(weights[ix], :, n_kernels)'

                # Get the local learning parameter beta
                local_beta = get_beta(art, local_out)
                # @info local_beta
                @debug "sizes going into conv: $(size(local_in)) $(size(local_weight)) $(size(local_beta))"
                result = DeepART.art_learn_cast(
                    local_in,
                    local_weight,
                    local_beta,
                )
                @debug "Conv before: \t$(sum(result .- local_weight))"

                local_weight .= result

                @debug "Conv after: \t$(sum(result .- local_weight))"
                # @debug result[1]
                # @debug local_weight[1]
                # @debug "types: " typeof(result) typeof(local_weight) typeof(weights[ix]) typeof(local_in) typeof(local_beta)

                # local_weight .= DeepART.art_learn_cast(
                #     local_in,
                #     local_weight,
                #     local_beta,
                # )
            else
                @debug "sizes going into dense: $(size(vec(ins[ix]))) $(size(weights[ix])) $(size(get_beta(art, outs[ix])))"
                result = DeepART.art_learn_cast(
                    vec(ins[ix]),
                    # ins[ix],
                    weights[ix],
                    get_beta(art, outs[ix]),
                )
                # @info result
                @debug "Dense before: \t$(sum(result - weights[ix]))"
                weights[ix] .= result
                @debug "Dense after: \t$(sum(result - weights[ix]))"

                # weights[ix] .= DeepART.art_learn_cast(
                #     ins[ix],
                #     weights[ix],
                #     get_beta(art, outs[ix]),
                # )
            end
        elseif art.opts.update == "instar"
            weights[ix] .+= DeepART.instar(
                ins[ix],
                outs[ix],
                weights[ix],
                art.opts.beta,
            )
        else
            error("Invalid update method: $(art.opts.update)")
        end
    end

    # @info weights

    # for ix in eachindex(trainables)
        # weights[ix] .+= DeepART.instar(inputs[ix], acts[ix], weights[ix], eta)
    # end
    # DeepART.instar(xf, acts, model, 0.0001)

    return acts
end