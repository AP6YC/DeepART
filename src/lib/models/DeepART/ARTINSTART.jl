"""
    ARTINSTART.jl

# Description
An implementation of a deep instar learning network with an existing ART module on top.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Options container for a [`ARTINSTART`](@ref) module.
"""
@with_kw struct opts_ARTINSTART
    """
    The vigilance parameter of the [`ARTINSTART`](@ref) module, rho ∈ (0.0, 1.0].
    """
    rho::Float = 0.6; @assert rho > 0.0 && rho <= 1.0

    """
    Instar learning rate.
    """
    eta::Float = 0.1

    """
    Choice parameter: alpha > 0.
    """
    alpha = 1e-3; @assert alpha > 0.0

    """
    Learning parameter: beta ∈ (0, 1].
    """
    beta = 1.0
    # beta = 1.0; @assert beta > 0.0 && beta <= 1.0

    """
    Flag to use an uncommitted node when learning.

    If true, new weights are created with ones(dim) and learn on the complement-coded sample.
    If false, fast-committing is used where the new weight is simply the complement-coded sample.
    """
    uncommitted::Bool = false

    """
    The dimension of the interaction field.
    """
    head_dim::Int = 128

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
Stateful information of an ARTINSTART model.
"""
mutable struct ARTINSTART{T <: Flux.Chain, U <: ART.ARTModule} <: DeepARTModule
    """
    The shared model.
    """
    model::T

    """
    The heads.
    """
    head::U

    """
    An [`opts_ARTINSTART`](@ref) options container.
    """
    opts::opts_ARTINSTART

    """
    Number of category weights (F2 nodes).
    """
    n_categories::Int

    """
    The statistics dictionary for logging.
    """
    stats::ARTStats
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Constructor for a [`ARTINSTART`](@ref) taking a [`opts_ARTINSTART`](@ref) for construction options.

# Arguments
- `opts::opts_ARTINSTART`: the [`opts_ARTINSTART`](@ref) that specifies the construction options.
"""
function ARTINSTART(
    model,
    opts::opts_ARTINSTART
)
    # Create the head
    head = ART.SFAM(
        rho=opts.rho,
    )
    head.config = ART.DataConfig(0.0, 1.0, opts.head_dim)

    opts.gpu && model |> gpu

    # Construct and return the field
    return ARTINSTART(
        model,
        head,
        opts,
        0,                              # n_categories
        build_art_stats(),
    )
end

"""
Keyword argument constructor for a [`ARTINSTART`](@ref) module passing the keyword arguments to the [`opts_ARTINSTART`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function ARTINSTART(model; kwargs...)
    # Create the options from the keyword arguments
    opts = opts_ARTINSTART(;kwargs...)

    # Instantiate and return a constructed module
    return ARTINSTART(
        model,
        opts,
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# function get_beta(art::DeepARTModule, outs::RealArray)
#     local_beta = if art.opts.softwta == true
#         # art.opts.beta .* (1 .- outs[ix])
#         art.opts.beta .* Flux.softmax(
#             outs,
#         )
#         # art.opts.beta .* (1 .- Flux.softmax(outs[ix]))
#     else
#         art.opts.beta
#     end
#     return local_beta
# end

# function learn_model(art::ARTINSTART, xf)
#     weights = Flux.params(art.model)
#     acts = Flux.activations(art.model, xf)

#     n_layers = length(weights)

#     # trainables = weights
#     ins = [acts[jx] for jx = 1:2:(n_layers*2)]
#     outs = [acts[jx] for jx = 2:2:(n_layers*2)]

#     # @info "sizes:" n_layers size(ins) size(outs)

#     # for ix in eachindex(ins)
#     for ix = 1:n_layers
#         # @info "sizes:" size(ins[ix]) size(outs[ix]) size(trainables[ix])
#         if art.opts.update == "art"
#             # trainables[ix] .= DeepART.art_learn_cast(ins[ix], trainables[ix], art.opts.beta)
#             # Get the local learning parameter beta
#             # local_beta = get_beta(art, outs[ix])

#             # If the layer is a convolution
#             if ndims(weights[ix]) == 4
#                 full_size = size(weights[ix])
#                 n_kernels = full_size[4]
#                 kernel_shape = full_size[1:3]

#                 unfolded = Flux.NNlib.unfold(ins[ix], full_size)
#                 local_in = reshape(
#                     mean(
#                         reshape(unfolded, :, kernel_shape...),
#                         dims=1,
#                     ),
#                     :
#                 )

#                 # Get the averaged and reshaped local output
#                 local_out = reshape(mean(outs[ix], dims=(1, 2)), n_kernels)
#                 # Reshape the weights to be (n_kernels, n_features)
#                 local_weight = reshape(weights[ix], :, n_kernels)'
#                 # Get the local learning parameter beta
#                 local_beta = get_beta(art, local_out)
#                 # new_weight = DeepART.art_learn_cast(
#                 #     local_in,
#                 #     local_weight,
#                 #     local_beta,
#                 # )
#                 # @info sum(new_weight .- local_weight)
#                 # local_weight .= new_weight

#                 local_weight .= DeepART.art_learn_cast(
#                     local_in,
#                     local_weight,
#                     local_beta,
#                 )
#             else
#                 local_weight = weights[ix]
#                 local_in = ins[ix]
#                 local_out = outs[ix]
#                 local_beta = get_beta(art, local_out)

#                 local_weight .= DeepART.art_learn_cast(
#                     local_in,
#                     local_weight,
#                     local_beta,
#                 )
#             end

#             # weights[ix] .= DeepART.art_learn_cast(
#             #     ins[ix],
#             #     weights[ix],
#             #     local_beta,
#             # )
#         elseif art.opts.update == "instar"
#             weights[ix] .+= DeepART.instar(
#                 ins[ix],
#                 outs[ix],
#                 weights[ix],
#                 art.opts.beta,
#             )
#         else
#             error("Invalid update method: $(art.opts.update)")
#         end
#     end

#     # for ix in eachindex(trainables)
#         # weights[ix] .+= DeepART.instar(inputs[ix], acts[ix], weights[ix], eta)
#     # end
#     # DeepART.instar(xf, acts, model, 0.0001)

#     return acts
# end

"""
Copies the statistics from the head module to the top of the [`ARTINSTART`](@ref) module.
"""
function copy_stats!(art::ARTINSTART)
    art.stats["bmu"] = art.head.stats["bmu"]
    art.stats["M"] = art.head.stats["M"]
    art.stats["T"] = art.head.stats["T"]
    art.stats["mismatch"] = art.head.stats["mismatch"]
    art.n_categories = art.head.n_categories
end

function train!(
    art::ARTINSTART,
    x;
    y::Integer=0,
)
    # Compute the activations
    # acts = Flux.activations(art.model, x)
    acts = learn_model(art, x)

    # Train the head
    if art.opts.leader
        # y_hat = ART.train!(art.head, acts[end], y)
        y_hat = argmax(acts[end])
    else
        y_hat = ART.train!(art.head, acts[end], y)
    end


    copy_stats!(art)

    return y_hat
end

# COMMON DOC: FuzzyART incremental classification method
function classify(
    art::ARTINSTART,
    x::RealArray;
    preprocessed::Bool=false,
    get_bmu::Bool=true,
)

    acts = Flux.activations(art.model, x)

    if art.opts.leader
    else
        y_hat = ART.classify(art.head, acts[end], get_bmu=get_bmu)
    end

    copy_stats!(art)

    # Return the inferred label
    return y_hat
end

# -----------------------------------------------------------------------------
# EXPERIMENTAL
# -----------------------------------------------------------------------------

# function mergeart(art)
#     weights = [head[2].weight for head in art.heads]
#     la = ART.FuzzyART(
#         rho=0.99
#     )
#     la.config = ART.DataConfig(0, 1, art.opts.head_dim)

#     for weight in weights
#         ART.train!(la, weight)
#     end
#     return la
# end

# function trimart!(art)
#     inds = findall(x -> x == 1, art.n_instance)
#     deleteat!(art.heads, inds)
#     deleteat!(art.labels, inds)
#     art.n_categories = length(art.heads)
#     return
# end
