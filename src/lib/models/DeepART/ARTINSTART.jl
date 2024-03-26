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
        epsilon=1e-4,
    )
    head.config = ART.DataConfig(0.0, 1.0, opts.head_dim)

    if opts.gpu
        model = gpu(model)
    end

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
    acts = learn_model(art, x, y=y)

    # Train the head
    if art.opts.leader
        # y_hat = ART.train!(art.head, acts[end], y)
        y_hat = argmax(acts[end])
    else
        # head_input = if art.opts.gpu
        #     vec(cpu(acts[end]))
        # else
        #     vec(acts[end])
        # end
        head_input = vec(cpu(acts[end]))
        # y_hat = ART.train!(art.head, vec(cpu(acts[end])), y)
        y_hat = ART.train!(art.head, head_input, y)
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

    # acts = Flux.activations(art.model, x)
    out = art.model(x)

    if art.opts.leader
        # @info size(acts[end])
        y_hat = argmax(acts[end])
    else
        # head_input = if art.opts.gpu
        #     vec(cpu(acts[end]))
        # else
        #     acts[end]
        # end
        # head_input = vec(cpu(acts[end]))
        head_input = vec(cpu(out))
        # y_hat = ART.classify(art.head, acts[end], get_bmu=get_bmu)
        y_hat = ART.classify(art.head, head_input, get_bmu=get_bmu)
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
