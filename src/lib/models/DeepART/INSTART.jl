"""
    INSTART.jl

# Description
An implementation of a deep instar learning network.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Options container for a [`INSTART`](@ref) module.
"""
@with_kw struct opts_INSTART
    """
    The vigilance parameter of the [`INSTART`](@ref) module, rho ∈ (0.0, 1.0].
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
    List of the layer entries with trainable weights.
    """
    trainables::Vector{Int} = [1, 3, 5]

    """
    The activations indices to use.
    """
    activations::Vector{Int} = [1, 3, 5]

    """
    Update method ∈ ["art", "instar"].
    """
    update::String = "art"

    """
    Soft WTA update rule flag.
    """
    softwta::Bool = false

    """
    Head layer type ∈ ["fuzzy", "hypersphere"].
    """
    head::String = "fuzzy"
    # """
    # Flux activation function.
    # """
    # activation_function::Function = relu

    # """
    # Flag for if the model is convolutional.
    # """
    # conv::Bool = false
end

"""
Stateful information of an INSTART model.
"""
mutable struct INSTART{T <: Flux.Chain, U <: Flux.Chain}
    """
    The shared model.
    """
    model::T

    """
    The heads.
    """
    heads::Vector{U}

    """
    An [`opts_INSTART`](@ref) options container.
    """
    opts::opts_INSTART

    """
    Incremental list of labels corresponding to each F2 node, self-prescribed or supervised.
    """
    labels::Vector{Int}

    """
    Activation values for every weight for a given sample.
    """
    T::Vector{Float}

    """
    Match values for every weight for a given sample.
    """
    M::Vector{Float}

    """
    Number of weights associated with each category.
    """
    n_instance::Vector{Int}

    """
    Number of category weights (F2 nodes).
    """
    n_categories::Int
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Keyword argument constructor for a [`INSTART`](@ref) module passing the keyword arguments to the [`opts_INSTART`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function INSTART(model; kwargs...)
    # Create the options from the keyword arguments
    opts = opts_INSTART(;kwargs...)

    # Instantiate and return a constructed module
    return INSTART(
        model,
        opts,
    )
end

function get_hypersphere_head(head_dim, weights=nothing)
    head = if isnothing(weights)
        Flux.@autosize (head_dim,) Chain(
            Flux.identity,
            DeepART.HypersphereLayer(_),
        )
    else
        Flux.@autosize (head_dim,) Chain(
            Flux.identity,
            DeepART.HypersphereLayer(
                weights,
            ),
        )
    end
    return head
end

function get_fuzzy_head(head_dim, weights=nothing)
    head = if isnothing(weights)
        Flux.@autosize (head_dim,) Chain(
            DeepART.CC(),
            DeepART.SingleFuzzy(_),
        )
    else
        Flux.@autosize (head_dim,) Chain(
            DeepART.CC(),
            DeepART.SingleFuzzy(
                DeepART.complement_code(weights),
            ),
        )
    end
    return head
end

"""
Constructs an INSTART head node.
"""
function get_head(opts, weights=nothing)
    # Dense(_, 128, sigmoid),
    # DeepART.Fuzzy(_, 1),
    # DeepART.SingleFuzzy(_),
    head = if opts.head == "fuzzy"
        get_fuzzy_head(opts.head_dim, weights)
    elseif opts.head == "hypersphere"
        get_hypersphere_head(opts.head_dim, weights)
    else
        error("Invalid head type: $(opts.head)")
    end

    return head
end

"""
Constructor for a [`INSTART`](@ref) taking a [`opts_INSTART`](@ref) for construction options.

# Arguments
- `opts::opts_INSTART`: the [`opts_INSTART`](@ref) that specifies the construction options.
"""
function INSTART(
    model,
    opts::opts_INSTART
)
    # Create the heads
    heads = Vector{Flux.Chain}()

    opts.gpu && model |> gpu
    opts.gpu && heads |> gpu

    # Construct and return the field
    return INSTART(
        model,
        heads,
        opts,
        Vector{Int}(undef, 0),
        Vector{Float}(undef, 0),        # T
        Vector{Float}(undef, 0),        # M
        Vector{Int}(undef, 0),          # n_instance
        0,                              # n_categories
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

# function learn_model(model, xf)
function learn_model(art::INSTART, xf)
    # eta = 0.1
    weights = Flux.params(art.model)
    acts = Flux.activations(art.model, xf)

    # n_layers = length(weights)
    # trainables = [weights[jx] for jx = 1:2:n_layers]
    # ins = [acts[jx] for jx = 1:2:n_layers]
    # outs = [acts[jx] for jx = 2:2:n_layers]

    trainables = [weights[jx] for jx in art.opts.trainables]
    # ins = [acts[jx] for jx = art.opts.trainables]
    ins = [acts[jx] for jx = art.opts.activations]
    outs = [acts[jx] for jx = art.opts.activations .+ 1]

    for ix in eachindex(ins)
        if art.opts.update == "art"
            # trainables[ix] .= DeepART.art_learn_cast(ins[ix], trainables[ix], art.opts.beta)
            local_beta = if art.opts.softwta == true
                # art.opts.beta .* (1 .- outs[ix])
                art.opts.beta .* Flux.softmax(outs[ix])
                # art.opts.beta .* (1 .- Flux.softmax(outs[ix]))
            else
                art.opts.beta
            end
            trainables[ix] .= DeepART.art_learn_cast(ins[ix], trainables[ix], local_beta)
        elseif art.opts.update == "instar"
            trainables[ix] .+= DeepART.instar(ins[ix], outs[ix], trainables[ix], art.opts.beta)
        else
            error("Invalid update method: $(art.opts.update)")
        end
    end

    # for ix in eachindex(trainables)
        # weights[ix] .+= DeepART.instar(inputs[ix], acts[ix], weights[ix], eta)
    # end
    # DeepART.instar(xf, acts, model, 0.0001)

    return acts
end

function art_learn_basic(x, W, beta)
    return beta .* min.(x, W) + W .* (1.0 .- beta)
end

function art_learn_cast(x, W, beta)
    Wy, Wx = size(W)
    _x = repeat(x', Wy, 1)
    _beta = if !isempty(size(beta))
        repeat(beta, 1, Wx)
    else
        beta
    end
    # @info size(_beta)
    # @info size(_x)
    # @info size(W)
    # return beta * min.(_x, W) + W * (1.0 - beta)
    return art_learn_basic(_x, W, _beta)
end

function art_learn_head(xf, head, beta)
    W = Flux.params(head[2])[1]
    _x = head[1](xf)
    # W .= beta * min.(_x, W) + W * (1.0 - beta)
    W .= art_learn_basic(_x, W, beta)
    return
end

function add_node!(
    art::INSTART,
    x::RealArray,
)
    # push!(art.heads, get_head(art.opts.head_dim, x))
    push!(art.heads, get_head(art.opts, x))
    return
end

function create_category!(
    art::INSTART,
    x::RealArray,
    y::Integer,
)
    # Increment the number of categories
    art.n_categories += 1

    # Increment number of samples associated with new category
    push!(art.n_instance, 1)

    # Update the model
    # acts = learn_model(art.model, x)
    acts = Flux.activations(art.model, x)

    # Then add a new head
    # add_node!(art, acts[end])

    # If we use an uncommitted node
    if art.opts.uncommitted
        # Add a new weight of ones
        # append!(art.W, ones(art.config.dim_comp, 1))
        add_node!(art, ones(Float32, art.opts.head_dim))
        # Learn the uncommitted node on the sample
        # learn!(art, x, art.n_categories)
        art_learn_head(acts[end], art.heads[art.n_categories], art.opts.beta)
    else
        # Fast commit the sample
        # append!(art.W, x)
        add_node!(art, acts[end])
    end

    # Add the label for the category
    push!(art.labels, y)

    # Update the head
    # art_learn_head(x, heads[bmu], art.opts.beta)
    # art_learn_head(acts[end], art.heads[art.n_categories], art.opts.beta)
    return
end

function initialize!(
    art::INSTART,
    x::RealArray;
    y::Integer=0,
)
    # Set the threshold
    # set_threshold!(art)
    # Initialize the feature dimension of the weights
    # art.W = ARTMatrix{Float}(undef, art.config.dim_comp, 0)
    # Set the label to either the supervised label or 1 if unsupervised
    label = !iszero(y) ? y : 1

    # Create a category with the given label
    # create_category!(art, x, label)
    create_category!(art, x, label)

    return
end

function train!(
    art::INSTART,
    x;
    y::Integer=0,
)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    if isempty(art.heads)
        y_hat = supervised ? y : 1
        initialize!(art, x, y=y_hat)
        return y_hat
    end

    acts = Flux.activations(art.model, x)
    MT = [head(acts[end]) for head in art.heads]
    M = [m[1] for m in MT]
    T = [m[2] for m in MT]

    # Sort activation function values in descending order
    index = sortperm(T, rev=true)

    # Initialize mismatch as true
    mismatch_flag = true

    # Loop over all categories
    for j = 1:art.n_categories
        # Best matching unit
        bmu = index[j]
        # Vigilance check - pass
        if M[bmu] >= art.opts.rho
            # If supervised and the label differed, force mismatch
            if supervised && (art.labels[bmu] != y)
                break
            end
            # Update the model
            # acts = learn_model(art.model, x)
            acts = learn_model(art, x)

            # Update the head
            art_learn_head(acts[end], art.heads[bmu], art.opts.beta)

            # Increment the instance counting
            art.n_instance[bmu] += 1

            # Save the output label for the sample
            y_hat = art.labels[bmu]

            # No mismatch
            mismatch_flag = false
            break
        end
    end

    # If there was no resonant category, make a new one
    if mismatch_flag
        # Keep the bmu as the top activation despite creating a new category
        bmu = index[1]
        # Get the correct label for the new category
        y_hat = supervised ? y : art.n_categories + 1
        # Create a new category
        create_category!(art, x, y_hat)
    end

    # return T, M
    return y_hat
end

# COMMON DOC: FuzzyART incremental classification method
function classify(
    art::INSTART,
    x::RealVector;
    preprocessed::Bool=false,
    get_bmu::Bool=false,
)
    # Compute activation and match functions
    # activation_match!(art, x)
    # M = [basic_activation(art, f1, f2[ix]) for ix in eachindex(f2)]
    # T = [basic_match(art, f1, f2[ix]) for ix in eachindex(f2)]
    acts = Flux.activations(art.model, x)
    MT = [head(acts[end]) for head in art.heads]
    M = [m[1] for m in MT]
    T = [m[2] for m in MT]

    # Sort activation function values in descending order
    index = sortperm(T, rev=true)

    # Default is mismatch
    mismatch_flag = true
    y_hat = -1

    # Iterate over all categories
    for jx in 1:art.n_categories
        # Set the best matching unit
        bmu = index[jx]

        # Vigilance check - pass
        if M[bmu] >= art.opts.rho
            # Current winner
            y_hat = art.labels[bmu]
            mismatch_flag = false
            break
        end
    end

    # If we did not find a match
    if mismatch_flag
        # Report either the best matching unit or the mismatch label -1
        bmu = index[1]

        # Report either the best matching unit or the mismatch label -1
        y_hat = get_bmu ? art.labels[bmu] : -1
    end

    # Update the stored match and activation values
    # log_art_stats!(art, bmu, mismatch_flag)

    # Return the inferred label
    return y_hat
end

# -----------------------------------------------------------------------------
# EXPERIMENTAL
# -----------------------------------------------------------------------------

function mergeart(art)
    weights = [head[2].weight for head in art.heads]
    la = ART.FuzzyART(
        rho=0.99
    )
    la.config = ART.DataConfig(0, 1, art.opts.head_dim)

    for weight in weights
        ART.train!(la, weight)
    end
    return la
end

function trimart!(art)
    inds = findall(x -> x == 1, art.n_instance)
    deleteat!(art.heads, inds)
    deleteat!(art.labels, inds)
    art.n_categories = length(art.heads)
    return
end
