"""
    DeepHeadART.jl

# Description
TODO
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

"""
Options container for a [`DeepHeadART`](@ref) module.
"""
@with_kw struct opts_DeepHeadART
    """
    The vigilance parameter of the [`DeepHeadART`](@ref) module, rho ∈ (0.0, 1.0].
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
    beta = 1.0; @assert beta > 0.0 && beta <= 1.0

    """
    Simple dense specifier for the F1 layer.
    """
    F1_spec::DenseSpecifier = [2, 5, 3]

    """
    Shared dense specifier for the F2 layer.
    """
    F2_shared::DenseSpecifier = [3, 6, 3]

    """
    Shared dense specifier for the F2 layer.
    """
    F2_heads::DenseSpecifier = [3, 5, 3]

    # """
    # Flux activation function.
    # """
    # activation_function::Function = relu
end

"""
Stateful information of a DeepHeadART module.
"""
mutable struct DeepHeadART{T <: Flux.Chain, U <: Flux.Chain, V <: Flux.Chain} <: ARTModule
    """
    Feature presentation layer.
    """
    F1::T

    """
    Feedback expectancy layer.
    """
    F2::MultiHeadField{U, V}

    """
    An [`opts_DeepHeadART`](@ref) options container.
    """
    opts::opts_DeepHeadART

    """
    Data configuration struct.
    """
    config::DataConfig

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
Keyword argument constructor for a [`DeepHeadART`](@ref) module passing the keyword arguments to the [`opts_DeepHeadART`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function DeepHeadART(;kwargs...)
    # Create the options from the keyword arguments
    opts = opts_DeepHeadART(;kwargs...)

    # Instantiate and return a constructed module
    return DeepHeadART(
        opts,
    )
end

"""
Constructor for a [`DeepHeadART`](@ref) taking a [`opts_DeepHeadART`](@ref) for construction options.

# Arguments
- `opts::opts_DeepHeadART`: the [`opts_DeepHeadART`](@ref) that specifies the construction options.
"""
function DeepHeadART(
    opts::opts_DeepHeadART
)
    # # Create the shared network base
    # shared = get_dense(opts.shared_spec)

    # # Create the heads
    # heads = [get_dense(opts.head_spec) for _ = 1:5]

    F1 = get_dense(opts.F1_spec)
    # F2 = get_dense(opts.F2_spec)
    F2 = MultiHeadField(
        shared_spec=opts.F2_shared,
        head_spec=opts.F2_heads,
    )

    # field_dim = opts.F1_spec[1]
    field_dim = opts.F1_spec[end]

    config = DataConfig(
        -1.0,
        1.0,
        field_dim,
    )

    # Construct and return the field
    return DeepHeadART(
        F1,
        F2,
        opts,
        config,
        Vector{Int}(undef, 0),
        Vector{Float}(undef, 0),        # T
        Vector{Float}(undef, 0),        # M
        # Matrix{Float}(undef, 0, 0),     # W
        Vector{Int}(undef, 0),          # n_instance
        0,                              # n_categories
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Forward pass for a [`DeepHeadART`](@ref) module.

# Arguments
$ARG_DEEPHEADART
$ARG_X
"""
function forward(
    art::DeepHeadART,
    x::RealArray,
)
    # Forward pass on the F1 layer
    f1 = art.F1(x)

    # Use the MultiHeadField forward pass for F2
    f2 = forward(art.F2, f1)

    # Return both forward pass values
    return f1, f2
end

"""
Forward pass for a [`DeepHeadART`](@ref) module with activations.

# Arguments
$ARG_DEEPHEADART
$ARG_X
"""
function multi_activations(
    art::DeepHeadART,
    x::RealArray,
)
    # Get the activations for F1
    f1 = Flux.activations(art.F1, x)

    # Get the activations for the MultiHeadField
    f2 = multi_activations(art.F2, f1[end])
    # f1 = Flux.activations(field.F1, x)
    # f2 = Flux.activations(field.F2, f1[end])

    # Return both activation tuples
    return f1, f2
end

"""
Adds a node to the F2 layer of the [`DeepHeadART`](@ref) module.

# Arguments
$ARG_DEEPHEADART
$ARG_X
"""
function add_node!(
    art::DeepHeadART,
    x::RealArray,
)
    add_node!(art.F2, x)

    return
end

"""
Updates the weights of both the F1 layer and F2 layer (at the index) of the [`DeepHeadART`](@ref) module.

# Arguments
$ARG_DEEPHEADART
- `activations::Tuple`: the activations tuple.
- `index::Integer`: the index of the node to update.
"""
function learn!(
    art::DeepHeadART,
    x::RealArray,
    # activations::Tuple,
    f1a::Tuple,
    f2a::Tuple,
    index::Integer,
)
    # Instar learning for F1
    instar(x, f1a, art.F1, art.opts.eta)

    # F2 shared
    instar(f1a[end], f2a[1], art.F2.shared, art.opts.eta)

    # F2 index
    instar(f2a[1][end], f2a[2][index], art.F2.heads[index], art.opts.eta)

    return
end

# COMMON DOC: create_category! function
# """
# """
function create_category!(
    art::DeepHeadART,
    # x::RealVector,
    x::RealArray,
    # f1a::Tuple,
    # f2a::Tuple,
    y::Integer,
)
    # Increment the number of categories
    art.n_categories += 1

    # Increment number of samples associated with new category
    push!(art.n_instance, 1)

    # # If we use an uncommitted node
    # if art.opts.uncommitted
    #     # Add a new weight of ones
    #     append!(art.W, ones(art.config.dim_comp, 1))
    #     # Learn the uncommitted node on the sample
    #     learn!(art, x, art.n_categories)
    # else
    #     # Fast commit the sample
    #     append!(art.W, x)
    # end
    add_node!(art.F2)

    # Add the label for the category
    push!(art.labels, y)

    # Recompute F2 activations
    local_f1a, local_f2a = multi_activations(art, x)

    # learn!(art, x, art.n_categories)
    learn!(art, x, local_f1a, local_f2a, art.n_categories)
end

function initialize!(
    art::DeepHeadART,
    x::RealArray;
    # f1a::Tuple,
    # f2a::Tuple;
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
    create_category!(
        art,
        x,
        # f1a,
        # f2a,
        label,
    )
end

"""
Returns the last activation of the F1 layer.

# Arguments
- `a::Tuple`: the activations tuple.
"""
function get_last_f1(a::Tuple)
    return a[end]
end

"""
Returns the last activations of the F2 layer.

# Arguments
- `a::Tuple`: the activations tuple.
"""
function get_last_f2(a::Tuple)
    return [a[end][ix][end] for ix in eachindex(a[end])]
end

"""
Trains the [`DeepHeadART`](@ref) module on the provided sample `x`.

# Arguments
$ARG_DEEPHEADART
$ARG_X
"""
function train!(
    art::DeepHeadART,
    x::RealArray;
    # preprocessed::Bool=false,
    y::Integer=0,
)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # sample = ART.init_train!(x, art, preprocessed)

    # f1, f2 = forward(art, x)
    f1a, f2a = multi_activations(art, x)
    f1 = ART.init_train!(get_last_f1(f1a), art, false)
    f2 = get_last_f2(f2a)
    f2 = [ART.init_train!(f2[ix], art, false) for ix in eachindex(f2)]

    # If no prototypes exist, initialize the module
    if isempty(art.F2.heads)
        y_hat = supervised ? y : 1
        # initialize!(art, x, y=y_hat)
        initialize!(
            art,
            x,
            # f1a,
            # f2a,
            y=y_hat,
        )
        return y_hat
    end

    M = [basic_activation(art, f1, f2[ix]) for ix in eachindex(f2)]
    T = [basic_match(art, f1, f2[ix]) for ix in eachindex(f2)]

    # Sort activation function values in descending order
    index = sortperm(T, rev=true)

    # Initialize mismatch as true
    mismatch_flag = true

    # Loop over all categories
    n_categories = length(art.F2.heads)
    for j = 1:n_categories
        # Best matching unit
        bmu = index[j]
        # Vigilance check - pass
        if M[bmu] >= art.opts.rho
            # If supervised and the label differed, force mismatch
            if supervised && (art.labels[bmu] != y)
                break
            end

            # Learn the sample
            # ART.learn!(art, sample, bmu)
			learn!(art, x, f1a, f2a, bmu)

            # Increment the instance counting
            art.n_instance[bmu] += 1

            # Save the output label for the sample
            y_hat = art.labels[bmu]
            # y_hat = bmu

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
        y_hat = supervised ? y : n_categories + 1

        # Create a new category
        # ART.create_category!(art, sample, y_hat)
        # create_category!(art, sample, y_hat)
        create_category!(
            art,
            x,
            # f1a,
            # f2a,
            y_hat,
        )
    end

    # return T, M
    return y_hat
end

# COMMON DOC: FuzzyART incremental classification method
function classify(
    art::DeepHeadART,
    x::RealVector;
    preprocessed::Bool=false,
    get_bmu::Bool=false,
)
    f1a, f2a = multi_activations(art, x)
    # f1 = ART.init_train!(get_last_f1(f1a), art, false)
    f1 = ART.init_classify!(get_last_f1(f1a), art, false)
    f2 = get_last_f2(f2a)
    f2 = [ART.init_classify!(f2[ix], art, false) for ix in eachindex(f2)]

    # Preprocess the data

    # Compute activation and match functions
    # activation_match!(art, x)
    M = [basic_activation(art, f1, f2[ix]) for ix in eachindex(f2)]
    T = [basic_match(art, f1, f2[ix]) for ix in eachindex(f2)]

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
# OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`DeepHeadART`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `field::DeepHeadART`: the [`DeepHeadART`](@ref) to print/display.
"""
function Base.show(
    io::IO,
    field::DeepHeadART,
)
    print(io, "DeepHeadART(F1: $(field.opts.F1_spec), F2: $(field.F2))")
end
