"""
    hebbmodel.jl

# Description

"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

# struct HebbModel{T <: Flux.Chain}
#     model::T
#     opts::ModelOpts
# end

struct HebbModel{T <: CCChain}
    model::T
    opts::ModelOpts
end

function HebbModel(
    data::DeepART.DataSplit,
    opts::ModelOpts,
)
    return HebbModel(
        construct_model(data, opts),
        opts,
    )
end

# -----------------------------------------------------------------------------
#  INSPECTION FUNCTIONS
# -----------------------------------------------------------------------------

function inspect_weights(model::HebbModel, layer::Integer)
    # weights = get_weights(model.model)
    # return weights[layer]
    return model.model.chain[layer][2].weight
end

# function get_weights(model::HebbModel)
#     return Flux.params(model.model)
# end

function get_weight_slice(
    model::HebbModel,
    layer::Integer,
    index::Integer,
)

    # weights = get_weights(model.model)

    # weights = Flux.params(model.model.chain)
    weights = get_weights(model.model)
    dim = Int(size(weights[layer])[2])

    if model.opts["bias"]
        dim -= 1
        local_weights = weights[layer][index, 2:end]
    else
        local_weights = weights[layer][index, :]
    end

    if model.opts["cc"]
        dim = Int(dim / 2)
    end

    dim = Int(sqrt(dim))
    local_weight = reshape(
        # weights[layer][index, :],
        local_weights,
        dim,
        model.opts["cc"] ? dim*2 : dim,
    )

    return local_weight
end

function view_weight(
    model::HebbModel,
    index::Integer;
    layer::Integer=1
)
    # if model.opts["bias"]
    #     dim_x -= 1
    # end

    if model.model.chain[layer][2] isa Flux.Conv
        weights = model.model.chain[layer][2].weight
        lmax = maximum(weights)
        lmin = minimum(weights)
        img = DeepART.Gray.(weights[:, :, 1, index] .- lmin ./ (lmax - lmin))
    else
        # # weights = Flux.params(model.model.chain)
        # weights = get_weights(model.model)
        # dim = Int(size(weights[layer])[2])
        # if model.opts["cc"]
        #     dim = Int(dim / 2)
        # end

        # dim = Int(sqrt(dim))
        # local_weight = reshape(
        #     weights[layer][index, :],
        #     dim,
        #     model.opts["cc"] ? dim*2 : dim,
        # )
        local_weight = get_weight_slice(model, layer, index)

        lmax = maximum(local_weight)
        lmin = minimum(local_weight)
        img = DeepART.Gray.(local_weight .- lmin ./ (lmax - lmin))
    end

    return img
end

function view_weight_grid(model::Hebb.HebbModel, n_grid::Int; layer=1)
    # Infer the size of the weight matrix
    a = Hebb.view_weight(model, 1, layer=layer)
    (dim_x, dim_y) = size(a)

    # if model.opts["bias"]
    #     dim_x -= 1
    # end

    # Create the output grid
    out_grid = zeros(DeepART.Gray{Float32}, dim_x * n_grid, dim_y * n_grid)

    # Populate the grid iteratively
    for ix = 1:n_grid
        for jx = 1:n_grid
            local_weight = Hebb.view_weight(
                model,
                n_grid * (ix - 1) + jx,
                layer=layer,
            )
            out_grid[(ix - 1) * dim_x + 1:ix * dim_x,
                     (jx - 1) * dim_y + 1:jx * dim_y] = local_weight
        end
    end

    # Return the tranpose for visualization
    return out_grid'
end

# -----------------------------------------------------------------------------
# TRAINING FUNCTIONS
# -----------------------------------------------------------------------------

function train_hebb(
    model::HebbModel{T},
    x,
    y;
) where T <: AlternatingCCChain
    # chain = model.model.chain
    # params = Flux.params(chain)
    # acts = Flux.activations(chain, x)

    params = get_weights(model.model)
    acts = get_activations(model.model, x)
    n_layers = length(params)
    n_acts = length(acts)

    # Caches
    # caches = []
    # for p in params
    #     push!(caches, zeros(Float32, (size(p)..., 2)))
    # end

    # if bias
    #     n_layers = Int(length(params) / 2)
    # else
    #     n_layers = length(params)
    #     ins = [x, acts[1:end-1]...]
    #     outs = [acts...]
    # end

    ins = [acts[jx] for jx = 1:2:n_acts-1]
    outs = [acts[jx] for jx = 2:2:n_acts]

    target = zeros(Float32, size(outs[end]))
    # target = -ones(Float32, size(outs[end]))
    target[y] = 1.0
    if model.opts["gpu"]
        target = target |> gpu
    end

    for ix = 1:n_layers
        weights = params[ix]
        out = outs[ix]
        input = ins[ix]
        # cache = caches[ix]

        if ix == n_layers
            widrow_hoff_learn!(
                input,
                out,
                weights,
                target,
                model.opts,
            )
        else
            deepart_learn!(
                input,
                out,
                weights,
                model.opts,
            )
        end
    end

    return
end


function train_hebb(
    model::HebbModel{T},
    x,
    y;
) where T <: GroupedCCChain
    # Get the names for weights and iteration
    params = get_weights(model.model)
    n_layers = length(params)

    # Get the correct inputs and outputs for actuall learning
    ins, outs = get_incremental_activations(model.model, x)

    # Create the target vector
    target = zeros(Float32, size(outs[end]))
    # target = -ones(Float32, size(outs[end]))
    target[y] = 1.0

    if model.opts["gpu"]
        target = target |> gpu
    end

    for ix = 1:n_layers
        weights = params[ix]
        out = outs[ix]
        input = ins[ix]

        if ix == n_layers
            widrow_hoff_learn!(
                input,
                out,
                weights,
                target,
                model.opts,
            )
        else
            deepart_learn!(
                input,
                out,
                weights,
                model.opts,
            )
        end
    end

    return
end

function train_hebb_immediate(
    model::HebbModel{AlternatingCCChain},
    x,
    y;
)
    chain = model.model
    params = Flux.params(chain)
    n_layers = length(params)

    input = []
    out = []

    for ix = 1:n_layers
        weights = params[ix]

        # If the first layer, set up the recursion
        if ix == 1
            input = chain[1](x)
            out = chain[2](input)
        # If the last layer, set the supervised target
        elseif ix == n_layers
            input = chain[2*ix-1](out)
            out = chain[2*ix](input)
            target = zeros(Float32, size(out))
            target[y] = 1.0
        # Otherwise, recursion
        else
            input = chain[2*ix-1](out)
            out = chain[2*ix](input)
        end

        # If we are in the top supervised layer, use the supervised rule
        if ix == n_layers
            widrow_hoff_learn!(
                input,
                out,
                weights,
                target,
                model.opts,
            )
        # Otherwise, use the unsupervised rule(s)
        else
            deepart_learn!(
                input,
                out,
                weights,
                model.opts,
            )
        end
    end

    return
end
