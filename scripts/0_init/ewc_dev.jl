"""
    ewc_dev.jl

# Description
This file is a workspace for developing an EWC optimizer for DeepART modules.
"""

using Revise
using Flux
using DeepART
using Parameters
using ProgressMeter

# Make a simple multilayer perceptron
model = Chain(
    Dense(784, 128, relu),
    Dense(128, 64, relu),
    Dense(64, 10),
    sigmoid,
)

# Load the datasets
all_data = DeepART.load_all_datasets()
data = all_data["moon"]
mnist = DeepART.get_mnist()
# function train(model, data, params)
#     opt = Descent(params.learning_rate)
#     for (x, y) in data
#         gradient = Flux.gradient(() -> loss(x, y), Flux.params(model))
#         update!(opt, Flux.params(model), gradient)
#     end
#     return model
# end

"""
Function to update Fisher Information Matrix based on a Flux model's gradients.
"""
function update_fim(FIM, gradients)
    alpha = 0.1
    if isnothing(FIM)
        # FIM = Dict(key => (g .* g) for (key, g) in gradients)
        # FIM = []
        # for layer in gradients
        #     push!(FIM, layer .* layer)
        # end
        new_FIM = [layer .* layer for layer in gradients]
        # new_FIM = [layer .* layer for layer in gradients]
    else
        # for (key, g) in gradients
        #     FIM[key] = (1 - alpha) * FIM[key] + alpha * (g .* g)
        # end
        # for ix = 1:length(gradients)
        # for ix in eachindex(gradients)
        #     new_FIM[ix] = (1 - alpha) * FIM[ix] + alpha * (gradients[ix] .* gradients[ix])
        # end
        new_FIM = [(1 - alpha) * FIM[ix] + alpha * (gradients[ix] .* gradients[ix]) for ix in eachindex(gradients)]
    end
    return new_FIM
end

"""
Function to update model parameters with respect to the old Fisher Information Matrix.
"""
function update_ewc!(model, FIM, lambda, mu)
    # for (key, param) in Flux.params(model)
    #     param .-= (lambda / 2) * (param .- mu[key]) .* FIM[key]
    # end
    n_layers = length(Flux.params(model))
    # for ix in eachindex(Flux.params(model))
    for ix = 1:n_layers
        local_FIM = FIM[ix]
        # Flux.params(model)[ix] .-= (lambda / 2) * (Flux.params(model)[ix] .- mu[ix]) .* FIM[ix]
        Flux.params(model)[ix] .-= (lambda / 2) * (Flux.params(model)[ix] .- mu[ix]) .* local_FIM
    end
end

# """
# Simple crossentropy loss function.
# """
# loss(x, y) = Flux.crossentropy(model(x), y)

loss(x, y) = Flux.crossentropy(x, y)

"""
EWC training routine.
"""
function train(
    model,
    data,
    params,
)
    # Define an empty Dict for storing mu values
    # mu_dict = Dict()
    opt = Descent(params.learning_rate)
    s = Flux.setup(opt, model);

    # Fisher information matrix approximation diagonal
    FIM = nothing
    # mu = nothing
    mu = copy(Flux.params(model))

    # for (x, y) in data
    n_data = length(data.y)
    @showprogress for ix  = 1:n_data
        x = reshape(data.x[:, :, ix], 784)
        @info size(x)
        y = data.y[ix]
        # gradient = Flux.gradient(() -> loss(x, y), Flux.params(model))
        grads = Flux.gradient(model) do m
            result = m(x)
            loss(result, y)
        end

        # # Get the previous model parameters if available
        # if haskey(mu_dict, :previous_model_params)
        #     prev_params = mu_dict[:previous_model_params]
        #     for (param, prev_param) in zip(Flux.params(model), prev_params)
        #         param.data .= prev_param.data
        #     end
        # end

        FIM = update_fim(FIM, grads[1])
        # @info FIM
        update_ewc!(model, FIM, params.lambda, mu)

        mu = copy(Flux.params(model))
        # mu_dict[:previous_model_params] = [copy(param) for param in Flux.params(model)]
        # mu_dict = Dict(key => )

        Flux.update!(s, model, grads[1])
    end

    return model
end

"""
EWC parameters struct.
"""
@with_kw struct EWCParameters
    learning_rate::Float64 = 0.01
    lambda::Float64 = 0.1
end

params = EWCParameters()

ix = 1
x = reshape(mnist.train.x[:, :, ix], 784)
y = mnist.train.y[ix]
# gradient = Flux.gradient(() -> loss(x, y), Flux.params(model))
grads = Flux.gradient(model) do m
    result = m(x)
    loss(result, y)
end

# Train the model
trained_model = train(model, mnist.train, params)
