"""
    ewc_dev.jl

# Description
This file is a workspace for developing an EWC optimizer for DeepART modules.
"""

using Revise
using Flux
using DeepART
using Parameters

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
function update_fim!(FIM, gradients)
    alpha = 0.1
    if isnothing(FIM)
        FIM = Dict(key => (g .* g) for (key, g) in gradients)
    else
        for (key, g) in gradients
            FIM[key] = (1 - alpha) * FIM[key] + alpha * (g .* g)
        end
    end
end

"""
Function to update model parameters with respect to the old Fisher Information Matrix.
"""
function update_ewc!(model, FIM, lambda, mu)
    for (key, param) in Flux.params(model)
        param .-= (lambda / 2) * (param .- mu[key]) .* FIM[key]
    end
end

"""
Simple crossentropy loss function.
"""
loss(x, y) = Flux.crossentropy(model(x), y)

"""
EWC training routine.
"""
function train(
    model,
    data,
    params,
)
    # Define an empty Dict for storing mu values
    mu_dict = Dict()
    opt = Descent(params.learning_rate)

    # Fisher information matrix approximation diagonal
    FIM = nothing

    for (x, y) in data
        gradient = Flux.gradient(() -> loss(x, y), Flux.params(model))

        # # Get the previous model parameters if available
        # if haskey(mu_dict, :previous_model_params)
        #     prev_params = mu_dict[:previous_model_params]
        #     for (param, prev_param) in zip(Flux.params(model), prev_params)
        #         param.data .= prev_param.data
        #     end
        # end

        update_fim!(FIM, gradient)
        update_ewc!(model, FIM, params.lambda, mu_dict)

        # mu_dict[:previous_model_params] = [copy(param) for param in Flux.params(model)]
        # mu_dict = Dict(key => )

        update!(opt, Flux.params(model), gradient)
    end

    return model
end

"""
Parameters struct
"""
@with_kw struct Parameters
    learning_rate::Float64
    lambda::Float64
end

params = Parameters(0.01, 0.1)

# Train the model
trained_model = train(model, data, params)
