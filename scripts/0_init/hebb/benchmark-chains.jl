# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using BenchmarkTools
using Flux

# -----------------------------------------------------------------------------
# DIRECT MODEL
# -----------------------------------------------------------------------------

struct DirectModel{T <: Flux.Chain}
    chain::T
end

function DirectModel(n_input::Int, n_output::Int, n_hidden::Int, n_layers::Int)
    chain = Chain(
        Dense(n_input, n_hidden, relu),
        (Dense(n_hidden, n_hidden, relu) for ix = 1:n_layers)...,
        Dense(n_hidden, n_output, sigmoid_fast),
    )
    return DirectModel(chain)
end

function forward(model::DirectModel, x)
    return model.chain(x)
end

# -----------------------------------------------------------------------------
# SEGMENTED MODEL
# -----------------------------------------------------------------------------

struct SegmentedModel{T <: Flux.Chain, U <: Flux.Chain}
    chain::T
    head::U
end

function SegmentedModel(n_input::Int, n_output::Int, n_hidden::Int, n_layers::Int)
    chain = Chain(
        Dense(n_input, n_hidden, relu),
        (Dense(n_hidden, n_hidden, relu) for ix = 1:n_layers)...,
        # Dense(n_hidden, n_hidden, relu),
    )
    head = Chain(
        Dense(n_hidden, n_output, sigmoid_fast),
    )
    return SegmentedModel(chain, head)
end

function forward(model::SegmentedModel, x)
    y = model.chain(x)
    return model.head(y)
end

# -----------------------------------------------------------------------------
# VECTORMODEL
# -----------------------------------------------------------------------------

struct VectorModel{T <: Flux.Chain}
    chain::Vector{T}
end

function VectorModel(n_input::Int, n_output::Int, n_hidden::Int, n_layers::Int)
    chain = [
        Chain(Dense(n_input, n_hidden, relu)),
        (Chain(Dense(n_hidden, n_hidden, relu)) for ix = 1:n_layers)...,
        Chain(Dense(n_hidden, n_output, sigmoid_fast)),
    ]
    return VectorModel(chain)
end

function forward(model::VectorModel, x)
    y = x
    for layer in model.chain
        y = layer(y)
    end
    return y
end

# -----------------------------------------------------------------------------
# BENCHMARKING
# -----------------------------------------------------------------------------

n_input = 10
n_hidden = 20
n_output = 3
n_layers = 10

x = rand(Float32, n_input)

m1 = DirectModel(n_input, n_output, n_hidden, n_layers)
m2 = SegmentedModel(n_input, n_output, n_hidden, n_layers)
m3 = VectorModel(n_input, n_output, n_hidden, n_layers)

# Compile
forward(m1, x)
forward(m2, x)
forward(m3, x)

@info "Direct model"
@benchmark forward(m1, x)

@info "Segmented model"
@benchmark forward(m2, x)
# @benchmark

@info "Vector model"
@benchmark forward(m3, x)
