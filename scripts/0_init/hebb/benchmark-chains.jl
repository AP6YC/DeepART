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

function DirectModel(n_input::Int, n_output::Int, n_hidden::Int)
    chain = Chain(
        Dense(n_input, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
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

function SegmentedModel(n_input::Int, n_output::Int, n_hidden::Int)
    chain = Chain(
        Dense(n_input, n_hidden, relu),
        Dense(n_hidden, n_hidden, relu),
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
# BENCHMARKING
# -----------------------------------------------------------------------------

n_input = 10
n_hidden = 20
n_output = 3

x = rand(Float32, n_input)

m1 = DirectModel(n_input, n_output, n_hidden)
m2 = SegmentedModel(n_input, n_output, n_hidden)

# Compile
forward(m1, x)
forward(m2, x)

@info "Direct model"
@benchmark forward(m1, x)

@info "Segmented model"
@benchmark forward(m2, x)
# @benchmark
