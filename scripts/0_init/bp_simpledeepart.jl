"""
    init.jl

# Description
This script is a development zone for common workflow elements of the `CART` project.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using ProgressMeter

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

N_TRAIN = 3000
N_TEST = 1000

# -----------------------------------------------------------------------------
# CONVOLUTIONAL
# -----------------------------------------------------------------------------

# Create a convolutional module
model = DeepART.SimpleDeepART(
    size_tuple=(28, 28, 1, 1),
    conv=true,
)
model.art.opts.rho = 0.1

data = DeepART.get_mnist()

begin
    DeepART.supervised_train!(model, data.train, N_TRAIN)
    @info "n categories: " model.art.n_categories
end

y_hats = Vector{Int}()
# @showprogress for ix = 1:length(data.test.y)
@showprogress for ix = 1:N_TEST
    y_hat = DeepART.classify(model, data.test, ix)
    push!(y_hats, y_hat)
end

perf = DeepART.ART.performance(y_hats, data.test.y[1:N_TEST])

@info perf unique(y_hats) model.art.n_categories

# -----------------------------------------------------------------------------
# DENSE
# -----------------------------------------------------------------------------

b = DeepART.SimpleDeepART(
    size_tuple=(2,),
    conv=false,
)
b.art.opts.rho = 0.4

all_data = DeepART.load_all_datasets()
data = all_data["moon"]

DeepART.supervised_train!(b, data.train, N_TRAIN)

@info "n categories: " b.art.n_categories
