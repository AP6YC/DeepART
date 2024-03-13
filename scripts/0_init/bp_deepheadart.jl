"""
Development space for DeeperART.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using ProgressMeter

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# all_data = DeepART.load_all_datasets()
# data = all_data["moon"]
data = DeepART.get_mnist()
data = DeepART.flatty(data)

ix = 20
x = data.train.x[:, ix]

dim = length(x)
n_train = length(data.train.y)
n_test = length(data.test.y)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

N_TRAIN = min(1000, n_train)
N_TEST = min(1000, n_test)

# -----------------------------------------------------------------------------
# MODULE
# -----------------------------------------------------------------------------

b = DeepART.DeepHeadART(
    F1_spec = [
        dim,
        128,
        64,
        10,
    ],
    F2_shared = [
        10,
        20,
        12,
    ],
    F2_heads = [
        12,
        # 20,
        10,
    ],
    rho=0.1,
    eta=0.001,
)
@info minimum(b.F1[3].weight)
@info maximum(b.F1[3].weight)

# out = DeepART.forward(b, x)
# multi = DeepART.multi_activations(b, x)
f1a, f2a = DeepART.multi_activations(b, x)

# forward = DeepART.forward(b, x)
# trained = DeepART.train!(b, x)

y_hats = Vector{Int}()
@showprogress for ix in eachindex(data.train.y[1:N_TRAIN])
    # outs = DeepART.forward(b, data.train.x[:, ix])
    sample = data.train.x[:, ix]
    label = data.train.y[ix]
    y_hat = DeepART.train!(b, sample, y=label)
    push!(y_hats, y_hat)
end

y_hats_test = Vector{Int}()
@showprogress for ix in eachindex(data.test.y[1:N_TEST])
    # outs = DeepART.forward(b, data.train.x[:, ix])
    sample = data.test.x[:, ix]
    # label = data.test.y[ix]
    y_hat = DeepART.classify(b, sample, get_bmu=true)
    push!(y_hats_test, y_hat)
end

# @info y_hats_test
@info unique(y_hats_test)
@info b.n_categories

DeepART.ART.performance(y_hats_test, data.test.y[1:N_TEST])

# b.F1(x)

# f1 = ART.init_train!(get_last_f1(f1a), art, false)

# DeepART.add_node!(b, x)

# for ix = 1:length(data.train.y)
#     outs = DeepART.forward(m, data.train.x[:, ix])
#     # @info outs
# end
