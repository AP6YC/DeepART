"""
    init.jl

# Description
This script is a development zone for common workflow elements of the `CART` project.
"""

using Revise
using DeepART
# using Flux
using AdaptiveResonance

# data = DeepART.get_mnist()

# a = DeepART.tryit()
a = DeepART.SimpleDeepART((28, 28, 1, 1), true)
# a = DeepART.SimpleDeepART((10,), false)
data = DeepART.get_mnist()

a.art.opts.rho = 0.4

for ix = 1:15
    local_y = data.train.y[ix]
    # features = vec(DeepART.get_features(a, local_data))
    features = DeepART.get_features(a, data.train, ix)

    @info size(features)
    @info typeof(features)

    # bmu = AdaptiveResonance.train!(a.art, features, y=local_y)
    bmu = DeepART.train_deepART!(a.art, features, y=local_y)
    # bmu = AdaptiveResonance.train!(a.art, features)
    @info bmu
end

@info "n categories: " a.art.n_categories

# AdaptiveResonance.art_learn(a.art, features,)
# n_kernels = 6

# model = Chain(
#     Conv((5,5), 1=>n_kernels, sigmoid; init=Flux.glorot_normal),
#     MaxPool((2,2)),
#     Flux.flatten,
#     # softmax
# )

# dim = 28
# dev_sample = data.train.x[:, :, 1]
# dev_sample = reshape(dev_sample, dim, dim, 1, :)
# out = model(dev_sample)

# feature_dim = size(out)[1]
# art = DDVFA()
# art.config = DataConfig(0, 1, feature_dim)

# # for ix = 1:length(data.train.x[1, 1, :])
# n_train = 1000
# for ix = 1:n_train
#     sample = reshape(data.train.x[:,:, ix], dim, dim, 1, :)
#     train_sample = vec(model(sample))
#     label = data.train.y[ix]
#     train!(art, train_sample, y=label)
# end

# n_test = 1000
# y = data.test.y[1:n_test]
# y_hat = zeros(Int, n_test)
# for jx = 1:n_test
#     sample = reshape(data.test.x[:,:, jx], dim, dim, 1, :)
#     test_sample = vec(model(sample))
#     y_hat[jx] = classify(art, test_sample)
# end

# @info performance(y, y_hat)
