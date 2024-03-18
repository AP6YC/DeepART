"""
Development script for deep instar learning.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
# using StatsBase: norm

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

N_TRAIN = 1000
N_TEST = 1000
N_BATCH = 128
# N_BATCH = 1
N_EPOCH = 1
ACC_ITER = 10

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# all_data = DeepART.load_all_datasets()
# data = all_data["moon"]
data = DeepART.get_mnist()
fdata = DeepART.flatty_hotty(data)

n_classes = length(unique(data.train.y))
n_train = min(N_TRAIN, length(data.train.y))
n_test = min(N_TEST, length(data.test.y))

# model = Chain(

# )

# model = @autosize (28, 28, 1, 1) Chain(
#     Conv((5,5),1=>6,relu),
#     Flux.flatten,
#     Dense(_=>15,relu),
#     Dense(15=>10,sigmoid),
#     softmax
# )

size_tuple = (28, 28, 1, 1)

# Create a LeNet model
model = @Flux.autosize (size_tuple,) Chain(
    Conv((5,5),1 => 6, relu),
    MaxPool((2,2)),
    Conv((5,5),6 => 16, relu),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(256=>120,relu),
    Dense(120=>84, relu),
    Dense(84=>10, sigmoid),
    softmax
)

ix = 1
# x = fdata.train.x[:, ix]
x = reshape(data.train.x[:, :, ix], size_tuple)
y = data.train.y[ix]

acts = Flux.activations(model, x)
