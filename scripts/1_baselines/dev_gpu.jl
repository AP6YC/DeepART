"""
Development script for deep instar learning.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
using CUDA
using ProgressMeter
using AdaptiveResonance
using Plots

# theme(:dark)
# theme(:juno)
theme(:dracula)
# using StatsBase: norm

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Accept data downloads
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Fix plotting on headless
ENV["GKSwstype"] = "100"

# N_TRAIN = 10000
# N_TRAIN = 4000
N_TRAIN = 1000
N_TEST = 1000
N_BATCH = 128
# N_BATCH = 1
N_EPOCH = 1
ACC_ITER = 10
GPU = true

EXP_TOP = ["singles", "gpus"]

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "----------------- LOADING DATA -----------------"

# all_data = DeepART.load_all_datasets()
# data = all_data["moon"]
# data = DeepART.get_mnist()
data = DeepART.get_cifar10(
    gray=false,
    n_train=N_TRAIN,
    n_test=N_TEST,
)
# fdata = DeepART.flatty_hotty(data)
fdata = DeepART.flatty(data)

n_classes = length(unique(data.train.y))
n_train = min(N_TRAIN, length(data.train.y))
n_test = min(N_TEST, length(data.test.y))

n_input = size(fdata.train.x)[1]

loader = DeepART.get_loader(data.train)

# for (xf,label)in loader
#     println(size(xf))
#     println(label)
#     # print(label)
#     break
# end

@info "----------------- DENSE -----------------"

# F2 layer size
# head_dim = 256
head_dim = 1024

# Model definition
model = DeepART.get_rep_dense(n_input, head_dim)

# art = DeepART.INSTART(
art = DeepART.ARTINSTART(
    model,
    head_dim=head_dim,
    beta=0.1,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
    # gpu=false,
)

# dev_xf = fdata.train.x[:, 1]
# GPU && (dev_xf = dev_xf |> gpu)
# prs = Flux.params(art.model)
# acts = Flux.activations(art.model, dev_xf)

# Train/test
results = DeepART.tt_basic!(art, fdata, n_train, n_test)

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y[1:n_test],
    results["y_hats"],
    string.(collect(0:9)),
    "dense_basic_confusion",
    EXP_TOP,
)


@info "----------------- CONVOLUTIONAL -----------------"

head_dim = 1024
size_tuple = (size(data.train.x)[1:3]..., 1)
conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

art = DeepART.ARTINSTART(
    conv_model,
    head_dim=head_dim,
    beta=0.1,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
)

results = DeepART.tt_basic!(art, data, n_train, n_test)
# results = DeepART.tt_basic!(art, data, 4000, n_test)

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y[1:n_test],
    results["y_hats"],
    string.(collect(0:9)),
    "conv_basic_confusion",
    EXP_TOP,
)

# @info "----------------- L2M DENSE -----------------"

# cidata = DeepART.ClassIncrementalDataSplit(fdata)
# # cidata = DeepART.ClassIncrementalDataSplit(data)
# # groupings = [collect(1:5), collect(6:10)]
# # groupings = [collect(1:4), collect(5:7), collect(8:10)]
# groupings = [collect(1:2), collect(3:4), collect(5:6), collect(7:8), collect(9:10)]
# tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
# n_tasks = length(tidata.train)
# GPU && tidata |> gpu

# # Model definition
# head_dim = 1024
# model = DeepART.get_rep_dense(n_input, head_dim)

# art = DeepART.ARTINSTART(
#     model,
#     head_dim = head_dim,
#     beta = 0.01,
#     rho=0.3,
#     update="art",
#     softwta=true,
#     # uncommitted=true,
#     gpu=GPU,
# )

# dev_xf = fdata.train.x[:, 1]
# prs = Flux.params(art.model)
# acts = Flux.activations(model, dev_xf)

# results = DeepART.tt_inc!(art, tidata, fdata, n_train, n_test)

# # Create the confusion matrix from this experiment
# DeepART.plot_confusion_matrix(
#     data.test.y[1:n_test],
#     results["y_hats"],
#     string.(collect(0:9)),
#     "dense_ti_confusion",
#     EXP_TOP,
# )

