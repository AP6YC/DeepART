"""
Development script for deep instar learning.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
# using CUDA
# using ProgressMeter
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

# Dataset selection
DATASET = "mnist"
DISPLAY = true

# Separate development and cluster settings
DEV = Sys.iswindows()
N_TRAIN = DEV ? 500 : 10000
N_TEST = DEV ? 500 : 4000
GPU = !DEV

EXP_TOP = ["singles"]

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "----------------- LOADING DATA -----------------"

data = DeepART.load_one_dataset(
    DATASET,
    n_train=N_TRAIN,
    n_test=N_TEST,
)
fdata = DeepART.flatty(data)

n_classes = length(unique(data.train.y))

n_input = size(fdata.train.x)[1]

# -----------------------------------------------------------------------------
# BASELINE WITHOUT TRAINING THE EXTRACTOR
# -----------------------------------------------------------------------------

@info "----------------- BASELINE -----------------"

# F2 layer size
head_dim = 256

# Model definition
model = DeepART.get_rep_dense(n_input, head_dim)

art = DeepART.ARTINSTART(
# art = DeepART.ARTINSTART(
    model;
    head_dim=head_dim,
    beta=0.0,
    rho=0.6,
    update="art",
    softwta=true,
    gpu=GPU,
)

# Train/test
results = DeepART.tt_basic!(
    art,
    fdata,
    display=DISPLAY,
)
@info "Results: " results["perf"] results["n_cat"]

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    string.(collect(0:9)),
    "baseline_confusion",
    EXP_TOP,
)

# -----------------------------------------------------------------------------
# DENSE
# -----------------------------------------------------------------------------

@info "----------------- DENSE -----------------"

# F2 layer size
head_dim = 1024

# Model definition
model = DeepART.get_rep_dense(n_input, head_dim)

# art = DeepART.INSTART(
art = DeepART.ARTINSTART(
    model,
    head_dim=head_dim,
    beta=0.01,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
)

dev_xf = fdata.train.x[:, 1]
prs = Flux.params(art.model)
acts = Flux.activations(model, dev_xf)

# Train/test
results = DeepART.tt_basic!(
    art,
    fdata,
    display=DISPLAY,
)
@info "Results: " results["perf"] results["n_cat"]

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    string.(collect(0:9)),
    "dense_basic_confusion",
    EXP_TOP,
)

# -----------------------------------------------------------------------------
# CONVOLUTIONAL
# -----------------------------------------------------------------------------

@info "----------------- CONVOLUTIONAL -----------------"

head_dim = 1024
size_tuple = (size(data.train.x)[1:3]..., 1)
conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

art = DeepART.ARTINSTART(
    conv_model,
    head_dim=head_dim,
    beta=0.01,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
)

results = DeepART.tt_basic!(
    art,
    data,
    display=DISPLAY
)
@info "Results: " results["perf"] results["n_cat"]

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    string.(collect(0:9)),
    "conv_basic_confusion",
    EXP_TOP,
)

# -----------------------------------------------------------------------------
# L2M DENSE
# -----------------------------------------------------------------------------

@info "----------------- L2M DENSE -----------------"

cidata = DeepART.ClassIncrementalDataSplit(fdata)
# cidata = DeepART.ClassIncrementalDataSplit(data)
# groupings = [collect(1:5), collect(6:10)]
# groupings = [collect(1:4), collect(5:7), collect(8:10)]
groupings = [collect(1:2), collect(3:4), collect(5:6), collect(7:8), collect(9:10)]
tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
n_tasks = length(tidata.train)
GPU && tidata |> gpu

# Model definition
head_dim = 1024
model = DeepART.get_rep_dense(n_input, head_dim)

art = DeepART.ARTINSTART(
    model,
    head_dim = head_dim,
    beta = 0.01,
    rho=0.3,
    update="art",
    softwta=true,
    # uncommitted=true,
    gpu=GPU,
)

dev_xf = fdata.train.x[:, 1]
prs = Flux.params(art.model)
acts = Flux.activations(model, dev_xf)

results = DeepART.tt_inc!(
    art,
    tidata,
    fdata,
    display=DISPLAY,
)
@info "Results: " results["perf"] results["n_cat"]

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    string.(collect(0:9)),
    "dense_ti_confusion",
    EXP_TOP,
)

# cidata = DeepART.ClassIncrementalDataSplit(data)
# # cidata = DeepART.ClassIncrementalDataSplit(data)
# groupings = [collect(1:2), collect(3:4), collect(5:6), collect(7:8), collect(9:10)]
# tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
# n_tasks = length(tidata.train)
# GPU && tidata |> gpu

# results = DeepART.tt_inc!(art, tidata, data, n_train, n_test)

# -----------------------------------------------------------------------------
# L2M CONV
# -----------------------------------------------------------------------------

@info "----------------- L2M CONVOLUTIONAL -----------------"

cidata = DeepART.ClassIncrementalDataSplit(data)
groupings = [collect(1:2), collect(3:4), collect(5:6), collect(7:8), collect(9:10)]
tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
n_tasks = length(tidata.train)

head_dim = 1024
size_tuple = (size(data.train.x)[1:3]..., 1)
conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

art = DeepART.ARTINSTART(
    conv_model,
    head_dim=head_dim,
    beta=0.01,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
)

# results = DeepART.tt_basic!(art, data, n_train, n_test)
results = DeepART.tt_inc!(
    art,
    tidata,
    data,
    display=DISPLAY,
)
@info "Results: " results["perf"] results["n_cat"]

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    string.(collect(0:9)),
    "conv_ti_confusion",
    EXP_TOP,
)

# -----------------------------------------------------------------------------
# INSPECT WEIGHTS
# -----------------------------------------------------------------------------

# function normalize_mat(m)
#     local_eps = 1e-12
#     return (m .- minimum(m)) ./ (maximum(m) .- minimum(m) .+ local_eps)
# end

# function view_layer(art, i_layer)
#     im = art.model.layers[i_layer].weight
#     Gray.(normalize_mat(im))
# end

# view_layer(art, 2)
# view_layer(art, 4)
# # view_layer(old_art, 2)

# function view_weight(art, i_layer, i_weight, cc=false)
#     local_weight = art.model.layers[i_layer].weight[i_weight, :]
#     l_weight = Int(length(local_weight) / 2)
#     im = if cc
#         local_weight[l_weight + 1:end]
#     else
#         local_weight[1:l_weight]
#     end
#     Gray.(normalize_mat(reshape(im, (28, 28)))')
#     # Gray.(reshape(im, (28, 28)))
# end

# view_weight(art, 2, 100, true)
# view_weight(art, 2, 100, false)
