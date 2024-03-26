"""
Development script for deep instar learning.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
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
DATASET = "cifar10"
DISPLAY = true

# Separate development and cluster settings
DEV = Sys.iswindows()
N_TRAIN = DEV ? 500 : 4000
N_TEST = DEV ? 500 : 4000
# GPU = !DEV
GPU = true

BETA_S = 0.5
BETA_D = 0.01

EXP_TOP = ["singles", "gpus"]

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

@info "----------------- LOADING DATA -----------------"

data = DeepART.load_one_dataset(
    DATASET,
    n_train=N_TRAIN,
    n_test=N_TEST,
)

# fdata = DeepART.flatty_hotty(data)
fdata = DeepART.flatty(data)

gpudata = DeepART.gputize(fdata)

n_classes = length(unique(data.train.y))

# -----------------------------------------------------------------------------
# DENSE
# -----------------------------------------------------------------------------

@info "----------------- DENSE -----------------"

# F2 layer size
head_dim = 1024

# Model definition
n_input = size(fdata.train.x)[1]
model = DeepART.get_rep_dense(n_input, head_dim)

# art = DeepART.INSTART(
art = DeepART.ARTINSTART(
    model,
    head_dim=head_dim,
    beta=BETA_D,
    beta_s=BETA_S,
    # rho=0.6,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
    # gpu=false,
)

xf = DeepART.get_sample(gpudata.train, 1)
art.model(xf)

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
# CONVOLUTION
# -----------------------------------------------------------------------------

@info "----------------- CONVOLUTIONAL -----------------"

head_dim = 1024
size_tuple = (size(data.train.x)[1:3]..., 1)
conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

# Construct the module
art = DeepART.ARTINSTART(
    conv_model,
    head_dim=head_dim,
    beta=BETA_D,
    beta_s=BETA_S,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
)

# Run the experiment
results = DeepART.tt_basic!(
    art,
    data,
    display=DISPLAY,
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

# loader = DeepART.get_loader(data.train)

# for (xf,label)in loader
#     println(size(xf))
#     println(label)
#     # print(label)
#     break
# end

# dev_xf = fdata.train.x[:, 1]
# GPU && (dev_xf = dev_xf |> gpu)
# prs = Flux.params(art.model)
# acts = Flux.activations(art.model, dev_xf)

# -----------------------------------------------------------------------------
# L2M DENSE
# -----------------------------------------------------------------------------

@info "----------------- L2M DENSE -----------------"

cidata = DeepART.ClassIncrementalDataSplit(fdata)
groupings = [collect(1:2), collect(3:4), collect(5:6), collect(7:8), collect(9:10)]
tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
n_tasks = length(tidata.train)

# tigpudata = DeepART.gputize(tidata)
# sample, label = tigpudata.train[1][2]
# y_hat = DeepART.incremental_supervised_train!(art, sample, label)

# Model definition
head_dim = 1024
model = DeepART.get_rep_dense(n_input, head_dim)

art = DeepART.ARTINSTART(
    model,
    head_dim = head_dim,
    beta=BETA_D,
    beta_s=BETA_S,
    rho=0.3,
    update="art",
    softwta=true,
    gpu=GPU,
)

# dev_xf = fdata.train.x[:, 1]
# prs = Flux.params(art.model)
# acts = Flux.activations(model, dev_xf)

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


# -----------------------------------------------------------------------------
# L2M CONV
# -----------------------------------------------------------------------------

@info "----------------- L2M CONVOLUTIONAL -----------------"

cidata = DeepART.ClassIncrementalDataSplit(data)
groupings = [collect(1:2), collect(3:4), collect(5:6), collect(7:8), collect(9:10)]
tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
n_tasks = length(tidata.train)

tigpudata = DeepART.gputize(tidata)
sample, label = tigpudata.train[1][2]
y_hat = DeepART.incremental_supervised_train!(art, sample, label)


head_dim = 1024
size_tuple = (size(data.train.x)[1:3]..., 1)
conv_model = DeepART.get_rep_conv(size_tuple, head_dim)

art = DeepART.ARTINSTART(
    conv_model,
    head_dim=head_dim,
    beta=BETA_D,
    beta_s=BETA_S,
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
    names,
    "conv_ti_confusion",
    EXP_TOP,
)


# # -----------------------------------------------------------------------------
# # LEADER NEURON
# # -----------------------------------------------------------------------------

# @info "----------------- LEADER NEURON -----------------"

# head_dim = 10

# # Model definition
# model = DeepART.get_rep_dense(n_input, head_dim)

# art = DeepART.ARTINSTART(
#     model,
#     head_dim=head_dim,
#     # beta=1.0,
#     beta=0.1,
#     rho=0.65,
#     update="art",
#     softwta=true,
#     gpu=GPU,
#     leader=true,
# )

# dev_xf = fdata.train.x[:, 1]
# prs = Flux.params(art.model)
# acts = Flux.activations(art.model, dev_xf)
# out = art.model(dev_xf)
# local_data = DeepART.get_mnist(
#     flatten=true,
#     n_train=2000,
#     n_test=1000,
# )

# # Train/test
# results = DeepART.tt_basic!(
#     art,
#     local_data,
#     display=DISPLAY,
# )
# @info "Results: " results["perf"] results["n_cat"]

# # Create the confusion matrix from this experiment
# DeepART.plot_confusion_matrix(
#     local_data.test.y,
#     results["y_hats"],
#     string.(collect(0:9)),
#     "leader_basic_confusion",
#     EXP_TOP,
# )
