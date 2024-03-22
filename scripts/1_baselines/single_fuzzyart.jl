"""
    single_fuzzyart.jl

# Description
Simple FuzzyART training and testing.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using ProgressMeter
import AdaptiveResonance as ART
using Plots

# theme(:dark)
# theme(:juno)
theme(:dracula)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Accept data downloads
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
# Fix plotting on headless
ENV["GKSwstype"] = "100"

# Train/test config
N_TRAIN = 1000
N_TEST = 1000

# Print to the paper dir only if on Windows, assuming that unix means the cluster
PAPER = Sys.iswindows()

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# Load the dataset
data = DeepART.get_mnist()
# Flatten the dataset
fdata = DeepART.flatty(data)
# Get the dimension
dim = size(fdata.train.x)[1]
# Infer other aspects of the data
n_classes = length(unique(data.train.y))

# Get the number of samples to use for training and testing based
n_train = min(N_TRAIN, length(data.train.y))
n_test = min(N_TEST, length(data.test.y))

# -----------------------------------------------------------------------------
# TASK-HOMOGENOUS TRAIN/TEST
# -----------------------------------------------------------------------------

# Init the FuzzyART module
art = ART.FuzzyART(
    rho=0.7,
    display=true,
)

# Set the data config
art.config = ART.DataConfig(0, 1, dim)

# Train the FuzzyART model in simple supervised mode
results = DeepART.tt_basic!(
    art,
    fdata,
    n_train,
    n_test,
)

DeepART.plot_confusion_matrix(
    data.test.y[1:n_test],
    results["y_hats"],
    string.(collect(0:9)),
    "confusion_matrix",
    ["single_fuzzyart"],
)

# # Confusion matrix
# p = DeepART.create_confusion_heatmap(
#     string.(collect(0:9)),
#     data.test.y[1:n_test],
#     results["y_hats"],
# )
# p = DeepART.create_unicode_confusion_heatmap(
#     string.(collect(0:9)),
#     data.test.y[1:n_test],
#     results["y_hats"];
#     # colormap=DeepART.COLORSCHEME,
#     # colormap=:viridis,
#     # blend=true,
#     # fix_ar=true,
#     # height=20,
#     # blend=false,
#     # xfact=200,
#     # yfact=200,
# )

# -----------------------------------------------------------------------------
# TASK-INCREMENTAL TRAIN/TEST
# -----------------------------------------------------------------------------

# Get a class-incremental data split (1 class per task)
cidata = DeepART.ClassIncrementalDataSplit(fdata)
# Declare the ways that we are grouping the classes (two classes per task here)
groupings = [collect(2*ix - 1: 2*ix) for ix = 1:5]
# Create a split now from these groupings
tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
# Get the resulting number of tasks
n_tasks = length(tidata.train)

# Init a new FuzzyART module
tiart = ART.FuzzyART(
    rho=0.7,
    display=true,
)

# Set the data config
tiart.config = ART.DataConfig(0, 1, dim)

# # Train over each task
# for ix = 1:n_tasks
#     # Get the local batch of training data
#     task_x = tidata.train[ix].x
#     task_y = tidata.train[ix].y
#     # Train the FuzzyART model in simple supervised mode
#     ART.train!(
#         tiart,
#         task_x,
#         y=task_y,
#     )
# end

# Simple incremental train/test loop for quick validation
DeepART.tt_inc!(
    tiart,
    tidata,
    fdata,
    N_TRAIN,
    N_TEST,
)

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# # Run the scenario
# for (data_key, data_value) in data
#     CFAR.full_scenario(
#         data_value,
#         CFAR.config_dir("l2", data_key),
#     )
# end
