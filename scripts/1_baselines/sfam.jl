"""
    single_sfam.jl

# Description
Simple FuzzyARTMAP training and testing.
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
# theme(:dracula)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Accept data downloads
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Fix plotting on headless
ENV["GKSwstype"] = "100"

# Train/test config
N_TRAIN = 60000
N_TEST = 10000
RHO = 0.0

# Print to the paper dir only if on Windows, assuming that unix means the cluster
PAPER = Sys.iswindows()

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# Load the dataset
data = DeepART.get_mnist(
    flatten=true,
    n_train=N_TRAIN,
    n_test=N_TEST,
)

# Get the dimension
dim = size(data.train.x)[1]

# Infer other aspects of the data
n_classes = length(unique(data.train.y))

# -----------------------------------------------------------------------------
# TASK-HOMOGENOUS TRAIN/TEST
# -----------------------------------------------------------------------------

# Init the SFAM module
art = ART.SFAM(
    # rho=0.6,
    rho = RHO,
    display=true,
    # epsilon=1e-4,
)

# Set the data config
art.config = ART.DataConfig(0, 1, dim)

# Train the FuzzyART model in simple supervised mode
results = DeepART.tt_basic!(
    art,
    data,
    display=true,
)

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    string.(collect(0:9)),
    "th_conf",
    ["single_fuzzyart"],
    PAPER,
)

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
tiart = ART.SFAM(
    rho=RHO,
    display=true,
)

# Set the data config
tiart.config = ART.DataConfig(0, 1, dim)

# Simple incremental train/test loop for quick validation
DeepART.tt_inc!(
    tiart,
    tidata,
    fdata,
    n_train,
    n_train,
)

# Create the confusion matrix from this experiment
DeepART.plot_confusion_matrix(
    data.test.y,
    results["y_hats"],
    string.(collect(0:9)),
    "ti_conf",
    ["single_fuzzyart"],
)

# -----------------------------------------------------------------------------
# COMPLEX TASK-INCREMENTAL TRAIN/TEST
# -----------------------------------------------------------------------------

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
