"""
    single_fuzzyart.jl

# Description

"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using ProgressMeter
import AdaptiveResonance as ART
# using Flux
# using CUDA
# using UnicodePlots
# using Plots

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

N_TRAIN = 1000
N_TEST = 1000

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# Load the dataset
data = DeepART.get_mnist()
# Flatten the dataset
fdata = DeepART.flatty(data)
# Get the dimension
dim = size(fdata.train.x)[1]

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
DeepART.tt_basic!(art, fdata, N_TRAIN, N_TEST)

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
    rho=0.6,
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

DeepART.tt_inc!(
    tiart,
    tidata,
    fdata,
    n_train,
    n_test,
)