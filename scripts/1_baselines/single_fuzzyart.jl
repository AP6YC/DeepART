"""
    single_fuzzyart.jl

# Description

"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
# using Flux
# using CUDA
using ProgressMeter
# using UnicodePlots
# using Plots

import AdaptiveResonance as ART

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

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
    rho=0.6,
)

# Set the data config
art.config = ART.DataConfig(0, 1, dim)

# Train the FuzzyART model in simple supervised mode
# ART.train!(art, fdata.train.x, y=fdata.train.y)

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
art = ART.FuzzyART(
    rho=0.6,
)
# Set the data config
art.config = ART.DataConfig(0, 1, dim)

