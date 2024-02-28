"""
    mnist.jl

# Description
Boilerplate MNIST training and testing with no modifications.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using Flux
using CUDA
using ProgressMeter
using UnicodePlots
using Plots

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

tt_dict = Dict(
    # N_BATCH = 12
    "N_BATCH" => 128,
    "N_EPOCH" => 1,
    "ACC_ITER" => 10,
)

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

# Load the dataset
data = DeepART.get_mnist()
# all_data = DeepART.load_all_datasets()
# data = all_data["spiral"]
# data = all_data["moon"]
# data = all_data["ring"]

DeepART.train_test!(model, data)

lineplot(
    acc_log,
    title="Accuracy Trend",
    xlabel="Iteration",
    ylabel="Test Accuracy",
)

# plot(
#     acc_log,
#     title="Accuracy Trend",
#     xlabel="Iteration",
#     ylabel="Test Accuracy",
# )

# Flux.Optimisers.adjust!(optim, enabled = false)

# function plot_f(x, y)
#     classes = collect(1:n_classes)
#     y_hat = model([x, y])
#     return Flux.onecold(y_hat, classes)
# end

# p = plot()

# ccol = cgrad([RGB(1,.3,.3), RGB(.4,1,.4), RGB(.3,.3,1), RGB(.3,.6,.1)])
# r = 0:.05:1

# contour!(
#     p,
#     r,
#     r,
#     plot_f,
#     f=true,
#     nlev=4,
#     c=ccol,
#     # c=DeepART.COLORSCHEME,
#     leg=:none
# )

# p = scatter!(
#     p,
#     data.train.x[1, :],
#     data.train.x[2, :],
#     group=data.train.y,
# )