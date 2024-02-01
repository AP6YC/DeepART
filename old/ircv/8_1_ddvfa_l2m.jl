using Revise            # Editing this file
using DrWatson          # Project directory functions, etc.
using Logging           # Printing diagnostics
using AdaptiveResonance # ART modules
using Random            # Random subsequence
using ProgressMeter     # Progress bar
using Plots             # All plots
using PlotThemes        # Themes for the plots
using DelimitedFiles
# using Clustering

# Plotting style
# pyplot()
theme(:dark)

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)

# Load source files
include(projectdir("experiments/lib_sim.jl"))

# Custom directories
data_dir = projectdir("work/results/flir_compressed")
results_dir(args...) = projectdir("work/results/flir_l2m", args...)

# Define all extractors
extractors = readdir(joinpath(data_dir, "train"))

flir_split = FLIRSplit(data_dir, extractors[1])

# Create the DDVFA options
opts = opts_DDVFA()
opts.gamma = 5
opts.gamma_ref = 1
opts.display = false
# opts.rho_lb = 0.35
# opts.rho = 0.35
# opts.rho_ub = 0.45
# opts.method = "weighted"
# opts.rho_lb = 0.10
# opts.rho_ub = 0.20
# opts.method = "average"
# opts.rho_lb = 0.45
# opts.rho = 0.45
# opts.rho_ub = 0.7
# opts.method = "average"

# Create the ART module
art = DDVFA(opts)

# Get the data stats
dim, n_train = size(flir_split.train_x)
_, n_test = size(flir_split.test_x)

# Set the DDVFA config Manually
art.config = DataConfig(0.0, 1.0, dim)

# Create results destination and train/test
test_accuracies = zeros(n_train)
true_test_y = convert(Array{Int}, flir_split.test_y)
@showprogress for ix = 1:n_train
    train!(art, flir_split.train_x[:, ix], y=[flir_split.train_y[ix]])
    y_hat_test = classify(art, flir_split.test_x, get_bmu=true)
    test_accuracies[ix] = AdaptiveResonance.performance(y_hat_test, true_test_y)
end

# Save the results to a file
open(results_dir("test_accuracies"), "w") do io
    writedlm(io, test_accuracies)
end

# Plot the results
plot(1:n_train, test_accuracies)

# # Testing performance, timed
# test_stats = @timed classify(art, test_x)
# y_hat_test = test_stats.value
# local_test_y = convert(Array{Int}, test_y)
# test_perf = AdaptiveResonance.performance(y_hat_test, local_test_y)
# @info test_perf

# # BMU
# test_stats_bmu = @timed classify(art, flir_split.test_x, get_bmu=true)
# y_hat_test_bmu = test_stats_bmu.value
# local_test_y = convert(Array{Int}, flir_split.test_y)
# test_perf_bmu = AdaptiveResonance.performance(y_hat_test_bmu, local_test_y)
# @info test_perf_bmu

# points = randn(3, 10000)
# DBSCAN clustering, clusters with less than 20 points will be discarded:
# clusters = dbscan(points, 0.05, min_neighbors = 3, min_cluster_size = 20)
# clusters = dbscan(train_x, 0.05, min_neighbors = 3, min_cluster_size = 20)
