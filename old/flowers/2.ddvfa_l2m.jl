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
# using MLDataUtils
using StatsBase
using HDF5
using MultivariateStats

# Plotting style
# pyplot()
theme(:dark)

# flag_preprocess = true
preprocess = "PCA"

# Set the logging level to Info and standardize the random seed
LogLevel(Logging.Info)
Random.seed!(0)

# Load source files
# include(projectdir("flowers/lib_sim.jl"))

top_dir(args...) = projectdir("work", "results", "flowers", args...)

# Custom directories
data_dir(args...) = top_dir("activations", args...)
@info data_dir()
# results_dir(args...) = projectdir("work/results/flowers", args...)
results_dir(args...) = top_dir(args...)

# Define all extractors
extractors = readdir(data_dir())

# Load x and y from each h5 file
train_data = data_dir(extractors[1])
data_x = h5read(train_data, "df")["block0_values"]
data_y = vec(h5read(train_data, "classes")["block0_values"])
# data_x = convert(Matrix{Float64}, data_x)
# data_y = convert(Vector{Int64}, data_y)

"""
    sigmoid(x::Real)

Return the sigmoid function on x.
"""
function sigmoid(x::Real)
# return 1.0 / (1.0 + exp(-x))
    return one(x) / (one(x) + exp(-x))
end

# if flag_preprocess
if preprocess == "ZScore"
    # Standardize with the data transformer
    transformer = fit(ZScoreTransform, data_x, dims=2)
    data_x = StatsBase.transform(transformer, data_x)

    # Sigmoidally squash the data to normalize between [0, 1] scaled by sigmoid_scaling
    sigmoid_scaling = 3.0
    data_x = sigmoid.(sigmoid_scaling.*(data_x.*2 .- 1))
    config = DataConfig(0.0, 1.0, dim)
elseif preprocess == "PCA"
    # train a PCA model
    M = fit(PCA, data_x; maxoutdim=1000)
    # # apply PCA model to testing set
    # Yte = transform(M, Xte)
    data_x = transform(M, data_x)
    # # reconstruct testing observations (approximately)
    # Xr = reconstruct(M, Yte)
    config = DataConfig()
    data_setup!(config, data_x)
elseif preprocess == "None"
    config = DataConfig()
    data_setup!(config, data_x)
end

# Get the data stats
dim, _ = size(data_x)
# dim, n_train = size(flir_split.train_x)
# _, n_test = size(flir_split.test_x)


# Get the classes
lm = labelmap(data_y)
ind_a = [lm[0]; lm[1]; lm[2]]
ind_b = [lm[3]; lm[4]]

# Splice
data_a_x = data_x[:, ind_a]
data_b_x = data_x[:, ind_b]
data_a_y = data_y[ind_a]
data_b_y = data_y[ind_b]

# Split
(x_train_a, y_train_a), (x_test_a, y_test_a) = stratifiedobs((data_a_x, data_a_y))
(x_train_b, y_train_b), (x_test_b, y_test_b) = stratifiedobs((data_b_x, data_b_y))

# Standardize data types
x_train_a = convert(Matrix{Float64}, x_train_a)
x_test_a = convert(Matrix{Float64}, x_test_a)
y_train_a = convert(Vector{Int}, y_train_a)
y_test_a = convert(Vector{Int}, y_test_a)

# Standardize data types
x_train_a = convert(Matrix{Float64}, x_train_a)
x_test_a = convert(Matrix{Float64}, x_test_a)
y_train_a = convert(Vector{Int}, y_train_a)
y_test_a = convert(Vector{Int}, y_test_a)

# Create the DDVFA options
opts = opts_DDVFA()
opts.gamma = 5.0
opts.gamma_ref = 1.0
opts.display = true

# opts.rho_lb = 0.35
# opts.rho = 0.35
# opts.rho_ub = 0.45
# opts.method = "weighted"

# opts.rho_lb = 0.10
# opts.rho_ub = 0.20
# opts.method = "average"

opts.rho_lb = 0.45
opts.rho = 0.45
opts.rho_ub = 0.7
# opts.method = "average"
opts.method = "single"

# Create the ART module
art = DDVFA(opts)
# Set the DDVFA config Manually
# art.config = DataConfig(0.0, 1.0, dim)
# art.config = data_setup!(DataConfig(), data_x)
art.config = config

# # Create results destination and train/test
# test_accuracies = zeros(n_train)
# true_test_y = convert(Array{Int}, flir_split.test_y)
# @showprogress for ix = 1:n_train
#     train!(art, flir_split.train_x[:, ix], y=[flir_split.train_y[ix]])
#     y_hat_test = classify(art, flir_split.test_x, get_bmu=true)
#     test_accuracies[ix] = AdaptiveResonance.performance(y_hat_test, true_test_y)
# end

train!(art, x_train_a, y=y_train_a)
y_hat_a = AdaptiveResonance.classify(art, x_test_a)
perf = performance(y_hat_a, y_test_a)
@info "Train A, Test A: $perf"

train!(art, x_train_b, y=y_train_b)
y_hat_b = AdaptiveResonance.classify(art, x_test_b)
perf = performance(y_hat_b, y_test_b)
@info "Train B, Test B: $perf"

y_hat_a = AdaptiveResonance.classify(art, x_test_a)
perf = performance(y_hat_a, y_test_a)
@info "Train B, Test A: $perf"

# # Save the results to a file
# open(results_dir("test_accuracies"), "w") do io
#     writedlm(io, test_accuracies)
# end

# # Plot the results
# plot(1:n_train, test_accuracies)

# # # Testing performance, timed
# # test_stats = @timed classify(art, test_x)
# # y_hat_test = test_stats.value
# # local_test_y = convert(Array{Int}, test_y)
# # test_perf = AdaptiveResonance.performance(y_hat_test, local_test_y)
# # @info test_perf

# # # BMU
# # test_stats_bmu = @timed classify(art, flir_split.test_x, get_bmu=true)
# # y_hat_test_bmu = test_stats_bmu.value
# # local_test_y = convert(Array{Int}, flir_split.test_y)
# # test_perf_bmu = AdaptiveResonance.performance(y_hat_test_bmu, local_test_y)
# # @info test_perf_bmu

# # points = randn(3, 10000)
# # DBSCAN clustering, clusters with less than 20 points will be discarded:
# # clusters = dbscan(points, 0.05, min_neighbors = 3, min_cluster_size = 20)
# # clusters = dbscan(train_x, 0.05, min_neighbors = 3, min_cluster_size = 20)
