"""
    condensed_complex.jl

# Description
This script runs a complex single condensed scenario iteration.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# -----------------------------------------------------------------------------
# PREAMBLE
# -----------------------------------------------------------------------------

using Revise
using DeepART

# -----------------------------------------------------------------------------
# ADDITIONAL DEPENDENCIES
# -----------------------------------------------------------------------------

using AdaptiveResonance
using ProgressMeter
using JLD2
using Flux

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

N_TRAIN = 1000
N_TEST = 1000
GPU = true

# Experiment save directory name
experiment_top = "3_single_condensed"

# Saving names
# plot_name = "4_condensed_complex.png"
mkpath(DeepART.results_dir(experiment_top))
data_file = DeepART.results_dir(experiment_top, "condensed_complex_data.jld2")

DATASET = "mnist"

DISPLAY = true
DEV = Sys.iswindows()
N_TRAIN = DEV ? 500 : 1000
N_TEST = DEV ? 500 : 1000
GPU = !DEV

BETA_S = 1.0
BETA_D = 0.01


# -----------------------------------------------------------------------------
# EXPERIMENT SETUP
# -----------------------------------------------------------------------------

data = DeepART.get_mnist(
    flatten=true,
    n_train=N_TRAIN,
    n_test=N_TEST,
)
cidata = DeepART.ClassIncrementalDataSplit(data)
n_classes = length(cidata.train)

groupings = [collect(1:2), collect(3:4), collect(5:6), collect(7:8), collect(9:10)]
tidata = DeepART.TaskIncrementalDataSplit(cidata, groupings)
n_tasks = length(tidata.train)

# # Load the default simulation options
# opts = DeepART.load_sim_opts(opts_file)

# # Load the data names and class labels from the selection
# data_dirs, class_labels = DeepART.get_orbit_names(opts["data_selection"])

# # Number of classes
# n_classes = length(data_dirs)

# # Load the data
# data = DeepART.load_orbits(DeepART.data_dir, data_dirs, opts["scaling"])

# # Sort/reload the data as indexed components
# data_indexed = DeepART.get_indexed_data(data)

n_input = size(data.train.x, 1)

# # Model definition
# head_dim = 2048
# model = Flux.@autosize (n_input,) Chain(
#     DeepART.CC(),
#     Dense(_, 256, sigmoid, bias=false),
#     # Dense(_, 128, sigmoid, bias=false),
#     DeepART.CC(),
#     # Dense(_, 128, sigmoid, bias=false),
#     # DeepART.CC(),
#     # Dense(_, 64, sigmoid, bias=false),
#     # DeepART.CC(),
#     Dense(_, head_dim, sigmoid, bias=false),
# )

# Model definition
head_dim = 1024
model = DeepART.get_rep_dense(n_input, head_dim)

art = DeepART.ARTINSTART(
    model,
    head_dim = head_dim,
    beta = BETA_D,
    beta_s=BETA_S,
    rho=0.3,
    update="art",
    softwta=true,
    # uncommitted=true,
    gpu=GPU,
)

# art = DeepART.ARTINSTART(
#     model,
#     head_dim=head_dim,
#     beta=0.01,
#     softwta=true,
#     gpu=true,
#     rho=0.6,
# )

# -----------------------------------------------------------------------------
# TRAIN/TEST
# -----------------------------------------------------------------------------

# Get the data dimensions
# dim, n_train = size(data.train.x)
# _, n_test = size(data.test.x)

# Create the estimate containers
perfs = [[] for i = 1:n_tasks]
# vals = [[] for i = 1:n_classes]

# Initial testing block
for j = 1:n_tasks
    push!(perfs[j], 0.0)
end

vals = []
# test_interval = 20
test_interval = 10

# Iterate over each class
# for ix = 1:n_classes
for ix = 1:n_tasks
    # Learning block
    _, n_samples_local = size(tidata.train[ix].x)
    # local_vals = zeros(n_classes, n_samples_local)
    # local_vals = zeros(n_classes, 0)
    # local_vals = Array{Float64}(undef, n_classes, 0)
    local_vals = Array{Float64}(undef, n_tasks, 0)

    # Iterate over all samples
    @showprogress for jx = 1:n_samples_local
        sample = DeepART.get_sample(tidata.train[ix], jx)
        label = tidata.train[ix].y[jx]
        # train!(ddvfa, sample, y=label)
        DeepART.incremental_supervised_train!(art, sample, label)

        # Validation intervals
        if jx % test_interval == 0
            # Validation data
            # local_y_hat = AdaptiveResonance.classify(ddvfa, data.val_x, get_bmu=true)
            # local_val = get_accuracies(data.val_y, local_y_hat, n_classes)
            # Training data
            # local_y_hat = AdaptiveResonance.classify(ddvfa, data.train.x, get_bmu=true)

            # local_y_hat = basic_test(art, data.train, n_test=N_TEST)
            # local_val = DeepART.get_accuracies(data.train.y, local_y_hat, n_classes)

            # local_y_hat = DeepART.basic_test(art, tidata.train[ix], N_TEST)
            # local_val = DeepART.get_accuracies(tidata.train[ix].y, local_y_hat, n_classes)

            # local_y_hat = DeepART.basic_test(art, data.train, N_TEST)
            # local_val = DeepART.get_accuracies(data.train.y, local_y_hat, n_classes)
            local_y_hat = DeepART.basic_test(art, tidata.train[ix])
            local_val = DeepART.get_accuracies(tidata.train[ix].y, local_y_hat, n_classes)
            # @info local_val
            # local_val = [local_val[i]*local_val[i+1] for i in groupings[ix]]
            # local_vals = hcat(local_vals, local_val')
            local_val = [prod([local_val[i] for i in groupings[j]]) for j in eachindex(groupings)]
            local_vals = hcat(local_vals, local_val)
        end
    end

    push!(vals, local_vals)

    # Experience block
    # for jx = 1:n_classes
    for jx = 1:n_tasks
        # local_y_hat = AdaptiveResonance.classify(ddvfa, data_indexed.test.x[j], get_bmu=true)
        # local_y_hat = DeepART.basic_test(art, tidata.test[jx], N_TEST)
        local_y_hat = DeepART.basic_test(art, tidata.test[jx])
        push!(perfs[jx], performance(local_y_hat, tidata.test[jx].y))
    end
end

# Clean the NaN vals
# for ix = 1:n_classes
for ix = 1:n_tasks
    replace!(vals[ix], NaN => 0.0)
end

# Save the data
# DeepART.save_sim_results(data_file, perfs, vals, class_labels)
class_labels = [join(string.(group), "-") for group in groupings]
jldsave(data_file; perfs, vals, class_labels)
