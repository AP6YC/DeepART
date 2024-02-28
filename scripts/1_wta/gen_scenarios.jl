"""
    gen_scenarios.jl

# Description
Generates the scenario and config files for l2logger and l2metrics experiments.
**NOTE** Must be run before any l2 experiments.

# Authors
- Sasha Petrenko <petrenkos@mst.edu>
"""

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------

# Include Revise first to track changes everywhere else
using Revise
using DeepART
# Include remainder of dependencies
using Combinatorics
# using DrWatson

all_data = DeepART.load_all_datasets()

mnist = DeepART.get_mnist()
cifar10 = DeepART.get_cifar10()

all_data["mnist"] = mnist
all_data["cifar10"] = cifar10

# Iterate over all datasets
for (key, datasplit) in all_data

    local_ci_data = DeepART.ClassIncrementalDataSplit(datasplit)
    @info local_ci_data
    # # Get a list of the order indices
    # orders = collect(1:6)

    # # Create an iterator for all permutations and make it into a list
    # orders = collect(permutations(orders))

    # # Iterate over every permutation
    # for order in orders
    #     # Point to the permutation's own folder
    #     exp_dir(args...) = configs_dir(join(order), args...)
    #     # Make the permutation folder
    #     mkpath(exp_dir())
    #     # Point to the config and scenario files within the experiment folder
    #     config_file = exp_dir("config.json")
    #     scenario_file = exp_dir("scenario.json")

    #     # -------------------------------------------------------------------------
    #     # CONFIG FILE
    #     # -------------------------------------------------------------------------

    #     DIR = results_dir("logs", join(order))
    #     NAME = "l2metrics_logger"
    #     COLS = Dict(
    #         # "metrics_columns" => "reward",
    #         "metrics_columns" => [
    #             "performance",
    #             "art_match",
    #             "art_activation",
    #         ],
    #         "log_format_version" => "1.0",
    #     )
    #     META = Dict(
    #         "author" => "Sasha Petrenko",
    #         "complexity" => "1-low",
    #         "difficulty" => "2-medium",
    #         "scenario_type" => "custom",
    #     )

    #     # Create the config dict
    #     config_dict = Dict(
    #         "DIR" => DIR,
    #         "NAME" => NAME,
    #         "COLS" => COLS,
    #         "META" => META,
    #     )

    #     # Write the config file
    #     json_save(config_file, config_dict)

    #     # -------------------------------------------------------------------------
    #     # SCENARIO FILE
    #     # -------------------------------------------------------------------------

    #     # Build the scenario vector
    #     SCENARIO = []
    #     # for ix = 1:n_classes
    #     for ix in order
    #         # Create a train step and push
    #         train_step = Dict(
    #             "type" => "train",
    #             "regimes" => [Dict(
    #                 # "task" => class_labels[ix],
    #                 "task" => data_selection[ix],
    #                 "count" => length(data_indexed.train.y[ix]),
    #             )],
    #         )
    #         push!(SCENARIO, train_step)

    #         # Create all test steps and push
    #         regimes = []
    #         for jx = 1:n_classes
    #             local_regime = Dict(
    #                 # "task" => class_labels[jx],
    #                 "task" => data_selection[jx],
    #                 "count" => length(data_indexed.test.y[jx]),
    #             )
    #             push!(regimes, local_regime)
    #         end

    #         test_step = Dict(
    #             "type" => "test",
    #             "regimes" => regimes,
    #         )

    #         push!(SCENARIO, test_step)
    #     end

    #     # Make scenario list into a dict entry
    #     scenario_dict = Dict(
    #         "scenario" => SCENARIO,
    #     )

    #     # Save the scenario
    #     json_save(scenario_file, scenario_dict)

    # end

end

# Experiment save directory name
# experiment_top = "10_l2m_dist"

# DCCR project files
# include(projectdir("src", "setup.jl"))

# Special folders for this experiment
# include(projectdir("src", "setup_l2.jl"))

# -----------------------------------------------------------------------------
# SETUP ORDERS
# -----------------------------------------------------------------------------

# Get a list of the order indices
# orders = collect(1:6)

# Create an iterator for all permutations and make it into a list
# orders = collect(permutations(orders))

# Load the default data configuration
# data, data_indexed, class_labels, data_selection, n_classes = load_default_orbit_data(data_dir)

# -----------------------------------------------------------------------------
# ITERATE
# -----------------------------------------------------------------------------
