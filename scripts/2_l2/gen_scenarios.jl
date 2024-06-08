"""
    gen_scenarios.jl

# Description
Generates all scenarios for the DeepART condensed learning scenarios.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Define which scenarios to generate and how
N_TRAIN = 1000
N_TEST = 1000
N_MAX = 5
GROUPINGS = Dict(
    "mnist" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => N_TRAIN,
        "n_test" => N_TEST,
    ),
    "fashionmnist" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => N_TRAIN,
        "n_test" => N_TEST,
    ),
    "cifar10" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => N_TRAIN,
        "n_test" => N_TEST,
    ),
    # "cifar100_fine" => Dict(
    #     "random" => true,
    #     "group_size" => 20,
    #     "n_train" => N_TRAIN,
    #     "n_test" => N_TEST,
    # ),
    # "cifar100_coarse" => Dict(
    #     "random" => true,
    #     "group_size" => 4,
    #     "n_train" => N_TRAIN,
    #     "n_test" => N_TEST,
    # ),
    "usps" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => N_TRAIN,
        "n_test" => N_TEST,
    ),
    # "omniglot" => Dict("random" => true, "group_size" => 6),
    # "CBB-R15" => Dict(
    #     "random" => true,
    #     "group_size" => 5,
    #     "n_train" => 1000,
    #     "n_test" => 1000,
    # ),
)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

# all_data = DeepART.load_all_datasets()
all_data = Dict{String, DeepART.DataSplit}()
for (key, group) in GROUPINGS
    all_data[key] = DeepART.load_one_dataset(
        key,
        n_train=group["n_train"],
        n_test=group["n_test"],
    )
end

# Inspect the number of unique labels in each dataset
["$key => $(length(unique(data.train.y)))" for (key, data) in all_data]

# -----------------------------------------------------------------------------
# GENERATE SCENARIOS
# -----------------------------------------------------------------------------

# Generate all scenario files
DeepART.gen_all_scenarios(all_data, GROUPINGS, N_MAX)
