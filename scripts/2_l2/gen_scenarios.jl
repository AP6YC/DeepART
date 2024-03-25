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
GROUPINGS = Dict(
    "mnist" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => 1000,
        "n_test" => 1000,
    ),
    "fashionmnist" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => 1000,
        "n_test" => 1000,
    ),
    "cifar10" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => 1000,
        "n_test" => 1000,
    ),
    "cifar100_fine" => Dict(
        "random" => true,
        "group_size" => 20,
        "n_train" => 1000,
        "n_test" => 1000,
    ),
    "cifar100_coarse" => Dict(
        "random" => true,
        "group_size" => 4,
        "n_train" => 1000,
        "n_test" => 1000,
    ),
    "usps" => Dict(
        "random" => true,
        "group_size" => 2,
        "n_train" => 1000,
        "n_test" => 1000,
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

# DeepART.gen_all_scenarios()
all_data = DeepART.load_all_datasets()

all_data["mnist"] = DeepART.get_mnist(
    flatten=true,
    n_train=GROUPINGS["mnist"]["n_train"],
    n_test=GROUPINGS["mnist"]["n_test"],
)
all_data["fashionmnist"] = DeepART.get_fashionmnist(
    flatten=true,
    n_train=GROUPINGS["fashionmnist"]["n_train"],
    n_test=GROUPINGS["fashionmnist"]["n_test"],
)
all_data["cifar10"] = DeepART.get_cifar10(
    flatten=true,
    n_train=GROUPINGS["cifar10"]["n_train"],
    n_test=GROUPINGS["cifar10"]["n_test"],
)
all_data["cifar100_fine"] = DeepART.get_cifar100_fine(
    flatten=true,
    n_train=GROUPINGS["cifar100_fine"]["n_train"],
    n_test=GROUPINGS["cifar100_fine"]["n_test"],
)
all_data["cifar100_coarse"] = DeepART.get_cifar100_coarse(
    flatten=true,
    n_train=GROUPINGS["cifar100_coarse"]["n_train"],
    n_test=GROUPINGS["cifar100_coarse"]["n_test"],
)
# all_data["omniglot"] = DeepART.get_omniglot(
#     flatten=true,
# )
all_data["usps"] = DeepART.get_usps(
    flatten=true,
    n_train=GROUPINGS["usps"]["n_train"],
    n_test=GROUPINGS["usps"]["n_test"],
)

# Inspect the number of unique labels in each dataset
["$key => $(length(unique(data.train.y)))" for (key, data) in all_data]

# -----------------------------------------------------------------------------
# GENERATE SCENARIOS
# -----------------------------------------------------------------------------

# Generate all scenario files
DeepART.gen_all_scenarios(all_data, GROUPINGS, 10)
