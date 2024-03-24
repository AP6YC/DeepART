"""
    gen_scenarios.jl

# Description
Generates all scenarios for the DeepART condensed learning scenarios.
"""

using Revise
using DeepART

# DeepART.gen_all_scenarios()
all_data = DeepART.load_all_datasets()

all_data["mnist"] = DeepART.get_mnist(
    flatten=true,
)
all_data["cifar10"] = DeepART.get_cifar10(
    flatten=true,
)
all_data["cifar100_fine"] = DeepART.get_cifar100_fine(
    flatten=true,
)
all_data["cifar100_coarse"] = DeepART.get_cifar100_coarse(
    flatten=true,
)
# all_data["omniglot"] = DeepART.get_omniglot(
#     flatten=true,
# )
all_data["usps"] = DeepART.get_usps(
    flatten=true,
)

# Inspect the number of unique labels in each dataset
["$key => $(length(unique(data.train.y)))" for (key, data) in all_data]

# Define which scenarios to generate and how
GROUPINGS = Dict(
    "mnist" => Dict("random" => true, "group_size" => 2),
    "cifar10" => Dict("random" => true, "group_size" => 2),
    "cifar100_fine" => Dict("random" => true, "group_size" => 20),
    "cifar100_coarse" => Dict("random" => true, "group_size" => 4),
    # "omniglot" => Dict("random" => true, "group_size" => 6),
    "CBB-R15" => Dict("random" => true, "group_size" => 5),
)

# Generate all scenario files
DeepART.gen_all_scenarios(all_data, GROUPINGS, 10)
