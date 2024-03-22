using Revise
using DeepART

# DeepART.gen_all_scenarios()
all_data = DeepART.load_all_datasets()

["$key => $(length(unique(data.train.y)))" for (key, data) in all_data]

all_data = DeepART.load_all_datasets()
all_data["mnist"] = DeepART.get_mnist()
all_data["cifar10"] = DeepART.get_cifar10()
all_data["cifar100_fine"] = DeepART.get_cifar100_fine()
all_data["cifar100_coarse"] = DeepART.get_cifar100_coarse()

"""
The groupings metadata for each dataset as a dictionary of tuples.
"""
GROUPINGS = Dict(
    "mnist" => Dict("random" => true, "group_size" => 2),
    "cifar10" => Dict("random" => true, "group_size" => 2),
    "cifar100_fine" => Dict("random" => true, "group_size" => 20),
    "cifar100_coarse" => Dict("random" => true, "group_size" => 4),
    "CBB-R15" => Dict("random" => true, "group_size" => 5),
)

