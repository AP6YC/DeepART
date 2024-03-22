using Revise
using DeepART

# DeepART.gen_all_scenarios()
all_data = DeepART.load_all_datasets()

["$key => $(length(unique(data.train.y)))" for (key, data) in all_data]

# mnist = DeepART.get_mnist()
# cifar10 = DeepART.get_cifar10()
# all_data["mnist"] = mnist
# all_data["cifar10"] = cifar10
all_data = DeepART.load_all_datasets()

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