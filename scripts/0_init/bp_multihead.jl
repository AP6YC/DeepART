using Revise
using DeepART

using Flux
# d = DeepART.DeeperART()

m = DeepART.MultiHeadField(
    shared_spec = [2, 10, 3],
    head_spec = [3, 10, 2],
)

# j = DeepART.get_dense([1,2,3])
# typeof(j)
# j([1])

# @info m

Flux.activations(m.shared, [1,2])

# m = DeepART.MultiHeadField(
#     shared_spec = [],
#     head_spec = [3, 10, 2],
# )

outs = DeepART.forward(m, [1,2])

all_data = DeepART.load_all_datasets()
data = all_data["moon"]

for ix = 1:length(data.train.y)
    outs = DeepART.forward(m, data.train.x[:, ix])
    # @info outs
end
