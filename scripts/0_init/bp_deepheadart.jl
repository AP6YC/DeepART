"""
Development space for DeeperART.
"""

using Revise
using DeepART

all_data = DeepART.load_all_datasets()
data = all_data["moon"]
size(data.train.x)

ix = 20
x = data.train.x[:, ix]

b = DeepART.DeepHeadART()

out = DeepART.forward(b, x)
multi = DeepART.multi_activations(b, x)

DeepART.forward(b, x)
DeepART.train!(b, x)

# DeepART.add_node!(b, x)

# for ix = 1:length(data.train.y)
#     outs = DeepART.forward(m, data.train.x[:, ix])
#     # @info outs
# end
