"""
Development space for DeeperART.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

all_data = DeepART.load_all_datasets()
data = all_data["moon"]
size(data.train.x)

ix = 20
x = data.train.x[:, ix]

# -----------------------------------------------------------------------------
# MODULE
# -----------------------------------------------------------------------------

b = DeepART.DeepHeadART()

out = DeepART.forward(b, x)
multi = DeepART.multi_activations(b, x)

forward = DeepART.forward(b, x)
trained = DeepART.train!(b, x)
# b.F1[1].weight
b.F1(x)

f1a, f2a = DeepART.multi_activations(b, x)
# f1 = ART.init_train!(get_last_f1(f1a), art, false)

# DeepART.add_node!(b, x)

# for ix = 1:length(data.train.y)
#     outs = DeepART.forward(m, data.train.x[:, ix])
#     # @info outs
# end
