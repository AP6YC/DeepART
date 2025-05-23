"""
Development space for DeeperART.
"""

using Revise
using DeepART
using Flux

all_data = DeepART.load_all_datasets()
data = all_data["moon"]

a = DeepART.DeeperART()

size(data.train.x)

ix = 20
x = data.train.x[:, ix]
# x = [1.0, 1.0]

f1 = a.F1(x)
f2 = a.F2(f1)
for jx = 1:4
    f2 = a.F2(f1 .+ f2)
    @info f2
end

f1 = Flux.activations(a.F1, x)
f2 = Flux.activations(a.F2, f1[end])

# for ix = 1:length(data.train.y)
#     outs = DeepART.forward(m, data.train.x[:, ix])
#     # @info outs
# end
