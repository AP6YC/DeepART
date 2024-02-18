using Revise
using DeepART
using Flux
# d = DeepART.DeeperART()

m = DeepART.MultiHeadField(
    n_shared = [2, 10, 3],
    n_heads = [3, 10, 2],
)

# j = DeepART.get_dense([1,2,3])
# typeof(j)
# j([1])

@info m

Flux.activations(m.shared, [1,2])
