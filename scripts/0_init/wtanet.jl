"""
Development script for a WTANet module.
"""

using Revise
using DeepART

all_data = DeepART.load_all_datasets()
data = all_data["moon"]


n_classes = length(unique(data.train.y))

m = DeepART.WTANet(model_spec=[2, 10, n_classes])


ix = 1
x = data.train.x[:, ix]
f1 = m.model(x)

argmax(f1)
