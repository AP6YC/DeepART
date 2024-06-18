"""
    dev_cvpr.jl

# Description
Development script for the Indoor Scene Recognition dataset.
"""

using Revise
using DeepART


data = DeepART.load_one_dataset(
    "isr",
    # dir=data_dir,
)

data[3]

model = DeepART.get_rep_conv((64, 64, 3, 1), 15)

model(data[3])
