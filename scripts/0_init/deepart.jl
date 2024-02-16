"""
    init.jl

# Description
This script is a development zone for common workflow elements of the `CART` project.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

n_train = 10

# -----------------------------------------------------------------------------
# CONVOLUTIONAL
# -----------------------------------------------------------------------------

# Create a convolutional module
a = DeepART.SimpleDeepART(
    size_tuple=(28, 28, 1, 1),
    conv=true,
)
a.art.opts.rho = 0.4

mnist = DeepART.get_mnist()

DeepART.supervised_train!(a, mnist.train, n_train)

@info "n categories: " a.art.n_categories

# -----------------------------------------------------------------------------
# DENSE
# -----------------------------------------------------------------------------

b = DeepART.SimpleDeepART(
    size_tuple=(2,),
    conv=false,
)
b.art.opts.rho = 0.4

all_data = DeepART.load_all_datasets()
data = all_data["moon"]

DeepART.supervised_train!(b, data.train, n_train)

@info "n categories: " b.art.n_categories
