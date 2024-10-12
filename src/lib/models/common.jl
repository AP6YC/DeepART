"""
    common.jl

# Description
Common model code for the DeepART project.
"""

# -----------------------------------------------------------------------------
# ABSTRACT TYPES
# -----------------------------------------------------------------------------

"""
Supertype of all DeepART modules that adhere to the `train!` and `classify` usages.
"""
abstract type DeepARTModule end

"""
Union for functions accepting both [`DeepARTModule`](@ref)s and ART.ARTModules.
"""
const CommonARTModule = Union{DeepARTModule, ART.ARTModule, Hebb.BlockNet, Hebb.HebbModel}

# -----------------------------------------------------------------------------
# DOCSTRINGS
# -----------------------------------------------------------------------------

"""
Common docstring; the configuration tuple.
"""
const ARG_CONFIG = """
- `config::NamedTuple`: the contruction configuration options.
"""

"""
[`SimpleDeepART`](@ref) argument docstring.
"""
const ARG_SIMPLEDEEPART = """
- `model::SimpleDeepART`: the [`SimpleDeepART`](@ref) model.
"""

"""
The model input size tuple argument docstring.
"""
const ARG_SIZE_TUPLE = """
- `size_tuple::Tuple{Int}`: a tuple of the model input dimensions.
"""

"""
Type alias for the model input size tuple.
"""
const SizeTuple = Tuple

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
A specifier for the number of nodes per layer in a dense feedforward network.
"""
const DenseSpecifier = Vector{Int}

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Creates a Flux.Chain of Flux.Dense layers according to the hidden layers [`DenseSpecifier`](@ref).

# Arguments
- `n_neurons::DenseSpecifier`: the [`DenseSpecifier`](@ref) that specifies the number of neurons per layer, including the input and output layers.
"""
function get_dense(
    n_neurons::DenseSpecifier,
	# activation=relu,
)
    chain_list = [
        Dense(
            n_neurons[ix] => n_neurons[ix + 1],
			# activation
            # sigmoid,
			tanh,
			# relu,
        ) for ix in range(1, length(n_neurons) - 1)
    ]

    # Broadcast until the types are more stable
    # https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain
    local_chain = Chain(chain_list...)

    # Return the chain
    return local_chain
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Generates the feature extractor model for the ART network.

# Arguments
$ARG_SIZE_TUPLE
"""
function get_conv_model(
	size_tuple::SizeTuple
)
	n_kernels = 4
	conv_width = 4
	pad_width = 2
	pool_width = 2

    # total_n_kernels = config.n_kernels * config.n_scales
	# model = @autosize (size_tuple,) Chain(
	#     Conv(
	# 		# (config.conv_width, config.conv_width),
	# 		(conv_width, conv_width),
	# 		1 => n_kernels,
	# 		sigmoid;
	# 		# init=Flux.glorot_normal,
	# 		# init = ones_function,
	# 		init = Flux.orthogonal,
	# 		pad=(pad_width, pad_width),
	# 	),
	# 	MaxPool((pool_width, pool_width)),
	# 	Flux.flatten,
	#     # softmax
	# )

	model = @autosize (size_tuple,) Chain(
		Conv((5,5),1=>6,relu),
		Flux.flatten,
		# Dense(4704=>15,relu),
		Dense(_=>15,relu),
		Dense(15=>10,sigmoid),
		# softmax
	)

	return model
end

"""
Constructs a dense model.

# Arguments
$ARG_SIZE_TUPLE
"""
function get_dense_model(
	size_tuple::SizeTuple
)
	return Chain(
		Dense(size_tuple[1]=>120, sigmoid),
		Dense(120=>84, sigmoid),
		Dense(84=>10, sigmoid),
		# softmax
	)
end


# Flux.Optimisers.@def struct Instar <: Flux.Optimisers.AbstractRule
# 	eta = 0.01
# end

# function Flux.Optimisers.apply!(o::Instar, state, x, dx)
# 	eta = convert(float(eltype(x)), o.eta)

# 	return state, Flux.Optimisers.@lazy dx * eta  # @lazy creates a Broadcasted, will later fuse with x .= x .- dx
# end

# function Flux.Optimisers.init(o::Instar, x::AbstractArray)
# 	return zero(x)
# end

# model = Chain(
# 	Conv((5,5),1 => 6, relu),
# 	MaxPool((2,2)),
# 	Conv((5,5),6 => 16, relu),
# 	MaxPool((2,2)),
# 	Flux.flatten,
# 	Dense(256=>120,relu),
# 	Dense(120=>84, relu),
# 	Dense(84=>10, sigmoid),
# 	softmax
# )

# function init_node()
# 	# size_tuple = (28, 28, 1, 1)
# 	size_tuple = (28, 28, 1, 1)
# 	conv_width = 4
# 	pad_width = 2
# 	pool_width = 2
# 	n_kernels = 1
# 	model = @autosize (size_tuple,) Chain(
# 	    Conv(
# 			(conv_width, conv_width),
#             1 => n_kernels,
# 			sigmoid;
# 			# init=Flux.glorot_normal,
# 			# init = ones_function,
# 			init = Flux.orthogonal,
#             pad=(pad_width, pad_width),
# 		),
# 		MaxPool((pool_width, pool_width)),
# 	)

# 	return model
# end

# function add_node!(model::SimpleDeepART)
# 	push!(model.F2, init_node())
# end

# """
# TEMP: development function.
# """
# function tryit()
# 	a = SimpleDeepART()
# 	# add_node!(a)
# 	data = get_mnist()

# 	dim = 28
# 	local_data = reshape(data.train.x[:, :, 1], dim, dim, 1, :)

# 	features = get_features(a, local_data)

# 	train!(a.art, features)
# 	return a
# end


# """
# Generates the convolutional feature extractor model for the ART network.

# # Arguments
# $ARG_CONFIG
# """
# function get_model(config::NamedTuple)
#     size_tuple = (28, 28, 1, 1)
#     total_n_kernels = config.n_kernels * config.n_scales
# 	model = @autosize (size_tuple,) Chain(
# 	    Conv(
# 			(config.conv_width, config.conv_width),
# 			# 1=>config.n_kernels,
#             1 => total_n_kernels,
# 			sigmoid;
# 			# init=Flux.glorot_normal,
# 			# init = ones_function,
# 			init = Flux.orthogonal,
# 			# pad=(2,2),
#             pad=(config.pad_width, config.pad_width),
# 		),
# 		MaxPool((config.pool_width, config.pool_width)),
# 	    # softmax
# 	)



# 	# if "second_conv" in config.model_bools
# 	# 	model = @autosize (size_tuple,) Chain(model,
# 	# 		Conv(
# 	# 			(config.conv_width, config.conv_width),
# 	# 			# _=>config.n_kernels,
# 	# 			_ => config.n_kernels,
# 	# 			sigmoid;
# 	# 			init=Flux.glorot_normal,
# 	# 			# init = Flux.orthogonal,
# 	# 			# pad=(2,2),
#     #             pad=(config.pad_width, config.pad_width),
# 	# 		),
# 	# 	    MaxPool((config.pool_width, config.pool_width)),
# 	# 		Flux.flatten,
# 	# 	)
# 	# else
# 	# 	model = Chain(model,
# 	# 		Flux.flatten,
# 	# 	)
# 	# end

# 	# if "dense" in config.model_bools
# 	# 	model = @autosize (size_tuple,) Chain(model,
# 	# 		Dense(
#     #             _,
#     #             # 100,
#     #             config.dense_size,
#     #             sigmoid,
#     #             # init=Flux.orthogonal,
#     #             init=Flux.glorot_normal,
#     #         )
# 	# 	)
# 	# end
#     # return model
# end


# """
# Utility function to examine the filters of a feature extractor.

# # Arguments
# - `model::Flux.Chain`: the Flux object contiaining the filters
# """
# function view_filts(model::Flux.Chain)
#     # Init a vector for viewing the filters
#     view_filts = []

#     # Iterate over each orientation
#     n_filts = size(Flux.params(model)[1])[4]
#     # for m_ix = 1:4
#     for m_ix = 1:n_filts
#         local_filt = Flux.params(model)[1][:, :, 1, m_ix]
#         push!(view_filts, Gray.(local_filt)./maximum(Gray.(local_filt)))
#     end
#     return view_filts
# end
