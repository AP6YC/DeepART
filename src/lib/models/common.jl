"""
    common.jl

# Description
Common model code for the DeepART project.
"""

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

"""
Options for the construction and usage of a [`SimpleDeepART`](@ref) module.
"""
@with_kw struct opts_SimpleDeepART
	"""
	The model input size tuple.
	"""
	size_tuple::SizeTuple = (28, 28, 1, 1)

	"""
	Flag for if the model is convolutional.
	"""
	conv::Bool = true

	"""
	The FuzzyART module options.
	"""
	opts_fuzzyart::opts_FuzzyART = opts_FuzzyART()
end

"""
Container for a simple DeepART module.
"""
struct SimpleDeepART{T <: Flux.Chain}
	"""
    The `Flux.Chain` feature extractor model.
    """
    model::T

	"""
	The FuzzyART module.
	"""
	art::FuzzyART

	"""
	The [`opts_SimpleDeepART`](@ref) options and flags for the module.
	"""
	opts::opts_SimpleDeepART
end

"""
Runs inference on the [`SimpleDeepART`](@ref) model's feature extractor.

# Arguments
$ARG_SIMPLEDEEPART
- `data::SupervisedDataset`: the [`SupervisedDataset`](@ref) dataset with the features to run inference on.
- `index::Integer`: the sample index to extract features of.
"""
function get_features(model::SimpleDeepART, data::SupervisedDataset, index::Integer)
    # local_data = reshape(data.train.x[:, :, index], dim, dim, 1, :)
	if model.opts.conv
		dim = 28
		local_data = reshape(data.x[:, :, index], dim, dim, 1, :)
		# local_data = data.x[:, :, :, index]
		# @info local_data
		# @info size(local_data)
		features = vec(get_features(model, local_data))
	else
		local_data = data.x[:, index]
		features = get_features(model, local_data)
	end

    return features
	# return get_features(model, data.x[:, :, index])
end

"""
Runs inference on the feature extractor of a [`SimpleDeepART`](@ref) model on a provided sample array.

# Arguments
$ARG_SIMPLEDEEPART
- `x::RealArray`: the sample to process with the deep model.
"""
function get_features(model::SimpleDeepART, x::RealArray)
	return model.model(x)
end

"""
Returns the weights of a `model` at the layer `index`.

# Arguments
$ARG_SIMPLEDEEPART
- `index::Integer`: the layer index to return weights for.
"""
function get_weights(model::SimpleDeepART, index::Integer)
	# return Flux.params(model[:, :, 1, 1])
	return Flux.params(model.model)[index]
end
# Flux.params(model)[1][:, :, 1, m_ix] = new_filt

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
	model = @autosize (size_tuple,) Chain(
	    Conv(
			# (config.conv_width, config.conv_width),
			(conv_width, conv_width),
			1 => n_kernels,
			sigmoid;
			# init=Flux.glorot_normal,
			# init = ones_function,
			init = Flux.orthogonal,
			pad=(pad_width, pad_width),
		),
		MaxPool((pool_width, pool_width)),
		Flux.flatten,
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

"""
Keyword argument constructor for a [`SimpleDeepART`](@ref) module passing the keyword arguments to the [`opts_SimpleDeepART`](@ref) for the module.

# Arguments
- `kwargs...`: the options keyword arguments.
"""
function SimpleDeepART(;kwargs...)
    # Create the options from the keyword arguments
    opts = opts_SimpleDeepART(;kwargs...)

    # Instantiate and return a constructed module
    return SimpleDeepART(
		opts,
	)
end

"""
Main constructor for a [`SimpleDeepART`](@ref) module.

# Arguments
- `opts::opts_SimpleDeepART`: the [`opts_SimpleDeepART`] options driving the construction.
"""
function SimpleDeepART(
	opts::opts_SimpleDeepART
)
	# Create the deep model
	model = if opts.conv
		get_conv_model(opts.size_tuple)
	else
		get_dense_model(opts.size_tuple)
	end

	# opts_fuzzyart = opts_FuzzyART()
	art = FuzzyART(opts.opts_fuzzyart)
	model_dim = Flux.outputsize(model, opts.size_tuple)
	art.config = DataConfig(0, 1, model_dim[1])

	return SimpleDeepART(
		model,
		art,
		opts,
	)
end

function supervised_train!(
	model::SimpleDeepART,
	data::SupervisedDataset,
	n_train::Integer=0,
)
	# Determine how much data to train on
	n_samples = length(data.y)
	if n_train > 0
		local_n_train = min(n_train, n_samples)
	else
		local_n_train = n_samples
	end

	for ix = 1:n_train
		# local_y = data.train.y[ix]
		local_y = data.y[ix]
		# features = vec(DeepART.get_features(a, local_data))
		# features = DeepART.get_features(model, data.train, ix)
		features = DeepART.get_features(model, data, ix)

		# @info size(features)
		# @info typeof(features)

		# bmu = AdaptiveResonance.train!(a.art, features, y=local_y)
		bmu = DeepART.train_deepART!(model.art, features, y=local_y)
		# bmu = AdaptiveResonance.train!(a.art, features)
		# @info bmu
	end

	return
end

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
