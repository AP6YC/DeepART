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
Container for a simple DeepART module
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
end

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

function get_features(model::SimpleDeepART, data::SupervisedDataset, index::Integer)
    # local_data = reshape(data.train.x[:, :, index], dim, dim, 1, :)
    dim = 28
    local_data = reshape(data.x[:, :, index], dim, dim, 1, :)
    # local_data = data.x[:, :, :, index]
    # @info local_data
    # @info size(local_data)
    features = vec(get_features(model, local_data))
    return features
	# return get_features(model, data.x[:, :, index])
end

function get_features(model::SimpleDeepART, x::RealArray)
	return model.model(x)
end

function get_weights(model::SimpleDeepART, index::Integer)
	return Flux.params(model[:, :, 1, 1])
end

# Flux.params(model)[1][:, :, 1, m_ix] = new_filt

"""
Generates the feature extractor model for the ART network.

# Arguments
$ARG_CONFIG
"""
function get_model()
# function get_model(config::NamedTuple)
    size_tuple = (28, 28, 1, 1)
    # size_tuple = (28, 28, 1)

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

	return model
end

"""
Empty constructor for a SimpleDeepART.
"""
function SimpleDeepART()
	size_tuple = (28, 28, 1, 1)
	# total_n_kernels = config.n_kernels
	model = get_model()

	opts = opts_FuzzyART()
	art = FuzzyART(opts)
	model_dim = Flux.outputsize(model, size_tuple)
	# @info model_dim
	art.config = DataConfig(0, 1, model_dim[1])

	return SimpleDeepART(
		model,
		art,
	)
end

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
