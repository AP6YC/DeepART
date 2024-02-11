"""
    models.jl

# Description
Machine learning models and their functions for training and testing.
"""

"""
Common docstring; the configuration tuple.
"""
const ARG_CONFIG = """
- `config::NamedTuple`: the contruction configuration options.
"""

"""
Container for the full convolutional ART model.
"""
struct ConvART{T <: Flux.Chain}
# struct ConvART{T <: Flux.Chain}
	"""
    The `Flux.Chain` feature extractor model.
    """
    F2::Vector{T}
end

function init_node()
	# size_tuple = (28, 28, 1, 1)
	size_tuple = (28, 28, 1, 1)
	conv_width = 4
	pad_width = 2
	pool_width = 2
	n_kernels = 1
	model = @autosize (size_tuple,) Chain(
	    Conv(
			(conv_width, conv_width),
            1 => n_kernels,
			sigmoid;
			# init=Flux.glorot_normal,
			# init = ones_function,
			init = Flux.orthogonal,
            pad=(pad_width, pad_width),
		),
		MaxPool((pool_width, pool_width)),
	)

	return model
end

function add_node!(model::ConvART)
	push!(model.F2, init_node())
end

function get_features(model::ConvART, data::SupervisedDataset, index::Integer)
	return get_features(model, data.x[:, :, index])
end

function get_features(model::ConvART, x::RealArray)
	return model.F2[1](x)
end

function get_weights(model::ConvART, index::Integer)
	return Flux.params(model.F2[index][:, :, 1, 1])
end

function tryit()
	a = ConvART()
	add_node!(a)
	data = get_mnist()

	dim = 28
	local_data = reshape(data.train.x[:, :, 1], dim, dim, 1, :)

	return get_features(a, local_data)
end

# Flux.params(model)[1][:, :, 1, m_ix] = new_filt

function ConvART()
	# size_tuple = (28, 28, 1, 1)
	# total_n_kernels = config.n_kernels

	F2 = Vector{Chain}()

	return ConvART(
		F2,
	)
end

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


"""
Utility function to examine the filters of a feature extractor.

# Arguments
- `model::Flux.Chain`: the Flux object contiaining the filters
"""
function view_filts(model::Flux.Chain)
    # Init a vector for viewing the filters
    view_filts = []

    # Iterate over each orientation
    n_filts = size(Flux.params(model)[1])[4]
    # for m_ix = 1:4
    for m_ix = 1:n_filts
        local_filt = Flux.params(model)[1][:, :, 1, m_ix]
        push!(view_filts, Gray.(local_filt)./maximum(Gray.(local_filt)))
    end
    return view_filts
end
