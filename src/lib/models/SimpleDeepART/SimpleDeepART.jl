"""
Implements the SimpleDeepART module.
"""

# -----------------------------------------------------------------------------
# TYPES
# -----------------------------------------------------------------------------

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

	"""
	The model output dimension for reference.
	"""
	model_dim::Int
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

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
	@info model_dim
	art.config = DataConfig(0, 1, model_dim[1])
	# art.config = DataConfig(0, 1, model_dim[end])

	return SimpleDeepART(
		model,
		art,
		opts,
		model_dim[1],
	)
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

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
Runs the supervised training of a [`SimpleDeepART`](@ref) module.

# Arguments
$ARG_SIMPLEDEEPART
$ARG_SUPERVISEDDATASET
- `n_train::Integer`: the upper-bound of number of samples to train, default 0.
If this is not manually set, all samples are trained upon.
"""
function supervised_train!(
	model::SimpleDeepART,
	data::SupervisedDataset,
	n_train::Integer=0,
)
	# Determine how much data to train on
	n_samples = length(data.y)
	local_n_train = if n_train > 0
		min(n_train, n_samples)
	else
		n_samples
	end

	optim = Flux.setup(Flux.Adam(), model.model)

	for ix = 1:local_n_train
		# local_y = data.train.y[ix]
		local_y = data.y[ix]
		# features = vec(DeepART.get_features(a, local_data))
		# features = DeepART.get_features(model, data.train, ix)
		features = DeepART.get_features(model, data, ix)

		# @info size(features)
		# @info typeof(features)

		# bmu = AdaptiveResonance.train!(a.art, features, y=local_y)
		bmu, w_diff = DeepART.train_SimpleDeepART!(model.art, features, y=local_y)
		# bmu = AdaptiveResonance.train!(a.art, features)
		# @info bmu
		# @info size(w_diff)
		@info sum(w_diff)
		# Compute gradients from the forward pass
		dim = 28
        val, grads = Flux.withgradient(model.model) do m
		# val, grads = Flux.withjacobian(model.model) do m
			# features = DeepART.get_features(model, data, ix)
			local_data = reshape(data.x[:, :, ix], dim, dim, 1, :)
			result = m(local_data)
            # loss(result, ly)
            # Flux.logitcrossentropy(result, ly)
			sum(w_diff)
        end

		# @info grads

        Flux.update!(optim, model, grads[1])

		# @info w_diff
	end

	return
end

function classify(
	model::SimpleDeepART,
	data::SupervisedDataset,
	# data::RealArray,
	index::Integer
)
	# features = DeepART.get_features(model, data.train, index)
	# dim = 28
	# local_data = reshape(data.x[:, :, ix], dim, dim, 1, :)
	# features = DeepART.get_features(model, local_data, index)
	features = DeepART.get_features(model, data, index)
	return ART.classify(model.art, features, get_bmu=true)
end


"""
In place learning function.

# Arguments
- `art::AbstractFuzzyART`: the FuzzyART module to update.
- `x::RealVector`: the sample to learn from.
- `index::Integer`: the index of the FuzzyART weight to update.
"""
function learn_SimpleDeepART!(art::ART.AbstractFuzzyART, x::RealVector, index::Integer)
    # Compute the updated weight W
    new_vec = ART.art_learn(art, x, index)

    # NEW: get the weight update difference
    w_diff = new_vec - art.W[:, index]

    # Replace the weight in place
    ART.replace_mat_index!(art.W, new_vec, index)

    # Return the weight diff
    return w_diff
end

# Taken from AdaptiveResonance.jl
function train_SimpleDeepART!(art::FuzzyART, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
    # Flag for if training in supervised mode
    supervised = !iszero(y)

    # Run the sequential initialization procedure
    sample = ART.init_train!(x, art, preprocessed)

    # Initialization
    if isempty(art.W)
        # Set the first label as either 1 or the first provided label
        y_hat = supervised ? y : 1
        # Initialize the module with the first sample and label
        ART.initialize!(art, sample, y=y_hat)
        # Return the selected label
        w_diff = zero(art.W[:, 1])
        return y_hat, w_diff
    end

    # If we have a new supervised category, create a new category
    if supervised && !(y in art.labels)
        ART.create_category!(art, sample, y)
		w_diff = zero(art.W[:, 1])
        return y, w_diff
    end

    # Compute activation/match functions
    ART.activation_match!(art, sample)

    # Sort activation function values in descending order
    index = sortperm(art.T, rev=true)

    # Initialize mismatch as true
    mismatch_flag = true

    w_diff = zero(art.W[:, 1])

    # Loop over all categories
    for j = 1:art.n_categories
        # Best matching unit
        bmu = index[j]
        # Vigilance check - pass
        if art.M[bmu] >= art.threshold
            # If supervised and the label differed, force mismatch
            if supervised && (art.labels[bmu] != y)
                break
            end

            # Learn the sample
            # ART.learn!(art, sample, bmu)
			@info "LEARNING"
			w_diff = learn_SimpleDeepART!(art, sample, bmu)
			# @info sum(w_diff[1:784])
			@info size(w_diff)
            # @info size(w_diff)
            # Increment the instance counting
            art.n_instance[bmu] += 1

            # Save the output label for the sample
            y_hat = art.labels[bmu]

            # No mismatch
            mismatch_flag = false
            break
        end
    end

    # If there was no resonant category, make a new one
    if mismatch_flag
        # Keep the bmu as the top activation despite creating a new category
        bmu = index[1]

        # Get the correct label for the new category
        y_hat = supervised ? y : art.n_categories + 1

        # Create a new category
        ART.create_category!(art, sample, y_hat)
    end

    # Update the stored match and activation values
    ART.log_art_stats!(art, bmu, mismatch_flag)

    # Return the training label
    return y_hat, w_diff
end

# # COMMON DOC: FuzzyART incremental classification method
# function classify_SimpleDeepART(art::FuzzyART, x::RealVector ; preprocessed::Bool=false, get_bmu::Bool=false)
#     # Preprocess the data
#     x = init_classify!(x, art, preprocessed)

#     # Compute activation and match functions
#     activation_match!(art, x)

#     # Sort activation function values in descending order
#     index = sortperm(art.T, rev=true)

#     # Default is mismatch
#     mismatch_flag = true
#     y_hat = -1

#     # Iterate over all categories
#     for jx in 1:art.n_categories
#         # Set the best matching unit
#         bmu = index[jx]

#         # Vigilance check - pass
#         if art.M[bmu] >= art.threshold
#             # Current winner
#             y_hat = art.labels[bmu]
#             mismatch_flag = false
#             break
#         end
#     end
#     # If we did not find a match
#     if mismatch_flag
#         # Report either the best matching unit or the mismatch label -1
#         bmu = index[1]

#         # Report either the best matching unit or the mismatch label -1
#         y_hat = get_bmu ? art.labels[bmu] : -1
#     end

#     # Update the stored match and activation values
#     log_art_stats!(art, bmu, mismatch_flag)

#     # Return the inferred label
#     return y_hat
# end
