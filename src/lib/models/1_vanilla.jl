"""
    vanilla.jl

1. Deep extractor, FuzzyART field

# Description
TODO
"""

"""
In place learning function.

# Arguments
- `art::AbstractFuzzyART`: the FuzzyART module to update.
- `x::RealVector`: the sample to learn from.
- `index::Integer`: the index of the FuzzyART weight to update.
"""
function learn_deepART!(art::ART.AbstractFuzzyART, x::RealVector, index::Integer)
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
function train_deepART!(art::FuzzyART, x::RealVector ; y::Integer=0, preprocessed::Bool=false)
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
        return y
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
			w_diff = learn_deepART!(art, sample, bmu)
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
