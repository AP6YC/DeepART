"""
    train-test.jl

# Description
Implements the variety of training/testing start-to-finish experiments.
"""

# -----------------------------------------------------------------------------
# EXPERIMENTS
# -----------------------------------------------------------------------------

# function basic_train!(
#     art::ART.ARTModule,
#     fdata::DataSplit,
#     n_train::Integer,
# )
#     pr = Progress(n_train; desc="Task-Homogenous Training")
#     for ix = 1:n_train
#         xf = fdata.train.x[:, ix]
#         label = fdata.train.y[ix]
#         ART.train!(art, xf, y=label)
#         next!(pr)
#         # DeepART.train!(art, xf)
#     end
# end

# function basic_test(
#     art::ART.ARTModule,
#     fdata::DataSplit,
#     n_test::Integer,
# )
#     # Get the estimates on the test data
#     y_hats = Vector{Int}()
#     pr = Progress(n_test; desc="Task-Homogenous Testing")
#     for ix = 1:n_test
#         xf = fdata.test.x[:, ix]
#         y_hat = DeepART.classify(art, xf, get_bmu=true)
#         push!(y_hats, y_hat)
#         next!(pr)
#     end

#     # Calculate the performance and log
#     perf = DeepART.ART.performance(y_hats, fdata.test.y[1:n_test])
#     @info "Perf: $perf, n_cats: $(art.n_categories), uniques: $(unique(y_hats))"

#     # Return the estimates
#     return y_hats
# end

# function tt_basic!(
#     art,
#     fdata,
#     n_train,
#     n_test,
# )
#     # Train
#     basic_train!(art, fdata, n_train)

#     # Test
#     y_hats = basic_test(art, fdata, n_test)

#     # Confusion matrix
#     p = DeepART.create_confusion_heatmap(
#         string.(collect(0:9)),
#         fdata.test.y[1:n_test],
#         y_hats,
#     )

#     # Return the plot
#     return p
# end
