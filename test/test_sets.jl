"""
    test_sets.jl

# Description
Aggregates all tests for the DeepART project.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Test
using DeepART
using Logging

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

# Set the log level
LogLevel(Logging.Info)

# -----------------------------------------------------------------------------
# TESTSETS
# -----------------------------------------------------------------------------

@testset "Data" begin
    @info "----------------- DATA -----------------"
    # Accept data downloads
    ENV["DATADEPS_ALWAYS_ACCEPT"] = true
    # Fix plotting on headless
    ENV["GKSwstype"] = "100"
    # Dataset selection
    DATASET = "mnist"
    N_TRAIN = 100
    N_TEST = 100

    data = DeepART.load_one_dataset(
        DATASET,
        n_train=N_TRAIN,
        n_test=N_TEST,
    )
    fdata = DeepART.flatty(data)
end
