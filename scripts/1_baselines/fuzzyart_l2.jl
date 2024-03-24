"""
    fuzzyart_l2.jl

# The L2 logging and metrics generation for FuzzyART.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using PythonCall

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

top_dir = DeepART.results_dir("l2metrics", "scenarios")
ONE_SCENARIO = true

# -----------------------------------------------------------------------------
# PYTHONCALL SETUP
# -----------------------------------------------------------------------------

# Why on Earth isn't this included in the PythonCall package?
PythonCall.Py(T::AbstractDict) = pydict(T)
PythonCall.Py(T::AbstractVector) = pylist(T)
PythonCall.Py(T::Symbol) = pystr(String(T))


# l2logger = Ref{PythonCall.Py}()
# l2logger[] = PythonCall.pyimport("l2logger.l2logger")
l2l = PythonCall.pyimport("l2logger.l2logger")

# -----------------------------------------------------------------------------
# ALL SCENARIOS
# -----------------------------------------------------------------------------

# Iterate over every scenario top directory
for scenario_top_dir in readdir(top_dir)
    # Get the full path to the scenario directory
    scenario_top_dir_full = joinpath(top_dir, scenario_top_dir)

    # Load the dataset associated with the top scenario name
    data = DeepART.load_one_dataset(scenario_top_dir)
    # Convert this dataset to its class-incremental version, letting the scenario determine the grouping
    data = DeepART.ClassIncrementalDataSplit(data)
    # @info data

    # Create the FuzzyART model
    opts_fuzzyart = DeepART.ART.opts_FuzzyART(
        rho=0.6,
        display=true,
    )
    art = DeepART.ART.FuzzyART(opts_fuzzyart)
    art.config = DeepART.ART.DataConfig(0, 1, size(data.train[1].x, 1))

    # Iterate over the scenarios
    for scenario_dir in readdir(scenario_top_dir_full, join=true)
        # Run the scenario
        DeepART.full_scenario(
            art,
            opts_fuzzyart,
            data,
            scenario_dir,
            l2l,
        )
        ONE_SCENARIO && break
    end
    ONE_SCENARIO && break
end
