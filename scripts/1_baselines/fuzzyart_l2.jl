"""
    fuzzyart_l2.jl

# The L2 logging and metrics generation for FuzzyART.
"""
# -----------------------------------------------------------------------------
# L2 METRICS
# -----------------------------------------------------------------------------

using PythonCall

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

# Why on Earth isn't this included in the PythonCall package?
PythonCall.Py(T::AbstractDict) = pydict(T)
PythonCall.Py(T::AbstractVector) = pylist(T)
PythonCall.Py(T::Symbol) = pystr(String(T))

# l2logger = Ref{PythonCall.Py}()
# l2logger[] = PythonCall.pyimport("l2logger.l2logger")
l2l = PythonCall.pyimport("l2logger.l2logger")

# for dir in readdir(DeepART.results_dir("l2metrics", "scenarios"), join=true)
top_dir = DeepART.results_dir("l2metrics", "scenarios")
for scenario_top_dir in readdir(top_dir)
    scenario_top_dir_full = joinpath(top_dir, scenario_top_dir)
    # ti_data =
    for scenario_dir in readdir(dir, join=true)

        # Run the scenario
        DeepART.full_scenario(
            tidata,
            scenario_dir,
            l2l,
        )
    end
end
