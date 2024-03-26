"""
    fuzzyart_l2.jl

# Description
The L2 logging and metrics generation for FuzzyART.
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

using Revise
using DeepART
using PythonCall
using DrWatson

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

top_dir = DeepART.results_dir("l2metrics", "scenarios")


EXP_TOP = "2_l2"
EXP_NAME = "l2logs"

# -----------------------------------------------------------------------------
# DEPENDENT OPTIONS
# -----------------------------------------------------------------------------

DEV = Sys.iswindows()
# ONE_SCENARIO = DEV
ONE_SCENARIO = false
# DISPLAY = DEV
DISPLAY = true
# GPU = !DEV
GPU = true

# -----------------------------------------------------------------------------
# SIM PARAMETERS
# -----------------------------------------------------------------------------

# Set the simulation parameters
sim_params = Dict{String, Any}(
    "m" => [
        "SFAM",
        "DeepARTDense",
        "DeepARTConv",
    ],
    "rho" => [
        @onlyif("m" == "SFAM", 0.6),
        @onlyif("m" == "DeepARTDense", 0.3),
        @onlyif("m" == "DeepARTConv", 0.3),
    ],
    # "beta" => 0.01,
    "beta_d" => 0.01,
    "beta_s" => 1.0,
    "head_dim" => 1024,
    "display" => DISPLAY,
    "gpu" => GPU,
)

# Log the simulation scale
@info "Running $(dict_list_count(sim_params)) algorithms on the generated scenarios."

# Turn the dictionary of lists into a list of dictionaries
dicts = dict_list(sim_params)

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

    # Iterate over every each model and parameter combination
    for d in dicts

        isconv = (d["m"] == "DeepARTConv")

        # Load the dataset associated with the top scenario name
        data = DeepART.load_one_dataset(
            scenario_top_dir,
            flatten=!isconv,
        )

        # Convert this dataset to its class-incremental version, letting the scenario determine the grouping
        data = DeepART.ClassIncrementalDataSplit(data)

        # Construct the module from the options and properties of the data
        art = DeepART.get_module_from_options(d, data)

        # Iterate over the scenarios
        for scenario_dir in readdir(scenario_top_dir_full, join=true)
            # Run the scenario
            DeepART.full_scenario(
                art,
                art.opts,
                data,
                scenario_dir,
                l2l,
                d,
            )

            ONE_SCENARIO && break
        end

        ONE_SCENARIO && break

    end

    ONE_SCENARIO && break

end

# # Construct the agent from the scenario
# global scenario = DeepART.json_load(joinpath(scenario_dir, "scenario.json"))
# global agent = DeepART.Agent(
#     art,
#     opts_fuzzyart,
#     scenario,
# )
# global config = DeepART.json_load(joinpath(scenario_dir, "config.json"))

# # Setup the scenario_info dictionary as a function of the config and scenario
# scenario_info = config["META"]
# scenario_info["input_file"] = scenario

# # Extract the groupings order from the config file
# global groupings = DeepART.string_to_orders(config["META"]["task-orders"])

# # Construct a dataset from the grouping
# # tidata = TaskIncrementalDataSplit(data, groupings)
# global tidata, name_map = DeepART.L2TaskIncrementalDataSplit(data, groupings)


# configs = []
# scenarios = []
# # Iterate over every scenario top directory
# for scenario_top_dir in readdir(top_dir)
#     # Get the full path to the scenario directory
#     scenario_top_dir_full = joinpath(top_dir, scenario_top_dir)
#     for scenario_dir in readdir(scenario_top_dir_full, join=true)
#         config = DeepART.json_load(joinpath(scenario_dir, "config.json"))
#         scenario = DeepART.json_load(joinpath(scenario_dir, "scenario.json"))
#         push!(configs, config)
#         push!(scenarios, scenario)
#     end
# end

# using JSON
# function show_json(dict, index)
#     JSON.print(dict[index], 2)
# end
