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

# -----------------------------------------------------------------------------
# OPTIONS
# -----------------------------------------------------------------------------

top_dir = DeepART.results_dir("l2metrics", "scenarios")

ONE_SCENARIO = true

EXP_TOP = "2_l2"
EXP_NAME = "l2logs"
# N_PROCS = Sys.iswindows() ? 0 : 31

# N_SIMS = Sys.iswindows() ? 1 : 1000
N_TRAIN = 1000
N_TEST = 1000

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
    "beta" => 0.01,
    "rng_seed" => collect(1:N_SIMS),
    "n_train" => N_TRAIN,
    "n_test" => N_TEST,
    "head_dim" => 1024
    "dataset" => [
        "mnist",
        "fashionmnist",
        "cifar10",
        "cifar100_fine",
        "cifar100_coarse",
        "usps",
    ],
)




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

# Iterate over every scenario top directory
for scenario_top_dir in readdir(top_dir)
    # Get the full path to the scenario directory
    scenario_top_dir_full = joinpath(top_dir, scenario_top_dir)

    # Load the dataset associated with the top scenario name
    data = DeepART.load_one_dataset(
        scenario_top_dir,
        flatten=true,
        # n_train=100,
        # n_test=100,
    )
    # Convert this dataset to its class-incremental version, letting the scenario determine the grouping
    data = DeepART.ClassIncrementalDataSplit(data)

    # Create the FuzzyART model
    # opts_fuzzyart = DeepART.ART.opts_FuzzyART(
    #     rho=0.6,
    #     display=true,
    # )
    # art = DeepART.ART.FuzzyART(opts_fuzzyart)
    opts = DeepART.ART.opts_SFAM(
        rho=0.6,
    )
    art = DeepART.ART.SFAM(opts)
    art.config = DeepART.ART.DataConfig(0.0, 1.0, size(data.train[1].x, 1))

    # Iterate over the scenarios
    for scenario_dir in readdir(scenario_top_dir_full, join=true)
        # Run the scenario
        DeepART.full_scenario(
            art,
            opts,
            data,
            scenario_dir,
            l2l,
        )
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

        ONE_SCENARIO && break
    end
    ONE_SCENARIO && break
end
