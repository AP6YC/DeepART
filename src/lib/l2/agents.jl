"""
    agents.jl

# Description
Definitions of agents and their evaluation.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------

"""
L2 agent supertype.
"""
abstract type AbstractAgent end

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
L2 [`AbstractAgent`](@ref) struct.
"""
struct Agent{T} <: AbstractAgent
    """
    The DDVFA module.
    """
    agent::T

    """
    Parameters used for l2logging.
    """
    params::Dict

    """
    Container for the [`Experience`](@ref) Queue.
    """
    scenario::ExperienceQueueContainer
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Creates an agent with an empty experience queue.

# Arguments
- `agent::T`: the agent module.
- `ddvfa_opts::opts_DDVFA`: the options struct used to initialize the DDVFA module and set the logging params.
"""
function Agent(
    agent,
    opts,
)
# function DDVFAAgent(ddvfa_opts::opts_DDVFA)
    # Create the DDVFA object from the opts
    # ddvfa = DDVFA(ddvfa_opts)

    # Create the experience dequeue
    # exp_container = ExperienceQueueContainer(scenario_dict)
    exp_container = ExperienceQueueContainer()

    # Create the params object for Logging
    params = StatsDict()
    fields_to_dict!(params, opts)

    # Construct and return the DDVFAAgent
    return Agent(
        agent,
        params,
        exp_container,
    )
end

"""
Constructor for a [`Agent`](@ref) using the scenario dictionary and optional DDVFA keyword argument options.

# Arguments
- `scenario::AbstractDict`: l2logger scenario as a dictionary.
"""
function Agent(
    agent,
    opts,
    scenario_dict::AbstractDict,
)
    # Create an agent with an empty queue
    agent = Agent(agent, opts)
    # Initialize the agent's scenario container with the dictionary
    initialize_exp_queue!(agent.scenario, scenario_dict)
    # Return the agent with an initialized queue
    return agent
end

# -----------------------------------------------------------------------------
# TYPE OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`Agent`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `cont::AbstractAgent`: the [`Agent`](@ref) to print/display.
"""
function Base.show(io::IO, agent::Agent)
# function Base.show(io::IO, ::MIME"text/plain", agent::DDVFAAgent)
    # compact = get(io, :compact, false)
    print(io, "--- Agent{$(typeof(agent.agent))}:\n")
    # print(io, agent.agent.opts)
    print(io, "\n--- Params: \n")
    print(io, agent.params)
    print(io, "\n--- Scenario: \n")
    print(io, agent.scenario)
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Checks if the [`Agent`](@ref) is done with its scenario queue.

# Arguments
- `agent::Agent`: the agent to test scenario completion on.
"""
function is_complete(agent::Agent)::Bool
    # Return a bool if the agent is complete with the scenario
    # return (length(agent.scenario.queue) == 0)
    return isempty(agent.scenario.queue)
end

"""
Gets an integer index of where a string name appears in a list of strings.

# Arguments
- `labels::Vector{T} where T <: AbstractString`: the list of strings to search.
- `name::AbstractString`: the name to search for in the list of labels.
"""
function get_index_from_name(
    labels::Vector{T},
    name::AbstractString,
) where T <: AbstractString
    # Findall results in a list, even of only one entry
    results = findall(x -> x == name, labels)
    # If the list is longer than 1, error
    if length(results) > 1
        error("Labels list contains multiple instances of $name")
    end
    # If no error, return the first (and only) entry of the reverse search
    return results[1]
end

# const GroupingMap = Dict{String, Int}

"""
Evaluates a single agent on a single experience, training or testing as needed.

# Arguments
- `agent::Agent`: the [`Agent`](@ref) to evaluate.
- `exp::Experience`: the [`Experience`](@ref) to use for training/testing.
"""
function evaluate_agent!(
    agent::Agent,
    experience::Experience,
    # data::VectoredData,
    data::ClassIncrementalDataSplit,
    name_map::Dict{String, Int},
)
    # Disect the experience
    # dataset_index = get_index_from_name(
    #     data.train.labels,
    #     experience.task_name
    # )

    # Get the index of the task from the task name
    dataset_index = name_map[experience.task_name]
    # Get the index of the datum from the sequence number
    datum_index = experience.seq_nums.task_num

    # If we are updating the model, run the training function
    if experience.update_model
        # sample = data.train[dataset_index].x[:, datum_index]
        # label = data.train[dataset_index].y[datum_index]
        sample, label = data.train[dataset_index][datum_index]
        # y_hat = AdaptiveResonance.train!(agent.agent, sample, y=label)
        y_hat = incremental_supervised_train!(agent.agent, sample, label)
    # elseif experience.block_type == "test":
    else
        # sample = data.test[dataset_index].x[:, datum_index]
        # label = data.test[dataset_index].y[datum_index]
        sample, label = data.train[dataset_index][datum_index]
        # y_hat = AdaptiveResonance.classify(agent.agent, sample)
        y_hat = incremental_classify(agent.agent, sample)
    end

    # Create the results dictionary
    results = Dict(
        "performance" => y_hat == label ? 1.0 : 0.0,
        "art_match" => agent.agent.stats["M"],
        "art_activation" => agent.agent.stats["T"],
    )

    return results
end

"""
Logs data from an L2 [`Experience`](@ref).

# Arguments
- `data_logger::PythonCall.Py`: the l2logger DataLogger.
- `exp::Experience`: the [`Experience`](@ref) that the [`AbstractAgent`](@ref) just processed.
- `results::Dict`: the results from the [`AbstractAgent`](@ref)'s [`Experience`](@ref).
- `status::AbstractString`: string expressing if the [`Experience`](@ref) was processed.
"""
function log_data(
    # data_logger::PythonCall.Py,
    data_logger,
    experience::Experience,
    results::Dict,
    params::Dict ;
    status::AbstractString="complete",
)
    seq = experience.seq_nums
    worker = "l2metrics"
    record = Dict(
        "block_num" => seq.block_num,
        "block_type" => experience.block_type,
        # "task_params" => exp.params,
        "task_params" => params,
        "task_name" => experience.task_name,
        "exp_num" => seq.exp_num,
        "exp_status" => status,
        "worker_id" => worker,
    )
    merge!(record, results)
    data_logger.log_record(record)
end

"""
Runs an agent's scenario.

# Arguments
- `agent::Agent`: a struct that contains an [`Agent`](@ref) and `scenario`.
- `data_logger::PythonCall.Py`: a l2logger object.
"""
function run_scenario(
    agent::Agent,
    # data::VectoredData,
    # groupings,
    name_map,
    data::ClassIncrementalDataSplit,
    # data_logger::PythonCall.Py,
    data_logger,
)
    # Initialize the "last sequence"
    # last_seq = SequenceNums(-1, -1, -1)

    # Initialize the progressbar
    n_exp = length(agent.scenario.queue)
    # block_log_string = "Block 1"
    p = Progress(n_exp; showspeed=true)
    # @info agent.scenario.queue
    agent_name = typeof(agent.agent).name
    # Iterate while the agent's scenario is incomplete
    while !is_complete(agent)
        # Get the next experience
        exp = popfirst!(agent.scenario.queue)
        # @info "exp: " exp
        # @info "next: " first(agent.scenario.queue)

        # Get the current sequence number
        # cur_seq = exp.seq_nums
        # Logging
        next!(p; showvalues = [
            # (:Block, cur_seq.block_num),
            (:Alg, agent_name),
            (:Block, exp.seq_nums.block_num),
            (:Type, exp.block_type),
            (:NCat, agent.agent.n_categories),
        ])
        # Evaluate the agent on the experience
        results = evaluate_agent!(
            agent,
            exp,
            data,
            name_map,
        )

        # Log the data
        log_data(data_logger, exp, results, agent.params)

        # Loop reflection
        # last_seq = cur_seq
    end

    return
end

# # Why on Earth isn't this included in the PythonCall package?
# PythonCall.Py(T::AbstractDict) = pydict(T)
# PythonCall.Py(T::AbstractVector) = pylist(T)
# PythonCall.Py(T::Symbol) = pystr(String(T))

"""
Runs a full scenario for a given dataset.

A CommonARTModule here needs to have:
- incremental_supervised_train!(...)
- incremental_classify(...)
- stats["M"] and stats["T"] for ART match and activation.

# Arguments
- `art::CommonARTModule`: the ART module to use.
- `opts`: the options used to create the ART module.
- `data::ClassIncrementalDataSplit`: the data to use.
- `exp_dir::AbstractString`: the directory to containing the config and scenario files for each permutation.
- `l2logger::PythonCall.Py`: the l2logger Python library module, used for instantiating the specific `DataLogger` itself here.
"""
function full_scenario(
    art::CommonARTModule,
    opts,
    data::ClassIncrementalDataSplit,
    # exp_dir::AbstractString=DeepART.config_dir("l2")
    exp_dir::AbstractString,
    # l2logger::PythonCall.Py,
    l2logger,
)
    # Load the config and scenario
    # config = DeepART.json_load(DeepART.config_dir("l2", data_key, "config.json"))
    # scenario = DeepART.json_load(DeepART.config_dir("l2", data_key, "scenario.json"))
    config = DeepART.json_load(joinpath(exp_dir, "config.json"))
    scenario = DeepART.json_load(joinpath(exp_dir, "scenario.json"))

    # Setup the scenario_info dictionary as a function of the config and scenario
    scenario_info = config["META"]
    scenario_info["input_file"] = scenario

    # Instantiate the data logger
    # data_logger = l2logger[].DataLogger(
    data_logger = l2logger.DataLogger(
        config["DIR"],
        config["NAME"],
        config["COLS"],     # This one right here, officer
        scenario_info,
    )

    # Construct the agent from the scenario
    agent = DeepART.Agent(
        art,
        opts,
        scenario,
    )

    # Extract the groupings order from the config file
    groupings = string_to_orders(config["META"]["task-orders"])

    # Construct a dataset from the grouping
    # tidata = TaskIncrementalDataSplit(data, groupings)
    tidata, name_map = L2TaskIncrementalDataSplit(data, groupings)

    # Put the x data onto the GPU if required
    if agent.agent.opts.gpu
        tidata = gputize(tidata)
    end

    # # Run the scenario for this dataset
    DeepART.run_scenario(
        agent,
        name_map,
        tidata,
        data_logger,
    )

    # Finally close the logger
    data_logger.close()

    return
end

function gputize(
    data::SupervisedDataset
)
    return SupervisedDataset(
        gpu(data.x),
        data.y
    )
end

function gputize(
    data::DataSplit,
)
    return DataSplit(
        gputize(data.train),
        gputize(data.test),
    )
end

function gputize(
    tidata::ClassIncrementalDataSplit
)
    return ClassIncrementalDataSplit(
        [gputize(t) for t in tidata.train],
        [gputize(t) for t in tidata.test],
    )
end
