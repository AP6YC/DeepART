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
function Agent(agent, opts, scenario_dict::AbstractDict)
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
    print(io, "--- Agent with options:\n")
    # print(io, agent.agent.opts)
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
)
    # Disect the experience
    dataset_index = get_index_from_name(data.train.labels, experience.task_name)
    datum_index = experience.seq_nums.task_num

    # If we are updating the model, run the training function
    if experience.update_model
        sample = data.train.x[dataset_index][:, datum_index]
        label = data.train.y[dataset_index][datum_index]
        # y_hat = AdaptiveResonance.train!(agent.agent, sample, y=label)
        y_hat = incremental_supervised_train!(agent.agent, sample, label)
    # elseif experience.block_type == "test":
    else
        sample = data.test.x[dataset_index][:, datum_index]
        label = data.test.y[dataset_index][datum_index]
        # y_hat = AdaptiveResonance.classify(agent.agent, sample)
        y_hat = incremental_classify(agent.agent, sample)
    end

    # Create the results dictionary
    results = Dict(
        "performance" => y_hat == label ? 1.0 : 0.0,
        "art_match" => agent.agent.stats["M"],
        "art_activation" => agent.agent.stats["T"],
    )

    # agent.agent
    # # Artificially create some results
    # results = Dict(
    #     "performance" => 0.0,
    # )

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
    data_logger::PythonCall.Py,
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
    data::ClassIncrementalDataSplit,
    data_logger::PythonCall.Py,
)
    # Initialize the "last sequence"
    # last_seq = SequenceNums(-1, -1, -1)

    # Initialize the progressbar
    n_exp = length(agent.scenario.queue)
    # block_log_string = "Block 1"
    p = Progress(n_exp; showspeed=true)
    # Iterate while the agent's scenario is incomplete
    while !is_complete(agent)
        # Get the next experience
        exp = popfirst!(agent.scenario.queue)
        # Get the current sequence number
        # cur_seq = exp.seq_nums
        # Logging
        next!(p; showvalues = [
            # (:Block, cur_seq.block_num),
            (:Block, exp.seq_nums.block_num),
            (:Type, exp.block_type),
        ])
        # Evaluate the agent on the experience
        results = evaluate_agent!(agent, exp, data)

        # Log the data
        log_data(data_logger, exp, results, agent.params)

        # Loop reflection
        # last_seq = cur_seq
    end

    return
end

# Why on Earth isn't this included in the PythonCall package?
PythonCall.Py(T::AbstractDict) = pydict(T)
PythonCall.Py(T::AbstractVector) = pylist(T)
PythonCall.Py(T::Symbol) = pystr(String(T))

function full_scenario(
    # data::DSIC,
    # data::VectoredData,
    data::ClassIncrementalDataSplit,
    exp_dir::AbstractString=DeepART.config_dir("l2")
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
    data_logger = l2logger[].DataLogger(
        config["DIR"],
        config["NAME"],
        config["COLS"],     # This one right here, officer
        scenario_info,
    )

    # Create the DDVFA options for both initialization and logging
    opts = opts_DDVFA(
        # DDVFA options
        gamma = 5.0,
        gamma_ref = 1.0,
        # rho=0.45,
        rho_lb = 0.45,
        rho_ub = 0.7,
        similarity = :single,
        display = false,
    )

    # Instantiate the art module
    local_art = DDVFA(opts)

    # Infer the data dimension
    dim = size(data.train.x[1])[1]

    # Specify the input data configuration
    local_art.config = DataConfig(0, 1, dim)

    # Construct the agent from the scenario
    agent = DeepART.Agent(
        local_art,
        opts,
        scenario,
    )

    # Run the scenario for this dataset
    DeepART.run_scenario(agent, data, data_logger)
end
