"""
    scenario.jl

# Description
Definitions of collections of experiences and how they are created from scenarios.

# Authors
- Sasha Petrenko <petrenkos@mst.edu> @AP6YC
"""

# -----------------------------------------------------------------------------
# ALIASES
# -----------------------------------------------------------------------------

"""
Alias for a queue of [`Experience`](@ref)s.
"""
const ExperienceQueue = Deque{Experience}

"""
Alias for a statistics dictionary being string keys mapping to any object.
"""
const StatsDict = Dict{String, Any}

# -----------------------------------------------------------------------------
# STRUCTS
# -----------------------------------------------------------------------------

"""
Container for the [`ExperienceQueue`](@ref) and some statistics about it.
"""
struct ExperienceQueueContainer
    """
    The [`ExperienceQueue`](@ref) itself.
    """
    queue::ExperienceQueue

    """
    The statistics about the queue.
    **NOTE** These statistics reflect the queue at construction, not after any processing.
    """
    stats::StatsDict
end

# -----------------------------------------------------------------------------
# CONSTRUCTORS
# -----------------------------------------------------------------------------

"""
Initializes an [`ExperienceQueueContainer`](@ref) from the provided scenario dictionary.

# Arguments
- `eqc::ExperienceQueueContainer`: the container with the queue and stats to initialize.
- `scenario_dict::AbstractDict`: the dictionary with the scenario regimes and block types.
"""
function initialize_exp_queue!(
    eqc::ExperienceQueueContainer,
    scenario_dict::AbstractDict,
)
    # Initialize the incremented counts and stats
    eqc.stats["n_train"] = 0    # Number of training blocks
    eqc.stats["n_test"] = 0     # Number of testing blocks
    exp_num = 0                 # Experience count
    block_num = 0               # Block count

    # Iterate over each learning/testing block
    for block in scenario_dict["scenario"]
        # Increment the block count
        block_num += 1
        # Get the block type as "train" or "test"
        block_type = block["type"]
        # Verify the block type
        sanitize_block_type(block_type)
        # Stats on the blocks
        if block_type == "train"
            eqc.stats["n_train"] += 1
        elseif block_type == "test"
            eqc.stats["n_test"] += 1
        end
        # Iterate over the regimes of the block
        for regime in block["regimes"]
            # Reinitialize the task-specific count
            task_num = 0
            # Iterate over each count within the current regime
            for _ in 1:regime["count"]
                # Increment the experience count
                exp_num += 1
                # Increment the task-specific count
                task_num += 1
                # Get the task name
                task_name = regime["task"]
                # Create a sequence number container for the block and experience
                seq = SequenceNums(block_num, exp_num, task_num)
                # Create an experience for all of the above
                exp = Experience(task_name, seq, block_type)
                # Add the experience to the top of the Deque
                push!(eqc.queue, exp)
            end
        end
    end

    # Post processing stats
    eqc.stats["length"] = length(eqc.queue)
    eqc.stats["n_blocks"] = block_num

    # Exit with no return
    return
end

"""
Creates an empty [`ExperienceQueueContainer`](@ref) with an empty queue and zeroed stats.
"""
function ExperienceQueueContainer()
    # Create an empty statistics container
    stats = StatsDict(
        "length" => 0,
        "n_blocks" => 0,
        "n_train" => 0,
        "n_test" => 0,
    )

    # Create an empty experience Deque
    exp_queue = Deque{Experience}()

    # Return the a container with the experiences
    return ExperienceQueueContainer(
        exp_queue,
        stats,
    )
end

"""
Creates a queue of [`Experience`](@ref)s from the scenario dictionary.

# Arguments
- `scenario_dict::AbstractDict`: the scenario dictionary.
"""
function ExperienceQueueContainer(scenario_dict::AbstractDict)
    # Create the empty queue container
    eqc = ExperienceQueueContainer()

    # Add the scenario to the queue
    initialize_exp_queue!(eqc, scenario_dict)

    # Return the populated queue.
    return eqc
end

# -----------------------------------------------------------------------------
# TYPE OVERLOADS
# -----------------------------------------------------------------------------

"""
Overload of the show function for [`ExperienceQueue`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `cont::ExperienceQueueContainer`: the [`ExperienceQueueContainer`](@ref) to print/display.
"""
function Base.show(io::IO, queue::ExperienceQueue)
    # compact = get(io, :compact, false)
    print(
        io,
        """
        ExperienceQueue of type $(ExperienceQueue)
        Length: $(length(queue))
        """
    )
end

"""
Overload of the show function for [`ExperienceQueueContainer`](@ref).

# Arguments
- `io::IO`: the current IO stream.
- `cont::ExperienceQueueContainer`: the [`ExperienceQueueContainer`](@ref) to print/display.
"""
function Base.show(io::IO, cont::ExperienceQueueContainer)
    # compact = get(io, :compact, false)
    print(
        io,
        """
        ExperienceQueueContainer
        ExperienceQueue
        \tType: $(ExperienceQueue)
        Stats:
        \tLength: $(cont.stats["length"])
        \tBlocks: $(cont.stats["n_blocks"])
        \tTrain blocks: $(cont.stats["n_train"])
        \tTest blocks: $(cont.stats["n_test"])
        """
    )
end

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Adds entry to a dictionary from a struct with fields.

Meant to be used with [`StatsDict`](@ref).

# Arguments
- `dict::AbstractDict`: the [`StatsDict`](@ref) dictionary to add entries to.
- `opts::Any`: a struct containing fields, presumably of options, to add as key-value entries to the dict.
"""
function fields_to_dict!(dict::AbstractDict, opts::Any)
    # Iterate over every fieldname of the type
    for name in fieldnames(typeof(opts))
        # Add an entry with a string name of the field and the value from the opts.
        dict[string(name)] = getfield(opts, name)
    end
end

function get_grouping(
    groupings::Vector{Int},
    group_size::Int,
)
    n_classes = length(groupings)
    return [groupings[ix : ix + group_size - 1] for ix = 1:group_size:n_classes]
end

function random_grouping(
    data::DataSplit,
    group_size::Int,
)
    classes = unique(data.train.y)
    # n_classes = length(classes)
    groupings = shuffle(classes)

    return get_grouping(groupings, group_size)
end

function gen_permutation_groupings(
    data::DataSplit,
)
    classes = unique(data.train.y)
    orders = collect(permutations(classes))
    return [[[suborder] for suborder in order] for order in orders]
end

function gen_random_groupings(
    data::DataSplit,
    group_size::Int,
    n_groupings::Int,
)
    return [random_grouping(data, group_size) for _ = 1:n_groupings]
end

"""
Generates a single scenario according to a grouping.
"""
function gen_scenario_from_group(
    key::AbstractString,
    cidata::ClassIncrementalDataSplit,
    order::Vector{Vector{Int}},
)
    # Create a task-incremental data split according to the prescribed task/class order
    tidata = DeepART.TaskIncrementalDataSplit(cidata, order)

    # Get the number of tasks
    n_tasks = length(order)

    # Point to the permutation's own folder
    # exp_dir(args...) = DeepART.configs_dir(key, join(order), args...)
    file_dir = string(hash(join(order)))
    exp_dir(args...) = DeepART.results_dir(
        "l2metrics",
        "scenarios",
        key,
        file_dir,
        args...
    )
    # Make the permutation folder
    mkpath(exp_dir())
    @info exp_dir()

    # Point to the config and scenario files within the experiment folder
    config_file = exp_dir("config.json")
    scenario_file = exp_dir("scenario.json")

    # -----------------------------------------------------------------
    # CONFIG FILE
    # -----------------------------------------------------------------

    DIR = DeepART.results_dir(
        "l2metrics",
        "logs",
        file_dir,
    )
    NAME = "l2metrics_logger"
    COLS = Dict(
        # "metrics_columns" => "reward",
        "metrics_columns" => [
            "performance",
            "art_match",
            "art_activation",
        ],
        "log_format_version" => "1.0",
    )
    META = Dict(
        "author" => "Sasha Petrenko",
        "complexity" => "1-low",
        "difficulty" => "2-medium",
        "scenario_type" => "custom",
    )

    # Create the config dict
    config_dict = Dict(
        "DIR" => DIR,
        "NAME" => NAME,
        "COLS" => COLS,
        "META" => META,
    )

    # Write the config file
    DeepART.json_save(config_file, config_dict)

    # -----------------------------------------------------------------
    # SCENARIO FILE
    # -----------------------------------------------------------------

    # Build the scenario vector
    SCENARIO = []
    for ix = 1:n_tasks
        # @info "stuff:"
        # @info unique(tidata.train[ix].y)
        # @info join(unique(tidata.train[ix].y), "-")
        # Create a train step and push
        train_step = Dict(
            "type" => "train",
            "regimes" => [Dict(
                # "task" => class_labels[ix],
                "task" => join(unique(tidata.train[ix].y), "-"),
                "count" => length(tidata.train[ix].y),
            )],
        )
        push!(SCENARIO, train_step)

        # Create all test steps and push
        regimes = []
        for jx = 1:n_tasks
            local_regime = Dict(
                # "task" => class_labels[jx],
                # "task" => tidata.test[1].y[1],
                # "task" => tidata.test[1].y,
                "task" => join(unique(tidata.test[jx].y), "-"),
                "count" => length(tidata.test[jx].y),
            )
            push!(regimes, local_regime)
        end

        test_step = Dict(
            "type" => "test",
            "regimes" => regimes,
        )

        push!(SCENARIO, test_step)
    end

    # Make scenario list into a dict entry
    scenario_dict = Dict(
        "scenario" => SCENARIO,
    )

    # Save the scenario
    DeepART.json_save(scenario_file, scenario_dict)
end

"""
Generates scenarios for one dataset.
"""
function gen_scenarios(
    key::AbstractString,
    datasplit::DataSplit,
    grouping_dict::AbstractDict,
    n_max::Int=10,
)

    # If the groupings should be random, generate n_max groupings from random permutations
    if grouping_dict["random"]
        group_size = grouping_dict["group_size"]
        groupings = gen_random_groupings(datasplit, group_size, n_max)
        # Otherwise, generate all of the permutations, assuming one class per task
    else
        groupings = gen_permutation_groupings(datasplit)
    end

    cidata = DeepART.ClassIncrementalDataSplit(datasplit)
    # Iterate over every permutation
    for order in groupings
        gen_scenario_from_group(key, cidata, order)
    end

end

"""
Generates all scenarios.
"""
function gen_all_scenarios(
    datasets::Dict{String, DataSplit},
    groupings_dict::AbstractDict,
    n_max::Int=10,
)
    # Iterate over all datasets in the prescribed groupings
    # for (key, datasplit) in datasets
    for (key, grouping_subdict) in groupings_dict
        # gen_scenario(key, datasplit, groupings_dict[key],n_max)
        gen_scenarios(key, datasets[key], grouping_subdict, n_max)
    end
end
