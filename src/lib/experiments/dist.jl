"""
    dist.jl

# Description
Distributed experiment drivers.
"""

function get_sim_name(
    d::AbstractDict,
    dir_func::Function,
)
    # Point to the correct save file for the results dictionary
    sim_save_name = dir_func(savename(
        d,
        "jld2";
        # digits=4,
        ignores=[
            "display",
        #     "rng_seed",
        #     "m",
        ],
    ))
    return sim_save_name
end

"""
Common save function for simulations.

# Arguments
"""
function save_sim(
    dir_func::Function,
    d::AbstractDict,
    fulld::AbstractDict,
)
    # # Point to the correct save file for the results dictionary
    # sim_save_name = dir_func(savename(
    #     d,
    #     "jld2";
    #     # digits=4,
    #     ignores=[
    #         "display",
    #     #     "rng_seed",
    #     #     "m",
    #     ],
    # ))
    sim_save_name = get_sim_name(d, dir_func)

    # Log completion of the simulation
    # @info "Worker $(myid()): saving to $(sim_save_name)"
    @info "Saving to $(sim_save_name)"

    # DrWatson function to save the results with an additional tag entry
    tagsave(sim_save_name, fulld)

    # Empty return
    return
end

"""
Generates a new grouping from the classes vector and the group size, assuming that the length of the classes is evenly divisible by `group_size`.
"""
function get_dist_grouping(
    classes::Vector{Int},
    group_size::Int,
)
    n_classes = length(classes)
    return [classes[ix : ix + group_size - 1] for ix = 1:group_size:n_classes]
end

"""
Generates a random grouping from the provided dataset and selected group size.
"""
function random_dist_grouping(
    data::DataSplit,
    group_size::Int,
)
    classes = unique(data.train.y)
    groupings = shuffle(classes)

    return get_grouping(groupings, group_size)
end


"""
Trains and classifies a START module on the provided statements.

# Arguments
"""
function tt_dist(
    d::AbstractDict,
    # data::DataSplit,
    dir_func::Function;
    # opts::AbstractDict,
    rerun::Bool=true
)
    # Initialize the random seed at the beginning of the experiment
    Random.seed!(d["rng_seed"])

    # Idempotent check for sim
    if !rerun
        sim_save_name = get_sim_name(d, dir_func)
        if isfile(sim_save_name)
            @warn "Simulation $(sim_save_name) already exists. Skipping."
            return
        end
    end

    # Load the dataset with the provided options
    isconv = (d["m"] == "DeepARTConv") || (d["m"] == "DeepARTConvHebb") || (d["m"] == "DeepARTConvBlock")
    data = load_one_dataset(
        d["dataset"],
        flatten=!isconv,
        gray=true,
        # gray=!isconv,
        n_train=d["n_train"],
        n_test=d["n_test"],
    )

    # n_input = size(data.train.x, 1)

    # Construct the module from the options
    art = get_module_from_options(d, data.train)

    # Process the statements
    results = if d["scenario"] == "task-homogenous"
        @info "Task-Homogenous: Training $(d["m"]) on $(d["dataset"]) with seed $(d["rng_seed"])"
        tt_basic!(art, data, display=d["display"])
    else
        @info "Task-Incremental: Training $(d["m"]) on $(d["dataset"]) with seed $(d["rng_seed"])"
        grouping = random_dist_grouping(data, d["group_size"])
        tidata = ClassIncrementalDataSplit(data)
        tidata = TaskIncrementalDataSplit(tidata, grouping)
        tt_inc!(art, tidata, data, display=d["display"])
    end

    # Compute the confusion while we have the true y for this dataset shuffle
    n_classes = length(unique(data.test.y))

    # Computes the confusion matrix
    conf = get_normalized_confusion(
        data.test.y,
        results["y_hats"],
        n_classes,
    )

    # Copy the input sim dictionary
    fulld = deepcopy(d)

    # Add entries for the results
    # fulld["n_cat"] = art.n_categories
    # n_cat = if haskey(results, "n_cat")
    #     results
    # else
    #     art.n_cat
    # end
    n_cat = get(results, "n_cat", 0)
    # fulld["n_cat"] = results["n_cat"]
    fulld["n_cat"] = n_cat
    fulld["perf"] = results["perf"]
    fulld["conf"] = conf

    # Save the results
    save_sim(dir_func, d, fulld)

    # Explicitly empty return
    return
end
