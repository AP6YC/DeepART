"""
    dist.jl

# Description
Distributed experiment drivers.
"""

"""
Common save function for simulations.

# Arguments
"""
function save_sim(
    dir_func::Function,
    d::AbstractDict,
    fulld::AbstractDict,
)
    # Point to the correct save file for the results dictionary
    sim_save_name = dir_func(savename(
        d,
        "jld2";
        digits=4,
        # ignores=[
        #     "rng_seed",
        #     "m",
        # ],
    ))

    # Log completion of the simulation
    @info "Worker $(myid()): saving to $(sim_save_name)"

    # DrWatson function to save the results with an additional tag entry
    tagsave(sim_save_name, fulld)

    # Empty return
    return
end

"""
Trains and classifies a START module on the provided statements.

# Arguments
"""
function tt_dist(
    d::AbstractDict,
    # data::DataSplit,
    dir_func::Function,
    # opts::AbstractDict,
)
    # Initialize the random seed at the beginning of the experiment
    Random.seed!(d["rng_seed"])

    # Initialize the ART module
    art = if d["m"] == "SFAM"
        ART.SFAM(
            rho=d["rho"],
        )
    end

    # Load the dataset with the provided options
    data = load_one_dataset(
        d["dataset"],
        flatten=true,
        n_train=d["n_train"],
        n_test=d["n_test"],
    )

    # Process the statements
    results = tt_basic!(art, data, 1000, 1000)

    # Compute the confusion while we have the true y for this dataset shuffle
    n_classes = length(unique(data.test.y))
    conf = get_confusion_matrix(
        data.test.y[1:n_test],
        results["y_hats"],
        n_classes,
    )

    # Copy the input sim dictionary
    fulld = deepcopy(d)

    # Add entries for the results
    fulld["perf"] = results["perf"]
    fulld["conf"] = conf

    # Save the results
    save_sim(dir_func, d, fulld)

    # Explicitly empty return
    return
end
