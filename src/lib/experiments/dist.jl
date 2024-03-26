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
        # digits=4,
        # ignores=[
        #     "rng_seed",
        #     "m",
        # ],
    ))

    # Log completion of the simulation
    # @info "Worker $(myid()): saving to $(sim_save_name)"
    @info "Saving to $(sim_save_name)"

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

    # Load the dataset with the provided options
    isconv = !(d["m"] == "DeepARTConv")
    data = load_one_dataset(
        d["dataset"],
        flatten=isconv,
        n_train=d["n_train"],
        n_test=d["n_test"],
    )

    n_input = size(data.train.x, 1)

    # Initialize the ART module
    art = if d["m"] == "SFAM"
        local_art = ART.SFAM(
            rho=d["rho"],
        )
        local_art.config = ART.DataConfig(0.0, 1.0, n_input)
        local_art
    elseif d["m"] == "DeepARTDense"
        # Model definition
        head_dim = 2048
        model = Flux.@autosize (n_input,) Chain(
            DeepART.CC(),
            Dense(_, 256, sigmoid, bias=false),
            # Dense(_, 128, sigmoid, bias=false),
            DeepART.CC(),
            # Dense(_, 128, sigmoid, bias=false),
            # DeepART.CC(),
            # Dense(_, 64, sigmoid, bias=false),
            # DeepART.CC(),
            Dense(_, head_dim, sigmoid, bias=false),
        )
        local_art = DeepART.ARTINSTART(
            model,
            head_dim=head_dim,
            beta=0.01,
            softwta=true,
            gpu=true,
            rho=d["rho"],
        )
        local_art
    elseif d["m"] == "DeepARTConv"
        # Model definition
        head_dim = 2048

        size_tuple = (size(data.train.x)[1:3]..., 1)
        # size_tuple = (28, 28, 1, 1)
        conv_model = DeepART.get_rep_conv(size_tuple, head_dim)
        local_art = DeepART.ARTINSTART(
            conv_model,
            head_dim=head_dim,
            # beta=0.01,
            beta=0.1,
            # rho=0.65,
            rho=0.65,
            update="art",
            softwta=true,
            gpu=true,
        )

        local_art
    else
        error("Unknown model: $(d["m"])")
    end

    # Process the statements
    # @info "Worker $(myid()): training $(d["m"]) on $(d["dataset"]) with seed $(d["rng_seed"])"
    @info "Training $(d["m"]) on $(d["dataset"]) with seed $(d["rng_seed"])"
    results = tt_basic!(art, data, 1000, 1000)

    # Compute the confusion while we have the true y for this dataset shuffle
    n_classes = length(unique(data.test.y))

    conf = get_normalized_confusion(
        data.test.y,
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
