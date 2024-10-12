"""
    data.jl

# Description
Definitions of utilities for loading data.
"""

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

const DATASETS = Dict(
    "high_dimensional" => [
        "fashionmnist",
        "mnist",
        "usps",
        "cifar10",
        "cifar100",
    ],
    "low_dimensional" => [
        "wine",
        "iris",
        "wave",
        "face",
        "flag",
        "halfring",
        "moon",
        "ring",
        "spiral",
    ]
)

function _get_data_v1(opts::SimOpts)
    data = if opts["dataset"] in DATASETS["high_dimensional"]
        DeepART.load_one_dataset(
            opts["dataset"],
            n_train=opts["n_train"],
            n_test=opts["n_test"],
            # flatten=opts["flatten"],
            flatten = !(opts["model_opts"]["model"] in ["conv", "conv_new"]),
        )
    else
        DeepART.load_one_dataset(
            opts["dataset"],
        )
    end
    return data
end

function _get_data_v2(opts::SimOpts)
    # local_opts = opts["sim_opts"]
    local_opts = get_simopts(opts)

    to_flatten = !(get_model_opts(opts, 1)["model"] in ["conv", "conv_new", "lenet"])

    data = if local_opts["dataset"] in DATASETS["high_dimensional"]
        DeepART.load_one_dataset(
            local_opts["dataset"],
            n_train = local_opts["n_train"],
            n_test = local_opts["n_test"],
            flatten = to_flatten,
        )
    else
        DeepART.load_one_dataset(
            local_opts["dataset"],
        )
    end

    return data
end

function get_data(opts::SimOpts)
    data = if opts[OPTS_VERSION] == 1
        _get_data_v1(opts)
    elseif opts[OPTS_VERSION] == 2
        _get_data_v2(opts)
    else
        error("Invalid version: $(opts[OPTS_VERSION])")
    end

    return data
end
