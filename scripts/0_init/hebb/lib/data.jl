
# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

const DATASETS = Dict(
    "high_dimensional" => [
        "fashionmnist",
        "mnist",
        "usps",
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

function get_data(opts::SimOpts)
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
