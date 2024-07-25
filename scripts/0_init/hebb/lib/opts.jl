"""
    opts.jl

# Description
Definitions for loading options.
"""

# -----------------------------------------------------------------------------
# TYPE ALIASES
# -----------------------------------------------------------------------------

"""
Alias for the simulation options dictionary.
"""
const SimOpts = Dict{String, Any}

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

"""
Key for the model options in the options dictionary.
"""
const MODEL_OPTS = "model_opts"

"""
Elements of model_opts that should be converted to Float32.
"""
const TO_FLOAT32 = [
    "eta",
    "beta_d",
    "sigma",
]

"""
Elements of model_opts that should be converted to Flux functions.
"""
const TO_FLUX = [
    "init",
    "middle_activation",
]

const SWITCHES = Dict{String, Any}(
    "model" => [
        "dense",
        "small_dense",
        "fuzzy",
        "conv",
        "fuzzy_new",
        "dense_new",
        "dense_spec",
        "conv_new",
    ],

    "conv_strategy" => [
        "unfold",
        "patchwise",
    ],

    "learning_rule" => [
        "hebb",
        "oja",
        "instar",
        "fuzzyart",
    ],

    # Enumeration of beta rules.
    "beta_rule" => [
        "wta",
        "contrast",
        "softmax",
        "wavelet",
        "gaussian",
    ],
)

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

"""
Sanitize the options dictionary.
"""
function sanitize_opts!(opts::SimOpts)
    # Convert to Float32
    for key in keys(opts[MODEL_OPTS])
        if key in TO_FLOAT32
            opts[MODEL_OPTS][key] = Float32(opts[MODEL_OPTS][key])
        end
    end

    # Convert to Flux functions
    for key in keys(opts[MODEL_OPTS])
        if key in TO_FLUX
            # opts[key] = eval("Flux." * opts[key])
            opts[MODEL_OPTS][key] = getfield(Flux, Symbol(opts[MODEL_OPTS][key]))
        end
    end

    # Check for valid switches
    for key in keys(SWITCHES)
        if !(opts[MODEL_OPTS][key] in SWITCHES[key])
            error("Invalid value for key $key: $(opts[MODEL_OPTS][key])")
        end
    end

    # Wavelet correction
    if opts["model_opts"]["beta_rule"] == "wavelet"
        n_samples = 10000
        plot_range = -0.5

        x = range(-plot_range, plot_range, length=n_samples)
        y = Hebb.ricker_wavelet.(x, opts["model_opts"]["sigma"])

        min_y = minimum(y)
        inds = findall(x -> x == min_y, y)
        # @info x[inds]
        opts["model_opts"]["wavelet_offset"] = Float32(abs(x[inds][1]))
    end
end

"""
Load the options from a YAML file.
"""
function load_opts(name::AbstractString)::SimOpts
    # Load the options
    opts = YAML.load_file(
        joinpath(@__FILE__, "..", "..", "opts", name);
        dicttype=Dict{String, Any}
    )

    # Santize the options and do post-processing
    sanitize_opts!(opts)

    # Return the options
    return opts
end
