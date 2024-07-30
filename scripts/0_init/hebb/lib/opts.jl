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
const ModelOpts = Dict{String, Any}

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

"""
All valid string options for the model.
"""
const SWITCHES = Dict{String, Any}(
    "model" => [
        "dense",
        "small_dense",
        "fuzzy",
        "conv",
        "fuzzy_new",
        "dense_new",
        "conv_new",
        "dense_spec",
        "fuzzy_spec",
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

function _sanitize_floats!(opts::ModelOpts)
    # Convert to Float32
    for key in keys(opts)
        if key in TO_FLOAT32
            opts[key] = Float32(opts[key])
        end
    end
    return
end

function _sanitize_flux!(opts::ModelOpts)
    # Convert to Flux functions
    for key in keys(opts)
        if key in TO_FLUX
            # opts[key] = eval("Flux." * opts[key])
            opts[key] = getfield(Flux, Symbol(opts[key]))
        end
    end
    return
end

function _sanitize_switches!(opts::ModelOpts)
    # Check for valid switches
    for key in keys(SWITCHES)
        if !(opts[key] in SWITCHES[key])
            error("Invalid value for key $key: $(opts[key])")
        end
    end
    return
end

function _wavelet_correction(opts::ModelOpts)
    # Wavelet correction
    if opts["beta_rule"] == "wavelet"
        n_samples = 10000
        plot_range = -0.5

        x = range(-plot_range, plot_range, length=n_samples)
        y = Hebb.ricker_wavelet.(x, opts["sigma"])

        min_y = minimum(y)
        inds = findall(x -> x == min_y, y)
        # @info x[inds]
        opts["wavelet_offset"] = Float32(abs(x[inds][1]))
    end

    return
end

"""
Sanitize the options dictionary.
"""
function sanitize_opts_v1!(opts::SimOpts)
    _sanitize_floats!(opts[MODEL_OPTS])

    _sanitize_flux!(opts[MODEL_OPTS])

    _sanitize_switches!(opts[MODEL_OPTS])

    _wavelet_correction(opts[MODEL_OPTS])
end


"""
Sanitize the options dictionary.
"""
function sanitize_opts_v2!(opts::SimOpts)
    # for (_, value) in opts["block_opts"]["blocks"]
        # local_model_opts = value["model_opts"]
    for model_opts in opts["block_opts"]["blocks"]
        _sanitize_floats!(model_opts)
        _sanitize_flux!(model_opts)
        _sanitize_switches!(model_opts)
        _wavelet_correction(model_opts)
    end
end

function load_opts(name::AbstractString)::SimOpts
    # Load the options
    opts = YAML.load_file(
        joinpath(@__FILE__, "..", "..", "opts", name);
        dicttype=SimOpts
    )

    if opts["opts_version"] == 1
        # Santize the options and do post-processing
        sanitize_opts_v1!(opts)
    elseif opts["opts_version"] == 2
        sanitize_opts_v2!(opts)
    else
        error("Invalid opts_version: $(opts["opts_version"])")
    end

    # Return the options
    return opts
end

# """
# Load the options from a YAML file.
# """
# function load_opts(name::AbstractString)::SimOpts
# end